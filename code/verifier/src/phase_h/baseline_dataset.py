"""Phase H — supervised baseline dataset.

Constructs (label, C, E) examples for binary classification:
    label=1 (positive): real (C_i, E_i) — both at Phase G eval resolution.
    label=0 (negative class A, shuffled): (C_i, E_j), |i-j|>60, cross-session OK.
    label=0 (negative class B, perturbed): (C_i, E_i_perturbed) using train-pool
        perturbations (Type 1, 2, 4, 6 general).

CRITICAL: ALL E inputs go through xof_perturb.render_E_for_phase_h to maintain
identical render path between positives and negatives. The CNN must not learn
"PNG-loaded artifact" vs "tensor-rendered artifact" as a shortcut.

Source rows: D2 + V10 Phase G training blocks (i.e. the rows NOT in held-out).

Negative composition: 50% shuffled, 50% perturbed.
"""
from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from phase_g.diffusion_diagnostic_dataset import (  # noqa: E402
    _crop_and_resize_C, _load_packed_cfa_float01, train_rows,
)
from phase_g.xof_perturb import (  # noqa: E402
    render_E_for_phase_h, load_chain_log, TRAIN_POOL_LABELS,
)


SESSIONS_AVAILABLE = ("D2", "V10")


def _stable_session_seed_int(seed: int, label: str, *parts) -> int:
    """Deterministic seed derived from (seed, label, *parts) via blake3-style
    digest. Avoids Python's randomized hash() on strings."""
    h = hashlib.sha256()
    h.update(f"{seed}|{label}".encode())
    for p in parts:
        h.update(b"|"); h.update(str(p).encode())
    return int.from_bytes(h.digest()[:8], "big") & ((1 << 63) - 1)


class PhaseHBaselineDataset(Dataset):
    """One sample = (input_tensor, label).
        input_tensor: (7, 768, 1024) — 4 CFA + 3 RGB
        label: 1 (positive) or 0 (negative)

    Per `__getitem__`, deterministically decides:
      - 50% chance positive: real (C_i, E_i_identity).
      - 25% chance shuffled negative: (C_i, E_j_identity), |i-j|>60, cross-session.
      - 25% chance perturbed negative: (C_i, E_i_perturbed) with random
        train-pool label.
    Decision RNG seeded by (epoch_seed, sample_idx).
    """

    def __init__(self,
                 session_dirs: dict[str, Path],
                 stride: int = 3,
                 epoch_seed: int = 0,
                 lag_min: int = 60):
        self.session_dirs = {k: Path(v) for k, v in session_dirs.items()}
        self.stride = stride
        self.epoch_seed = epoch_seed
        self.lag_min = lag_min

        # Per-session list of valid training rows (in Phase G train blocks,
        # post stride-3). Filter to rows present in chain_log (any row missing
        # from chain log can't render real OR perturbed E).
        self.session_rows: dict[str, list[int]] = {}
        self.chain_logs: dict[str, dict[int, str]] = {}
        for sess, sd in self.session_dirs.items():
            tr = train_rows(sess, stride=stride)
            chain = load_chain_log(sd)
            self.chain_logs[sess] = chain
            valid = [r for r in tr if r in chain]
            self.session_rows[sess] = valid

        # Flat (session, row) list for indexing
        self.flat: list[tuple[str, int]] = []
        for sess, rows in self.session_rows.items():
            for r in rows:
                self.flat.append((sess, r))

    def __len__(self) -> int:
        return len(self.flat)

    def _load_C(self, session: str, row: int) -> torch.Tensor:
        sd = self.session_dirs[session]
        return _crop_and_resize_C(_load_packed_cfa_float01(
            sd / "Recordings" / f"frame_{row:06d}.raw"))

    def _render_E(self, session: str, row: int, condition_label: str,
                  donor_session: str | None = None,
                  donor_row: int | None = None) -> torch.Tensor:
        chain_log = self.chain_logs[session]
        donor_chain_log = (self.chain_logs[donor_session]
                           if donor_session is not None else None)
        return render_E_for_phase_h(
            session=session,
            target_frame_id=row,
            condition_label=condition_label,
            chain_log=chain_log,
            donor_chain_log=donor_chain_log,
            donor_frame_id=donor_row,
            device="cpu",
        )

    def _pick_shuffled_partner(self, session: str, row: int,
                               rng: random.Random) -> tuple[str, int]:
        """Pick a shuffled partner (session', row') with cross-session allowed
        and |row - row'| > lag_min if same session."""
        # Cross-session pairing OK; same-session requires |gap| > lag_min.
        sessions = list(self.session_rows.keys())
        for _ in range(32):
            psess = rng.choice(sessions)
            prow = rng.choice(self.session_rows[psess])
            if psess != session:
                return psess, prow
            if abs(prow - row) > self.lag_min:
                return psess, prow
        # Last resort: deterministic fallback (rotated row) within same session
        n = len(self.session_rows[session])
        idx = self.session_rows[session].index(row)
        return session, self.session_rows[session][(idx + n // 2) % n]

    def _pick_perturbation(self, rng: random.Random) -> str:
        return rng.choice(TRAIN_POOL_LABELS)

    def _pick_intra_session_donor(self, session: str, row: int,
                                  rng: random.Random) -> int:
        """For type 4/5/6 perturbations within Phase H training, pick a donor
        row in the same session with |gap| > lag_min."""
        rows = self.session_rows[session]
        for _ in range(32):
            cand = rng.choice(rows)
            if abs(cand - row) > self.lag_min:
                return cand
        # Last resort
        idx = rows.index(row)
        return rows[(idx + len(rows) // 2) % len(rows)]

    def _build_sample(self, idx: int, retry_count: int = 0) -> dict:
        """Internal sample builder with bounded retry on render failure.

        Per Codex audit 2026-05-02: a render failure in a worker would
        crash the DataLoader and kill training. Wrap in try/except and
        fall back to a SAFE positive (identity render — proven bit-exact)
        on any unexpected exception. Bounded retry count prevents infinite
        recursion if data is fundamentally broken (which would manifest in
        loss-not-decreasing and trip the overfit canary).
        """
        session, row = self.flat[idx]
        seed = _stable_session_seed_int(self.epoch_seed, "phase_h_sample",
                                        session, row, idx, retry_count)
        rng = random.Random(seed)
        C = self._load_C(session, row)

        # Decide class
        u = rng.random()
        try:
            if u < 0.5:
                label = 1
                E = self._render_E(session, row, "identity")
                meta = {"class": "positive", "condition": "identity",
                        "donor_session": None, "donor_row": None,
                        "partner_session": None, "partner_row": None}
            elif u < 0.75:
                label = 0
                psess, prow = self._pick_shuffled_partner(session, row, rng)
                E = self._render_E(psess, prow, "identity")
                meta = {"class": "shuffled", "condition": "identity",
                        "donor_session": None, "donor_row": None,
                        "partner_session": psess, "partner_row": prow}
            else:
                label = 0
                cond = self._pick_perturbation(rng)
                from phase_g.xof_perturb import _spec_by_label
                spec = _spec_by_label(cond)
                if spec.needs_donor():
                    donor_row = self._pick_intra_session_donor(session, row, rng)
                    E = self._render_E(session, row, cond,
                                       donor_session=session, donor_row=donor_row)
                else:
                    donor_row = None
                    E = self._render_E(session, row, cond)
                meta = {"class": "perturbed", "condition": cond,
                        "donor_session": session if donor_row is not None else None,
                        "donor_row": donor_row,
                        "partner_session": None, "partner_row": None}
        except Exception as e:
            # Render failure (e.g. PNG missing, donor row bad). Log once and
            # fall back to a safe positive identity render of the same row.
            # If even THAT fails, escalate up; the parity gate would have caught
            # systemic identity-render failure already.
            if retry_count >= 3:
                # Last resort: re-raise (training will halt; better than wrong data)
                raise
            print(f"[dataset WARN] render failure at idx={idx} session={session} "
                  f"row={row} u={u:.3f}: {e!r} — falling back to identity positive",
                  flush=True)
            label = 1
            E = self._render_E(session, row, "identity")  # if this fails, parity gate should have caught it
            meta = {"class": "positive", "condition": "identity_fallback",
                    "donor_session": None, "donor_row": None,
                    "partner_session": None, "partner_row": None}

        x = torch.cat([C, E], dim=0)  # (7, 768, 1024)
        return {
            "x": x,
            "label": float(label),
            "session": session,
            "row": row,
            **{f"meta_{k}": (v if v is not None else "") for k, v in meta.items()},
        }

    def __getitem__(self, idx: int) -> dict:
        return self._build_sample(idx, retry_count=0)


def collate_phase_h(batch: list[dict]) -> dict:
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.float32),
        "session": [b["session"] for b in batch],
        "row": torch.tensor([b["row"] for b in batch]),
        "meta_class": [b["meta_class"] for b in batch],
        "meta_condition": [b["meta_condition"] for b in batch],
    }

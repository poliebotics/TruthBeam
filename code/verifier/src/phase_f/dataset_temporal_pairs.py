"""Phase F dataset: (C_source, E_source, E_target, C_target) tuples for
target-swap supervision.

For each training sample at row t:
  source_t  = t - k   (some k frames earlier)
  target_t  = t       (matched ground truth)

Returns:
  C_source: capture at row source_t  → packed CFA (4, 2300, 2660) float [0,1]
  E_source: emission at row source_t → (3, 1080, 1920) float [0,1]
  E_target: emission at row target_t → (3, 1080, 1920) float [0,1]
  C_target: capture at row target_t  → packed CFA (4, 2300, 2660) float [0,1]

Augmentation (BayerRG-aware, applied identically to source AND target so the
pair stays consistent):
  - 50% horizontal flip with G1 ↔ G2 swap (preserves Bayer pattern)
  - even-aligned random crop with cropped emission resize to fixed size

Splits — match Phase E:
  D2  train [0, 4194)   val [5394, 5992)
  V10 train [0, 2500)   val [2993, 3743)

Configurable k via constructor `k` parameter; multiple k values supported via
`k_choices` list (sampled uniformly per __getitem__).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from data.emission_dataset import load_emission_at, EMISSION_H_NATIVE, EMISSION_W_NATIVE
from phase_f.cfa_roundtrip import bayer_rg8_to_packed_cfa, EXPECTED_BYTES, HALF_H, HALF_W


def _load_packed_cfa_float01(raw_path: Path) -> torch.Tensor:
    """Load .raw → packed CFA (4, 2300, 2660) float32 in [0, 1]."""
    raw = raw_path.read_bytes()
    cfa = bayer_rg8_to_packed_cfa(raw)
    return torch.from_numpy(cfa.astype(np.float32) / 255.0)


def _flip_pair_horizontal(cfa: torch.Tensor, em: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Horizontal flip with BayerRG-aware G1↔G2 swap.

    BayerRG layout has R at (0,0), G1 at (0,1), G2 at (1,0), B at (1,1) of each
    2×2 block. After horizontal flip:
      - The 2×2 block visited from right-to-left becomes (G1, R) on top row
        and (B, G2) on bottom row. Equivalently, flipping the whole image
        horizontally and swapping G1 ↔ G2 channels recovers the BayerGB
        layout, which is rotationally equivalent to RGGB modulo channel
        labelling. To keep the CHANNEL ORDER unchanged (R, G1, G2, B) AND
        the visual content flipped, we flip horizontally then swap G1<->G2.
    """
    cfa_flipped = torch.flip(cfa, dims=[-1])
    cfa_swapped = cfa_flipped.clone()
    cfa_swapped[1] = cfa_flipped[2]
    cfa_swapped[2] = cfa_flipped[1]
    em_flipped = torch.flip(em, dims=[-1])
    return cfa_swapped, em_flipped


class TemporalPairDataset(Dataset):
    """Returns (C_source, E_source, E_target, C_target) for source=t-k, target=t.

    Args:
      session_dir: e.g. /path/to/poliebotics_phase_b/d2 (must have Recordings/ + derived/Emissions/)
      rows: list of valid target_t rows (for which both target_t-k and target_t exist)
      k_choices: list of allowable k values; sampled uniformly per item
      augment: bool, enable hflip + crop
      seed: rng seed for augmentation
    """

    def __init__(
        self,
        session_dir: Path,
        rows: Sequence[int],
        k_choices: Sequence[int] = (1,),
        emission_h: int = EMISSION_H_NATIVE,
        emission_w: int = EMISSION_W_NATIVE,
        augment: bool = False,
        seed: int = 0,
    ):
        self.session_dir = Path(session_dir)
        self.recordings = self.session_dir / "Recordings"
        self.emissions = self.session_dir / "derived" / "Emissions"
        self.rows = sorted(int(r) for r in rows)
        self.k_choices = list(k_choices)
        if not self.k_choices:
            raise ValueError("k_choices must be non-empty")
        self.emission_h = emission_h
        self.emission_w = emission_w
        self.augment = augment
        self._seed = seed

    def __len__(self) -> int:
        return len(self.rows)

    def _can_load(self, t: int) -> bool:
        return ((self.recordings / f"frame_{t:06d}.raw").exists()
                and (self.emissions / f"tile_{t:06d}.png").exists())

    def __getitem__(self, idx: int) -> dict:
        target_t = self.rows[idx]
        rng = random.Random((self._seed, target_t))
        # Pick k that produces a valid source frame
        for _ in range(8):
            k = rng.choice(self.k_choices)
            source_t = target_t - k
            if source_t >= 0 and self._can_load(source_t):
                break
        else:
            # Fallback: smallest k
            k = min(self.k_choices)
            source_t = max(0, target_t - k)

        C_target = _load_packed_cfa_float01(self.recordings / f"frame_{target_t:06d}.raw")
        C_source = _load_packed_cfa_float01(self.recordings / f"frame_{source_t:06d}.raw")
        E_target = load_emission_at(self.emissions / f"tile_{target_t:06d}.png",
                                     self.emission_h, self.emission_w)
        E_source = load_emission_at(self.emissions / f"tile_{source_t:06d}.png",
                                     self.emission_h, self.emission_w)

        if self.augment:
            do_flip = rng.random() < 0.5
            if do_flip:
                C_source, E_source = _flip_pair_horizontal(C_source, E_source)
                C_target, E_target = _flip_pair_horizontal(C_target, E_target)
            # Even-aligned crop: random crop of size (HALF_H - 64, HALF_W - 64) on packed CFA
            # with corresponding emission crop. Skip implementation in F-A scaffolding;
            # full-frame for now. TODO: revisit if memory pressures require crops.

        return {
            "C_source": C_source,    # (4, 2300, 2660) float [0,1]
            "E_source": E_source,    # (3, 1080, 1920) float [0,1]
            "E_target": E_target,    # (3, 1080, 1920) float [0,1]
            "C_target": C_target,    # (4, 2300, 2660) float [0,1]
            "source_t": source_t,
            "target_t": target_t,
            "k": target_t - source_t,
        }


def collate_temporal_pairs(batch: list[dict]) -> dict:
    return {
        "C_source": torch.stack([b["C_source"] for b in batch]),
        "E_source": torch.stack([b["E_source"] for b in batch]),
        "E_target": torch.stack([b["E_target"] for b in batch]),
        "C_target": torch.stack([b["C_target"] for b in batch]),
        "source_t": torch.tensor([b["source_t"] for b in batch]),
        "target_t": torch.tensor([b["target_t"] for b in batch]),
        "k":        torch.tensor([b["k"] for b in batch]),
    }


def split_rows(session: str, role: str) -> list[int]:
    """Phase F-aligned splits, post-2026-04-29 three-role contract.

    'val' here = selection_gate_normal slice (used by causality probes /
    checkpoint selection). The threshold_calibration_normal slice is
    NEVER returned here — it's reserved for verifier-side held-out binder
    threshold setting and must not be touched by Phase F training scripts.
    See `experiments/phase_f_prep/three_role_split.md` for the contract.

    session in {"D2", "V10"}; role in {"train", "val"}."""
    splits = {
        ("D2", "train"):  list(range(0, 4194)),
        ("D2", "val"):    list(range(4792, 5392)),    # selection_gate_normal
        ("V10", "train"): list(range(0, 2500)),
        ("V10", "val"):   list(range(2993, 3368)),    # selection_gate_normal
    }
    return splits[(session, role)]

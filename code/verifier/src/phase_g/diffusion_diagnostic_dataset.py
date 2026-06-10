"""Phase G — dataset for the diffusion diagnostic.

Three modes:
  - real:               (C, E) from same row. Standard pairing.
  - shuffled:           C from row r, E from row perm[r] where |r - perm[r]| > 60.
                        Permutation is fixed (seed) and within-session (D2 stays
                        with D2, V10 stays with V10).
  - synthetic_positive: C is the real C plus an E-dependent perturbation:
                        C' = clip(C + α · M · blur(E_repeated_to_4ch, σ=8))
                        α ∈ {0.05, 0.1, 0.2}, sampled randomly per item.

Preprocessing (LOCKED — operator spec):
  - C: BayerRG8 raw 5320×4600 → packed 4-channel 2300×2660 → crop
       y=[0:1704], x=[155:2433] → resize to 768×1024 via area interpolation.
  - E: 1080×1920 RGB → resize to 768×1024 (no crop).
  - No black level correction (BL=0); divide by 255 to map to [0,1].
       The "fixed black level subtraction, fixed normalization" clause in the
       spec is satisfied by BL=0 + /255, identical to Phase E/F preprocessing.
  - Same crop applied to D2 and V10 (rig_hash 3984699a504fb798 confirmed
       identical for both via verification_bundle.pretty.json).

Held-out blocks (per spec):
  D2 (5992 frames):  3 blocks of 400 at ~25/50/75% with 60-frame guards.
  V10 (3743 frames): 2 blocks of 250 at ~33/66% with 60-frame guards.

Stride-3 sampling: training rows are decimated by 3 to reduce temporal
autocorrelation (each `__getitem__` index maps to a stride-3 row).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from phase_f.cfa_roundtrip import bayer_rg8_to_packed_cfa  # noqa: E402
from data.emission_dataset import load_emission_at  # noqa: E402


# ----- LOCKED preprocessing constants -----
PACKED_H, PACKED_W = 2300, 2660           # packed CFA
CROP_Y0, CROP_Y1 = 0, 1704                 # crop in packed-CFA coords
CROP_X0, CROP_X1 = 155, 2433               # crop in packed-CFA coords
TARGET_H, TARGET_W = 768, 1024             # resize target for both C and E
EMISSION_NATIVE_H, EMISSION_NATIVE_W = 1080, 1920
BLACK_LEVEL = 0.0                          # documented BL=0
NORM_DIV = 255.0                           # raw bytes / 255 → [0, 1]


# ----- Held-out block definitions -----
# Centers placed at the requested fractions; symmetric block widths.
EVAL_BLOCKS = {
    "D2": [(1298, 1698),  (2796, 3196),  (4294, 4694)],   # 400-frame blocks at ~25/50/75%
    "V10": [(1110, 1360), (2345, 2595)],                  # 250-frame blocks at ~33/66%
}
GUARD = 60                                                 # frames excluded around each block


def session_total(name: str) -> int:
    return {"D2": 5992, "V10": 3743}[name]


def held_out_rows(session: str) -> list[int]:
    """Rows in the held-out blocks (used for evaluation)."""
    rows = []
    for a, b in EVAL_BLOCKS[session]:
        rows.extend(range(a, b))
    return sorted(rows)


def train_rows(session: str, stride: int = 3) -> list[int]:
    """Training rows = full session minus blocks ± GUARD, then strided."""
    total = session_total(session)
    excluded = set()
    for a, b in EVAL_BLOCKS[session]:
        for r in range(max(0, a - GUARD), min(total, b + GUARD)):
            excluded.add(r)
    valid = [r for r in range(total) if r not in excluded]
    return valid[::stride]


def build_shuffle_permutation(rows: Sequence[int], min_gap: int, seed: int) -> dict[int, int]:
    """Return a permutation perm[r] = r' such that |r - r'| >= min_gap.

    Uses a deterministic rotation by `len(rows) // 2 + (seed % rotation_max)`,
    where rotation_max is constrained so the offset is always large enough to
    guarantee `|rows[i] - rows[(i + offset) % N]| >= min_gap` for all i.

    For the diagnostic's training rows (post-stride-3 over a session of
    several thousand frames), rotating by N//2 produces gaps of roughly
    session_length/2 (>> 60 always). We add a small seeded perturbation
    around N//2 for variability without risking gap violation.

    Guaranteed: every (r, perm[r]) pair satisfies |r - perm[r]| >= min_gap.
    Raises if the input is too small to satisfy the constraint.
    """
    rows = sorted(rows)
    n = len(rows)
    if n < 2:
        raise ValueError(f"need >= 2 rows for shuffle, got {n}")

    # Find the smallest valid rotation offset that guarantees the gap.
    # For sorted rows, |rows[i] - rows[(i+k) % n]| is monotonically non-decreasing
    # in k for k in [0, n//2], then symmetric. We try offsets near n//2 + perturbation.
    rng = random.Random(seed)
    perturb = rng.randint(-n // 4, n // 4)
    base_offset = n // 2 + perturb
    base_offset = max(1, min(n - 1, base_offset))

    # Verify the chosen offset gives the required gap across ALL i; if not,
    # fall back to n//2 exactly (always at least roughly half the row span).
    def offset_ok(off: int) -> bool:
        for i in range(n):
            if abs(rows[i] - rows[(i + off) % n]) < min_gap:
                return False
        return True

    if not offset_ok(base_offset):
        if not offset_ok(n // 2):
            # Try widening offsets
            for cand in range(1, n):
                if offset_ok(cand):
                    base_offset = cand
                    break
            else:
                raise ValueError(
                    f"cannot find rotation offset satisfying min_gap={min_gap} "
                    f"on {n} rows; the row set is too dense or the gap requirement "
                    f"is unsatisfiable"
                )
        else:
            base_offset = n // 2

    return {rows[i]: rows[(i + base_offset) % n] for i in range(n)}


# Stable per-session seed offsets to avoid Python's randomized string hashing.
SESSION_SEED_OFFSET = {"D2": 0, "V10": 1}


# ----- Loaders -----

def _load_packed_cfa_float01(raw_path: Path) -> torch.Tensor:
    """BayerRG8 raw → packed CFA (4, 2300, 2660) float32 in [0, 1]."""
    raw = raw_path.read_bytes()
    cfa = bayer_rg8_to_packed_cfa(raw)
    return torch.from_numpy(cfa.astype(np.float32) / NORM_DIV - BLACK_LEVEL)


def _crop_and_resize_C(packed_cfa: torch.Tensor) -> torch.Tensor:
    """packed_cfa (4, 2300, 2660) → cropped & resized (4, 768, 1024) via area interp."""
    cropped = packed_cfa[:, CROP_Y0:CROP_Y1, CROP_X0:CROP_X1]  # (4, 1704, 2278)
    # area interpolation (downsample, anti-aliased average pooling)
    out = F.interpolate(cropped.unsqueeze(0), size=(TARGET_H, TARGET_W),
                        mode="area").squeeze(0)
    return out  # (4, 768, 1024)


def _resize_E_to_target(emission: torch.Tensor) -> torch.Tensor:
    """E (3, em_h, em_w) → (3, 768, 1024) via area interpolation."""
    out = F.interpolate(emission.unsqueeze(0), size=(TARGET_H, TARGET_W),
                        mode="area").squeeze(0)
    return out


# ----- Dataset class -----

class DiffusionDiagnosticDataset(Dataset):
    """Returns (C, E) pairs at TARGET resolution (768×1024) per the spec.

    Args:
      session_dirs: dict mapping session name → session dir path.
                    Must contain at least one of {"D2", "V10"}.
      mode: "real" | "shuffled" | "synthetic_positive"
      role: "train" | "eval"
      stride: stride for training row decimation (default 3 per spec; ignored for eval)
      shuffle_seed: seed for the per-session permutation when mode="shuffled"
      synth_alphas: tuple of α values to sample uniformly when mode="synthetic_positive"
    """

    def __init__(
        self,
        session_dirs: dict[str, Path],
        mode: str = "real",
        role: str = "train",
        stride: int = 3,
        shuffle_seed: int = 12345,
        synth_alphas: Sequence[float] = (0.05, 0.10, 0.20),
        synth_blur_sigma: float = 8.0,
        seed: int = 0,
    ):
        if mode not in ("real", "shuffled", "synthetic_positive"):
            raise ValueError(f"unknown mode: {mode}")
        if role not in ("train", "eval"):
            raise ValueError(f"unknown role: {role}")
        self.mode = mode
        self.role = role
        self.session_dirs = {k: Path(v) for k, v in session_dirs.items()}
        self.synth_alphas = tuple(synth_alphas)
        self.synth_blur_sigma = float(synth_blur_sigma)
        self._seed = int(seed)

        # Per-session row lists at this role.
        self.session_rows: dict[str, list[int]] = {}
        self.shuffle_perm: dict[str, dict[int, int]] = {}
        for sess in self.session_dirs:
            if role == "train":
                rows = train_rows(sess, stride=stride)
            else:
                rows = held_out_rows(sess)
            self.session_rows[sess] = rows
            if mode == "shuffled":
                # Deterministic per-session permutation. Min gap = 60 (matches spec).
                # Use a stable session-seed offset (NOT Python hash) to ensure
                # all DDP ranks build identical permutations.
                self.shuffle_perm[sess] = build_shuffle_permutation(
                    rows, min_gap=60,
                    seed=shuffle_seed + SESSION_SEED_OFFSET.get(sess, 99))

        # Flat index over (session, row).
        self.flat: list[tuple[str, int]] = []
        for sess, rows in self.session_rows.items():
            for r in rows:
                self.flat.append((sess, r))

        # Synthetic perturbation needs a Gaussian blur kernel; build once.
        if mode == "synthetic_positive":
            self.blur_kernel = self._build_gaussian_kernel_2d(self.synth_blur_sigma)
        else:
            self.blur_kernel = None

    @staticmethod
    def _build_gaussian_kernel_2d(sigma: float) -> torch.Tensor:
        """Return a (1, 1, ksz, ksz) Gaussian kernel for depthwise blur."""
        ksz = max(3, int(2 * round(3 * sigma) + 1))
        ax = torch.arange(ksz, dtype=torch.float32) - (ksz - 1) / 2
        gauss = torch.exp(-(ax ** 2) / (2 * sigma * sigma))
        gauss = gauss / gauss.sum()
        kernel_2d = gauss[:, None] * gauss[None, :]
        return kernel_2d.view(1, 1, ksz, ksz)

    def __len__(self) -> int:
        return len(self.flat)

    def _load_C(self, sess: str, row: int) -> torch.Tensor:
        raw_path = self.session_dirs[sess] / "Recordings" / f"frame_{row:06d}.raw"
        return _crop_and_resize_C(_load_packed_cfa_float01(raw_path))

    def _load_E(self, sess: str, row: int) -> torch.Tensor:
        em_path = self.session_dirs[sess] / "derived" / "Emissions" / f"tile_{row:06d}.png"
        em = load_emission_at(em_path, EMISSION_NATIVE_H, EMISSION_NATIVE_W)
        return _resize_E_to_target(em)

    def _apply_synthetic_perturbation(self, C: torch.Tensor, E: torch.Tensor,
                                      rng: random.Random) -> torch.Tensor:
        """C' = clip(C + α · M · blur(E_to_4ch, σ)). M = ones (we already cropped
        to the projection region; the entire 768×1024 IS the bbox interior)."""
        alpha = rng.choice(self.synth_alphas)
        # Map E (3, H, W) → 4-ch matching CFA pattern (R, G, G, B).
        E_4ch = torch.stack([E[0], E[1], E[1], E[2]], dim=0)
        kernel = self.blur_kernel.to(E_4ch.dtype)
        ksz = kernel.shape[-1]
        pad = ksz // 2
        # Depthwise blur per channel.
        E_blur = F.conv2d(E_4ch.unsqueeze(0),
                          kernel.expand(4, 1, ksz, ksz),
                          padding=pad, groups=4).squeeze(0)
        return torch.clamp(C + alpha * E_blur, 0.0, 1.0)

    def __getitem__(self, idx: int) -> dict:
        sess, row = self.flat[idx]
        rng = random.Random(f"{self._seed}|{sess}|{row}")
        C = self._load_C(sess, row)
        if self.mode == "real":
            E = self._load_E(sess, row)
            alpha = 0.0
        elif self.mode == "shuffled":
            partner_row = self.shuffle_perm[sess][row]
            E = self._load_E(sess, partner_row)
            alpha = 0.0
        elif self.mode == "synthetic_positive":
            # 2026-05-01: moved synthetic perturbation OUT of the CPU dataloader
            # because the 49×49 depthwise CPU conv at σ=8 caused a 100× DDP
            # slowdown via stragglers (8-rank DDP at 71 s/step instead of 0.7).
            # Dataset now returns clean (C, E) plus a per-sample alpha; the
            # training step applies the perturbation on GPU.
            E = self._load_E(sess, row)
            alpha = float(rng.choice(self.synth_alphas))
        else:
            raise RuntimeError(self.mode)
        return {"C": C, "E": E, "session": sess, "row": row, "alpha": alpha}


def collate_diagnostic(batch: list[dict]) -> dict:
    return {
        "C":       torch.stack([b["C"] for b in batch]),
        "E":       torch.stack([b["E"] for b in batch]),
        "alpha":   torch.tensor([b.get("alpha", 0.0) for b in batch], dtype=torch.float32),
        "session": [b["session"] for b in batch],
        "row":     torch.tensor([b["row"] for b in batch]),
    }


def gpu_synthetic_perturbation(
    C: torch.Tensor,
    E: torch.Tensor,
    alpha: torch.Tensor,
    blur_kernel: torch.Tensor,  # (1, 1, ksz, ksz)
) -> torch.Tensor:
    """GPU-side synthetic_positive perturbation.

    C: (B, 4, H, W) capture
    E: (B, 3, H, W) emission
    alpha: (B,) per-sample magnitude
    blur_kernel: (1, 1, ksz, ksz) Gaussian on the same device/dtype as inputs

    Returns: C' = clamp(C + α · blur(E_to_4ch, σ), 0, 1) (same shape as C).
    Equivalent to the previous CPU dataset perturbation but on GPU."""
    B, _, H, W = E.shape
    # Map E (3) → 4-ch matching CFA RGGB pattern: [R, G, G, B]
    E_4ch = torch.stack([E[:, 0], E[:, 1], E[:, 1], E[:, 2]], dim=1)
    ksz = blur_kernel.shape[-1]
    pad = ksz // 2
    kernel_4 = blur_kernel.expand(4, 1, ksz, ksz)
    E_blur = F.conv2d(E_4ch, kernel_4, padding=pad, groups=4)
    a = alpha.view(-1, 1, 1, 1).to(C.dtype)
    return torch.clamp(C + a * E_blur, 0.0, 1.0)

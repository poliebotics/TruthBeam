"""Dataset for emission-RGB recovery (exp001c).

Each sample for row t returns:
  - capture: (4, capture_h, capture_w) float32 in [0, 1] — CFA-split Bayer
    (R, G1, G2, B), resized to requested resolution via INTER_AREA.
  - emission: (3, emission_h, emission_w) float32 in [0, 1] — the rendered
    emission tile (the ground-truth image projected onto the cloth at row t).
    Loaded from `derived/Emissions/tile_NNNNNN.png` (1920x1080 RGB on disk),
    resized to requested resolution via INTER_AREA if smaller.
  - meta: t, session_id

CFA layout matches raw_bayer_dataset.py (RGGB).

Last row constraint from raw_bayer_dataset (S_{t+1} requirement) is NOT
needed here — emission prediction does not use XOF — but we keep the same
row range filtering so train/val splits match exp001a/b for comparability.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

WIDTH_RAW = 5320
HEIGHT_RAW = 4600
EXPECTED_RAW_BYTES = WIDTH_RAW * HEIGHT_RAW
HALF_H, HALF_W = HEIGHT_RAW // 2, WIDTH_RAW // 2  # (2300, 2660)

EMISSION_H_NATIVE = 1080
EMISSION_W_NATIVE = 1920


def _split_cfa_rggb(raw2d: np.ndarray) -> np.ndarray:
    R  = raw2d[0::2, 0::2]
    G1 = raw2d[0::2, 1::2]
    G2 = raw2d[1::2, 0::2]
    B  = raw2d[1::2, 1::2]
    return np.stack([R, G1, G2, B], axis=0)


def load_capture_at(raw_path: Path, target_h: int, target_w: int) -> torch.Tensor:
    """Load raw, split CFA at half-res, resize each channel to (target_h, target_w)."""
    raw_bytes = np.fromfile(raw_path, dtype=np.uint8)
    if raw_bytes.size != EXPECTED_RAW_BYTES:
        raise ValueError(f"unexpected raw size {raw_bytes.size}: {raw_path}")
    raw2d = raw_bytes.reshape(HEIGHT_RAW, WIDTH_RAW)
    cfa = _split_cfa_rggb(raw2d)  # (4, 2300, 2660) uint8
    if (target_h, target_w) == (HALF_H, HALF_W):
        return torch.from_numpy(cfa.astype(np.float32) / 255.0)
    out = np.empty((4, target_h, target_w), dtype=np.float32)
    for c in range(4):
        resized = cv2.resize(
            cfa[c], (target_w, target_h), interpolation=cv2.INTER_AREA
        )
        out[c] = resized.astype(np.float32) / 255.0
    return torch.from_numpy(out)


def load_emission_at(emission_path: Path, target_h: int, target_w: int) -> torch.Tensor:
    """Load 1920x1080 RGB PNG, resize to (target_h, target_w), return (3, H, W) float32."""
    img = cv2.imread(str(emission_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(emission_path)
    # cv2 loads BGR; convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if (target_h, target_w) != (img.shape[0], img.shape[1]):
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    arr = img.astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3, H, W)


def load_chain_log_rows(path: Path) -> set[int]:
    """Returns set of row indices present in chain_log."""
    out: set[int] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("t,"):
                continue
            out.add(int(line.split(",", 1)[0]))
    return out


class EmissionDataset(Dataset):
    """One session, restricted to rows in [row_start, row_end] (inclusive).

    Resolution-configurable: capture goes to (capture_h, capture_w), emission
    to (emission_h, emission_w). Both via INTER_AREA when downsizing.
    """

    def __init__(
        self,
        session_dir: Path,
        row_start: int,
        row_end: int,
        capture_h: int,
        capture_w: int,
        emission_h: int = EMISSION_H_NATIVE,
        emission_w: int = EMISSION_W_NATIVE,
        session_id: str = "",
        augment: bool = False,
        seed: int = 0,
    ) -> None:
        self.session_dir = Path(session_dir)
        self.session_id = session_id or self.session_dir.name
        self.recordings = self.session_dir / "Recordings"
        self.emissions = self.session_dir / "derived" / "Emissions"
        self.capture_h = capture_h
        self.capture_w = capture_w
        self.emission_h = emission_h
        self.emission_w = emission_w
        self.augment = augment
        self._seed = seed
        chain_rows = load_chain_log_rows(self.session_dir / "chain_log.csv")
        max_t = max(chain_rows)
        usable_end = min(row_end, max_t - 1)
        self.rows = [
            t for t in range(row_start, usable_end + 1)
            if t in chain_rows
            and (self.recordings / f"frame_{t:06d}.raw").exists()
            and (self.emissions / f"tile_{t:06d}.png").exists()
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        t = self.rows[idx]
        capture = load_capture_at(
            self.recordings / f"frame_{t:06d}.raw", self.capture_h, self.capture_w,
        )
        emission = load_emission_at(
            self.emissions / f"tile_{t:06d}.png", self.emission_h, self.emission_w,
        )
        if self.augment:
            # Per-sample RNG seeded by (seed, t) so shuffling order doesn't
            # affect which augmentation a row receives.
            rng = np.random.default_rng((self._seed, t))
            if rng.random() < 0.5:
                capture = torch.flip(capture, dims=[-1])   # horizontal flip
                emission = torch.flip(emission, dims=[-1])
            jitter = float(rng.uniform(0.9, 1.1))           # ±10% brightness on capture only
            capture = (capture * jitter).clamp(0.0, 1.0)
        return {
            "capture": capture,
            "emission": emission,
            "t": t,
            "session_id": self.session_id,
        }

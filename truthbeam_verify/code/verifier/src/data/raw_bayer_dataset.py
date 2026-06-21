"""Dataset for Phase-B Siamese verification training.

Each sample for row t returns:
  - capture: (4, 575, 665) float32 in [0, 1] — quarter-of-half-res CFA channels
    in order (R, G1, G2, B). PyTorch (C, H, W) convention.
  - xof_oct{0..3}: (3, h_oct, w_oct) float32 — XOF octaves derived from S_{t+1}
  - meta: t, session_id

CFA layout (RGGB per session_finalize.debayer_bayerrg_nn):
    R  = raw[0::2, 0::2]
    G1 = raw[0::2, 1::2]
    G2 = raw[1::2, 0::2]
    B  = raw[1::2, 1::2]

Last row of each session is excluded (no S_{t+1} available).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .xof_generation import xof_octaves_from_hex

WIDTH = 5320
HEIGHT = 4600
EXPECTED_BYTES = WIDTH * HEIGHT  # 24,472,000

QUARTER_HALF_H = HEIGHT // 8  # 575
QUARTER_HALF_W = WIDTH // 8   # 665

HALF_H = HEIGHT // 2  # 2300 — CFA-split native
HALF_W = WIDTH // 2   # 2660


def load_chain_log(path: Path) -> dict[int, str]:
    """Returns {t: S_t_hex}. Skips comment lines (#...) and CSV header."""
    out: dict[int, str] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("t,"):
                continue
            parts = line.split(",")
            t = int(parts[0])
            out[t] = parts[1]
    return out


def split_cfa_rggb(raw2d: np.ndarray) -> np.ndarray:
    """RGGB Bayer -> (4, H/2, W/2) uint8 in (R, G1, G2, B) order."""
    R  = raw2d[0::2, 0::2]
    G1 = raw2d[0::2, 1::2]
    G2 = raw2d[1::2, 0::2]
    B  = raw2d[1::2, 1::2]
    return np.stack([R, G1, G2, B], axis=0)


def load_raw_quarter_half(raw_path: Path) -> torch.Tensor:
    """Load one .raw, split CFA, resize each channel to (575, 665) via INTER_AREA."""
    return load_raw_at(raw_path, QUARTER_HALF_H, QUARTER_HALF_W)


def load_raw_at(raw_path: Path, target_h: int, target_w: int) -> torch.Tensor:
    """Load .raw, CFA-split at half-res, resize to (target_h, target_w) via INTER_AREA."""
    raw_bytes = np.fromfile(raw_path, dtype=np.uint8)
    if raw_bytes.size != EXPECTED_BYTES:
        raise ValueError(
            f"unexpected raw size {raw_bytes.size} vs {EXPECTED_BYTES}: {raw_path}"
        )
    raw2d = raw_bytes.reshape(HEIGHT, WIDTH)
    cfa = split_cfa_rggb(raw2d)  # (4, 2300, 2660) uint8
    if (target_h, target_w) == (HALF_H, HALF_W):
        return torch.from_numpy(cfa.astype(np.float32) / 255.0)
    out = np.empty((4, target_h, target_w), dtype=np.float32)
    for c in range(4):
        resized = cv2.resize(
            cfa[c], (target_w, target_h), interpolation=cv2.INTER_AREA
        )
        out[c] = resized.astype(np.float32) / 255.0
    return torch.from_numpy(out)


class SessionDataset(Dataset):
    """Single session, restricted to rows in [row_start, row_end] (inclusive).

    Filters out any t where t or t+1 are missing from chain_log, and any t
    whose .raw file doesn't exist on disk.
    """

    def __init__(
        self,
        session_dir: Path,
        row_start: int,
        row_end: int,
        session_id: str = "",
        capture_h: int = QUARTER_HALF_H,
        capture_w: int = QUARTER_HALF_W,
    ) -> None:
        self.session_dir = Path(session_dir)
        self.session_id = session_id or self.session_dir.name
        self.recordings = self.session_dir / "Recordings"
        self.capture_h = capture_h
        self.capture_w = capture_w
        self.chain = load_chain_log(self.session_dir / "chain_log.csv")

        max_t = max(self.chain.keys())
        usable_end = min(row_end, max_t - 1)
        self.rows = [
            t for t in range(row_start, usable_end + 1)
            if t in self.chain and (t + 1) in self.chain
            and (self.recordings / f"frame_{t:06d}.raw").exists()
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        t = self.rows[idx]
        s_next_hex = self.chain[t + 1]
        capture = load_raw_at(
            self.recordings / f"frame_{t:06d}.raw", self.capture_h, self.capture_w,
        )
        xof_octs = xof_octaves_from_hex(s_next_hex)
        return {
            "capture": capture,
            "xof_oct0": xof_octs[0],
            "xof_oct1": xof_octs[1],
            "xof_oct2": xof_octs[2],
            "xof_oct3": xof_octs[3],
            "t": t,
            "session_id": self.session_id,
        }

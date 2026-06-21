"""Dataset that returns Bayer captures in **native sparse** 4-channel layout.

Unlike `raw_bayer_dataset.SessionDataset` (which CFA-splits at half-res and then
resizes), this loader keeps the full sensor resolution (4600 × 5320, PyTorch
H × W convention) and places each CFA-channel value at its true sensor position
with zeros elsewhere:

    channel 0 (R):  values at [0::2, 0::2], zero elsewhere
    channel 1 (G1): values at [0::2, 1::2], zero elsewhere
    channel 2 (G2): values at [1::2, 0::2], zero elsewhere
    channel 3 (B):  values at [1::2, 1::2], zero elsewhere

This preserves the original sample positions: 75% of every channel is exactly
zero. The advantage is no information loss from area-resize and no smoothing of
the per-cell signal that XOF prediction relies on. The cost is 4× the raw byte
count in the tensor (one full H×W per channel) and downstream compute.

Returns either:
- XOF octave targets (for XOF prediction tasks); or
- Emission RGB at 1080×1920 (for emission recovery tasks); or
- Both, depending on `targets` argument.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .emission_dataset import EMISSION_H_NATIVE, EMISSION_W_NATIVE, load_emission_at
from .raw_bayer_dataset import (
    EXPECTED_BYTES,
    HEIGHT,
    WIDTH,
    load_chain_log,
)
from .xof_generation import xof_octaves_from_hex


def load_raw_native(raw_path: Path) -> torch.Tensor:
    """Load raw Bayer, return (4, HEIGHT, WIDTH) float32 in [0, 1] sparse layout."""
    raw_bytes = np.fromfile(raw_path, dtype=np.uint8)
    if raw_bytes.size != EXPECTED_BYTES:
        raise ValueError(
            f"unexpected raw size {raw_bytes.size} vs {EXPECTED_BYTES}: {raw_path}"
        )
    raw2d = raw_bytes.reshape(HEIGHT, WIDTH)
    out = np.zeros((4, HEIGHT, WIDTH), dtype=np.float32)
    out[0, 0::2, 0::2] = raw2d[0::2, 0::2]
    out[1, 0::2, 1::2] = raw2d[0::2, 1::2]
    out[2, 1::2, 0::2] = raw2d[1::2, 0::2]
    out[3, 1::2, 1::2] = raw2d[1::2, 1::2]
    out /= 255.0
    return torch.from_numpy(out)


class NativeBayerDataset(Dataset):
    """Native-resolution sparse Bayer dataset.

    targets: any subset of {"xof", "emission"}.
    """

    def __init__(
        self,
        session_dir: Path,
        row_start: int,
        row_end: int,
        targets: tuple[str, ...] = ("xof",),
        emission_h: int = EMISSION_H_NATIVE,
        emission_w: int = EMISSION_W_NATIVE,
        session_id: str = "",
    ) -> None:
        self.session_dir = Path(session_dir)
        self.session_id = session_id or self.session_dir.name
        self.recordings = self.session_dir / "Recordings"
        self.emissions = self.session_dir / "derived" / "Emissions"
        self.targets = tuple(targets)
        if not self.targets:
            raise ValueError("targets must include at least one of 'xof', 'emission'")
        self.emission_h = emission_h
        self.emission_w = emission_w
        self.chain = load_chain_log(self.session_dir / "chain_log.csv")

        max_t = max(self.chain.keys())
        usable_end = min(row_end, max_t - 1)
        need_xof = "xof" in self.targets
        need_emission = "emission" in self.targets
        rows = []
        for t in range(row_start, usable_end + 1):
            if t not in self.chain:
                continue
            if need_xof and (t + 1) not in self.chain:
                continue
            if not (self.recordings / f"frame_{t:06d}.raw").exists():
                continue
            if need_emission and not (self.emissions / f"tile_{t:06d}.png").exists():
                continue
            rows.append(t)
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        t = self.rows[idx]
        capture = load_raw_native(self.recordings / f"frame_{t:06d}.raw")
        item: dict[str, Any] = {
            "capture": capture,
            "t": t,
            "session_id": self.session_id,
        }
        if "xof" in self.targets:
            xof_octs = xof_octaves_from_hex(self.chain[t + 1])
            for i, o in enumerate(xof_octs):
                item[f"xof_oct{i}"] = o
        if "emission" in self.targets:
            item["emission"] = load_emission_at(
                self.emissions / f"tile_{t:06d}.png",
                self.emission_h,
                self.emission_w,
            )
        return item

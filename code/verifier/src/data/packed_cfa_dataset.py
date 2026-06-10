"""Phase D dataset: packed CFA + median+IQR normalization + centered XOF target.

Returns per row:
    capture_norm: (4, 2300, 2660) float32, normalized via stats from `stats_path`
    capture_pre_norm: (4, 2300, 2660) float32 in raw byte space (for observability)
    xof_oct{0..3}: (3, h, w) float32 in [-1, 1] (centered: (byte - 127.5)/127.5)
    emission: (3, 1080, 1920) float32 in [0, 1] — only if `with_emission=True`
    t: int (capture row)
    target_chain_row: int = t + offset
    session_id: str

Design notes:
- Loads packed CFA from raw or from the staging cache (audit §4 — pre-normalization).
- Normalization stats are passed in by the caller (audit Q8: A0..A7 use
  per-session stats; A6 uses D2-train stats for everything).
- Boundary rows where `target_chain_row` is invalid are excluded.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.normalization import apply_normalization
from preprocessing.packed_cfa import (
    BLACK_LEVEL_DEFAULT,
    cache_path_for,
    load_packed_cfa,
    load_packed_cfa_cached,
)
from data.emission_dataset import EMISSION_H_NATIVE, EMISSION_W_NATIVE, load_emission_at
from data.raw_bayer_dataset import load_chain_log
from data.xof_generation import xof_octaves_from_hex


def xof_octaves_centered_from_hex(s_next_hex: str) -> tuple[torch.Tensor, ...]:
    """Centered XOF target: (byte - 127.5) / 127.5 ∈ [-1, 1]."""
    raw_octaves = xof_octaves_from_hex(s_next_hex)  # each in [0, 1] = byte/255
    # Convert back to bytes (multiply by 255), then center: (b - 127.5) / 127.5
    out = []
    for o in raw_octaves:
        b = o * 255.0
        out.append((b - 127.5) / 127.5)
    return tuple(out)


class PackedCFADataset(Dataset):
    def __init__(
        self,
        session_dir: Path,
        rows: Iterable[int],
        offset: int,
        normalization_stats: dict,
        *,
        cache_root: Path | None = None,
        session_id: str = "",
        black_level: int = BLACK_LEVEL_DEFAULT,
        with_xof: bool = True,
        with_emission: bool = False,
        emission_h: int = EMISSION_H_NATIVE,
        emission_w: int = EMISSION_W_NATIVE,
    ) -> None:
        if not (with_xof or with_emission):
            raise ValueError("must enable at least one of with_xof, with_emission")
        self.session_dir = Path(session_dir)
        self.session_id = session_id or self.session_dir.name
        self.recordings = self.session_dir / "Recordings"
        self.emissions = self.session_dir / "derived" / "Emissions"
        self.cache_root = Path(cache_root) if cache_root else None
        self.offset = int(offset)
        self.with_xof = with_xof
        self.with_emission = with_emission
        self.emission_h = emission_h
        self.emission_w = emission_w
        self.normalization_stats = normalization_stats
        self.black_level = black_level
        chain = load_chain_log(self.session_dir / "chain_log.csv")
        self.chain = chain
        max_t = max(chain.keys())

        valid_rows = []
        for t in rows:
            target = t + self.offset
            if target < 0 or target > max_t:
                continue
            if t not in chain:
                continue
            if with_xof and target not in chain:
                # Need chain[target] to derive S_next
                continue
            if not (self.recordings / f"frame_{t:06d}.raw").exists():
                continue
            if with_emission and not (self.emissions / f"tile_{target:06d}.png").exists():
                continue
            valid_rows.append(t)
        self.rows = valid_rows

    def __len__(self) -> int:
        return len(self.rows)

    def _load_packed(self, t: int) -> torch.Tensor:
        if self.cache_root is not None:
            cp = cache_path_for(self.cache_root, self.session_id, t)
            if cp.exists():
                return load_packed_cfa_cached(cp)
        return load_packed_cfa(
            self.recordings / f"frame_{t:06d}.raw", black_level=self.black_level
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        t = self.rows[idx]
        target = t + self.offset
        packed_pre = self._load_packed(t)  # uint8 (4, 2300, 2660)
        packed_pre_f = packed_pre.to(torch.float32)
        packed_norm = apply_normalization(packed_pre_f, self.normalization_stats)

        item: dict[str, Any] = {
            "capture_norm": packed_norm,
            "capture_pre_norm": packed_pre_f,
            "t": t,
            "target_chain_row": target,
            "session_id": self.session_id,
        }
        if self.with_xof:
            xof_octs = xof_octaves_centered_from_hex(self.chain[target])
            for i, o in enumerate(xof_octs):
                item[f"xof_oct{i}"] = o
        if self.with_emission:
            item["emission"] = load_emission_at(
                self.emissions / f"tile_{target:06d}.png",
                self.emission_h, self.emission_w,
            )
        return item

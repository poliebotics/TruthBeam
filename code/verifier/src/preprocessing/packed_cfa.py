"""Packed CFA helper (audit §4) — pre-normalization, deterministic.

Pipeline:
    raw Bayer (uint8, H×W=4600×5320, BayerRG8 / RGGB)
    → optional fixed black-level subtraction (no-op: BLACK_LEVEL=0 measured-or-default)
    → packed CFA (4 × H/2 × W/2): channels in order R, G1, G2, B
    → saved cache (raw bytes, no per-session statistics applied)

Normalization (median+IQR/1.349) is NEVER baked into the cache — it's applied
at training time using stats from the per-experiment declared source. See
`src/preprocessing/normalization.py`.

Cache file format: torch.save of a uint8 tensor of shape (4, 2300, 2660).
Filename: `<cache_root>/<session>/frame_{t:06d}.pt`
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

WIDTH = 5320
HEIGHT = 4600
EXPECTED_BYTES = WIDTH * HEIGHT  # 24,472,000
HALF_H, HALF_W = HEIGHT // 2, WIDTH // 2  # (2300, 2660)

# No measured pedestal exists for this rig (see docs/DATA_SCHEMA.md). Default to 0.
BLACK_LEVEL_DEFAULT = 0
BLACK_LEVEL_SOURCE_DEFAULT = "no_measurement_found"


def split_cfa_rggb(raw2d: np.ndarray) -> np.ndarray:
    """RGGB Bayer → (4, H/2, W/2) uint8 in (R, G1, G2, B) order.

    Layout per `notebooks/exemplars/_make_bayer_channels.py`:
        R  = raw[0::2, 0::2]
        G1 = raw[0::2, 1::2]
        G2 = raw[1::2, 0::2]
        B  = raw[1::2, 1::2]
    """
    R  = raw2d[0::2, 0::2]
    G1 = raw2d[0::2, 1::2]
    G2 = raw2d[1::2, 0::2]
    B  = raw2d[1::2, 1::2]
    return np.stack([R, G1, G2, B], axis=0)


def load_packed_cfa(
    raw_path: Path,
    *,
    black_level: int = BLACK_LEVEL_DEFAULT,
) -> torch.Tensor:
    """Load `frame_{t:06d}.raw`, return (4, 2300, 2660) torch.uint8 packed CFA.

    `black_level` is subtracted with saturation at 0 (no negative values). Default
    is 0 — see DATA_SCHEMA.md for why no pedestal is applied.
    """
    raw_bytes = np.fromfile(raw_path, dtype=np.uint8)
    if raw_bytes.size != EXPECTED_BYTES:
        raise ValueError(f"unexpected raw size {raw_bytes.size} vs {EXPECTED_BYTES}: {raw_path}")
    raw2d = raw_bytes.reshape(HEIGHT, WIDTH)
    cfa = split_cfa_rggb(raw2d)  # (4, 2300, 2660) uint8
    if black_level > 0:
        cfa = np.clip(cfa.astype(np.int16) - black_level, 0, 255).astype(np.uint8)
    return torch.from_numpy(cfa.copy())


def cache_path_for(cache_root: Path, session: str, t: int) -> Path:
    return Path(cache_root) / session / f"frame_{t:06d}.pt"


def stage_packed_cfa(
    raw_path: Path,
    cache_root: Path,
    session: str,
    t: int,
    *,
    black_level: int = BLACK_LEVEL_DEFAULT,
) -> Path:
    """Idempotent: stage one frame into the cache. Returns the cache path.

    Validates an existing cache file by attempting to load it; if the dtype or
    shape doesn't match, the cache file is deleted and re-staged. Uses a unique
    PID-tagged temp path to avoid collisions under parallel staging.
    """
    cp = cache_path_for(cache_root, session, t)
    if cp.exists():
        try:
            existing = load_packed_cfa_cached(cp)
            if existing.dtype == torch.uint8 and existing.shape == (4, HALF_H, HALF_W):
                return cp
        except Exception:
            pass
        # invalid cache file — remove and re-stage
        try:
            cp.unlink()
        except FileNotFoundError:
            pass
    cp.parent.mkdir(parents=True, exist_ok=True)
    cfa = load_packed_cfa(raw_path, black_level=black_level)
    tmp = cp.with_suffix(f".pt.tmp.{os.getpid()}")
    torch.save(cfa, tmp)
    os.replace(tmp, cp)
    return cp


def load_packed_cfa_cached(cache_path: Path) -> torch.Tensor:
    """Load a cached packed CFA tensor (uint8, (4, 2300, 2660))."""
    t = torch.load(cache_path, map_location="cpu", weights_only=True)
    if t.dtype != torch.uint8 or t.shape != (4, HALF_H, HALF_W):
        raise ValueError(f"cache dtype/shape mismatch: {cache_path} {t.dtype} {t.shape}")
    return t

"""generator_patterns.py — recognizable test-pattern drop-in for the tile generator.

Returns one of 10 hand-crafted 1920x1080 RGB patterns per call, cycling with
the session's emission counter. Ignores the BLAKE3-XOF seeds for pattern
selection; the chain still records the pixel hash and xof seeds normally.

Invoke by monkey-patching `tb_loop._load_tile_generator` to `load_tile_generator_patterns`
before constructing TBLoopV8 (see tb_main_patterns.py).

The module internally filters out the 10 prewarm calls and 1 determinism-check
call by seed-matching, so the in-session counter increments exactly once per
true emission: bootstrap (captured as frame t=0), then once per loop iteration
(captured as frame t=1, 2, …). Therefore frame_{t:06d}.raw captures pattern
index (t % 10).
"""
from __future__ import annotations

import inspect
import threading

import numpy as np
from blake3 import blake3

from tile_params import TILE_H, TILE_W


_SEED_DOMAIN_TAG_R = b"TB:SEED:R:v8"


def _compute_warmup_xof_r_set() -> set[bytes]:
    warmup = set()
    prewarm_seed = blake3(b"tb-v8-prewarm").digest()
    for i in range(10):  # GPU_PREWARM_TILES
        seed_i = blake3(prewarm_seed + i.to_bytes(4, "big")).digest()
        warmup.add(blake3(_SEED_DOMAIN_TAG_R + seed_i).digest())
    det_seed = blake3(b"tb-v8-determinism-check").digest()
    warmup.add(blake3(_SEED_DOMAIN_TAG_R + det_seed).digest())
    return warmup


def _pattern_white() -> np.ndarray:
    a = np.full((TILE_H, TILE_W, 3), 255, dtype=np.uint8)
    return a


def _pattern_black() -> np.ndarray:
    return np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)


def _pattern_solid(r: int, g: int, b: int) -> np.ndarray:
    a = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
    a[..., 0] = r
    a[..., 1] = g
    a[..., 2] = b
    return a


def _pattern_h_gradient() -> np.ndarray:
    col = (np.arange(TILE_W, dtype=np.int32) * 255 // (TILE_W - 1)).astype(np.uint8)
    a = np.broadcast_to(col[None, :, None], (TILE_H, TILE_W, 3)).copy()
    return a


def _pattern_v_gradient() -> np.ndarray:
    row = (np.arange(TILE_H, dtype=np.int32) * 255 // (TILE_H - 1)).astype(np.uint8)
    a = np.broadcast_to(row[:, None, None], (TILE_H, TILE_W, 3)).copy()
    return a


def _pattern_checker8() -> np.ndarray:
    a = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
    tile_w, tile_h = 240, 135
    for iy in range(8):
        for ix in range(8):
            if (ix + iy) % 2 == 0:
                a[iy * tile_h:(iy + 1) * tile_h,
                  ix * tile_w:(ix + 1) * tile_w, :] = 255
    return a


def _pattern_hstripes() -> np.ndarray:
    a = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
    band = 135
    for i in range(8):
        if i % 2 == 0:
            a[i * band:(i + 1) * band, :, :] = 255
    return a


def _pattern_cross() -> np.ndarray:
    a = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
    thick = 20
    cy = TILE_H // 2
    cx = TILE_W // 2
    a[cy - thick // 2:cy - thick // 2 + thick, :, :] = 255
    a[:, cx - thick // 2:cx - thick // 2 + thick, :] = 255
    return a


def _build_pattern_table() -> list[np.ndarray]:
    return [
        _pattern_white(),                # 0
        _pattern_black(),                # 1
        _pattern_solid(255, 0, 0),       # 2 red
        _pattern_solid(0, 255, 0),       # 3 green
        _pattern_solid(0, 0, 255),       # 4 blue
        _pattern_h_gradient(),           # 5
        _pattern_v_gradient(),           # 6
        _pattern_checker8(),             # 7
        _pattern_hstripes(),             # 8
        _pattern_cross(),                # 9
    ]


_PATTERN_TABLE = _build_pattern_table()
_WARMUP_XOF_R = _compute_warmup_xof_r_set()


class _PatternGen:
    """Stateful generator: filters warmup calls, counts session emissions."""

    def __init__(self):
        self._lock = threading.Lock()
        self._counter = 0

    def __call__(self, xof_r: bytes, xof_g: bytes, xof_b: bytes) -> np.ndarray:
        if xof_r in _WARMUP_XOF_R:
            return _PATTERN_TABLE[0]
        with self._lock:
            idx = self._counter % 10
            self._counter += 1
        return _PATTERN_TABLE[idx]


def load_tile_generator_patterns(cpu_only: bool):
    """Drop-in replacement for `tile_backend.load_tile_generator`.

    Returns (backend_name, gen_func, source_hash_hex, cpu_pool). cpu_only is
    accepted for signature compatibility but ignored: the pattern generator
    is pure numpy and has no CPU/GPU split.
    """
    del cpu_only
    gen = _PatternGen()
    src = (
        inspect.getsource(_compute_warmup_xof_r_set)
        + inspect.getsource(_build_pattern_table)
        + inspect.getsource(_pattern_white)
        + inspect.getsource(_pattern_black)
        + inspect.getsource(_pattern_solid)
        + inspect.getsource(_pattern_h_gradient)
        + inspect.getsource(_pattern_v_gradient)
        + inspect.getsource(_pattern_checker8)
        + inspect.getsource(_pattern_hstripes)
        + inspect.getsource(_pattern_cross)
        + inspect.getsource(_PatternGen)
    )
    source_hash = blake3(src.encode("utf-8")).hexdigest()
    return "generator_patterns", gen, source_hash, None

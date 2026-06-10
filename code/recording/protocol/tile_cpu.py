"""CPU tile generator. Thread-parallel fBm with integer bilinear upsampling.
See METHODS.md for generation logic.
"""
import numpy as np
from blake3 import blake3

from tile_params import (
    TILE_H, TILE_W, NUM_OCTAVES, GRID_H_TABLE, GRID_W_TABLE,
    TOTAL_XOF_BYTES_PER_CHANNEL as TOTAL,
)

SB = 16; S = 1 << SB; M = S - 1


def _tables(gh, gw, out_h, out_w):
    xs = (np.arange(out_w, dtype=np.int64) * (gw - 1) * S) // max(out_w - 1, 1)
    ix = (xs >> SB).astype(np.int64)
    fx = (xs & M).astype(np.int32)
    ix_n = np.minimum(ix + 1, gw - 1)
    ys = (np.arange(out_h, dtype=np.int64) * (gh - 1) * S) // max(out_h - 1, 1)
    iy = (ys >> SB).astype(np.int64)
    fy = (ys & M).astype(np.int32)
    iy_n = np.minimum(iy + 1, gh - 1)
    return ix, ix_n, fx, iy, iy_n, fy


_TABLES = [_tables(GRID_H_TABLE[o], GRID_W_TABLE[o], TILE_H, TILE_W)
           for o in range(NUM_OCTAVES)]


def gen_channel_v2(xof_seed):
    """Single uint8 (TILE_H, TILE_W) channel, bit-exact to generate_channel_xof."""
    stream = blake3(xof_seed).digest(length=TOTAL)
    frame = np.empty((TILE_H, TILE_W), dtype=np.int32)
    offset = 0
    for oct_i in range(NUM_OCTAVES):
        gh = GRID_H_TABLE[oct_i]; gw = GRID_W_TABLE[oct_i]
        n = gh * gw
        grid_s = np.frombuffer(stream, dtype=np.uint8, count=n, offset=offset
                               ).reshape(gh, gw).astype(np.int16) - np.int16(128)
        offset += n
        ix, ix_n, fx, iy, iy_n, fy = _TABLES[oct_i]
        # Horizontal: int16 × int32 sum → int32. Max |h| ≤ 128 * 65536 * 2 = 2^24.
        h_int32 = (grid_s[:, ix].astype(np.int32) * (S - fx)
                   + grid_s[:, ix_n].astype(np.int32) * fx)
        # Vertical: needs int64. 2^24 × 2^16 = 2^40.
        S_minus_fy = (S - fy).astype(np.int64)[:, None]
        fy64 = fy.astype(np.int64)[:, None]
        top = h_int32[iy, :].astype(np.int64)
        bot = h_int32[iy_n, :].astype(np.int64)
        v = ((top * S_minus_fy + bot * fy64) >> 16).astype(np.int32)
        if oct_i == 0:
            frame[:] = v
        else:
            frame += (v >> oct_i)
    return ((frame >> 16) + 128).clip(0, 255).astype(np.uint8)


def gen_rgb_v2(xof_r, xof_g, xof_b, pool):
    r, g, b = list(pool.map(gen_channel_v2, [xof_r, xof_g, xof_b]))
    return np.stack([r, g, b], axis=-1)

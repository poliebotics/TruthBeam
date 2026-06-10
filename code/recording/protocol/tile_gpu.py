"""GPU tile generator. Bit-exact equivalent of the CPU version (tile_cpu.py).
See METHODS.md for generation logic.
"""
import torch
from blake3 import blake3

from tile_params import (
    TILE_H, TILE_W, NUM_OCTAVES, GRID_H_TABLE, GRID_W_TABLE,
    TOTAL_XOF_BYTES_PER_CHANNEL as TOTAL,
)

SB = 16; S = 1 << SB; M = S - 1

DEV = torch.device("cuda:0")


def _tables_torch(gh, gw, out_h, out_w):
    xs = (torch.arange(out_w, dtype=torch.int64) * (gw - 1) * S) // max(out_w - 1, 1)
    ix = (xs >> SB).to(torch.int64)
    fx = (xs & M).to(torch.int32)
    ix_n = torch.clamp(ix + 1, max=gw - 1)
    ys = (torch.arange(out_h, dtype=torch.int64) * (gh - 1) * S) // max(out_h - 1, 1)
    iy = (ys >> SB).to(torch.int64)
    fy = (ys & M).to(torch.int32)
    iy_n = torch.clamp(iy + 1, max=gh - 1)
    return (ix.to(DEV), ix_n.to(DEV), fx.to(DEV),
            iy.to(DEV), iy_n.to(DEV), fy.to(DEV))


_TABLES = [_tables_torch(GRID_H_TABLE[o], GRID_W_TABLE[o], TILE_H, TILE_W)
           for o in range(NUM_OCTAVES)]

# Octave offsets into the XOF stream
_OFFSETS = [0]
for o in range(NUM_OCTAVES):
    _OFFSETS.append(_OFFSETS[-1] + GRID_H_TABLE[o] * GRID_W_TABLE[o])
# Ends with TOTAL

# Persistent pinned-host staging buffers — reused across tiles.
_stream_pinned = torch.empty((3, TOTAL), dtype=torch.uint8, pin_memory=True)
# Output buffer — (H, W, 3) uint8 on host, pinned.
_out_pinned    = torch.empty((TILE_H, TILE_W, 3), dtype=torch.uint8, pin_memory=True)
# Device-side scratch buffers allocated once
_stream_dev    = torch.empty((3, TOTAL), dtype=torch.uint8, device=DEV)


def _upsample_bilinear_int_cuda(grid_s, oct_i):
    """Bit-exact equivalent of _integer_bilinear_upsample applied batched
    over channels. grid_s: (3, gh, gw) int16 on DEV.
    Returns (3, TILE_H, TILE_W) int32 on DEV = v from the reference.
    """
    ix, ix_n, fx, iy, iy_n, fy = _TABLES[oct_i]
    # Horizontal int32 (bit-exact to numpy path).
    grid_i32 = grid_s.to(torch.int32)
    left  = grid_i32[:, :, ix]       # (3, gh, out_w)
    right = grid_i32[:, :, ix_n]
    S_minus_fx = (S - fx).to(torch.int32)
    h_int32 = left * S_minus_fx + right * fx   # ≤2^24, fits int32

    # Vertical int64.
    top = h_int32[:, iy, :].to(torch.int64)
    bot = h_int32[:, iy_n, :].to(torch.int64)
    S_minus_fy = (S - fy).to(torch.int64).unsqueeze(1)
    fy64 = fy.to(torch.int64).unsqueeze(1)
    v64 = top * S_minus_fy + bot * fy64
    v = (v64 >> 16).to(torch.int32)
    return v


def gen_rgb_tile_cuda(xof_r, xof_g, xof_b):
    """End-to-end: returns a numpy (1080,1920,3) uint8 tile."""
    stream_r = blake3(xof_r).digest(length=TOTAL)
    stream_g = blake3(xof_g).digest(length=TOTAL)
    stream_b = blake3(xof_b).digest(length=TOTAL)

    _stream_pinned[0].copy_(torch.frombuffer(bytearray(stream_r), dtype=torch.uint8))
    _stream_pinned[1].copy_(torch.frombuffer(bytearray(stream_g), dtype=torch.uint8))
    _stream_pinned[2].copy_(torch.frombuffer(bytearray(stream_b), dtype=torch.uint8))
    _stream_dev.copy_(_stream_pinned, non_blocking=True)

    frame = torch.empty((3, TILE_H, TILE_W), dtype=torch.int32, device=DEV)
    for oct_i in range(NUM_OCTAVES):
        gh = GRID_H_TABLE[oct_i]; gw = GRID_W_TABLE[oct_i]
        off0 = _OFFSETS[oct_i]; off1 = _OFFSETS[oct_i + 1]
        slab_u8 = _stream_dev[:, off0:off1].reshape(3, gh, gw)
        grid_s = slab_u8.to(torch.int16) - 128
        v = _upsample_bilinear_int_cuda(grid_s, oct_i)
        if oct_i == 0:
            frame.copy_(v)
        else:
            frame.add_(v >> oct_i)

    out_chw = ((frame >> 16) + 128).clamp_(0, 255).to(torch.uint8)
    out_hwc = out_chw.permute(1, 2, 0).contiguous()
    _out_pinned.copy_(out_hwc, non_blocking=True)
    torch.cuda.synchronize()
    return _out_pinned.numpy().copy()

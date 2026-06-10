"""Bit-exact PyTorch reimplementation of truth_beam.protocol.tile_gpu.gen_rgb_tile_cuda.

Original takes 32-byte BLAKE3 seeds and runs:
    streams = [blake3(seed_R).digest(43110), blake3(seed_G), blake3(seed_B)]
    for each octave o in 0..3:
        bytes (3, gh, gw) = streams reshaped → centered int16 (sub 128)
        upsampled = integer_bilinear(centered_int16, output (3, 1080, 1920)) → int32
        if o == 0: frame = upsampled
        else: frame += upsampled >> o     # divide by 2^o
    out_chw = ((frame >> 16) + 128).clamp(0, 255).to(uint8)
    out_hwc = out_chw.permute(1, 2, 0).contiguous()

This module exposes:
    render_from_streams(stream_r: bytes, stream_g: bytes, stream_b: bytes,
                        device='cpu') -> torch.Tensor (3, 1080, 1920) uint8

i.e. takes ALREADY-EXPANDED 43,110-byte streams (skips BLAKE3) so callers
can corrupt them at byte level and feed them directly. Pure integer math
on PyTorch tensors → deterministic on any device, bit-exact equivalent
to the original CUDA implementation.

Constants are hard-coded from truth_beam.protocol.tile_params (TILE_H=1080,
TILE_W=1920, NUM_OCTAVES=4, GRID_H_TABLE=[17,34,68,135],
GRID_W_TABLE=[30,60,120,240]).
"""
from __future__ import annotations

import numpy as np
import torch

# Constants (from truth_beam.protocol.tile_params)
TILE_W = 1920
TILE_H = 1080
NUM_OCTAVES = 4
GRID_H_TABLE = [17, 34, 68, 135]
GRID_W_TABLE = [30, 60, 120, 240]
TOTAL_BYTES_PER_CHANNEL = sum(h * w for h, w in zip(GRID_H_TABLE, GRID_W_TABLE))  # 43110
assert TOTAL_BYTES_PER_CHANNEL == 43110

SB = 16
S = 1 << SB  # 65536
M = S - 1


def _bilinear_tables(gh: int, gw: int, out_h: int, out_w: int, device):
    """Precomputed bit-exact bilinear lookup tables for one octave."""
    xs = (torch.arange(out_w, dtype=torch.int64, device=device) * (gw - 1) * S) // max(out_w - 1, 1)
    ix = (xs >> SB).to(torch.int64)
    fx = (xs & M).to(torch.int32)
    ix_n = torch.clamp(ix + 1, max=gw - 1)
    ys = (torch.arange(out_h, dtype=torch.int64, device=device) * (gh - 1) * S) // max(out_h - 1, 1)
    iy = (ys >> SB).to(torch.int64)
    fy = (ys & M).to(torch.int32)
    iy_n = torch.clamp(iy + 1, max=gh - 1)
    return ix, ix_n, fx, iy, iy_n, fy


_BILINEAR_TABLES_CACHE: dict[tuple, tuple] = {}


def _bilinear_tables_cached(gh: int, gw: int, out_h: int, out_w: int, device):
    """Module-level cache for _bilinear_tables. Tables depend only on
    (gh, gw, out_h, out_w, device); for a fixed renderer they are constant."""
    key = (gh, gw, out_h, out_w, str(device))
    cached = _BILINEAR_TABLES_CACHE.get(key)
    if cached is None:
        cached = _bilinear_tables(gh, gw, out_h, out_w, device)
        _BILINEAR_TABLES_CACHE[key] = cached
    return cached


def _upsample_octave_int(grid_s: torch.Tensor, oct_i: int, out_h: int, out_w: int, tables) -> torch.Tensor:
    """grid_s: (3, gh, gw) int16 (centered byte-128). Returns (3, TILE_H, TILE_W) int32."""
    ix, ix_n, fx, iy, iy_n, fy = tables[oct_i]
    grid_i32 = grid_s.to(torch.int32)
    # Horizontal int32 (≤2^24, fits int32)
    left = grid_i32[:, :, ix]
    right = grid_i32[:, :, ix_n]
    S_minus_fx = (S - fx).to(torch.int32)
    h_int32 = left * S_minus_fx + right * fx
    # Vertical int64
    top = h_int32[:, iy, :].to(torch.int64)
    bot = h_int32[:, iy_n, :].to(torch.int64)
    S_minus_fy = (S - fy).to(torch.int64).unsqueeze(1)
    fy64 = fy.to(torch.int64).unsqueeze(1)
    v64 = top * S_minus_fy + bot * fy64
    return (v64 >> 16).to(torch.int32)


def _build_offsets():
    offs = [0]
    for h, w in zip(GRID_H_TABLE, GRID_W_TABLE):
        offs.append(offs[-1] + h * w)
    return offs


_OFFSETS = _build_offsets()


def render_from_streams(
    stream_r: bytes,
    stream_g: bytes,
    stream_b: bytes,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Render emission RGB tile from already-expanded XOF streams.

    Bit-exact equivalent of tile_gpu.gen_rgb_tile_cuda's body AFTER the
    BLAKE3 expansion step. Inputs are 43,110-byte streams per channel
    (R, G, B). Output: (3, TILE_H, TILE_W) uint8 tensor on `device`.
    """
    if not (len(stream_r) == TOTAL_BYTES_PER_CHANNEL
            and len(stream_g) == TOTAL_BYTES_PER_CHANNEL
            and len(stream_b) == TOTAL_BYTES_PER_CHANNEL):
        raise ValueError(f"each stream must be {TOTAL_BYTES_PER_CHANNEL} bytes")
    dev = torch.device(device)

    # Stack channel streams into (3, TOTAL) uint8
    arr = np.stack([
        np.frombuffer(stream_r, dtype=np.uint8),
        np.frombuffer(stream_g, dtype=np.uint8),
        np.frombuffer(stream_b, dtype=np.uint8),
    ], axis=0).copy()
    streams = torch.from_numpy(arr).to(dev)

    tables = [_bilinear_tables_cached(GRID_H_TABLE[o], GRID_W_TABLE[o], TILE_H, TILE_W, dev)
              for o in range(NUM_OCTAVES)]

    frame = torch.empty((3, TILE_H, TILE_W), dtype=torch.int32, device=dev)
    for oct_i in range(NUM_OCTAVES):
        gh, gw = GRID_H_TABLE[oct_i], GRID_W_TABLE[oct_i]
        off0, off1 = _OFFSETS[oct_i], _OFFSETS[oct_i + 1]
        slab_u8 = streams[:, off0:off1].reshape(3, gh, gw)
        # int16 centered: byte_value − 128
        grid_s = slab_u8.to(torch.int16) - 128
        v = _upsample_octave_int(grid_s, oct_i, TILE_H, TILE_W, tables)
        if oct_i == 0:
            frame.copy_(v)
        else:
            frame.add_(v >> oct_i)

    out_chw = ((frame >> 16) + 128).clamp_(0, 255).to(torch.uint8)
    return out_chw  # (3, TILE_H, TILE_W) uint8


# ---------- helpers for the probe ----------

def expand_seeds_to_streams(s_next_bytes: bytes) -> tuple[bytes, bytes, bytes]:
    """Re-implement the protocol's seed → stream expansion (BLAKE3-XOF).

    Returns (stream_r, stream_g, stream_b), each TOTAL_BYTES_PER_CHANNEL bytes.
    Uses the same domain tags as truth_beam.protocol.session_schema.compute_xof_seeds.
    """
    from blake3 import blake3
    if len(s_next_bytes) != 32:
        raise ValueError(f"S_next must be 32 bytes, got {len(s_next_bytes)}")
    SEED_DOMAIN_TAG_R = b"TB:SEED:R:v8"
    SEED_DOMAIN_TAG_G = b"TB:SEED:G:v8"
    SEED_DOMAIN_TAG_B = b"TB:SEED:B:v8"
    seed_r = blake3(SEED_DOMAIN_TAG_R + s_next_bytes).digest()
    seed_g = blake3(SEED_DOMAIN_TAG_G + s_next_bytes).digest()
    seed_b = blake3(SEED_DOMAIN_TAG_B + s_next_bytes).digest()
    stream_r = blake3(seed_r).digest(length=TOTAL_BYTES_PER_CHANNEL)
    stream_g = blake3(seed_g).digest(length=TOTAL_BYTES_PER_CHANNEL)
    stream_b = blake3(seed_b).digest(length=TOTAL_BYTES_PER_CHANNEL)
    return stream_r, stream_g, stream_b


def render_from_chain_hex(s_next_hex: str, device: str | torch.device = "cpu") -> torch.Tensor:
    """Convenience: chain log row's S_{t+1}_hex → emission RGB (3, TILE_H, TILE_W) uint8."""
    streams = expand_seeds_to_streams(bytes.fromhex(s_next_hex))
    return render_from_streams(*streams, device=device)


def noise_streams_byte_replace(streams: tuple[bytes, bytes, bytes], p: float, seed: int) -> tuple[bytes, bytes, bytes]:
    """Each byte independently replaced with Uniform[0,255] with probability p,
    deterministic under seed. Returns 3-tuple of corrupted streams."""
    rng = np.random.default_rng(seed)
    out = []
    for s in streams:
        arr = np.frombuffer(s, dtype=np.uint8).copy()
        if p > 0:
            mask = rng.random(arr.shape[0]) < p
            n_flip = int(mask.sum())
            if n_flip > 0:
                replacement = rng.integers(0, 256, size=n_flip, dtype=np.uint8)
                arr[mask] = replacement
        out.append(arr.tobytes())
    return tuple(out)

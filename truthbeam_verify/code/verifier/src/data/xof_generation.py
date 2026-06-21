"""XOF re-derivation for verification training.

Per protocol v8 (truth_beam/protocol/session_schema.py:462):
    xof_seed_{R,G,B} = blake3(b"TB:SEED:{R,G,B}:v8" || S_{t+1})

Each per-channel seed is BLAKE3-XOF expanded to 43,110 bytes per channel,
which decomposes into 4 octaves:
    oct0:  17 ×  30 =    510 bytes/channel
    oct1:  34 ×  60 =  2,040
    oct2:  68 × 120 =  8,160
    oct3: 135 × 240 = 32,400
    sum                43,110 per channel

The XOF for row t is derived from row (t+1)'s S_t_hex, so row N-1 is unusable.
"""
from __future__ import annotations

import numpy as np
import torch
from blake3 import blake3

SEED_DOMAIN_TAG_R = b"TB:SEED:R:v8"
SEED_DOMAIN_TAG_G = b"TB:SEED:G:v8"
SEED_DOMAIN_TAG_B = b"TB:SEED:B:v8"

OCTAVE_SHAPES = ((17, 30), (34, 60), (68, 120), (135, 240))
OCTAVE_BYTES_PER_CHANNEL = tuple(h * w for (h, w) in OCTAVE_SHAPES)
TOTAL_BYTES_PER_CHANNEL = sum(OCTAVE_BYTES_PER_CHANNEL)
assert TOTAL_BYTES_PER_CHANNEL == 43110


def derive_xof_seeds(s_next_bytes: bytes) -> tuple[bytes, bytes, bytes]:
    if len(s_next_bytes) != 32:
        raise ValueError(f"S_next must be 32 bytes, got {len(s_next_bytes)}")
    return (
        blake3(SEED_DOMAIN_TAG_R + s_next_bytes).digest(),
        blake3(SEED_DOMAIN_TAG_G + s_next_bytes).digest(),
        blake3(SEED_DOMAIN_TAG_B + s_next_bytes).digest(),
    )


def expand_seed_to_octaves(seed: bytes) -> tuple[np.ndarray, ...]:
    """Expand a 32-byte channel seed into 4 octave 2-D arrays (uint8)."""
    raw = blake3(seed).digest(length=TOTAL_BYTES_PER_CHANNEL)
    arr = np.frombuffer(raw, dtype=np.uint8)
    out = []
    pos = 0
    for (h, w), nbytes in zip(OCTAVE_SHAPES, OCTAVE_BYTES_PER_CHANNEL):
        out.append(arr[pos:pos + nbytes].reshape(h, w))
        pos += nbytes
    return tuple(out)


def xof_octaves_from_s_next(s_next_bytes: bytes) -> tuple[torch.Tensor, ...]:
    """Returns 4 tensors with shapes (3, h, w), values in [0, 1] float32.
    Channel order is R, G, B per spec."""
    seeds = derive_xof_seeds(s_next_bytes)
    per_channel = [expand_seed_to_octaves(s) for s in seeds]
    octaves = []
    for i in range(4):
        stack = np.stack([per_channel[c][i] for c in range(3)], axis=0)
        # .copy() because np.frombuffer returns read-only views
        t = torch.from_numpy(stack.copy()).float() / 255.0
        octaves.append(t)
    return tuple(octaves)


def xof_octaves_from_hex(s_next_hex: str) -> tuple[torch.Tensor, ...]:
    return xof_octaves_from_s_next(bytes.fromhex(s_next_hex))

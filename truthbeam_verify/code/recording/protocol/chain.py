"""chain.py — thin wrappers that express chain operations in protocol terms.

Scope: protocol-facing operations on the BLAKE3 state chain
(S_0 derivation, S_{t+1} transition, XOF seeds, meta struct packing).

This module is a vocabulary layer over session_schema.py. session_schema
owns the dataclasses and the actual hash math; chain.py gives callers
domain-language function names (`advance_chain`, `pack_meta`, …) so
capture loops don't have to import the full schema module or repeat the
`struct.pack(META_STRUCT, …)` incantation inline.

Adapt this module for:
- A protocol version bump that changes the domain tag or field layout
- A swap of the capture-metadata struct (but remember: the struct format
  is hashed into generator_code_hash and into S_{t+1}, so changes are a
  protocol change, not a refactor)

Do NOT put here:
- The actual BLAKE3 hash math — that's session_schema
- Camera / projector / IO concerns
- Session manifest construction — that's also session_schema
"""
from __future__ import annotations

import struct

from session_schema import (
    compute_xof_seeds as _compute_xof_seeds,
)

# NOTE: S_0 derivation and the per-row chain transition are computed
# directly via session_schema.derive_s0() and session_schema.compute_chain_row()
# (both v9 signatures, drand-folded). Earlier thin wrappers here carried stale
# v8 signatures and were unused; they have been removed to avoid drift.


# Capture meta struct (28 bytes, big-endian). Hashed into S_{t+1}.
#   I  = t (uint32)
#   Q  = aravis_device_timestamp_ns (uint64)
#   Q  = capture_wall_ns (uint64)
#   I  = exposure_us (uint32)
#   4s = fourcc (4 ASCII bytes, e.g. b"RG08")
META_STRUCT = ">IQQI4s"
META_LEN = struct.calcsize(META_STRUCT)
assert META_LEN == 28, f"meta struct must be 28 bytes, got {META_LEN}"


def compute_xof_seeds(s_next: bytes) -> tuple[bytes, bytes, bytes]:
    """Per-channel XOF seeds (R, G, B) from S_{t+1}, domain-separated."""
    return _compute_xof_seeds(s_next)


def pack_meta(consumed_as_t: int, device_ts_ns: int, capture_wall_ns: int,
              exposure_us: int, fourcc: bytes) -> bytes:
    """Return the 28-byte meta payload for chain row `consumed_as_t`.

    fourcc is the raw pixel-format 4-byte code (e.g. b"RG08" for BayerRG8),
    pre-encoded to ASCII bytes by the caller.
    """
    if not isinstance(fourcc, (bytes, bytearray)) or len(fourcc) != 4:
        raise ValueError(f"fourcc must be exactly 4 bytes; got {fourcc!r}")
    return struct.pack(META_STRUCT, int(consumed_as_t), int(device_ts_ns),
                       int(capture_wall_ns), int(exposure_us), bytes(fourcc))

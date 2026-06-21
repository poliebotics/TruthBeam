"""Phase F infrastructure: bit-exact roundtrip between BayerRG8 raw bytes
and 4-channel packed CFA tensors.

The protocol's chain-byte commitment hashes the BayerRG8 raw bytes from the
camera (`bayer_hash = blake3(raw_bytes)`). For Phase F (digital-twin attack),
we generate fake captures in packed-CFA tensor form and need to convert back
to BayerRG8 bytes that, when re-hashed, produce a chain-consistent
`bayer_hash`. Without bit-exact roundtrip, the chain check fails immediately.

BayerRG8 layout (per session_finalize.debayer_bayerrg_nn / camera RGGB):
    R   at raw[0::2, 0::2]   (top-left of each 2×2 block)
    G1  at raw[0::2, 1::2]   (top-right)
    G2  at raw[1::2, 0::2]   (bottom-left)
    B   at raw[1::2, 1::2]   (bottom-right)

Native sensor: 5320 × 4600 = 24,472,000 bytes. Packed CFA: (4, 2300, 2660)
uint8.

Usage:
    cfa  = bayer_rg8_to_packed_cfa(raw_bytes)        # (4, 2300, 2660) uint8
    raw  = packed_cfa_to_bayer_rg8(cfa)              # bytes (24,472,000)
    assert raw == raw_bytes
"""
from __future__ import annotations

import numpy as np
import torch

WIDTH = 5320
HEIGHT = 4600
EXPECTED_BYTES = WIDTH * HEIGHT  # 24,472,000
HALF_H, HALF_W = HEIGHT // 2, WIDTH // 2  # 2300, 2660


def bayer_rg8_to_packed_cfa(raw_bytes: bytes | np.ndarray) -> np.ndarray:
    """BayerRG8 raw bytes → (4, 2300, 2660) uint8 packed CFA in (R, G1, G2, B) order.

    Inputs:
      - raw_bytes: 24,472,000 bytes (or numpy array of that size).
    Returns:
      - np.ndarray, shape (4, 2300, 2660), dtype uint8.
    """
    if isinstance(raw_bytes, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(bytes(raw_bytes), dtype=np.uint8)
    else:
        arr = np.asarray(raw_bytes, dtype=np.uint8).reshape(-1)
    if arr.size != EXPECTED_BYTES:
        raise ValueError(f"expected {EXPECTED_BYTES} bytes, got {arr.size}")
    raw2d = arr.reshape(HEIGHT, WIDTH)
    R  = raw2d[0::2, 0::2]
    G1 = raw2d[0::2, 1::2]
    G2 = raw2d[1::2, 0::2]
    B  = raw2d[1::2, 1::2]
    out = np.stack([R, G1, G2, B], axis=0)
    if out.shape != (4, HALF_H, HALF_W):
        raise RuntimeError(f"unexpected output shape: {out.shape}")
    return np.ascontiguousarray(out)


def packed_cfa_to_bayer_rg8(cfa: np.ndarray | torch.Tensor) -> bytes:
    """(4, 2300, 2660) uint8 packed CFA → BayerRG8 raw bytes.

    Inputs:
      - cfa: array or tensor of shape (4, 2300, 2660), dtype uint8 (or
             castable to uint8 without value loss).
    Returns:
      - 24,472,000 bytes in BayerRG8 layout.
    """
    if isinstance(cfa, torch.Tensor):
        cfa = cfa.detach().cpu().numpy()
    cfa = np.asarray(cfa)
    if cfa.shape != (4, HALF_H, HALF_W):
        raise ValueError(f"expected shape (4, {HALF_H}, {HALF_W}), got {cfa.shape}")
    if cfa.dtype != np.uint8:
        # Allow lossless cast from int16/int32 if values are in [0, 255]
        if cfa.min() < 0 or cfa.max() > 255:
            raise ValueError(f"non-uint8 input out of [0, 255] range: min={cfa.min()}, max={cfa.max()}")
        cfa = cfa.astype(np.uint8)
    R, G1, G2, B = cfa[0], cfa[1], cfa[2], cfa[3]
    raw2d = np.empty((HEIGHT, WIDTH), dtype=np.uint8)
    raw2d[0::2, 0::2] = R
    raw2d[0::2, 1::2] = G1
    raw2d[1::2, 0::2] = G2
    raw2d[1::2, 1::2] = B
    out_bytes = raw2d.tobytes()
    if len(out_bytes) != EXPECTED_BYTES:
        raise RuntimeError(f"unexpected output size: {len(out_bytes)}")
    return out_bytes


def verify_bit_exact_roundtrip(raw_path) -> dict:
    """Load a real .raw capture, roundtrip through packed CFA, confirm
    bit-exact match. Returns a dict with verification details (max_abs_diff,
    bayer_hash_before/after, ok).

    `raw_path` may be a Path or string.
    """
    from pathlib import Path
    from blake3 import blake3
    p = Path(raw_path)
    raw_bytes = p.read_bytes()
    if len(raw_bytes) != EXPECTED_BYTES:
        return {"path": str(p), "ok": False, "error": f"size {len(raw_bytes)} != {EXPECTED_BYTES}"}
    h_before = blake3(raw_bytes).hexdigest()
    cfa = bayer_rg8_to_packed_cfa(raw_bytes)
    raw_back = packed_cfa_to_bayer_rg8(cfa)
    h_after = blake3(raw_back).hexdigest()
    bitexact = (raw_back == raw_bytes)
    max_abs_diff = 0
    if not bitexact:
        a = np.frombuffer(raw_bytes, dtype=np.uint8)
        b = np.frombuffer(raw_back, dtype=np.uint8)
        max_abs_diff = int(np.abs(a.astype(np.int32) - b.astype(np.int32)).max())
    return {
        "path": str(p),
        "n_bytes": len(raw_bytes),
        "cfa_shape": list(cfa.shape),
        "cfa_dtype": str(cfa.dtype),
        "bayer_hash_before": h_before,
        "bayer_hash_after": h_after,
        "bitexact": bool(bitexact),
        "hashes_match": h_before == h_after,
        "max_abs_diff": max_abs_diff,
        "ok": bool(bitexact and h_before == h_after),
    }


def quantize_packed_cfa_to_uint8(cfa_float01) -> np.ndarray:
    """Float [0, 1] packed CFA → uint8 [0, 255] packed CFA.

    Phase F editor outputs are float32 in [0, 1] after sigmoid. The protocol's
    chain-byte commitment hashes uint8 BayerRG8 raw bytes — so the editor's
    float must be quantized to uint8 before `packed_cfa_to_bayer_rg8`.

    Quantization: clamp to [0, 1], multiply by 255, round-to-nearest,
    cast to uint8. The roundtrip uint8 → bytes → uint8 is bit-exact;
    the float → uint8 step is lossy (a one-way commitment).

    Args:
      cfa_float01: shape (4, 2300, 2660), torch.Tensor or np.ndarray, dtype
                   float, values nominally in [0, 1] (clamped if outside).
    Returns:
      np.ndarray shape (4, 2300, 2660) dtype uint8.
    """
    if isinstance(cfa_float01, torch.Tensor):
        cfa_float01 = cfa_float01.detach().cpu().numpy()
    cfa_float01 = np.asarray(cfa_float01)
    if cfa_float01.shape != (4, HALF_H, HALF_W):
        raise ValueError(f"expected shape (4, {HALF_H}, {HALF_W}), got {cfa_float01.shape}")
    arr = np.clip(cfa_float01, 0.0, 1.0)
    arr = (arr * 255.0).round().astype(np.uint8)
    return arr


def editor_output_to_chain_bytes(cfa_float01) -> tuple[bytes, np.ndarray]:
    """End-to-end: editor float [0,1] → uint8 → BayerRG8 raw bytes.

    Returns (raw_bytes, cfa_uint8) so callers can also keep the uint8 packed
    representation for any downstream consumers.
    """
    cfa_uint8 = quantize_packed_cfa_to_uint8(cfa_float01)
    raw_bytes = packed_cfa_to_bayer_rg8(cfa_uint8)
    return raw_bytes, cfa_uint8


def verify_uint8_stage_bitexact(cfa_float01) -> dict:
    """Verify the uint8 stage of the editor pipeline is bit-exact:
        float → quantize → uint8 packed CFA → bayer bytes → packed CFA → uint8 packed CFA
    The two uint8 packed CFAs MUST match byte-for-byte (only the float→uint8
    step is lossy; everything after is exact).
    """
    cfa_uint8_a = quantize_packed_cfa_to_uint8(cfa_float01)
    raw_bytes = packed_cfa_to_bayer_rg8(cfa_uint8_a)
    cfa_uint8_b = bayer_rg8_to_packed_cfa(raw_bytes)
    bitexact = bool(np.array_equal(cfa_uint8_a, cfa_uint8_b))
    max_diff = int(np.abs(cfa_uint8_a.astype(np.int32) - cfa_uint8_b.astype(np.int32)).max())
    return {
        "input_shape": list(cfa_float01.shape) if hasattr(cfa_float01, "shape") else None,
        "uint8_stage_bitexact": bitexact,
        "max_abs_diff": max_diff,
        "raw_byte_count": len(raw_bytes),
        "n_pixels": int(cfa_uint8_a.size),
    }


__all__ = [
    "WIDTH", "HEIGHT", "HALF_H", "HALF_W", "EXPECTED_BYTES",
    "bayer_rg8_to_packed_cfa", "packed_cfa_to_bayer_rg8",
    "verify_bit_exact_roundtrip",
    "quantize_packed_cfa_to_uint8",
    "editor_output_to_chain_bytes",
    "verify_uint8_stage_bitexact",
]

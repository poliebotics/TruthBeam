"""Per-octave XOF recovery metrics for exp001b.

Bit recovery: (matching bits) / (total bits). Random ≈ 0.5.
Byte recovery: (matching bytes) / (total bytes). Random ≈ 1/256.
L2 per byte: mean squared error in byte_value space (0..255), unsquared if reported as RMS.

All inputs are uint8 tensors of identical shape (B, 3, h, w).
"""
from __future__ import annotations

import numpy as np
import torch

# popcount lookup for 0..255
_POPCOUNT_NP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def floats_to_bytes(t: torch.Tensor) -> torch.Tensor:
    """[0, 1] float -> uint8 byte tensor (round, clamp)."""
    return (t * 255.0).round().clamp(0, 255).to(torch.uint8)


def bit_recovery(pred_bytes: torch.Tensor, true_bytes: torch.Tensor) -> float:
    """Mean bit accuracy. Random ≈ 0.5, perfect = 1.0."""
    assert pred_bytes.shape == true_bytes.shape
    assert pred_bytes.dtype == torch.uint8 and true_bytes.dtype == torch.uint8
    xor = (pred_bytes ^ true_bytes).cpu().numpy()
    n_diff_bits = int(_POPCOUNT_NP[xor].sum())
    n_total_bits = xor.size * 8
    return 1.0 - n_diff_bits / n_total_bits


def byte_recovery(pred_bytes: torch.Tensor, true_bytes: torch.Tensor) -> float:
    """Fraction of bytes exactly equal. Random ≈ 1/256."""
    return (pred_bytes == true_bytes).float().mean().item()


def rms_byte_error(pred_bytes: torch.Tensor, true_bytes: torch.Tensor) -> float:
    """RMS error in byte units (0..255)."""
    diff = pred_bytes.to(torch.int32) - true_bytes.to(torch.int32)
    return float(diff.float().pow(2).mean().sqrt().item())


def per_octave_metrics(
    pred_octs: list[torch.Tensor], true_octs: list[torch.Tensor]
) -> dict[str, float]:
    """Compute per-octave bit/byte recovery and RMS byte error.

    Inputs may be float tensors in [0, 1] (predictions/targets in training space)
    or uint8 byte tensors. Floats are rounded to bytes before comparison.

    Returns flat dict: {oct{i}_bit, oct{i}_byte, oct{i}_rms, all_bit, all_byte, all_rms}.
    """
    out: dict[str, float] = {}
    pred_b_all = []
    true_b_all = []
    for i, (p, t) in enumerate(zip(pred_octs, true_octs)):
        pb = floats_to_bytes(p) if p.dtype != torch.uint8 else p
        tb = floats_to_bytes(t) if t.dtype != torch.uint8 else t
        out[f"oct{i}_bit"] = bit_recovery(pb, tb)
        out[f"oct{i}_byte"] = byte_recovery(pb, tb)
        out[f"oct{i}_rms"] = rms_byte_error(pb, tb)
        pred_b_all.append(pb.flatten())
        true_b_all.append(tb.flatten())
    pred_cat = torch.cat(pred_b_all)
    true_cat = torch.cat(true_b_all)
    out["all_bit"] = bit_recovery(pred_cat, true_cat)
    out["all_byte"] = byte_recovery(pred_cat, true_cat)
    out["all_rms"] = rms_byte_error(pred_cat, true_cat)
    return out

"""Per-octave L2 loss for direct XOF-byte prediction (exp001b).

Each octave's loss is its per-byte MSE. Summing per-byte MSEs across octaves
makes each octave contribute equally regardless of its byte count — the large
oct3 (97,200 bytes) doesn't drown out oct0 (1,530 bytes).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def xof_l2_loss(
    preds: list[torch.Tensor], trues: list[torch.Tensor]
) -> tuple[torch.Tensor, dict[str, float]]:
    """preds: list of 4 sigmoid tensors (B, 3, h_i, w_i), values in [0, 1].
    trues: list of 4 ground-truth byte tensors (B, 3, h_i, w_i), values in [0, 1]
           (i.e., true_byte / 255).

    Returns (total_loss, parts_dict). total_loss is sum over octaves of per-byte MSE.
    """
    parts: dict[str, float] = {}
    total = preds[0].new_zeros(())
    for i, (p, t) in enumerate(zip(preds, trues)):
        mse = F.mse_loss(p, t)
        parts[f"l2_oct{i}"] = mse.item()
        total = total + mse
    parts["total"] = total.item()
    return total, parts

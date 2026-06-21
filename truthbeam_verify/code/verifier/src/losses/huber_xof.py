"""Per-octave Huber loss with weights {oct0: 1.0, oct1: 1.0, oct2: 0.5, oct3: 0.25}.

For A0 (oct0+oct1 only), the higher-octave heads are not constructed and the
loss is naturally limited to the predicted octaves. The weight tuple is
indexed by octave number; we only sum over present heads.

Targets are centered: target = (raw_byte - 127.5) / 127.5 ∈ [-1, 1].
Predictions are tanh-bounded ∈ [-1, 1].
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

OCTAVE_WEIGHTS = (1.0, 1.0, 0.5, 0.25)
HUBER_BETA = 1.0  # SmoothL1 default; symmetric quadratic-then-linear knee


def huber_xof_loss(
    preds: list[torch.Tensor | None],
    targets: list[torch.Tensor],
    weights: tuple[float, ...] = OCTAVE_WEIGHTS,
    beta: float = HUBER_BETA,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Per-octave SmoothL1 with operator-spec'd weights.

    `preds[i]` is the model output for octave i, or None if not predicted (A0).
    `targets[i]` is the centered ground-truth, always provided.

    Returns (total_loss, parts_dict). `total_loss` is a weighted sum of per-octave
    SmoothL1 losses; `parts_dict` has per-octave loss values + total + which were used.
    """
    if len(preds) != len(targets):
        raise ValueError(f"len(preds)={len(preds)} != len(targets)={len(targets)}")
    if len(weights) < len(preds):
        raise ValueError(f"len(weights)={len(weights)} < len(preds)={len(preds)}")

    parts: dict[str, float] = {}
    total = preds[0].new_zeros(()) if preds[0] is not None else None
    n_used = 0
    for i, (p, t) in enumerate(zip(preds, targets)):
        if p is None:
            parts[f"oct{i}_loss"] = float("nan")
            parts[f"oct{i}_used"] = False
            continue
        l = F.smooth_l1_loss(p, t, beta=beta)
        weighted = weights[i] * l
        parts[f"oct{i}_loss"] = l.item()
        parts[f"oct{i}_used"] = True
        if total is None:
            total = weighted
        else:
            total = total + weighted
        n_used += 1
    if total is None:
        raise ValueError("no octave heads enabled; cannot compute loss")
    parts["total"] = total.item()
    parts["n_octaves_used"] = n_used
    return total, parts

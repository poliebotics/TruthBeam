"""Emission RGB recovery loss: Charbonnier + multi-scale L1 (exp001c)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((pred - target) ** 2 + eps ** 2).mean()


def multiscale_l1(pred: torch.Tensor, target: torch.Tensor, scales: tuple[float, ...] = (0.5, 0.25)) -> torch.Tensor:
    """L1 at additional spatial scales (in addition to the full-resolution charbonnier)."""
    total = pred.new_zeros(())
    for s in scales:
        size = (max(1, int(pred.shape[-2] * s)), max(1, int(pred.shape[-1] * s)))
        p_s = F.interpolate(pred, size=size, mode="bilinear", align_corners=False)
        t_s = F.interpolate(target, size=size, mode="bilinear", align_corners=False)
        total = total + F.l1_loss(p_s, t_s)
    return total


def emission_loss(
    pred: torch.Tensor, target: torch.Tensor, ms_weight: float = 0.1, eps: float = 1e-3,
) -> tuple[torch.Tensor, dict[str, float]]:
    charb = charbonnier_loss(pred, target, eps)
    ms = multiscale_l1(pred, target)
    total = charb + ms_weight * ms
    return total, {
        "charb": charb.item(),
        "ms_l1": ms.item(),
        "total": total.item(),
    }

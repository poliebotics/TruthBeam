"""Per-epoch metrics for emission RGB recovery (exp001c).

PSNR computed in [0,1] space: 20 * log10(1 / RMSE).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> float:
    """PSNR in dB. pred/target in [0, 1]."""
    mse = ((pred - target) ** 2).mean().item()
    if mse < eps:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def per_channel_l1(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """L1 per RGB channel."""
    diff = (pred - target).abs()
    return {f"l1_{name}": diff[:, i].mean().item() for i, name in enumerate("rgb")}


def multiscale_psnr(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """PSNR at full res and downsampled (1/2, 1/4)."""
    out = {"psnr_full": psnr(pred, target)}
    for s, name in [(0.5, "half"), (0.25, "quarter")]:
        h = max(1, int(pred.shape[-2] * s))
        w = max(1, int(pred.shape[-1] * s))
        p_s = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
        t_s = F.interpolate(target, size=(h, w), mode="bilinear", align_corners=False)
        out[f"psnr_{name}"] = psnr(p_s, t_s)
    return out


def emission_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    out: dict[str, float] = {}
    out.update(multiscale_psnr(pred, target))
    out.update(per_channel_l1(pred, target))
    out["l1_all"] = (pred - target).abs().mean().item()
    return out

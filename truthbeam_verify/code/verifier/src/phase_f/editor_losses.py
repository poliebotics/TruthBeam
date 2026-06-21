"""Phase F editor losses.

L_recon          Charbonnier on (C_pred, C_target)
L_grad           multiscale gradient/high-freq loss (Sobel × multiple scales)
L_temporal       multi-frame consistency (placeholder for F-A which is single-frame)
L_disc           generator-side adversarial loss from a conditional patch
                 discriminator
L_binder         ensemble surrogate-binder margin loss: each surrogate scores
                 (C_pred → emission) and we want score(target) > score(source)
                 by at least `margin`.

Default coefficients (Phase F-A):
  L_recon = 1.0, L_grad = 0.1, L_temporal = 0.0, L_disc = 0.5, L_binder = 0.5
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def charbonnier(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((pred - target) ** 2 + eps ** 2).mean()


def _sobel_grad(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sobel gradients (gx, gy), grouped per-channel."""
    C = x.shape[1]
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    kx = kx.expand(C, 1, 3, 3)
    ky = ky.expand(C, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1, groups=C)
    gy = F.conv2d(x, ky, padding=1, groups=C)
    return gx, gy


def grad_loss(pred: torch.Tensor, target: torch.Tensor, scales: Sequence[float] = (1.0, 0.5, 0.25)) -> torch.Tensor:
    total = pred.new_zeros(())
    for s in scales:
        if s != 1.0:
            h = max(1, int(pred.shape[-2] * s))
            w = max(1, int(pred.shape[-1] * s))
            p = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
            t = F.interpolate(target, size=(h, w), mode="bilinear", align_corners=False)
        else:
            p, t = pred, target
        gxp, gyp = _sobel_grad(p)
        gxt, gyt = _sobel_grad(t)
        total = total + (gxp - gxt).abs().mean() + (gyp - gyt).abs().mean()
    return total / max(len(scales), 1)


def disc_g_loss(disc_fake_logits: torch.Tensor) -> torch.Tensor:
    """Hinge-style generator loss given discriminator's fake-pair logits."""
    return -disc_fake_logits.mean()


def binder_margin_loss(
    binder_outputs_target: Sequence[torch.Tensor],
    binder_outputs_source: Sequence[torch.Tensor],
    E_target: torch.Tensor,
    E_source: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Average across binders of:
        max(0, margin - (PSNR(binder(C_pred), E_target) - PSNR(binder(C_pred), E_source)))

    Each `binder_outputs_*` is a list of binder predictions on C_pred — the
    same prediction scored against both E_target (matched) and E_source
    (the wrong target the editor must NOT regress toward).
    """
    # We expect each binder to output a (B, 3, H, W) prediction matching emissions.
    # PSNR via L2; differentiable; lower L2 = higher PSNR.
    losses = []
    for o_t, o_s in zip(binder_outputs_target, binder_outputs_source):
        l_t = ((o_t - E_target) ** 2).mean(dim=(-3, -2, -1))   # (B,)
        l_s = ((o_s - E_source) ** 2).mean(dim=(-3, -2, -1))   # (B,)
        # We want l_t to be SMALL (binder recovers E_target from C_pred)
        # and l_s to be LARGE (binder cannot recover E_source from C_pred).
        # Margin: l_t + margin <= l_s. Loss = max(0, l_t + margin - l_s).
        diff = l_t + margin - l_s
        losses.append(diff.clamp(min=0).mean())
    return torch.stack(losses).mean() if losses else torch.zeros((), device=E_target.device)


def total_editor_loss(
    *,
    C_pred: torch.Tensor,
    C_target: torch.Tensor,
    disc_fake_logits: torch.Tensor | None = None,
    binder_outputs_target: Sequence[torch.Tensor] | None = None,
    binder_outputs_source: Sequence[torch.Tensor] | None = None,
    E_target: torch.Tensor | None = None,
    E_source: torch.Tensor | None = None,
    coef_recon: float = 1.0,
    coef_grad: float = 0.1,
    coef_disc: float = 0.5,
    coef_binder: float = 0.5,
    binder_margin: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    L_recon = charbonnier(C_pred, C_target)
    L_grad = grad_loss(C_pred, C_target)
    L_disc = disc_g_loss(disc_fake_logits) if disc_fake_logits is not None else torch.zeros((), device=C_pred.device)
    L_binder = (binder_margin_loss(binder_outputs_target, binder_outputs_source,
                                    E_target, E_source, margin=binder_margin)
                if (binder_outputs_target is not None and binder_outputs_source is not None)
                else torch.zeros((), device=C_pred.device))
    total = (coef_recon * L_recon + coef_grad * L_grad
             + coef_disc * L_disc + coef_binder * L_binder)
    parts = {
        "L_recon": L_recon.item(),
        "L_grad":  L_grad.item(),
        "L_disc":  L_disc.item(),
        "L_binder": L_binder.item(),
        "total":   total.item(),
    }
    return total, parts

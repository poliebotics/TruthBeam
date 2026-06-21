"""Constrained STN preprocessing module (audit Q9, A7).

Architecture (operator-specified):

  Localization network:
    Conv2d(in_channels → 16, kernel=7, stride=4)
    GELU
    Conv2d(16 → 32, kernel=5, stride=4)
    GELU
    AdaptiveAvgPool2d(8)

  FC head:
    Linear(32×8×8 = 2048 → 64)
    GELU
    Linear(64 → 6)

  Identity init: final Linear weight zeroed, bias = [1, 0, 0, 0, 1, 0]
                 (so theta is the identity affine at init).

  Forward: F.affine_grid + F.grid_sample with
           align_corners=False, padding_mode='border' (NOT zeros).

Position in pipeline: applied to packed CFA AFTER per-CFA-phase normalization,
BEFORE the encoder.

Logging hooks: `decompose_theta` returns translation/scale/rotation/det;
`out_of_bounds_fraction(grid)` returns the OOB share BEFORE grid_sample is run.

Identity regularizer: caller computes `F.mse_loss(theta, identity_2x3)` and adds
λ × it to the total loss every step (λ = 0.01 per Q9).

A7 is a reconstructor preprocessing ablation. **Do not use STN on the
raw-provenance / XOF path** — spatial warping is forbidden there per CEE spec.
The XOF path receives the un-warped packed CFA + normalization; the
reconstructor (emission predictor) receives the warped tensor.

If A7 ends up beating A1 via large transforms, high OOB fraction, or obvious
cropping: that's a failure mode (the model is "winning" by hiding hard
content), not a result. Caller is responsible for flagging this in findings.md.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _identity_2x3(batch: int, device, dtype) -> torch.Tensor:
    eye = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype)
    return eye.unsqueeze(0).expand(batch, -1, -1)


class ConstrainedSTN(nn.Module):
    def __init__(self, in_channels: int = 4):
        super().__init__()
        self.localizer = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=7, stride=4),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=4),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(8),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 8 * 8, 64),
            nn.GELU(),
            nn.Linear(64, 6),
        )
        # Identity init.
        with torch.no_grad():
            self.fc[-1].weight.zero_()
            self.fc[-1].bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (warped_x, theta(B,2,3), grid(B,H,W,2)).

        Caller adds the identity regularizer to total loss and logs grid OOB.
        """
        feat = self.localizer(x).flatten(1)
        theta = self.fc(feat).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        warped = F.grid_sample(
            x, grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return warped, theta, grid


def out_of_bounds_fraction(grid: torch.Tensor) -> float:
    """grid: (B, H, W, 2). Fraction of sampled positions outside [-1, 1] in either axis."""
    oob = ((grid[..., 0].abs() > 1) | (grid[..., 1].abs() > 1)).float().mean()
    return float(oob.item())


def decompose_theta(theta: torch.Tensor) -> dict[str, list[float]]:
    """Per-batch decomposition of a 2x3 affine theta into interpretable scalars.

    Logged per validation pass:
      tx, ty: translation (theta[:, :, 2])
      sx, sy: column-wise norms of the 2x2 linear part (≈ scale along x, y)
      shear:  off-diagonal coupling, theta[:, 0, 1] / max(|theta[:, 0, 0]|, eps)
      det:    determinant of the 2x2 linear part
    All returned as Python lists, one entry per batch sample.
    """
    A = theta[:, :, :2]   # (B, 2, 2)
    t = theta[:, :, 2]    # (B, 2)
    sx = A[:, :, 0].norm(dim=-1)
    sy = A[:, :, 1].norm(dim=-1)
    det = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]
    eps = 1e-6
    shear = A[:, 0, 1] / A[:, 0, 0].abs().clamp_min(eps)
    return {
        "tx": t[:, 0].tolist(),
        "ty": t[:, 1].tolist(),
        "sx": sx.tolist(),
        "sy": sy.tolist(),
        "shear": shear.tolist(),
        "det": det.tolist(),
    }


def identity_regularizer(theta: torch.Tensor) -> torch.Tensor:
    """Scalar MSE between theta and identity affine. Multiply by λ in caller (default 0.01)."""
    target = _identity_2x3(theta.shape[0], theta.device, theta.dtype)
    return F.mse_loss(theta, target)

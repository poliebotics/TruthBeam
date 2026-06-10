"""Phase F conditional patch discriminator.

D(C, E) → (B, 1, H', W') logits; predicts per-patch real-vs-fake.

The "condition" is the emission tile. We feed (C concatenated with
emission-resized-to-capture-resolution) so the discriminator sees the pair.
Spectral norm + LeakyReLU. PatchGAN-style 3-strided-conv backbone.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def _spectral_conv(in_ch: int, out_ch: int, k: int = 4, s: int = 2, p: int = 1) -> nn.Module:
    return spectral_norm(nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p))


class PatchDiscriminator(nn.Module):
    """Conditional patch discriminator on (C, E) pairs.

    Args:
        in_channels: 4 (packed CFA) + 3 (emission RGB) = 7 by default.
        base: base channel count.
    """

    def __init__(self, in_channels: int = 7, base: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            _spectral_conv(in_channels, base, k=4, s=2, p=1),  # /2
            nn.LeakyReLU(0.2, inplace=True),
            _spectral_conv(base, base * 2, 4, 2, 1),           # /4
            nn.GroupNorm(8, base * 2),
            nn.LeakyReLU(0.2, inplace=True),
            _spectral_conv(base * 2, base * 4, 4, 2, 1),       # /8
            nn.GroupNorm(16, base * 4),
            nn.LeakyReLU(0.2, inplace=True),
            _spectral_conv(base * 4, base * 8, 4, 1, 1),       # stride-1
            nn.GroupNorm(32, base * 8),
            nn.LeakyReLU(0.2, inplace=True),
            _spectral_conv(base * 8, 1, 4, 1, 1),
        )

    def forward(self, C: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        # Resize E to C's spatial dims, then concat along channel axis.
        if E.shape[-2:] != C.shape[-2:]:
            E_r = F.interpolate(E, size=C.shape[-2:], mode="bilinear", align_corners=False)
        else:
            E_r = E
        x = torch.cat([C, E_r], dim=1)
        return self.net(x)


def disc_d_loss_hinge(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """Hinge discriminator loss."""
    real_loss = F.relu(1.0 - real_logits).mean()
    fake_loss = F.relu(1.0 + fake_logits).mean()
    return 0.5 * (real_loss + fake_loss)

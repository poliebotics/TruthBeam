"""Fresh binder Family D — phase-aware (HELD OUT FROM TRAINING).

Architecture:
    input pre-processing: raw 4-ch CFA + DoG (3 scales × 4 ch = 12) = 16-ch
                          stack. Optionally + 1-ch per-pixel mean across the
                          4 CFA channels = 17-ch (default off, configurable).
    encoder: small CNN — 5 blocks @ (64, 128, 256, 512, 512), 3x3 kernels,
             GroupNorm (NOT BatchNorm), stride-2 between blocks, GELU.
             NO ImageNet pretrain.
    decoder: symmetric upsample with skip connections + 1x1 conv head
    head:    1x1 conv → 3-ch sigmoid
    final:   F.interpolate to (1080, 1920)

Distinguishing feature: deliberately maximally distinct from existing Phase E
binders. Hand-crafted DoG features as input augmentation, small kernels, no
ImageNet pretrain, GroupNorm. This is the manuscript's load-bearing held-out
architecture family — F-A v2 evaluated against Family D is the strongest
"unseen architecture transfer" claim available.

Estimated params: ~5M.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# DoG sigmas (at native resolution; scales over input)
DOG_SIGMAS = ((1.0, 2.0), (2.0, 4.0), (4.0, 8.0))


def _gn_groups(ch: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


def _make_gauss_kernel(sigma: float, ksize: int | None = None) -> torch.Tensor:
    """1D Gaussian kernel; normalized."""
    if ksize is None:
        ksize = max(3, int(2 * round(3 * sigma) + 1))
    half = ksize // 2
    x = torch.arange(-half, half + 1, dtype=torch.float32)
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


def _gauss_blur_2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Per-channel separable Gaussian blur (B, C, H, W)."""
    g1d = _make_gauss_kernel(sigma).to(x.device, dtype=x.dtype)
    k = g1d.view(1, 1, -1)
    C = x.shape[1]
    # Horizontal
    kh = k.expand(C, 1, k.shape[-1])
    pad = k.shape[-1] // 2
    x = F.conv2d(x, kh.unsqueeze(2), padding=(0, pad), groups=C)
    # Vertical
    kv = k.expand(C, 1, k.shape[-1])
    x = F.conv2d(x, kv.unsqueeze(3), padding=(pad, 0), groups=C)
    return x


def dog_features(cfa: torch.Tensor, sigmas: tuple = DOG_SIGMAS) -> torch.Tensor:
    """Compute Difference-of-Gaussians at multiple scales.
    Input: (B, 4, H, W) CFA.
    Output: (B, 4*len(sigmas), H, W) DoG features.
    """
    feats = []
    for sigma_lo, sigma_hi in sigmas:
        d = _gauss_blur_2d(cfa, sigma_lo) - _gauss_blur_2d(cfa, sigma_hi)
        feats.append(d)
    return torch.cat(feats, dim=1)


class _ConvBlock(nn.Module):
    """3x3 conv → GroupNorm → GELU, optionally followed by stride-2 downsample."""

    def __init__(self, in_ch: int, out_ch: int, downsample: bool):
        super().__init__()
        g = _gn_groups(out_ch)
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
        ]
        if downsample:
            layers.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1, bias=False))
            layers.append(nn.GroupNorm(g, out_ch))
            layers.append(nn.GELU())
        self.b = nn.Sequential(*layers)

    def forward(self, x): return self.b(x)


class FreshBinderD(nn.Module):
    """Family D — phase-aware DoG-augmented small CNN, no pretrain."""

    def __init__(self, emission_h: int = 1080, emission_w: int = 1920,
                 in_channels: int = 4,
                 use_dog: bool = True,
                 use_channel_mean: bool = False,
                 enc_dims: tuple = (64, 128, 256, 512, 512)):
        super().__init__()
        self.emission_h = emission_h
        self.emission_w = emission_w
        self.use_dog = use_dog
        self.use_channel_mean = use_channel_mean
        # Effective input channels: 4 raw + (12 DoG if use_dog) + (1 mean if use_channel_mean)
        eff_in = in_channels
        if use_dog: eff_in += in_channels * len(DOG_SIGMAS)
        if use_channel_mean: eff_in += 1
        self.eff_in = eff_in
        # 5 conv blocks: stem (no downsample) + 4 downsampling stages
        d0, d1, d2, d3, d4 = enc_dims
        self.b0 = _ConvBlock(eff_in, d0, downsample=False)
        self.b1 = _ConvBlock(d0, d1, downsample=True)
        self.b2 = _ConvBlock(d1, d2, downsample=True)
        self.b3 = _ConvBlock(d2, d3, downsample=True)
        self.b4 = _ConvBlock(d3, d4, downsample=True)
        # Decoder: symmetric upsample with skip from b3, b2, b1, b0
        gn_g = lambda c: _gn_groups(c)  # noqa: E731
        self.up3 = nn.Sequential(
            nn.Conv2d(d4 + d3, d3, 3, padding=1, bias=False),
            nn.GroupNorm(gn_g(d3), d3), nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(d3 + d2, d2, 3, padding=1, bias=False),
            nn.GroupNorm(gn_g(d2), d2), nn.GELU(),
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(d2 + d1, d1, 3, padding=1, bias=False),
            nn.GroupNorm(gn_g(d1), d1), nn.GELU(),
        )
        self.up0 = nn.Sequential(
            nn.Conv2d(d1 + d0, d0, 3, padding=1, bias=False),
            nn.GroupNorm(gn_g(d0), d0), nn.GELU(),
        )
        self.head = nn.Conv2d(d0, 3, 1)

    def forward(self, capture: torch.Tensor) -> torch.Tensor:
        x = capture
        if self.use_dog:
            dog = dog_features(capture)
            x = torch.cat([x, dog], dim=1)
        if self.use_channel_mean:
            mean = capture.mean(dim=1, keepdim=True)
            x = torch.cat([x, mean], dim=1)
        f0 = self.b0(x)
        f1 = self.b1(f0)
        f2 = self.b2(f1)
        f3 = self.b3(f2)
        f4 = self.b4(f3)
        # Decoder up
        u = F.interpolate(f4, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        u = self.up3(torch.cat([u, f3], dim=1))
        u = F.interpolate(u, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        u = self.up2(torch.cat([u, f2], dim=1))
        u = F.interpolate(u, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        u = self.up1(torch.cat([u, f1], dim=1))
        u = F.interpolate(u, size=f0.shape[-2:], mode="bilinear", align_corners=False)
        u = self.up0(torch.cat([u, f0], dim=1))
        x = self.head(u)
        if x.shape[-2:] != (self.emission_h, self.emission_w):
            x = F.interpolate(x, size=(self.emission_h, self.emission_w),
                              mode="bilinear", align_corners=False)
        return torch.sigmoid(x)

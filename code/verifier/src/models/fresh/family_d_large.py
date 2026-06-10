"""Fresh binder Family D-large — phase-aware, scaled-up architecture.

Iteration 1 of D-redesign per operator authorization 2026-05-04 (PART B).
D1 (the canonical small-D) failed acceptance at PSNR 15.55 dB (threshold
≥ 22.0 dB). Operator authorized iterative D-family redesign on idle GPUs
to find a D variant that passes acceptance and serves as the held-out
report binder.

Distinguishing features preserved (vs A/B/C surrogates):
    - NO ImageNet pretrain.
    - GroupNorm throughout (NOT BatchNorm).
    - DoG-augmented input (12-ch DoG features at 3 scales).
    - Small-kernel CNN (3×3 throughout, no patchify-style stems).

Scale-up vs D1:
    - Wider channels: (96, 192, 384, 768, 768) — 1.5× D1's (64, 128, 256, 512, 512).
    - Deeper encoder blocks: 3 conv layers per stage (D1: 2).
    - Deeper decoder up-blocks: 2 conv layers per up (D1: 1).
    - Same skip-connection topology (b0..b3 → up0..up3).

Estimated params: 50-70M (target ~50M; if overshoots, dial back width on
later iterations). Empirical param count printed on instantiation.

Inputs: (B, 4, capture_h, capture_w) packed CFA in [0, 1].
Output: (B, 3, 1080, 1920) RGB in [0, 1] via sigmoid.

Capture HW: 1150×1330 (matches D1; full-res 2300×2660 is infeasible on
A100 — D2 OOM precedent).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .family_d_phase_aware import (
    DOG_SIGMAS, _gn_groups, dog_features,
)


class _ConvBlockDeep(nn.Module):
    """3-conv variant: 3x3 conv → GN → GELU repeated 3 times, optional stride-2 downsample.

    D1 used 2-conv blocks. D-large uses 3-conv to add depth without changing
    receptive-field rules.
    """

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


class _UpBlockDeep(nn.Module):
    """2-conv up-block: bilinear interpolate to skip resolution, concat skip,
    then 2× (3x3 conv → GN → GELU). D1 used 1 conv; D-large uses 2 for
    decoder capacity."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        g = _gn_groups(out_ch)
        merge_in = in_ch + skip_ch
        self.block = nn.Sequential(
            nn.Conv2d(merge_in, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
        )

    def forward(self, x, skip):
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                              align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class FreshBinderDLarge(nn.Module):
    """Family D-large — scaled-up phase-aware DoG+small-kernel CNN, no pretrain."""

    def __init__(self, emission_h: int = 1080, emission_w: int = 1920,
                 in_channels: int = 4,
                 use_dog: bool = True,
                 use_channel_mean: bool = False,
                 enc_dims: tuple = (96, 192, 384, 768, 768)):
        super().__init__()
        self.emission_h = emission_h
        self.emission_w = emission_w
        self.use_dog = use_dog
        self.use_channel_mean = use_channel_mean
        eff_in = in_channels
        if use_dog: eff_in += in_channels * len(DOG_SIGMAS)
        if use_channel_mean: eff_in += 1
        self.eff_in = eff_in
        d0, d1, d2, d3, d4 = enc_dims
        # Encoder: 5 stages, stem (no downsample) + 4 downsampling
        self.b0 = _ConvBlockDeep(eff_in, d0, downsample=False)
        self.b1 = _ConvBlockDeep(d0, d1, downsample=True)
        self.b2 = _ConvBlockDeep(d1, d2, downsample=True)
        self.b3 = _ConvBlockDeep(d2, d3, downsample=True)
        self.b4 = _ConvBlockDeep(d3, d4, downsample=True)
        # Decoder: 4 up-blocks, deeper than D1 (2 conv layers per up)
        self.up3 = _UpBlockDeep(d4, d3, d3)
        self.up2 = _UpBlockDeep(d3, d2, d2)
        self.up1 = _UpBlockDeep(d2, d1, d1)
        self.up0 = _UpBlockDeep(d1, d0, d0)
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
        u = self.up3(f4, f3)
        u = self.up2(u, f2)
        u = self.up1(u, f1)
        u = self.up0(u, f0)
        x = self.head(u)
        if x.shape[-2:] != (self.emission_h, self.emission_w):
            x = F.interpolate(x, size=(self.emission_h, self.emission_w),
                              mode="bilinear", align_corners=False)
        return torch.sigmoid(x)

"""Fresh binder Family B — ConvNeXt-Tiny + FPN.

Architecture:
    encoder: ConvNeXt-Tiny with `in_chans=4` (modified stem), ImageNet pretrain
    decoder: FPN-style top-down + lateral 1x1 connections at 4 scales
    head:    3x3 conv → 1x1 conv → 3-ch sigmoid
    final:   F.interpolate to (1080, 1920)

Distinguishing feature: FPN-style multi-scale fusion (vs A's pure U-Net).
Encoder is ImageNet-pretrained ConvNeXt-Tiny (vs A's no-pretrain). Operator
flagged this is architecturally close to e3r-fam (which also uses FPN); the
encoder difference (ConvNeXt vs e3r's V2 architecture) is the family
separator.

Estimated params: ~30M.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


ENC_DIMS_B = (96, 192, 384, 768)
FPN_CH = 128


def _gn_groups(ch: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


class FreshBinderB(nn.Module):
    """Family B — ConvNeXt-Tiny + FPN."""

    def __init__(self, emission_h: int = 1080, emission_w: int = 1920,
                 in_channels: int = 4, fpn_channels: int = FPN_CH,
                 pretrained: bool = True):
        super().__init__()
        self.emission_h = emission_h
        self.emission_w = emission_w
        # Codex audit MED 2026-05-03: pretrained kwarg lets eval-time
        # reconstruction skip the timm download (weights will be overwritten
        # from the saved ckpt anyway).
        self.encoder = timm.create_model(
            "convnext_tiny", pretrained=pretrained, features_only=True,
            in_chans=in_channels, out_indices=(0, 1, 2, 3),
        )
        # Lateral 1x1 convs for each FPN level
        self.lat = nn.ModuleList([
            nn.Conv2d(c, fpn_channels, 1) for c in ENC_DIMS_B
        ])
        # 3x3 smoothing convs after merge at each level
        gn_g = _gn_groups(fpn_channels)
        self.smooth = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False),
                nn.GroupNorm(gn_g, fpn_channels),
                nn.GELU(),
            ) for _ in range(4)
        ])
        self.head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False),
            nn.GroupNorm(gn_g, fpn_channels),
            nn.GELU(),
            nn.Conv2d(fpn_channels, 3, 1),
        )

    def forward(self, capture: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(capture)
        f0, f1, f2, f3 = feats
        # Top-down FPN: start from deepest, upsample + add lateral
        p3 = self.lat[3](f3)
        p2 = self.lat[2](f2) + F.interpolate(p3, size=f2.shape[-2:],
                                              mode="bilinear", align_corners=False)
        p1 = self.lat[1](f1) + F.interpolate(p2, size=f1.shape[-2:],
                                              mode="bilinear", align_corners=False)
        p0 = self.lat[0](f0) + F.interpolate(p1, size=f0.shape[-2:],
                                              mode="bilinear", align_corners=False)
        # Smooth each level
        p0 = self.smooth[0](p0)
        p1 = self.smooth[1](p1)
        p2 = self.smooth[2](p2)
        p3 = self.smooth[3](p3)
        # Final head reads p0 (highest resolution), upsample to (1080, 1920)
        x = self.head(p0)
        if x.shape[-2:] != (self.emission_h, self.emission_w):
            x = F.interpolate(x, size=(self.emission_h, self.emission_w),
                              mode="bilinear", align_corners=False)
        return torch.sigmoid(x)

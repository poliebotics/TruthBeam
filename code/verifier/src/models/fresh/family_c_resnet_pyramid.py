"""Fresh binder Family C — ResNet-50 + dilated pyramid.

Architecture:
    encoder: ResNet-50 with `in_chans=4` (modified first conv), ImageNet pretrain
    decoder: dilated-pyramid head over the deepest feature, with lateral
             1x1 connections from earlier stages (4 scales)
    head:    3x3 group-conv refinement → 1x1 → 3-ch sigmoid
    final:   F.interpolate to (1080, 1920)

Distinguishing feature: deeper, ResNet-style backbone vs ConvNeXt (B). Pyramid
of dilated convs at the bottleneck gives multi-receptive-field context without
the explicit top-down/lateral structure of FPN.

Estimated params: ~30M.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


ENC_DIMS_C = (256, 512, 1024, 2048)
PYR_CH = 256


def _gn_groups(ch: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


class _DilatedBranch(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int):
        super().__init__()
        g = _gn_groups(out_ch)
        self.b = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
        )

    def forward(self, x): return self.b(x)


class FreshBinderC(nn.Module):
    """Family C — ResNet-50 + dilated pyramid."""

    def __init__(self, emission_h: int = 1080, emission_w: int = 1920,
                 in_channels: int = 4, pyr_channels: int = PYR_CH,
                 pretrained: bool = True):
        super().__init__()
        self.emission_h = emission_h
        self.emission_w = emission_w
        # pretrained kwarg lets eval-time
        # reconstruction skip the timm download.
        self.encoder = timm.create_model(
            "resnet50", pretrained=pretrained, features_only=True,
            in_chans=in_channels, out_indices=(1, 2, 3, 4),
        )
        # Lateral 1x1 reductions for each scale
        self.lat = nn.ModuleList([
            nn.Conv2d(c, pyr_channels, 1) for c in ENC_DIMS_C
        ])
        # Dilated pyramid at deepest scale: 4 parallel branches with dilations 1/2/4/8
        self.pyramid = nn.ModuleList([
            _DilatedBranch(pyr_channels, pyr_channels, d) for d in (1, 2, 4, 8)
        ])
        # Merge pyramid (4 × pyr_channels → pyr_channels)
        self.merge = nn.Sequential(
            nn.Conv2d(4 * pyr_channels, pyr_channels, 1, bias=False),
            nn.GroupNorm(_gn_groups(pyr_channels), pyr_channels),
            nn.GELU(),
        )
        # Refinement 3x3 group-conv at the merged scale
        self.refine = nn.Sequential(
            nn.Conv2d(pyr_channels, pyr_channels, 3, padding=1,
                      groups=_gn_groups(pyr_channels), bias=False),
            nn.GroupNorm(_gn_groups(pyr_channels), pyr_channels),
            nn.GELU(),
        )
        self.head = nn.Conv2d(pyr_channels, 3, 1)

    def forward(self, capture: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(capture)
        f0, f1, f2, f3 = feats   # strides (4, 8, 16, 32)
        # Reduce all to pyr_channels
        l0 = self.lat[0](f0)
        l1 = self.lat[1](f1)
        l2 = self.lat[2](f2)
        l3 = self.lat[3](f3)
        # Dilated pyramid at deepest scale
        p_branches = [b(l3) for b in self.pyramid]
        p = self.merge(torch.cat(p_branches, dim=1))
        # Upsample p step-by-step, adding lateral at each scale
        p = F.interpolate(p, size=l2.shape[-2:], mode="bilinear", align_corners=False) + l2
        p = F.interpolate(p, size=l1.shape[-2:], mode="bilinear", align_corners=False) + l1
        p = F.interpolate(p, size=l0.shape[-2:], mode="bilinear", align_corners=False) + l0
        p = self.refine(p)
        x = self.head(p)
        if x.shape[-2:] != (self.emission_h, self.emission_w):
            x = F.interpolate(x, size=(self.emission_h, self.emission_w),
                              mode="bilinear", align_corners=False)
        return torch.sigmoid(x)

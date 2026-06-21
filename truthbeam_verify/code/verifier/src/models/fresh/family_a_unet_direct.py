"""Fresh binder Family A — U-Net direct.

Architecture:
    encoder: ResNet-18 with `in_chans=4` (modified first conv), NO ImageNet pretrain
    decoder: U-Net-style with skip connections from encoder layers 1-4
    head:    1x1 conv → 3-ch sigmoid
    final:   F.interpolate to (1080, 1920) emission resolution

Distinguishing feature: NO ImageNet pretrain. The encoder learns from scratch
on the C→E task without ImageNet inductive biases — this is the architectural
lever that distinguishes A from B/C (which use pretrained encoders).

Estimated params: ~12M.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


# ResNet-18 per-stage output channels (from timm features_only output)
ENC_DIMS_A = (64, 128, 256, 512)


def _gn_groups(ch: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    g = _gn_groups(out_ch)
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.GroupNorm(g, out_ch),
        nn.GELU(),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.GroupNorm(g, out_ch),
        nn.GELU(),
    )


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.block = _conv_block(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:],
                                  mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.block(x)


class FreshBinderA(nn.Module):
    """Family A — U-Net direct. ResNet-18 backbone with no ImageNet pretrain."""

    def __init__(self, emission_h: int = 1080, emission_w: int = 1920,
                 in_channels: int = 4, decoder_dims: tuple = (256, 128, 64, 32, 16),
                 pretrained: bool = False):
        super().__init__()
        self.emission_h = emission_h
        self.emission_w = emission_w
        # Family A's distinguishing feature is NO ImageNet pretrain. Default
        # is False; the kwarg exists so eval can explicitly skip the timm
        # initialization download.
        self.encoder = timm.create_model(
            "resnet18", pretrained=pretrained, features_only=True,
            in_chans=in_channels, out_indices=(1, 2, 3, 4),
        )
        d3, d2, d1, d0, dx = decoder_dims
        self.up3 = _UpBlock(ENC_DIMS_A[3], ENC_DIMS_A[2], d3)
        self.up2 = _UpBlock(d3, ENC_DIMS_A[1], d2)
        self.up1 = _UpBlock(d2, ENC_DIMS_A[0], d1)
        self.up0 = _UpBlock(d1, 0, d0)
        self.up_extra = _UpBlock(d0, 0, dx)
        self.head = nn.Conv2d(dx, 3, 1)

    def forward(self, capture: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(capture)
        f0, f1, f2, f3 = feats
        x = self.up3(f3, f2)
        x = self.up2(x, f1)
        x = self.up1(x, f0)
        x = self.up0(x, None)
        x = self.up_extra(x, None)
        x = self.head(x)
        if x.shape[-2:] != (self.emission_h, self.emission_w):
            x = F.interpolate(x, size=(self.emission_h, self.emission_w),
                              mode="bilinear", align_corners=False)
        return torch.sigmoid(x)

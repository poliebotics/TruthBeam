"""exp001c: emission-RGB predictor — ConvNeXt-Tiny encoder + U-Net decoder.

Encoder produces multi-stage features via timm `features_only=True`.
Decoder mirrors with 4 upsampling stages, each using a skip connection from
the corresponding encoder stage. After the deepest decoder stage, a final
F.interpolate brings the output to the requested emission (H, W) — the
encoder's spatial output and the emission resolution have different aspect
ratios in general, so an explicit final resize is needed.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .siamese_xof import _adapt_convnext_stem_to_4ch

# ConvNeXt-Tiny per-stage output channels (stages 0..3)
ENC_DIMS = (96, 192, 384, 768)


def _make_encoder(pretrained: bool) -> nn.Module:
    """ConvNeXt-Tiny with 4-ch stem, returns features per stage."""
    backbone = timm.create_model(
        "convnext_tiny",
        pretrained=pretrained,
        features_only=True,
        out_indices=(0, 1, 2, 3),
    )
    _adapt_convnext_stem_to_4ch(backbone)
    return backbone


def _gn_groups(ch: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    g = _gn_groups(out_ch)
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(g, out_ch),
        nn.GELU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(g, out_ch),
        nn.GELU(),
    )


class _UpBlock(nn.Module):
    """Upsample 2x, optionally concat skip, then a small conv block."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        merge_in = in_ch + skip_ch
        self.block = _conv_block(merge_in, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            # Match shapes if off by one due to odd input dims
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.block(x)


class EmissionPredictor(nn.Module):
    def __init__(
        self,
        emission_h: int,
        emission_w: int,
        pretrained: bool = True,
        decoder_dims: tuple[int, int, int, int, int] = (384, 192, 96, 48, 24),
    ):
        super().__init__()
        self.emission_h = emission_h
        self.emission_w = emission_w
        self.encoder = _make_encoder(pretrained)
        d3, d2, d1, d0, dx = decoder_dims
        # Decoder: bottleneck -> up to stage 2 (skip ENC_DIMS[2]=384)
        self.up3 = _UpBlock(ENC_DIMS[3], ENC_DIMS[2], d3)   # 768 + 384 -> d3
        self.up2 = _UpBlock(d3, ENC_DIMS[1], d2)            # d3  + 192 -> d2
        self.up1 = _UpBlock(d2, ENC_DIMS[0], d1)            # d2  +  96 -> d1
        self.up0 = _UpBlock(d1, 0, d0)                      # no skip past stem -> d0
        self.up_extra = _UpBlock(d0, 0, dx)                 # one more 2x -> dx
        self.head = nn.Conv2d(dx, 3, kernel_size=1)

    def forward(self, capture: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(capture)  # list of 4 tensors at strides [4, 8, 16, 32]
        f0, f1, f2, f3 = feats
        x = self.up3(f3, f2)
        x = self.up2(x,  f1)
        x = self.up1(x,  f0)
        x = self.up0(x,  None)
        x = self.up_extra(x, None)
        x = self.head(x)
        if x.shape[-2:] != (self.emission_h, self.emission_w):
            x = F.interpolate(x, size=(self.emission_h, self.emission_w),
                               mode="bilinear", align_corners=False)
        return torch.sigmoid(x)

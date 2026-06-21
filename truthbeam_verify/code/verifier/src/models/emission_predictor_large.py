"""ConvNeXt-Large encoder + U-Net decoder for emission RGB recovery.

Same shape and skip-connection structure as `emission_predictor.EmissionPredictor`,
adapted for ConvNeXt-Large feature dims:
  Tiny:  stage 0..3 channels = (96, 192, 384, 768)
  Large: stage 0..3 channels = (192, 384, 768, 1536)

Decoder default channels are scaled accordingly.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .emission_predictor import _UpBlock, _conv_block  # noqa: F401  (reused)
from .siamese_xof import _adapt_convnext_stem_to_4ch

ENC_DIMS_LARGE = (192, 384, 768, 1536)


def _make_encoder_large(pretrained: bool) -> nn.Module:
    backbone = timm.create_model(
        "convnext_large",
        pretrained=pretrained,
        features_only=True,
        out_indices=(0, 1, 2, 3),
    )
    _adapt_convnext_stem_to_4ch(backbone)
    return backbone


class EmissionPredictorLarge(nn.Module):
    def __init__(
        self,
        emission_h: int,
        emission_w: int,
        pretrained: bool = True,
        decoder_dims: tuple[int, int, int, int, int] = (768, 384, 192, 96, 48),
    ):
        super().__init__()
        self.emission_h = emission_h
        self.emission_w = emission_w
        self.encoder = _make_encoder_large(pretrained)
        d3, d2, d1, d0, dx = decoder_dims
        self.up3 = _UpBlock(ENC_DIMS_LARGE[3], ENC_DIMS_LARGE[2], d3)   # 1536 + 768  -> d3
        self.up2 = _UpBlock(d3, ENC_DIMS_LARGE[1], d2)                  # d3   + 384  -> d2
        self.up1 = _UpBlock(d2, ENC_DIMS_LARGE[0], d1)                  # d2   + 192  -> d1
        self.up0 = _UpBlock(d1, 0, d0)                                   # no skip past stem
        self.up_extra = _UpBlock(d0, 0, dx)
        self.head = nn.Conv2d(dx, 3, kernel_size=1)

    def forward(self, capture: torch.Tensor) -> torch.Tensor:
        f0, f1, f2, f3 = self.encoder(capture)
        x = self.up3(f3, f2)
        x = self.up2(x, f1)
        x = self.up1(x, f0)
        x = self.up0(x, None)
        x = self.up_extra(x, None)
        x = self.head(x)
        if x.shape[-2:] != (self.emission_h, self.emission_w):
            x = F.interpolate(
                x, size=(self.emission_h, self.emission_w),
                mode="bilinear", align_corners=False,
            )
        return torch.sigmoid(x)

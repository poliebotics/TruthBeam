"""Phase D emission predictor v2 — encoder + coord-aware FPN + RGB decoder.

Used by experiment A4 (Tiny + emission). Output is sigmoid in [0, 1] over
shape (3, H_out, W_out). The structure mirrors the U-Net of `emission_predictor.py`
but routes encoder features through the same coord-aware FPN so the encoder
backbone is shared with A0/A1/A2/A6/A7.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cfa_stem import adapt_convnext_stem_4ch_half_g
from .coord_aware_fpn import CoordAwareFPN
from .xof_decoder_v2 import ENC_DIMS, _build_encoder
from .emission_predictor import _UpBlock, _gn_groups  # noqa: F401  (reuse decoder primitives)


class EmissionPredictorV2(nn.Module):
    def __init__(
        self,
        emission_h: int,
        emission_w: int,
        encoder_size: str = "tiny",
        pretrained: bool = True,
        fpn_out_channels: int = 256,
        decoder_dims: tuple[int, int, int, int, int] = (256, 192, 96, 48, 24),
    ):
        super().__init__()
        self.emission_h = emission_h
        self.emission_w = emission_w
        self.encoder = _build_encoder(encoder_size, pretrained)
        self.fpn = CoordAwareFPN(ENC_DIMS[encoder_size], out_channels=fpn_out_channels)
        cw = self.fpn.out_channels_with_coords  # = fpn_out_channels + 2
        d3, d2, d1, d0, dx = decoder_dims
        # Decode from P5 upwards, taking lateral skip from successive P-levels.
        self.up3 = _UpBlock(cw, cw, d3)   # input: P5_coord, skip: P4_coord
        self.up2 = _UpBlock(d3, cw, d2)   # skip: P3_coord
        self.up1 = _UpBlock(d2, cw, d1)   # skip: P2_coord
        self.up0 = _UpBlock(d1, 0, d0)
        self.up_extra = _UpBlock(d0, 0, dx)
        self.head = nn.Conv2d(dx, 3, kernel_size=1)

    def forward(self, packed_cfa: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(packed_cfa)
        p2, p3, p4, p5 = self.fpn(feats)
        x = self.up3(p5, p4)
        x = self.up2(x,  p3)
        x = self.up1(x,  p2)
        x = self.up0(x,  None)
        x = self.up_extra(x, None)
        x = self.head(x)
        if x.shape[-2:] != (self.emission_h, self.emission_w):
            x = F.interpolate(
                x, size=(self.emission_h, self.emission_w),
                mode="bilinear", align_corners=False,
            )
        return torch.sigmoid(x)

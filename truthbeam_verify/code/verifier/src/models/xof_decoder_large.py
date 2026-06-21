"""ConvNeXt-Large encoder + 4 MLP heads for direct XOF byte prediction.

Mirrors `xof_decoder.XOFDecoder` but with:
  - ConvNeXt-Large backbone (feature dim 1536, vs Tiny's 768)
  - Wider per-octave hidden dims (2048 / 2048 / 3072 / 4096)
  - Same 4-channel CFA stem adaptation (R, G1=G, G2=G, B mapped from
    pretrained R, G, G, B respectively)
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

from .siamese_xof import _adapt_convnext_stem_to_4ch

CAPTURE_FEAT_DIM = 1536  # ConvNeXt-Large

OCTAVE_SHAPES = ((3, 17, 30), (3, 34, 60), (3, 68, 120), (3, 135, 240))
OCTAVE_OUT_DIMS = tuple(c * h * w for (c, h, w) in OCTAVE_SHAPES)
OCTAVE_HIDDENS = (2048, 2048, 3072, 4096)
assert OCTAVE_OUT_DIMS == (1530, 6120, 24480, 97200)


class _XOFHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, shape: tuple[int, int, int]):
        super().__init__()
        self.shape = shape
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.body(feat).view(feat.shape[0], *self.shape)


class XOFDecoderLarge(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_large", pretrained=pretrained, num_classes=0, global_pool="avg",
        )
        # ConvNeXt-Large stem is 3 → 192 (not 3 → 96 as in Tiny). The adapter
        # walks named modules to find the unique 3→N stem conv regardless.
        _adapt_convnext_stem_to_4ch(self.backbone)
        self.heads = nn.ModuleList([
            _XOFHead(CAPTURE_FEAT_DIM, hidden, out_dim, shape)
            for hidden, out_dim, shape in zip(OCTAVE_HIDDENS, OCTAVE_OUT_DIMS, OCTAVE_SHAPES)
        ])

    def forward(self, capture: torch.Tensor) -> list[torch.Tensor]:
        feat = self.backbone(capture)  # (B, 1536)
        return [head(feat) for head in self.heads]

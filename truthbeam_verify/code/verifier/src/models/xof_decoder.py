"""exp001b: direct XOF byte prediction from a globally-pooled capture feature.

Architecture:
- Encoder: ConvNeXt-Tiny with the same 4-channel CFA stem as exp001a (R, G1, G2, B).
  GAP -> 768-dim global feature. No spatial structure preserved — by design,
  because the optical channel introduces severe spatial distortion (projection
  cone variation, surface reflectance, lens distortion, body movement).
- Decoder: 4 parallel MLP heads, one per octave. Each head:
    Linear(768 -> hidden) -> GELU -> LayerNorm -> Linear(hidden -> N_i) -> Sigmoid
  reshape to (3, h_i, w_i). Output values in [0, 1] = predicted byte / 255.

Hidden widths and output sizes per octave (channels-first 3 × h × w):
    head_oct0: 768 -> 1024 -> 1530    -> (3, 17, 30)
    head_oct1: 768 -> 1024 -> 6120    -> (3, 34, 60)
    head_oct2: 768 -> 1536 -> 24480   -> (3, 68, 120)
    head_oct3: 768 -> 2048 -> 97200   -> (3, 135, 240)
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

from .siamese_xof import _adapt_convnext_stem_to_4ch

CAPTURE_FEAT_DIM = 768

OCTAVE_SHAPES = ((3, 17, 30), (3, 34, 60), (3, 68, 120), (3, 135, 240))
OCTAVE_OUT_DIMS = tuple(c * h * w for (c, h, w) in OCTAVE_SHAPES)  # 1530, 6120, 24480, 97200
OCTAVE_HIDDENS = (1024, 1024, 1536, 2048)
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
        flat = self.body(feat)  # (B, prod(shape))
        return flat.view(flat.shape[0], *self.shape)


class XOFDecoder(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_tiny", pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        _adapt_convnext_stem_to_4ch(self.backbone)
        self.heads = nn.ModuleList([
            _XOFHead(CAPTURE_FEAT_DIM, hidden, out_dim, shape)
            for hidden, out_dim, shape in zip(OCTAVE_HIDDENS, OCTAVE_OUT_DIMS, OCTAVE_SHAPES)
        ])

    def forward(self, capture: torch.Tensor) -> list[torch.Tensor]:
        """Returns list of 4 sigmoid tensors, one per octave: (B, 3, h_i, w_i)."""
        feat = self.backbone(capture)  # (B, 768)
        return [head(feat) for head in self.heads]

"""Phase D XOF decoder v2 — encoder + coord-aware FPN + per-octave heads.

Used by experiments A0 / A1 / A2 / A6 / A7. Outputs are tanh-bounded ∈ [-1, 1]
matching the centered training target `(byte - 127.5) / 127.5`.

Encoder size: "tiny" (ConvNeXt-Tiny) or "large" (ConvNeXt-Large). Both adapted
to 4-channel packed CFA input via `adapt_convnext_stem_4ch_half_g` (G half-scaled).

For A0 (oct0 + oct1 heads only), pass `enabled_octaves=(0, 1)`.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

from .cfa_stem import adapt_convnext_stem_4ch_half_g
from .coord_aware_fpn import CoordAwareFPN
from .octave_heads import OctaveHeads
from .stn import ConstrainedSTN

ENC_DIMS = {
    "tiny":  (96, 192, 384, 768),
    "large": (192, 384, 768, 1536),
}


def _build_encoder(size: str, pretrained: bool) -> nn.Module:
    name = {"tiny": "convnext_tiny", "large": "convnext_large"}[size]
    backbone = timm.create_model(
        name, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3),
    )
    adapt_convnext_stem_4ch_half_g(backbone)
    return backbone


class XOFDecoderV2(nn.Module):
    def __init__(
        self,
        encoder_size: str = "tiny",
        pretrained: bool = True,
        fpn_out_channels: int = 256,
        head_hidden: int = 128,
        enabled_octaves: tuple[int, ...] = (0, 1, 2, 3),
        use_stn: bool = False,
    ):
        super().__init__()
        self.encoder_size = encoder_size
        self.use_stn = use_stn
        self.stn = ConstrainedSTN(in_channels=4) if use_stn else None
        self.encoder = _build_encoder(encoder_size, pretrained)
        self.fpn = CoordAwareFPN(ENC_DIMS[encoder_size], out_channels=fpn_out_channels)
        self.octave_heads = OctaveHeads(
            in_channels=self.fpn.out_channels_with_coords,
            hidden=head_hidden,
            enabled_octaves=enabled_octaves,
        )

    def forward(
        self, packed_cfa: torch.Tensor
    ) -> tuple[list[torch.Tensor | None], dict]:
        """Returns (predictions_list, info_dict).

        info_dict has:
            "stn": dict|None — when use_stn=True, contains theta, grid_oob_fraction
        """
        info: dict = {"stn": None}
        x = packed_cfa
        if self.stn is not None:
            x, theta, grid = self.stn(x)
            from .stn import out_of_bounds_fraction  # avoid circular hassles
            info["stn"] = {
                "theta": theta,           # (B, 2, 3) — caller computes regularizer + decompose
                "grid_oob_fraction": out_of_bounds_fraction(grid),
            }
        feats = self.encoder(x)
        fpn_levels = self.fpn(feats)
        preds = self.octave_heads(fpn_levels)
        return preds, info

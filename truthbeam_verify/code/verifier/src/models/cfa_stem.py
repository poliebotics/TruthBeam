"""4-channel CFA stem adapter with G half-scaling.

Pretrained ConvNeXt stems take 3-channel RGB. For our packed CFA input
(R, G1, G2, B at half-res) we want each pretrained channel's response to be
preserved in expectation: copy R→ch0, B→ch3, and **split** G's weight half
into ch1 (G1) and ch2 (G2) so the total green response is unchanged.

Without G half-scaling, a uniform-green input doubles the green activation
relative to the pretrained ImageNet network's expectation, perturbing
downstream LayerNorm statistics during the first training epochs.

The function `adapt_convnext_stem_4ch_half_g` walks the backbone's named
modules to find the unique 3→N patchify Conv2d (4×4 stride-4) and replaces
it in-place with a 4→N conv whose weights are:

    new_conv.weight[:, 0]  = old_conv.weight[:, 0]            # R
    new_conv.weight[:, 1]  = old_conv.weight[:, 1] * 0.5      # G1 half scale
    new_conv.weight[:, 2]  = old_conv.weight[:, 1] * 0.5      # G2 half scale
    new_conv.weight[:, 3]  = old_conv.weight[:, 2]            # B
    new_conv.bias          = old_conv.bias                    # unchanged

Works for ConvNeXt-Tiny (3→96), Small (3→96), Base (3→128), Large (3→192),
XL (3→256). Idempotent; calling it on an already-4-channel stem is a no-op.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def adapt_convnext_stem_4ch_half_g(backbone: nn.Module) -> None:
    """Replace ConvNeXt's 3-ch patchify stem with a 4-ch one, G weight half-scaled."""
    # Try direct attribute path first (timm plain ConvNeXt: backbone.stem[0]).
    parent = None
    attr: int | str | None = None
    old_conv: nn.Conv2d | None = None
    if hasattr(backbone, "stem") and isinstance(backbone.stem, nn.Module):
        cand = backbone.stem[0]
        if isinstance(cand, nn.Conv2d) and cand.in_channels == 3:
            parent, attr, old_conv = backbone.stem, 0, cand
    if old_conv is None:
        # FeatureListNet wrapper case (features_only=True).
        for name, mod in backbone.named_modules():
            if (
                isinstance(mod, nn.Conv2d)
                and mod.in_channels == 3
                and mod.kernel_size == (4, 4)
                and mod.stride == (4, 4)
            ):
                parts = name.split(".")
                p = backbone
                for piece in parts[:-1]:
                    p = p[int(piece)] if piece.isdigit() else getattr(p, piece)
                parent, attr, old_conv = p, parts[-1], mod
                break
    if old_conv is None:
        raise RuntimeError("could not locate the 3→N ConvNeXt patchify stem to adapt")

    new_conv = nn.Conv2d(
        4,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None),
    )
    with torch.no_grad():
        new_conv.weight[:, 0] = old_conv.weight[:, 0]            # R
        new_conv.weight[:, 1] = old_conv.weight[:, 1] * 0.5      # G1 half scale
        new_conv.weight[:, 2] = old_conv.weight[:, 1] * 0.5      # G2 half scale
        new_conv.weight[:, 3] = old_conv.weight[:, 2]            # B
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    if isinstance(attr, int):
        parent[attr] = new_conv
    else:
        setattr(parent, attr, new_conv)

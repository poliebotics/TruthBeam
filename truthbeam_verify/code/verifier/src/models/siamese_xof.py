"""Siamese XOF-binding verification network for exp001a.

Capture branch: ConvNeXt-Tiny adapted to 4-channel CFA input (R, G1, G2, B).
  - Stem conv: 4 -> 96 (was 3 -> 96 for ImageNet RGB).
    Init mapping: ch 0=R copies pretrained R, ch 1=G1 copies pretrained G,
    ch 2=G2 also copies pretrained G, ch 3=B copies pretrained B.
    No magnitude rescaling — downstream LayerNorm absorbs the scale shift.
  - GAP -> 768 -> linear -> 512 -> L2 normalize.

XOF branch: per-octave small CNN over each (3, h, w) tensor -> GAP -> 64 dims.
  Concatenate 4 octaves -> 256 dims -> MLP -> 512 -> L2 normalize.

Both branches output unit-norm 512-D embeddings.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

EMBED_DIM = 512
CAPTURE_FEAT_DIM = 768  # ConvNeXt-Tiny final feature dim


def _adapt_convnext_stem_to_4ch(backbone: nn.Module) -> None:
    """Replace ConvNeXt-Tiny's 3-ch stem with a 4-ch one, copying weights.

    Works for both timm's plain ConvNeXt (has `backbone.stem[0]`) and
    `features_only=True` FeatureListNet wrappers (where `.stem` is hidden).
    Fallback path locates the unique 3→96 Conv2d by walking modules.
    """
    parent = None
    attr = None
    old_conv = None
    if hasattr(backbone, "stem") and isinstance(backbone.stem, nn.Module):
        cand = backbone.stem[0]
        if isinstance(cand, nn.Conv2d) and cand.in_channels == 3:
            parent, attr, old_conv = backbone.stem, 0, cand
    if old_conv is None:
        # Match the 4×4 stride-4 patchify stem with 3 input channels (any out dim:
        # Tiny=96, Small=96, Base=128, Large=192, XL=256).
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
        raise RuntimeError("could not locate the 3→N ConvNeXt patchify stem conv to adapt")

    new_conv = nn.Conv2d(
        4,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None),
    )
    with torch.no_grad():
        new_conv.weight[:, 0] = old_conv.weight[:, 0]   # R
        new_conv.weight[:, 1] = old_conv.weight[:, 1]   # G1 from G
        new_conv.weight[:, 2] = old_conv.weight[:, 1]   # G2 from G
        new_conv.weight[:, 3] = old_conv.weight[:, 2]   # B
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    if isinstance(attr, int):
        parent[attr] = new_conv
    else:
        setattr(parent, attr, new_conv)


class CaptureBranch(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_tiny", pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        _adapt_convnext_stem_to_4ch(self.backbone)
        # LayerNorm before projection prevents the at-init collapse seen with
        # the bare linear: ConvNeXt features on similar-looking frames are
        # highly correlated, so a bias-dominated linear projects them all to
        # nearly the same direction, killing the InfoNCE gradient.
        self.norm = nn.LayerNorm(CAPTURE_FEAT_DIM)
        self.proj = nn.Linear(CAPTURE_FEAT_DIM, embed_dim, bias=False)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-L2-norm projected features (used for VICReg-style regularization)."""
        return self.proj(self.norm(self.backbone(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.forward_features(x), dim=-1)


_OCT_POOL = 4  # AdaptiveAvgPool2d target side


class OctaveCNN(nn.Module):
    """Per-octave CNN: (B, 3, h, w) -> (B, out_dim*4*4) via 3 convs + small grid pool.

    bias=False on convs because the input is uniform high-entropy noise; with
    biased convs, outputs are dominated by the bias term and nearly identical
    across batch (verified — caused total InfoNCE collapse at init).
    Pooling to a 4x4 grid (instead of GAP to 1x1) preserves enough per-sample
    spatial signal that random-init outputs are distinguishable across the
    batch — without that, GAP averages noise to ~zero and embeddings collapse.
    """

    def __init__(self, in_ch: int = 3, hidden: int = 32, out_dim: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden * 2, kernel_size=3, padding=1, stride=2, bias=False),
            nn.GroupNorm(8, hidden * 2),
            nn.GELU(),
            nn.Conv2d(hidden * 2, out_dim, kernel_size=3, padding=1, stride=2, bias=False),
            nn.GroupNorm(8, out_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(_OCT_POOL),
        )
        self.feat_dim = out_dim * _OCT_POOL * _OCT_POOL

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x).flatten(1)


class XOFBranch(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM, oct_dim: int = 64):
        super().__init__()
        self.octs = nn.ModuleList([OctaveCNN(out_dim=oct_dim) for _ in range(4)])
        in_dim = 4 * oct_dim * _OCT_POOL * _OCT_POOL  # 4 * 64 * 16 = 4096
        # LayerNorm + bias-free final proj prevents collapse.
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, embed_dim, bias=False),
        )

    def forward_features(self, octaves: list[torch.Tensor]) -> torch.Tensor:
        """Pre-L2-norm features."""
        feats = [m(x) for m, x in zip(self.octs, octaves)]
        f = torch.cat(feats, dim=-1)
        return self.mlp(f)

    def forward(self, octaves: list[torch.Tensor]) -> torch.Tensor:
        return F.normalize(self.forward_features(octaves), dim=-1)


class SiameseXOF(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM, pretrained_capture: bool = True):
        super().__init__()
        self.capture = CaptureBranch(embed_dim, pretrained=pretrained_capture)
        self.xof = XOFBranch(embed_dim)

    def encode_capture(self, x: torch.Tensor) -> torch.Tensor:
        return self.capture(x)

    def encode_xof(self, octaves: list[torch.Tensor]) -> torch.Tensor:
        return self.xof(octaves)

    def forward(
        self, capture: torch.Tensor, octaves: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_capture(capture), self.encode_xof(octaves)

    def forward_with_features(
        self, capture: torch.Tensor, octaves: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (z_cap, z_xof, h_cap, h_xof) where h_* are pre-L2-norm."""
        h_cap = self.capture.forward_features(capture)
        h_xof = self.xof.forward_features(octaves)
        z_cap = F.normalize(h_cap, dim=-1)
        z_xof = F.normalize(h_xof, dim=-1)
        return z_cap, z_xof, h_cap, h_xof

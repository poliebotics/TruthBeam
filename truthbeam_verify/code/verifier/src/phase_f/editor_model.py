"""Phase F editor: F(C_source, E_source, E_target) → C_target.

Architecture:
  source_encoder           ConvNeXt-Tiny + 4-ch CFA stem (legacy full-G dup,
                           matching exp001c). Returns multi-scale features
                           (C2, C3, C4, C5).
  source_emission_encoder  small 4-layer conv encoder over E_source (3-ch RGB).
                           Multi-scale features matching source encoder strides.
  target_emission_encoder  same architecture as source_emission_encoder,
                           separate weights, applied to E_target.
  conditioning             FiLM modulation at each source-encoder scale,
                           gamma/beta computed from a small MLP over
                           (pool(E_source_features) ‖ pool(E_target_features)).
  decoder                  U-Net upsample with skip connections from source
                           encoder. Outputs delta_C at packed CFA resolution.
  output                   C_target = (C_source + delta_C).clamp(0, 1)

Initialization mode:
  "random"             — random init for everything
  "exp001c-warm-start" — source_encoder loads exp001c's encoder weights;
                          everything else random.
"""
from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


ENC_DIMS_TINY = (96, 192, 384, 768)


def _adapt_convnext_stem_full_G(backbone: nn.Module) -> None:
    """Phase E + exp001c stem: full-magnitude G duplication (NOT half-scaled).
    Locates the unique 3-ch 4×4 stride-4 patchify conv and replaces with 4→N."""
    parent = None
    attr = None
    old = None
    if hasattr(backbone, "stem") and isinstance(backbone.stem, nn.Module):
        cand = backbone.stem[0]
        if isinstance(cand, nn.Conv2d) and cand.in_channels == 3:
            parent, attr, old = backbone.stem, 0, cand
    if old is None:
        for name, mod in backbone.named_modules():
            if (isinstance(mod, nn.Conv2d) and mod.in_channels == 3
                    and mod.kernel_size == (4, 4) and mod.stride == (4, 4)):
                parts = name.split(".")
                p = backbone
                for piece in parts[:-1]:
                    p = p[int(piece)] if piece.isdigit() else getattr(p, piece)
                parent, attr, old = p, parts[-1], mod
                break
    if old is None:
        raise RuntimeError("could not find 3→N ConvNeXt patchify stem")
    new = nn.Conv2d(4, old.out_channels, kernel_size=old.kernel_size,
                    stride=old.stride, padding=old.padding,
                    bias=(old.bias is not None))
    with torch.no_grad():
        new.weight[:, 0] = old.weight[:, 0]
        new.weight[:, 1] = old.weight[:, 1]   # G1 = full G (legacy)
        new.weight[:, 2] = old.weight[:, 1]   # G2 = full G (legacy)
        new.weight[:, 3] = old.weight[:, 2]
        if old.bias is not None:
            new.bias.copy_(old.bias)
    if isinstance(attr, int):
        parent[attr] = new
    else:
        setattr(parent, attr, new)


def _build_source_encoder(pretrained: bool = True) -> nn.Module:
    backbone = timm.create_model(
        "convnext_tiny", pretrained=pretrained,
        features_only=True, out_indices=(0, 1, 2, 3),
    )
    _adapt_convnext_stem_full_G(backbone)
    return backbone


class _SmallConvEncoder(nn.Module):
    """4-stage conv encoder over E (3-channel RGB). Strides 4,2,2,2 → /32 total
    to match ConvNeXt-Tiny's stride pattern. Output channel dims match
    ENC_DIMS_TINY = (96, 192, 384, 768)."""

    def __init__(self):
        super().__init__()
        c0, c1, c2, c3 = ENC_DIMS_TINY
        self.s0 = nn.Sequential(
            nn.Conv2d(3, c0, 4, stride=4), _gn(c0), nn.GELU(),
        )
        self.s1 = nn.Sequential(
            nn.Conv2d(c0, c1, 3, stride=2, padding=1), _gn(c1), nn.GELU(),
        )
        self.s2 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, stride=2, padding=1), _gn(c2), nn.GELU(),
        )
        self.s3 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, stride=2, padding=1), _gn(c3), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f0 = self.s0(x)
        f1 = self.s1(f0)
        f2 = self.s2(f1)
        f3 = self.s3(f2)
        return f0, f1, f2, f3


class FiLMConditioner(nn.Module):
    """For each scale, produces (gamma, beta) per-channel from
    (pool(E_source_feat) ‖ pool(E_target_feat))."""

    def __init__(self, dims=ENC_DIMS_TINY, hidden: int = 128):
        super().__init__()
        self.dims = dims
        self.heads = nn.ModuleList()
        for c in dims:
            self.heads.append(nn.Sequential(
                nn.Linear(2 * c, hidden), nn.GELU(),
                nn.Linear(hidden, 2 * c),
            ))

    def forward(self, src_feats, tgt_feats):
        """Returns list of (gamma, beta) tensors of shape (B, C, 1, 1) per scale."""
        out = []
        for i, (s, t) in enumerate(zip(src_feats, tgt_feats)):
            ps = s.mean(dim=(-2, -1))   # (B, C) global pool
            pt = t.mean(dim=(-2, -1))
            cat = torch.cat([ps, pt], dim=-1)
            params = self.heads[i](cat)   # (B, 2C)
            gamma, beta = params.chunk(2, dim=-1)
            out.append((gamma.unsqueeze(-1).unsqueeze(-1) + 1.0, beta.unsqueeze(-1).unsqueeze(-1)))
        return out


def _gn(out_ch: int) -> nn.GroupNorm:
    """GroupNorm with the largest group count that evenly divides out_ch (≤32)."""
    for g in (32, 16, 8, 4, 2, 1):
        if out_ch % g == 0:
            return nn.GroupNorm(g, out_ch)
    return nn.GroupNorm(1, out_ch)


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        merge_in = in_ch + skip_ch
        self.block = nn.Sequential(
            nn.Conv2d(merge_in, out_ch, 3, padding=1, bias=False),
            _gn(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            _gn(out_ch), nn.GELU(),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.block(x)


class Editor(nn.Module):
    """F(C_source, E_source, E_target) → C_target predicted (4-channel packed CFA)."""

    def __init__(
        self,
        capture_h: int = 2300,
        capture_w: int = 2660,
        emission_h: int = 1080,
        emission_w: int = 1920,
        init_mode: str = "exp001c-warm-start",
        decoder_dims: tuple[int, int, int, int, int] = (384, 192, 96, 48, 24),
    ):
        super().__init__()
        self.capture_h = capture_h
        self.capture_w = capture_w
        self.emission_h = emission_h
        self.emission_w = emission_w
        self.init_mode = init_mode

        # Encoders
        self.source_encoder = _build_source_encoder(
            pretrained=(init_mode == "exp001c-warm-start" or init_mode == "imagenet"))
        self.source_emission_encoder = _SmallConvEncoder()
        self.target_emission_encoder = _SmallConvEncoder()

        # FiLM conditioner produces (gamma, beta) at 4 scales
        self.film = FiLMConditioner(ENC_DIMS_TINY, hidden=128)

        # Decoder (mirror of source encoder, with skip connections)
        c0, c1, c2, c3 = ENC_DIMS_TINY
        d3, d2, d1, d0, dx = decoder_dims
        self.up3 = _UpBlock(c3, c2, d3)
        self.up2 = _UpBlock(d3, c1, d2)
        self.up1 = _UpBlock(d2, c0, d1)
        self.up0 = _UpBlock(d1, 0, d0)
        self.up_extra = _UpBlock(d0, 0, dx)
        self.head = nn.Conv2d(dx, 4, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)  # delta starts at zero → predicts C_source

    def load_warm_start(self, exp001c_ckpt: Path) -> None:
        """Load exp001c's encoder weights into source_encoder (matching layout)."""
        ck = torch.load(exp001c_ckpt, map_location="cpu", weights_only=False)
        state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        # exp001c's EmissionPredictor stores encoder under `encoder.*`. The 4-ch
        # stem was created by siamese_xof._adapt_convnext_stem_to_4ch (full G dup),
        # which matches our source_encoder layout — so encoder.* keys map directly.
        encoder_state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
        if encoder_state:
            missing, unexp = self.source_encoder.load_state_dict(encoder_state, strict=False)
            print(f"[editor] warm-start from {exp001c_ckpt}: encoder missing={len(missing)} unexp={len(unexp)}")

    @staticmethod
    def _resize_emission(em: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(em, size=target_hw, mode="bilinear", align_corners=False)

    def forward(
        self,
        C_source: torch.Tensor,   # (B, 4, capH, capW)
        E_source: torch.Tensor,   # (B, 3, emH, emW)
        E_target: torch.Tensor,   # (B, 3, emH, emW)
    ) -> torch.Tensor:
        # Source CFA features
        s2, s3, s4, s5 = self.source_encoder(C_source)   # strides /4, /8, /16, /32
        # Emission features (resized to capture-H/W first to align scales)
        E_source_at_cap = self._resize_emission(E_source, (self.capture_h, self.capture_w))
        E_target_at_cap = self._resize_emission(E_target, (self.capture_h, self.capture_w))
        es0, es1, es2, es3 = self.source_emission_encoder(E_source_at_cap)
        et0, et1, et2, et3 = self.target_emission_encoder(E_target_at_cap)

        # FiLM modulation at each scale
        film_params = self.film([es0, es1, es2, es3], [et0, et1, et2, et3])
        s2 = s2 * film_params[0][0] + film_params[0][1]
        s3 = s3 * film_params[1][0] + film_params[1][1]
        s4 = s4 * film_params[2][0] + film_params[2][1]
        s5 = s5 * film_params[3][0] + film_params[3][1]

        # U-Net decoder
        x = self.up3(s5, s4)
        x = self.up2(x,  s3)
        x = self.up1(x,  s2)
        x = self.up0(x,  None)
        x = self.up_extra(x, None)
        delta = self.head(x)
        if delta.shape[-2:] != (self.capture_h, self.capture_w):
            delta = F.interpolate(delta, size=(self.capture_h, self.capture_w),
                                   mode="bilinear", align_corners=False)
        C_pred = (C_source + delta).clamp(0, 1)
        return C_pred

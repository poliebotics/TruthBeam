"""Phase F editor — ControlNet-inspired architecture.

Replaces FiLM modulation (which proved bandwidth-limited; see
`experiments/phase_f_prep/f_a_mini_diagnosis.md`) with spatial hint
injection at multiple scales via zero-conv adapters.

Architecture per `CONTROLNET_DESIGN.md`:
    ┌─────────────────────────────┐         ┌────────────────────────┐
    │ Source encoder (ConvNeXt-T) │         │ Hint encoder           │
    │ 4-ch CFA + 2-ch coord stem  │         │ 11-ch input            │
    │ exp001c warm-start          │         │ (E_t ‖ E_s ‖ Δ ‖ coord)│
    │ outputs s2,s3,s4,s5         │         │ outputs h0,h1,h2,h3    │
    └────────┬────────────────────┘         └─────────┬──────────────┘
             │  per scale:                            │
             │   h_i → ZeroConv(c_i) → +              │
             │   s_i + ZeroConv(h_i)        ◄─────────┘
             │
             ▼
       Decoder (U-Net mirror with skip cnxs, zero-init delta head)
             │
             ▼
       C_pred = (C_source + delta).clamp(0, 1)

Cross-coordinate-system handling: E and C live in different coordinate
systems (projector vs camera). The architecture does NOT pre-warp
E_target — it relies on the binder's internalized warp during L_binder
loss computation. See CONTROLNET_DESIGN.md "How L_binder handles cross-
coordinate-system" for the rationale.
"""
from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


ENC_DIMS_TINY = (96, 192, 384, 768)


# ---------------------------- helpers ----------------------------

def _gn(out_ch: int) -> nn.GroupNorm:
    """GroupNorm with the largest group count that evenly divides out_ch (≤32)."""
    for g in (32, 16, 8, 4, 2, 1):
        if out_ch % g == 0:
            return nn.GroupNorm(g, out_ch)
    return nn.GroupNorm(1, out_ch)


def coord_grid(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return (2, h, w) tensor of normalized x, y coordinates in [-1, 1]."""
    y = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx, yy], dim=0)


# ---------------------------- source encoder (CFA + coord) ----------------------------

def _adapt_convnext_stem_cfa_plus_coord(backbone: nn.Module) -> None:
    """Replace the ConvNeXt-Tiny patchify stem with a 6→N stem.

    Channels: [R, G1, G2, B, coord_x, coord_y].
    The first 4 channels follow the legacy full-G duplication that exp001c
    uses (so warm-starting from exp001c works). The 2 coord channels are
    zero-initialized so the model behaves identically to a 4-ch model at
    init; the optimizer learns to use coord channels as needed.
    """
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
    new = nn.Conv2d(6, old.out_channels, kernel_size=old.kernel_size,
                    stride=old.stride, padding=old.padding,
                    bias=(old.bias is not None))
    with torch.no_grad():
        new.weight[:, 0] = old.weight[:, 0]      # R
        new.weight[:, 1] = old.weight[:, 1]      # G1 = full G (legacy)
        new.weight[:, 2] = old.weight[:, 1]      # G2 = full G (legacy)
        new.weight[:, 3] = old.weight[:, 2]      # B
        new.weight[:, 4].zero_()                  # coord_x — zero-init
        new.weight[:, 5].zero_()                  # coord_y — zero-init
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
    _adapt_convnext_stem_cfa_plus_coord(backbone)
    return backbone


# ---------------------------- hint encoder ----------------------------

class _HintEncoder(nn.Module):
    """4-stage conv encoder over the hint stack (11-channel input).

    Inputs (channel-wise concat):
      - E_target  (3 ch, projector RGB)
      - E_source  (3 ch, projector RGB)
      - Δ = E_t - E_s (3 ch, signed)
      - coord_x, coord_y (2 ch)
    Total: 11 channels.

    Strides 4, 2, 2, 2 → /32 total to match ConvNeXt-Tiny's stride pattern.
    Output channel dims match ENC_DIMS_TINY.
    """

    def __init__(self):
        super().__init__()
        c0, c1, c2, c3 = ENC_DIMS_TINY
        self.s0 = nn.Sequential(
            nn.Conv2d(11, c0, 4, stride=4), _gn(c0), nn.GELU(),
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

    def forward(self, hint_input: torch.Tensor) -> tuple[torch.Tensor, ...]:
        h0 = self.s0(hint_input)
        h1 = self.s1(h0)
        h2 = self.s2(h1)
        h3 = self.s3(h2)
        return h0, h1, h2, h3


# ---------------------------- zero-conv adapters ----------------------------

class _ZeroConvAdapter(nn.Module):
    """1×1 conv with all weights/bias zero-initialized.

    At init: outputs are zero, so the source-encoder features pass through
    unchanged. The optimizer learns to ramp up the hint contribution.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ---------------------------- decoder ----------------------------

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


# ---------------------------- main editor ----------------------------

class EditorControlNet(nn.Module):
    """F(C_source, E_source, E_target) → C_pred via ControlNet-inspired hint injection.

    Differences from `editor_model.Editor` (FiLM):
      - 6-ch source stem (CFA + coord) instead of 4-ch
      - Hint encoder takes 11-ch input (E_t ‖ E_s ‖ Δ ‖ coord)
      - Hint features are added to source features at each scale via
        zero-conv adapters (replaces FiLM modulation entirely)
      - No FiLM module
    """

    def __init__(
        self,
        capture_h: int = 2300,
        capture_w: int = 2660,
        emission_h: int = 1080,
        emission_w: int = 1920,
        init_mode: str = "exp001c-warm-start",
        decoder_dims: tuple[int, int, int, int, int] = (384, 192, 96, 48, 24),
        hint_use_source: bool = True,
    ):
        super().__init__()
        self.capture_h = capture_h
        self.capture_w = capture_w
        self.emission_h = emission_h
        self.emission_w = emission_w
        self.init_mode = init_mode
        self.hint_use_source = hint_use_source

        # Encoders
        self.source_encoder = _build_source_encoder(
            pretrained=(init_mode in ("exp001c-warm-start", "imagenet")))
        self.hint_encoder = _HintEncoder()

        # Zero-conv adapters: hint feature → source feature at each scale
        self.adapters = nn.ModuleList([
            _ZeroConvAdapter(c, c) for c in ENC_DIMS_TINY
        ])

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
        nn.init.zeros_(self.head.bias)  # delta starts at zero → C_pred = C_source at init

    def zero_init_source_channels(self) -> None:
        """Zero out the hint encoder's first-conv weights for the E(t-1)
        and Δ channels. With this zero-init, control and treatment runs
        produce identical step-0 outputs regardless of E_source value
        (operator's warm-start trick for the v1.5 ablation, 2026-04-30):

          channels 0-2: E(t)        — kept as-is
          channels 3-5: E(t-1)      — zeroed
          channels 6-8: Δ           — zeroed
          channels 9-10: coord      — kept as-is

        Effect at step 0: hint encoder behaves as a 5-channel encoder
        with input [E(t), coord_x, coord_y] only, regardless of whether
        the run is feeding real or zero E_source. Any divergence during
        training comes from the model learning to use E(t-1)/Δ via
        these channels' weights growing under gradient.
        """
        with torch.no_grad():
            w = self.hint_encoder.s0[0].weight  # Conv2d weight (c0, 11, 4, 4)
            w[:, 3:9, :, :].zero_()
        print("[editor-cn] zero-init hint encoder source/delta channels (3:9)")

    def load_warm_start(self, exp001c_ckpt: Path) -> None:
        """Load exp001c's encoder weights into source_encoder (matching layout)."""
        ck = torch.load(exp001c_ckpt, map_location="cpu", weights_only=False)
        state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        encoder_state = {k[len("encoder."):]: v for k, v in state.items()
                         if k.startswith("encoder.")}
        if encoder_state:
            # The exp001c stem is 4-ch (CFA only); ours is 6-ch (CFA + coord).
            # Drop the stem weight key from the loaded state if its in_channels
            # don't match — the new 6-ch stem was already initialized correctly.
            stem_key = None
            for k, v in encoder_state.items():
                if v.dim() == 4 and v.shape[2:] == (4, 4) and v.shape[1] == 4:
                    # 4×4 stride-4 patchify with in_channels=4 — that's the legacy stem
                    stem_key = k
                    break
            if stem_key is not None:
                del encoder_state[stem_key]
            missing, unexp = self.source_encoder.load_state_dict(encoder_state, strict=False)
            print(f"[editor-cn] warm-start from {exp001c_ckpt}: encoder "
                  f"missing={len(missing)} unexp={len(unexp)} (stem skipped: {stem_key})")

    @staticmethod
    def _resize_to_capture(x: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)

    def forward(
        self,
        C_source: torch.Tensor,   # (B, 4, capH, capW)
        E_source: torch.Tensor,   # (B, 3, emH, emW)
        E_target: torch.Tensor,   # (B, 3, emH, emW)
    ) -> torch.Tensor:
        B = C_source.shape[0]
        device = C_source.device
        dtype = C_source.dtype

        # --- Source path: append coord channels to 4-ch CFA → 6-ch input ---
        coord_cap = coord_grid(self.capture_h, self.capture_w, device, dtype).unsqueeze(0).expand(B, 2, -1, -1)
        C_input = torch.cat([C_source, coord_cap], dim=1)  # (B, 6, capH, capW)
        s2, s3, s4, s5 = self.source_encoder(C_input)

        # --- Hint path: build 11-ch hint at capture resolution ---
        # Resize emission tiles to capture H/W to align with source encoder strides.
        E_t_at_cap = self._resize_to_capture(E_target, (self.capture_h, self.capture_w))
        if self.hint_use_source:
            E_s_at_cap = self._resize_to_capture(E_source, (self.capture_h, self.capture_w))
            delta_E = E_t_at_cap - E_s_at_cap
        else:
            # v1.5 ablation control: zero out E(t-1) and Δ in the hint (channels 3-8).
            # Combined with `zero_init_source_channels()`, this guarantees those
            # weights receive no useful gradient during training, so the model is
            # blind to E(t-1) by construction.
            E_s_at_cap = torch.zeros_like(E_t_at_cap)
            delta_E = torch.zeros_like(E_t_at_cap)
        coord_hint = coord_grid(self.capture_h, self.capture_w, device, dtype).unsqueeze(0).expand(B, 2, -1, -1)
        hint = torch.cat([E_t_at_cap, E_s_at_cap, delta_E, coord_hint], dim=1)  # (B, 11, capH, capW)
        h0, h1, h2, h3 = self.hint_encoder(hint)

        # --- Zero-conv injection: source feature ← source feature + adapter(hint feature) ---
        # Hint encoder uses 3×3 stride-2 padded conv (ceil-div); ConvNeXt-Tiny
        # downsamplers use 2×2 stride-2 (floor-div). Spatial extents may
        # differ by 1 pixel at deeper scales — resize hint to match source.
        def _align(s, h):
            if s.shape[-2:] != h.shape[-2:]:
                h = F.interpolate(h, size=s.shape[-2:], mode="bilinear",
                                  align_corners=False)
            return h
        s2 = s2 + self.adapters[0](_align(s2, h0))
        s3 = s3 + self.adapters[1](_align(s3, h1))
        s4 = s4 + self.adapters[2](_align(s4, h2))
        s5 = s5 + self.adapters[3](_align(s5, h3))

        # --- Decoder ---
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

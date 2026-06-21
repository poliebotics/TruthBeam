"""Phase G — conditional ε-prediction U-Net for the diffusion diagnostic.

Direction: C ←  noised C, conditioned on E via ControlNet-style spatial hint
injection. NOT FiLM on E (per the 2026-04-29 F-A mini failure documented in
`experiments/phase_f_prep/f_a_mini_diagnosis.md`). FiLM is used only for the
diffusion timestep `t` (a genuinely scalar signal — the safe case).

Inputs:
    C_t       : (B, 4, 768, 1024) — noised C at timestep t
    E         : (B, 3, 768, 1024) — emission, in C-space resolution
    t         : (B,) integer or float timestep ∈ [0, T)
Output:
    eps_pred  : (B, 4, 768, 1024) — predicted noise

Conditioning:
    The hint encoder consumes an 11-channel input matching the F-A v1
    EditorControlNet layout exactly (so the architecture is a literal port):
        ch 0-2 : E                  (3 ch, RGB at C-res)
        ch 3-5 : zeros              (placeholders for E_source — unused here)
        ch 6-8 : zeros              (placeholders for Δ — unused here)
        ch 9-10: coord_x, coord_y   (2 ch, normalized to [-1, 1])
    Total: 11 channels.

    Hint features are added to U-Net features at each downsample scale via
    zero-conv 1×1 adapters (same as F-A v1 — at init, hint contribution is
    zero; the optimizer learns to ramp it up).

Condition dropout:
    At training time, with probability `cond_drop_prob` (default 0.2), the
    hint encoder receives zeros for E (channels 0-2 zeroed). This trains a
    shared model that can be queried both conditionally and unconditionally.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- helpers ----------------

def _gn(out_ch: int) -> nn.GroupNorm:
    """Largest-group GroupNorm (≤32) that evenly divides out_ch."""
    for g in (32, 16, 8, 4, 2, 1):
        if out_ch % g == 0:
            return nn.GroupNorm(g, out_ch)
    return nn.GroupNorm(1, out_ch)


def coord_grid(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """(2, h, w) tensor of normalized x, y coordinates in [-1, 1]."""
    y = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx, yy], dim=0)


# ---------------- time embedding (sinusoidal + MLP) ----------------

class TimeEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        target_dtype = self.mlp[0].weight.dtype
        t = t.to(target_dtype)
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=target_dtype) / half
        )
        ang = t.unsqueeze(-1) * freqs
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        return self.mlp(emb)


# ---------------- residual blocks ----------------

class ResBlock(nn.Module):
    """Standard DDPM-style ResBlock with t-FiLM (additive). The FiLM here is
    SCALAR (just t), which is the safe case per the 2026-04-29 finding."""

    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.norm1 = _gn(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.norm2 = _gn(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Attn(nn.Module):
    def __init__(self, ch: int, n_heads: int = 4):
        super().__init__()
        self.norm = _gn(ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.n_heads = n_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(b, self.n_heads, c // self.n_heads, h * w)
        k = k.reshape(b, self.n_heads, c // self.n_heads, h * w)
        v = v.reshape(b, self.n_heads, c // self.n_heads, h * w)
        a = torch.einsum("bhci,bhcj->bhij", q, k) / math.sqrt(c // self.n_heads)
        a = a.softmax(dim=-1)
        out = torch.einsum("bhij,bhcj->bhci", a, v)
        out = out.reshape(b, c, h, w)
        return x + self.proj(out)


# ---------------- hint encoder (ControlNet-style) ----------------

class HintEncoder(nn.Module):
    """4-stage strided conv encoder over the 11-channel hint input.
    Strides 2, 2, 2, 2 → /16 total (matches the U-Net's 4 downsample levels).
    Output feature dimensions match the U-Net's `chs`."""

    def __init__(self, hint_in_ch: int, chs: Sequence[int]):
        super().__init__()
        assert len(chs) == 4, "hint encoder expects 4 stages to match U-Net"
        c0, c1, c2, c3 = chs
        self.s0 = nn.Sequential(
            nn.Conv2d(hint_in_ch, c0, 3, stride=2, padding=1), _gn(c0), nn.GELU(),
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

    def forward(self, hint: torch.Tensor) -> tuple[torch.Tensor, ...]:
        h0 = self.s0(hint)  # /2
        h1 = self.s1(h0)    # /4
        h2 = self.s2(h1)    # /8
        h3 = self.s3(h2)    # /16
        return h0, h1, h2, h3


class ZeroConvAdapter(nn.Module):
    """1×1 conv with all weights/bias zero-initialized (ControlNet-style)."""
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ---------------- main model ----------------

class DiffusionDiagnosticUNet(nn.Module):
    """ε-prediction U-Net for C, conditioned on E via ControlNet hint injection.

    Args:
        in_ch:            number of channels in C (default 4 for packed CFA).
        base_ch:          base channel count.
        channel_mults:    multipliers per scale (length 4 = 4 down/up levels).
        attn_at:          tuple of bools, attention at scale i.
        cond_drop_prob:   probability of zeroing E at training time
                          (classifier-free-guidance-style cond dropout).
        hint_in_ch:       11 by default, matching F-A v1 EditorControlNet.
    """

    def __init__(
        self,
        in_ch: int = 4,
        base_ch: int = 96,
        channel_mults: tuple[int, ...] = (1, 2, 4, 4),
        attn_at: tuple[bool, ...] = (False, False, True, True),
        cond_drop_prob: float = 0.2,
        hint_in_ch: int = 11,
    ):
        super().__init__()
        assert len(channel_mults) == 4, "this hint encoder is wired for 4 levels"
        assert len(attn_at) == len(channel_mults)
        self.in_ch = in_ch
        self.base_ch = base_ch
        self.cond_drop_prob = cond_drop_prob
        self.hint_in_ch = hint_in_ch
        self.channel_mults = channel_mults
        self.attn_at = attn_at

        self.t_dim = base_ch
        self.t_emb = TimeEmb(base_ch)
        chs = [base_ch * m for m in channel_mults]

        # In conv (no condition concatenation — E enters via hint encoder)
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # Hint encoder + zero-conv adapters
        self.hint_encoder = HintEncoder(hint_in_ch, chs)
        self.adapters = nn.ModuleList([ZeroConvAdapter(c) for c in chs])

        # Down path
        self.downs = nn.ModuleList()
        self.down_attns = nn.ModuleList()
        prev = base_ch
        for i, c in enumerate(chs):
            blk = nn.ModuleList([
                ResBlock(prev, c, self.t_dim * 4),
                ResBlock(c, c, self.t_dim * 4),
            ])
            self.downs.append(blk)
            self.down_attns.append(Attn(c) if attn_at[i] else nn.Identity())
            prev = c

        # Mid
        self.mid_a = ResBlock(prev, prev, self.t_dim * 4)
        self.mid_attn = Attn(prev)
        self.mid_b = ResBlock(prev, prev, self.t_dim * 4)

        # Up path
        self.ups = nn.ModuleList()
        self.up_attns = nn.ModuleList()
        for i in reversed(range(len(chs))):
            c = chs[i]
            in_c = prev + c  # skip concat
            blk = nn.ModuleList([
                ResBlock(in_c, c, self.t_dim * 4),
                ResBlock(c, c, self.t_dim * 4),
            ])
            self.ups.append(blk)
            self.up_attns.append(Attn(c) if attn_at[i] else nn.Identity())
            prev = c

        self.out_norm = _gn(base_ch)
        self.out_conv = nn.Conv2d(base_ch, in_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    @staticmethod
    def _build_hint(E: torch.Tensor) -> torch.Tensor:
        """Build the 11-ch hint stack: [E, zeros(3), zeros(3), coord(2)]."""
        B, _, H, W = E.shape
        zeros3 = torch.zeros(B, 3, H, W, device=E.device, dtype=E.dtype)
        coord = coord_grid(H, W, E.device, E.dtype).unsqueeze(0).expand(B, 2, -1, -1)
        return torch.cat([E, zeros3, zeros3, coord], dim=1)

    def forward(
        self,
        C_t: torch.Tensor,   # (B, 4, H, W)
        E:   torch.Tensor,   # (B, 3, H, W)
        t:   torch.Tensor,   # (B,) timestep
        force_uncond: bool = False,
    ) -> torch.Tensor:
        B = C_t.shape[0]

        # Condition dropout. At eval, force_uncond=True → always zero E.
        if force_uncond:
            E_eff = torch.zeros_like(E)
        elif self.training and self.cond_drop_prob > 0:
            mask = (torch.rand(B, device=E.device) < self.cond_drop_prob).float()
            mask = mask.view(B, 1, 1, 1)
            E_eff = E * (1.0 - mask)
        else:
            E_eff = E

        hint = self._build_hint(E_eff)
        h0, h1, h2, h3 = self.hint_encoder(hint)
        hint_features = [h0, h1, h2, h3]

        t_emb = self.t_emb(t)
        x = self.in_conv(C_t)
        skips = []
        for i, (blocks, attn) in enumerate(zip(self.downs, self.down_attns)):
            for blk in blocks:
                x = blk(x, t_emb)
            x = attn(x)
            # Inject hint at the post-block features (matching scale via interp if needed)
            h_i = hint_features[i]
            if h_i.shape[-2:] != x.shape[-2:]:
                h_i = F.interpolate(h_i, size=x.shape[-2:], mode="bilinear",
                                    align_corners=False)
            x = x + self.adapters[i](h_i)
            skips.append(x)
            x = F.avg_pool2d(x, 2)

        x = self.mid_a(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_b(x, t_emb)

        for blocks, attn, skip in zip(self.ups, self.up_attns, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            for blk in blocks:
                x = blk(x, t_emb)
            x = attn(x)

        return self.out_conv(F.silu(self.out_norm(x)))


# ---------------- diffusion utilities (cosine schedule) ----------------

def cosine_alphas_cumprod(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule from Nichol & Dhariwal 2021."""
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T + s) / (1 + s)) * math.pi / 2) ** 2
    alphas_cum = (f / f[0]).clamp(min=1e-12)
    return alphas_cum[1:].float()  # length T


def build_diffusion_constants(T: int, device: torch.device, dtype: torch.dtype) -> dict:
    alphas_cum = cosine_alphas_cumprod(T).to(device=device, dtype=dtype)
    return {
        "T": T,
        "alphas_cum": alphas_cum,
        "sqrt_alphas_cum": torch.sqrt(alphas_cum),
        "sqrt_one_minus_alphas_cum": torch.sqrt(1.0 - alphas_cum),
    }


def q_sample(C0: torch.Tensor, t: torch.Tensor, dc: dict, noise: torch.Tensor) -> torch.Tensor:
    """Forward diffusion: C_t = sqrt(alphas_cum_t)·C0 + sqrt(1 - alphas_cum_t)·noise."""
    sa = dc["sqrt_alphas_cum"][t].view(-1, 1, 1, 1)
    so = dc["sqrt_one_minus_alphas_cum"][t].view(-1, 1, 1, 1)
    return sa * C0 + so * noise

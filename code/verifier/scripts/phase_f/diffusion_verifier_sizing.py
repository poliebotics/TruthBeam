"""Phase F prep — diffusion verifier compute sizing.

Profile a representative conditional U-Net diffusion model that maps
capture (4, H, W) → emission (3, h, w) via iterative denoising. The goal
is to estimate compute budget BEFORE building one — not to train to
convergence, just to surface "how expensive is this regime."

Architecture: standard DDPM-style U-Net with:
  - sinusoidal time embedding
  - ResNet blocks at each scale (4 down, 4 up)
  - self-attention at the two lowest scales
  - condition (capture) is concatenated to noisy emission as input
  - output: predicted noise

Profile:
  - Forward + backward step time at bs ∈ {1, 2, 4}
  - Peak GPU memory
  - Inference sampling time at K ∈ {10, 25, 50} denoising steps
  - Parameter count
  - Estimated training budget for N steps (default 5000)

Run on a single GPU. Single 80 GB A100 should handle bs=2 at native res
comfortably; bs=4 is borderline.

Run:
  CUDA_VISIBLE_DEVICES=1 python scripts/phase_f/diffusion_verifier_sizing.py \
    --capture-h 2300 --capture-w 2660 \
    --out experiments/phase_f_prep/diffusion_verifier_sizing
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- Tiny conditional DDPM U-Net ----------------

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
        # Cast t to the dtype of the MLP's first weight so concatenation +
        # matmul stay in the model's working dtype (bf16/fp16/fp32).
        target_dtype = self.mlp[0].weight.dtype
        t = t.to(target_dtype)
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=target_dtype) / half
        )
        ang = t.unsqueeze(-1) * freqs
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        return self.mlp(emb)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_ch) if in_ch >= 32 else 8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(32, out_ch) if out_ch >= 32 else 8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Attn(nn.Module):
    def __init__(self, ch: int, n_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, ch) if ch >= 32 else 8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.n_heads = n_heads

    def forward(self, x):
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


class CondDDPMUNet(nn.Module):
    """Conditional U-Net: input = (noisy_em ‖ capture_resized), output = predicted noise.

    Capture is resized to emission resolution and concatenated channel-wise.
    Capture resolution:  emission_h × emission_w  (typically 1080 × 1920).
    Channels: 3 (noisy emission) + 4 (capture) = 7 input channels.
    """

    def __init__(self, base_ch: int = 96,
                 channel_mults: tuple[int, ...] = (1, 2, 4, 4),
                 attn_at: tuple[bool, ...] = (False, False, True, True)):
        super().__init__()
        self.base_ch = base_ch
        self.t_dim = base_ch
        self.t_emb = TimeEmb(base_ch)
        chs = [base_ch * m for m in channel_mults]
        self.in_conv = nn.Conv2d(3 + 4, base_ch, 3, padding=1)

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

        self.out_norm = nn.GroupNorm(min(32, base_ch), base_ch)
        self.out_conv = nn.Conv2d(base_ch, 3, 3, padding=1)

    def forward(self, noisy_em: torch.Tensor, capture: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        # Resize capture (4, capH, capW) → (4, em_H, em_W) and concat
        cap_resized = F.interpolate(capture, size=noisy_em.shape[-2:],
                                     mode="bilinear", align_corners=False)
        x = torch.cat([noisy_em, cap_resized], dim=1)
        t_emb = self.t_emb(t)
        h = self.in_conv(x)
        skips = []
        for blocks, attn in zip(self.downs, self.down_attns):
            for blk in blocks:
                h = blk(h, t_emb)
            h = attn(h)
            skips.append(h)
            h = F.avg_pool2d(h, 2)

        h = self.mid_a(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_b(h, t_emb)

        for blocks, attn, skip in zip(self.ups, self.up_attns, reversed(skips)):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear",
                               align_corners=False)
            h = torch.cat([h, skip], dim=1)
            for blk in blocks:
                h = blk(h, t_emb)
            h = attn(h)

        return self.out_conv(F.silu(self.out_norm(h)))


# ---------------- Profiling ----------------

def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def profile_step(model: nn.Module, capture_h: int, capture_w: int,
                 em_h: int, em_w: int, bs: int,
                 device: torch.device, dtype: torch.dtype,
                 n_warmup: int = 2, n_iter: int = 5) -> dict:
    """Run forward + backward N times, return mean step time + peak memory."""
    cap = torch.randn(bs, 4, capture_h, capture_w, device=device, dtype=dtype)
    em = torch.randn(bs, 3, em_h, em_w, device=device, dtype=dtype)
    t = torch.randint(0, 1000, (bs,), device=device).float()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warmup
    for _ in range(n_warmup):
        optim.zero_grad(set_to_none=True)
        pred = model(em, cap, t)
        loss = ((pred - em) ** 2).mean()
        loss.backward()
        optim.step()

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iter):
        optim.zero_grad(set_to_none=True)
        pred = model(em, cap, t)
        loss = ((pred - em) ** 2).mean()
        loss.backward()
        optim.step()
    torch.cuda.synchronize()
    elapsed = (time.time() - t0) / n_iter

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return {
        "step_time_sec": round(elapsed, 4),
        "peak_mem_mb": round(peak_mb, 1),
    }


def profile_inference(model: nn.Module, capture_h: int, capture_w: int,
                      em_h: int, em_w: int, n_steps: int,
                      device: torch.device, dtype: torch.dtype) -> float:
    """Simulate K denoising steps at bs=1; return wall-clock seconds."""
    model.eval()
    cap = torch.randn(1, 4, capture_h, capture_w, device=device, dtype=dtype)
    em = torch.randn(1, 3, em_h, em_w, device=device, dtype=dtype)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for k in range(n_steps):
            t = torch.tensor([n_steps - 1 - k], device=device).float()
            em = em - 0.001 * model(em, cap, t)
    torch.cuda.synchronize()
    model.train()
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-h", type=int, default=2300)
    ap.add_argument("--capture-w", type=int, default=2660)
    ap.add_argument("--em-h", type=int, default=1080)
    ap.add_argument("--em-w", type=int, default=1920)
    ap.add_argument("--bs-list", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--base-ch", type=int, default=64)
    ap.add_argument("--mults", nargs="+", type=int, default=[1, 2, 4, 4, 8])
    ap.add_argument("--em-downsample", type=int, default=1,
                    help="Run U-Net at em_h/d × em_w/d (latent-style). d=1 native, d=2 540×960.")
    ap.add_argument("--attn-bottom-only", action="store_true",
                    help="Place attention only at the bottommost scale.")
    ap.add_argument("--inference-steps", nargs="+", type=int, default=[10, 25, 50])
    ap.add_argument("--training-steps-target", type=int, default=5000)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    print(f"[init] building CondDDPMUNet base_ch={args.base_ch} mults={tuple(args.mults)} "
          f"attn_bottom_only={args.attn_bottom_only}",
          flush=True)
    if args.attn_bottom_only:
        attn_at = tuple(i == len(args.mults) - 1 for i in range(len(args.mults)))
    else:
        attn_at = tuple(i >= len(args.mults) - 2 for i in range(len(args.mults)))
    model = CondDDPMUNet(base_ch=args.base_ch,
                         channel_mults=tuple(args.mults),
                         attn_at=attn_at)
    model = model.to(device, dtype=dtype)
    # If em_downsample > 1, run the U-Net at lower spatial resolution.
    em_h_eff = args.em_h // args.em_downsample
    em_w_eff = args.em_w // args.em_downsample
    n_params = count_params(model)
    print(f"[init] params = {n_params/1e6:.1f}M", flush=True)

    summary: dict = {
        "config": {
            "capture_hw": [args.capture_h, args.capture_w],
            "em_hw": [args.em_h, args.em_w],
            "em_hw_unet": [em_h_eff, em_w_eff],
            "em_downsample": args.em_downsample,
            "base_ch": args.base_ch,
            "channel_mults": list(args.mults),
            "attn_at": list(attn_at),
            "dtype": "bf16" if args.bf16 else "fp32",
        },
        "params_m": round(n_params / 1e6, 2),
        "training_profile": {},
        "inference_profile": {},
    }

    # Training profile at each batch size
    for bs in args.bs_list:
        print(f"\n[train] bs={bs}...", flush=True)
        try:
            res = profile_step(model, args.capture_h, args.capture_w,
                               em_h_eff, em_w_eff, bs, device, dtype)
            print(f"  step_time={res['step_time_sec']:.3f}s  peak_mem={res['peak_mem_mb']:.0f} MiB",
                  flush=True)
            res["training_steps_target"] = args.training_steps_target
            res["estimated_total_hours"] = round(
                res["step_time_sec"] * args.training_steps_target / 3600, 2)
            summary["training_profile"][f"bs_{bs}"] = res
        except torch.cuda.OutOfMemoryError as e:
            summary["training_profile"][f"bs_{bs}"] = {"error": "OOM"}
            print(f"  OOM at bs={bs}: {e}", flush=True)
            torch.cuda.empty_cache()

    # Inference profile at each K
    print("\n[inference] sampling profile (bs=1)...", flush=True)
    for K in args.inference_steps:
        try:
            sec = profile_inference(model, args.capture_h, args.capture_w,
                                    em_h_eff, em_w_eff, K, device, dtype)
            summary["inference_profile"][f"steps_{K}"] = {"sec_per_sample": round(sec, 3)}
            print(f"  K={K}: {sec:.2f}s/sample  ({1/sec:.2f} samples/sec)", flush=True)
        except torch.cuda.OutOfMemoryError as e:
            summary["inference_profile"][f"steps_{K}"] = {"error": "OOM"}
            print(f"  OOM at K={K}: {e}", flush=True)
            torch.cuda.empty_cache()

    # Write outputs
    (args.out / "diffusion_sizing.json").write_text(json.dumps(summary, indent=2))

    md_lines = [
        "# Phase F prep — diffusion verifier compute sizing",
        "",
        ("Profiles a representative conditional DDPM-style U-Net "
         "(capture → emission via iterative denoising) for memory + step "
         "time at multiple batch sizes. Goal: estimate compute envelope "
         "before deciding whether a diffusion verifier is feasible at "
         "scale."),
        "",
        f"## Architecture",
        f"- Base channels: {args.base_ch}",
        f"- Channel multipliers: {tuple(args.mults)}",
        f"- Attention at scales: {[i >= len(args.mults) - 2 for i in range(len(args.mults))]}",
        f"- Parameters: **{summary['params_m']} M**",
        f"- Capture resolution: {args.capture_h} × {args.capture_w}",
        f"- Emission resolution: {args.em_h} × {args.em_w}",
        f"- Precision: {'bf16' if args.bf16 else 'fp32'}",
        "",
        "## Training profile",
        "",
        "| bs | step time | peak mem | est. wall-clock for 5000 steps |",
        "|---:|---:|---:|---:|",
    ]
    for bs in args.bs_list:
        r = summary["training_profile"].get(f"bs_{bs}", {})
        if "error" in r:
            md_lines.append(f"| {bs} | OOM | — | — |")
        else:
            md_lines.append(
                f"| {bs} | {r['step_time_sec']:.3f}s | "
                f"{r['peak_mem_mb']:.0f} MiB | "
                f"{r['estimated_total_hours']:.1f} h |"
            )
    md_lines += [
        "",
        "## Inference profile (bs=1, single sample)",
        "",
        "| denoising K | sec / sample | samples / sec |",
        "|---:|---:|---:|",
    ]
    for K in args.inference_steps:
        r = summary["inference_profile"].get(f"steps_{K}", {})
        if "error" in r:
            md_lines.append(f"| {K} | OOM | — |")
        else:
            sec = r["sec_per_sample"]
            md_lines.append(f"| {K} | {sec:.2f}s | {1/sec:.2f} |")
    md_lines += [
        "",
        "## Comparison with current EmissionPredictor binders",
        "",
        ("E1 (EmissionPredictor) is ~70 M parameters and runs at ~0.16 s/step "
         "in a forward+backward at bs=4 single-GPU bf16 (per Phase E v5 "
         "throughput profile). A diffusion verifier of comparable parameter "
         "scale will be **K× slower at inference** (where K is the number "
         "of denoising steps) and similar speed at training."),
        "",
        "## Implications",
        "",
        ("- For threshold-calibration use, inference cost matters: "
         "scoring 600 val rows × K=25 denoising steps at the times above "
         "= the cost the verifier pays per FMR threshold update."),
        ("- For training-as-a-binder use, the conditional model is "
         "comparable in cost to the existing binders, so adding a "
         "diffusion verifier to the ensemble is feasible if the inference "
         "cost is acceptable."),
        ("- A higher-capacity model (larger base_ch or more attention "
         "levels) would require a separate sizing pass; this profile is for "
         "a 'reasonable starter' configuration."),
        "",
    ]
    (args.out / "diffusion_sizing.md").write_text("\n".join(md_lines))
    print(f"\n[done] wrote {args.out}/diffusion_sizing.{{json,md}}", flush=True)


if __name__ == "__main__":
    main()

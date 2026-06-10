"""V4 — E2 memory profile.

Loads E2 config (EmissionPredictor at 2300×2660 packed CFA input), runs
one forward + backward pass at per-GPU batch=2, reports peak memory.

Threshold: peak < 70 GB → DDP=2 allocation (saves 2 GPUs for parallel work).
Otherwise: fall back to DDP=4 with per-GPU bs=1.

We don't actually need to wrap in DDP — DDP gradient buckets add memory
proportional to model params (small relative to activations at this res),
so a single-process forward+backward at bs=2 is within ~10% of DDP=2's
per-GPU peak.

Run:
  python scripts/phase_e/v4_memory_profile.py --bs 2 \
    --capture-h 2300 --capture-w 2660 --bf16
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.emission_dataset import EmissionDataset  # noqa: E402
from losses.emission_loss import emission_loss  # noqa: E402
from models.emission_predictor import EmissionPredictor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=2, help="per-GPU batch size")
    ap.add_argument("--capture-h", type=int, default=2300)
    ap.add_argument("--capture-w", type=int, default=2660)
    ap.add_argument("--emission-h", type=int, default=1080)
    ap.add_argument("--emission-w", type=int, default=1920)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--n-warmup", type=int, default=2)
    ap.add_argument("--n-measure", type=int, default=3)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda:0")
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16

    print(f"[V4] config: bs={args.bs} capture={args.capture_h}x{args.capture_w} "
          f"emission={args.emission_h}x{args.emission_w} bf16={args.bf16}", flush=True)

    # Build model + optimizer (mimics training step)
    model = EmissionPredictor(emission_h=args.emission_h, emission_w=args.emission_w, pretrained=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[V4] params: {n_params/1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=(autocast_dtype == torch.float16))

    # Synthetic input + target
    cap = torch.rand((args.bs, 4, args.capture_h, args.capture_w), device=device)
    em = torch.rand((args.bs, 3, args.emission_h, args.emission_w), device=device)

    print(f"[V4] running {args.n_warmup} warmup + {args.n_measure} measure steps...", flush=True)
    torch.cuda.reset_peak_memory_stats()

    # Warmup steps to allocate any lazy buffers
    for i in range(args.n_warmup):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=autocast_dtype):
            pred = model(cap)
            loss, _parts = emission_loss(pred, em)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        torch.cuda.synchronize()
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"  warmup step {i+1}: peak={peak_gb:.2f} GB  loss={loss.item():.4f}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for i in range(args.n_measure):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=autocast_dtype):
            pred = model(cap)
            loss, _parts = emission_loss(pred, em)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    reserved_gb = torch.cuda.max_memory_reserved() / 1e9
    step_s = elapsed / args.n_measure

    print(f"\n[V4] PEAK MEMORY (allocated): {peak_gb:.2f} GB", flush=True)
    print(f"[V4] PEAK MEMORY (reserved):  {reserved_gb:.2f} GB", flush=True)
    print(f"[V4] Step time: {step_s:.2f} s/step  (model params {n_params/1e6:.1f}M)", flush=True)
    threshold = 70.0
    if peak_gb < threshold:
        print(f"[V4] DECISION: peak={peak_gb:.2f} GB < {threshold} GB → DDP=2 OK (saves 2 GPUs)", flush=True)
    else:
        print(f"[V4] DECISION: peak={peak_gb:.2f} GB >= {threshold} GB → fall back to DDP=4 per-GPU bs=1", flush=True)


if __name__ == "__main__":
    main()

"""V5 — throughput + GPU utilization profile.

For a given config, runs n_warmup + n_measure training steps with REAL
data loader (not synthetic — captures dataloader overhead). Concurrently
samples nvidia-smi for GPU utilization.

Reports:
  - step time mean / std (over measure window)
  - GPU SM utilization mean / max (sampled every 100 ms)
  - Estimated wall clock per epoch + per 30-epoch run
  - Verdict: dataloader-bound (util < 50%) or compute-bound

Run:
  python scripts/phase_e/v5_throughput_profile.py --config configs/e1.yaml --bf16
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

torch.set_num_threads(8)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "phase_e"))

from data.emission_dataset import EmissionDataset  # noqa: E402
from losses.emission_loss import emission_loss  # noqa: E402
from train_phase_e import build_model  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def _gpu_sampler(stop_evt, samples_q, gpu_idx, period_s=0.1):
    while not stop_evt.is_set():
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,memory.used",
                 "--format=csv,noheader,nounits", "-i", str(gpu_idx)],
                capture_output=True, text=True, timeout=2,
            )
            line = r.stdout.strip().split("\n")[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                samples_q.put((int(parts[0]), int(parts[1]), int(parts[2])))
        except Exception:
            pass
        time.sleep(period_s)


def collate(batch):
    return {
        "capture": torch.stack([b["capture"] for b in batch]),
        "emission": torch.stack([b["emission"] for b in batch]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--n-warmup", type=int, default=5)
    ap.add_argument("--n-measure", type=int, default=50)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--gpu-idx", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = torch.device("cuda")
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16

    print(f"[V5] config={args.config.name} arch={cfg['model']['arch']} "
          f"capture={cfg['data']['capture_h']}x{cfg['data']['capture_w']} bs={cfg['train']['batch_size']} "
          f"bf16={args.bf16} workers={args.num_workers}", flush=True)

    ds = EmissionDataset(
        session_dir=Path(cfg["data"]["d2_dir"]),
        row_start=cfg["data"]["d2_train_start"], row_end=cfg["data"]["d2_train_end"],
        capture_h=cfg["data"]["capture_h"], capture_w=cfg["data"]["capture_w"],
        emission_h=cfg["data"]["emission_h"], emission_w=cfg["data"]["emission_w"],
        session_id="D2", augment=False)
    print(f"[V5] dataset n={len(ds)}", flush=True)

    bs = cfg["train"]["batch_size"]
    loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=args.num_workers,
                        collate_fn=collate, pin_memory=True, drop_last=True,
                        persistent_workers=(args.num_workers > 0))

    model = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[V5] model params={n_params/1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=(autocast_dtype == torch.float16))

    # Start GPU sampler subprocess
    stop_evt = mp.Event()
    samples_q = mp.Queue()
    sampler = mp.Process(target=_gpu_sampler, args=(stop_evt, samples_q, args.gpu_idx, 0.1))
    sampler.start()

    # Warmup
    print(f"[V5] warmup {args.n_warmup} steps...", flush=True)
    train_iter = iter(loader)
    model.train()
    for _ in range(args.n_warmup):
        batch = next(train_iter)
        cap = batch["capture"].to(device, non_blocking=True)
        em = batch["emission"].to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=autocast_dtype):
            pred = model(cap)
            loss, _parts = emission_loss(pred, em)
        if scaler.is_enabled():
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        torch.cuda.synchronize()

    # Measure
    print(f"[V5] measuring {args.n_measure} steps...", flush=True)
    # drain GPU samples queue from warmup
    while not samples_q.empty():
        try: samples_q.get_nowait()
        except: break

    step_times = []
    t_start = time.time()
    for i in range(args.n_measure):
        t0 = time.time()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(loader); batch = next(train_iter)
        cap = batch["capture"].to(device, non_blocking=True)
        em = batch["emission"].to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=autocast_dtype):
            pred = model(cap)
            loss, _parts = emission_loss(pred, em)
        if scaler.is_enabled():
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        torch.cuda.synchronize()
        step_times.append(time.time() - t0)
    elapsed = time.time() - t_start

    # Stop sampler
    stop_evt.set()
    sampler.join(timeout=2)
    if sampler.is_alive(): sampler.terminate()
    samples = []
    while not samples_q.empty():
        try: samples.append(samples_q.get_nowait())
        except: break

    step_times = np.array(step_times)
    util_gpu = [s[0] for s in samples] if samples else [0]
    util_mem = [s[1] for s in samples] if samples else [0]

    # Wall clock
    n_train = len(ds)
    bs_per_step = bs * 2  # DDP=2 effective batch
    steps_per_epoch = n_train // bs_per_step
    step_mean = float(step_times.mean())
    epoch_s = step_mean * steps_per_epoch
    run_s = epoch_s * 30
    util_mean = float(np.mean(util_gpu))
    util_max = int(np.max(util_gpu))
    bound = "dataloader-bound" if util_mean < 50 else "compute-bound"

    out = {
        "config": str(args.config),
        "arch": cfg["model"]["arch"],
        "capture_hw": [cfg["data"]["capture_h"], cfg["data"]["capture_w"]],
        "bs_per_gpu": bs,
        "ddp_world_assumed": 2,
        "n_warmup": args.n_warmup, "n_measure": args.n_measure,
        "step_time_mean_s": step_mean,
        "step_time_std_s": float(step_times.std()),
        "step_time_p5_s":  float(np.percentile(step_times, 5)),
        "step_time_p95_s": float(np.percentile(step_times, 95)),
        "gpu_util_samples_n": len(util_gpu),
        "gpu_util_mean_pct": util_mean,
        "gpu_util_max_pct": util_max,
        "gpu_mem_util_mean_pct": float(np.mean(util_mem)),
        "verdict_bound": bound,
        "n_train": n_train,
        "steps_per_epoch_ddp2": steps_per_epoch,
        "epoch_wall_clock_min": epoch_s / 60,
        "30_epoch_wall_clock_h": run_s / 3600,
        "elapsed_measure_s": elapsed,
        "params_M": n_params / 1e6,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\n=== V5 results: {args.config.name} ===")
    print(f"  step time:        {step_mean*1000:.0f} ms (mean), {out['step_time_p5_s']*1000:.0f}-{out['step_time_p95_s']*1000:.0f} ms (p5-p95)")
    print(f"  GPU util:         {util_mean:.1f}% mean, {util_max}% max ({bound})")
    print(f"  GPU mem util:     {out['gpu_mem_util_mean_pct']:.1f}% mean")
    print(f"  steps/epoch:      {steps_per_epoch} (DDP=2, effective batch {bs_per_step})")
    print(f"  epoch wall clock: {epoch_s/60:.1f} min")
    print(f"  30-epoch run:     {run_s/3600:.1f} h", flush=True)
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

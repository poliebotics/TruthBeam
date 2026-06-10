"""Phase G — DDP trainer for the diffusion diagnostic.

torchrun-launchable. Trains a `DiffusionDiagnosticUNet` on (C, E) pairs in one
of three modes:
  - main:               real (C, E) pairs.
  - shuffled:           E paired randomly within session (fixed seed).
  - synthetic_positive: real C + E-dependent perturbation (positive control).

Logged: per-step loss, periodic checkpoints (every --ckpt-every).

Run:
  torchrun --nproc_per_node=8 scripts/phase_g/train_diffusion_diagnostic.py \
    --d2-dir <path>/data/d2 --v10-dir <path>/data/v10 \
    --out-dir <path>/experiments/phase_g_diffusion_diagnostic/main \
    --mode main --max-steps 25000 --bs 4 --bf16
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from phase_g.diffusion_diagnostic_model import (  # noqa: E402
    DiffusionDiagnosticUNet, build_diffusion_constants, q_sample,
)
from phase_g.diffusion_diagnostic_dataset import (  # noqa: E402
    DiffusionDiagnosticDataset, collate_diagnostic, gpu_synthetic_perturbation,
)


# ---------- DDP setup ----------

def setup_ddp() -> tuple[int, int, int, bool]:
    if "RANK" not in os.environ:
        return 0, 1, 0, True
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, (rank == 0)


def cleanup_ddp(world_size: int) -> None:
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def reduce_scalar(x: float) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return x
    t = torch.tensor([x], device=torch.cuda.current_device())
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return t.item()


def cosine_lr(step: int, max_steps: int, base_lr: float, warmup: int = 500) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, (max_steps - warmup))
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * min(1.0, progress)))


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d2-dir", type=Path, required=True)
    ap.add_argument("--v10-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--mode", choices=["main", "shuffled", "synthetic_positive"],
                    default="main")
    ap.add_argument("--max-steps", type=int, default=25_000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--base-ch", type=int, default=96)
    ap.add_argument("--mults", nargs="+", type=int, default=[1, 2, 4, 4])
    ap.add_argument("--cond-drop-prob", type=float, default=0.20)
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--shuffle-seed", type=int, default=12345)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--no-v10", action="store_true",
                    help="Train on D2 only (drop V10). Used for cross-session ablation.")
    ap.add_argument("--no-d2", action="store_true",
                    help="Train on V10 only (drop D2). Used for cross-session ablation. "
                         "Mutually exclusive with --no-v10.")
    args = ap.parse_args()
    if args.no_d2 and args.no_v10:
        raise SystemExit("--no-d2 and --no-v10 are mutually exclusive (would yield empty dataset)")

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rank, world_size, local_rank, is_main = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if world_size > 1 else "cuda")
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16

    if is_main:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "checkpoints").mkdir(exist_ok=True)
        cfg_path = args.out_dir / "training_config.json"
        cfg_path.write_text(json.dumps({
            **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "world_size": world_size,
        }, indent=2))

    # Dataset
    session_dirs = {}
    if not args.no_d2:
        session_dirs["D2"] = args.d2_dir
    if args.v10_dir is not None and args.v10_dir.exists() and not args.no_v10:
        session_dirs["V10"] = args.v10_dir
    if not session_dirs:
        raise SystemExit("No sessions selected (--no-d2 + --no-v10 + missing v10-dir all combined to empty)")
    # Map user-facing mode to dataset mode: "main" → "real" (real (C, E) pairs);
    # the other two pass through unchanged.
    ds_mode = {"main": "real", "shuffled": "shuffled",
               "synthetic_positive": "synthetic_positive"}[args.mode]
    train_ds = DiffusionDiagnosticDataset(
        session_dirs=session_dirs,
        mode=ds_mode,
        role="train",
        stride=args.stride,
        shuffle_seed=args.shuffle_seed,
    )
    if is_main:
        n_per_session = {s: len(rs) for s, rs in train_ds.session_rows.items()}
        print(f"[init] mode={args.mode} world_size={world_size} train_n={len(train_ds)} "
              f"per_session={n_per_session}", flush=True)

    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                                 shuffle=True, drop_last=True) if world_size > 1 else None
    loader = DataLoader(
        train_ds, batch_size=args.bs, sampler=sampler,
        shuffle=(sampler is None), num_workers=args.num_workers,
        collate_fn=collate_diagnostic, pin_memory=True, drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    # Model
    # Attention only at the deepest down/up scale (96×128 with 768×1024 input,
    # mults 1,2,4,4). Adding attention at scale 2 (192×256) would inflate the
    # attention QK^T matrix to ~2.4 G entries → ~38 GB at bs=2 bf16 = OOM on
    # A100-80GB. Mid block already adds attention at 48×64. (Codex audit
    # 2026-05-01 HIGH finding.)
    attn_at = tuple(i == len(args.mults) - 1 for i in range(len(args.mults)))
    model = DiffusionDiagnosticUNet(
        in_ch=4, base_ch=args.base_ch,
        channel_mults=tuple(args.mults),
        attn_at=attn_at,
        cond_drop_prob=args.cond_drop_prob,
        hint_in_ch=11,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"[init] model params={n_params/1e6:.1f}M base_ch={args.base_ch} "
              f"mults={tuple(args.mults)} cond_drop_prob={args.cond_drop_prob}",
              flush=True)

    if world_size > 1:
        for p in model.parameters():
            dist.broadcast(p.data, src=0)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Diffusion constants (use float32 internally for numerical stability;
    # cast on use within autocast region).
    dc = build_diffusion_constants(args.T, device, torch.float32)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    start_step = 0
    if args.resume is not None and args.resume.exists():
        if is_main:
            print(f"[init] resuming from {args.resume}", flush=True)
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        target = model.module if hasattr(model, "module") else model
        target.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start_step = int(ck.get("step", 0))

    if is_main:
        history_path = args.out_dir / "history.jsonl"
        if start_step == 0 and history_path.exists():
            history_path.unlink()

    step = start_step
    t_start = time.time()
    train_iter = iter(loader)
    if world_size > 1 and sampler is not None:
        sampler.set_epoch(0)

    while step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            if world_size > 1 and sampler is not None:
                epoch = step // max(1, len(train_ds) // (args.bs * world_size))
                sampler.set_epoch(epoch)
            train_iter = iter(loader)
            batch = next(train_iter)

        C = batch["C"].to(device, non_blocking=True)
        E = batch["E"].to(device, non_blocking=True)
        if args.mode == "synthetic_positive":
            # GPU-side synthetic perturbation (moved off the CPU dataloader on
            # 2026-05-01 to fix DDP straggler bottleneck).
            alpha = batch["alpha"].to(device, non_blocking=True)
            if not hasattr(main, "_synth_blur_kernel"):
                # Build kernel once per process; cache as function attribute.
                from phase_g.diffusion_diagnostic_dataset import (
                    DiffusionDiagnosticDataset as _DDS,
                )
                kern = _DDS._build_gaussian_kernel_2d(8.0).to(device, dtype=C.dtype)
                main._synth_blur_kernel = kern
            C = gpu_synthetic_perturbation(C, E, alpha, main._synth_blur_kernel)
        B = C.shape[0]

        # Sample timesteps + noise
        t = torch.randint(0, args.T, (B,), device=device).long()
        noise = torch.randn_like(C)
        # q_sample uses the constants in float32; keep in float32 then cast to autocast inside.
        C_t = q_sample(C.float(), t, dc, noise.float()).to(C.dtype)

        # LR cosine schedule
        lr_now = cosine_lr(step, args.max_steps, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr_now

        model.train()
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=autocast_dtype):
            eps_pred = model(C_t, E, t.float())
            loss = F.mse_loss(eps_pred.float(), noise.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        step += 1
        if step % args.log_every == 0:
            l_avg = reduce_scalar(loss.item())
            if is_main:
                elapsed = time.time() - t_start
                rec = {"step": step, "t": round(elapsed, 1), "loss": l_avg, "lr": lr_now}
                with open(args.out_dir / "history.jsonl", "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print(f"[step {step}] loss={l_avg:.4f} lr={lr_now:.2e} t={elapsed:.0f}s",
                      flush=True)

        if step % args.ckpt_every == 0 and is_main:
            target = model.module if hasattr(model, "module") else model
            ckpt = {
                "step": step,
                "model": target.state_dict(),
                "optimizer": opt.state_dict(),
                "args": {**{k: (str(v) if isinstance(v, Path) else v)
                             for k, v in vars(args).items()},
                         "attn_at": list(attn_at)},
            }
            ckpt_path = args.out_dir / "checkpoints" / f"step_{step:08d}.pt"
            torch.save(ckpt, ckpt_path)
            print(f"  [ckpt] saved {ckpt_path}", flush=True)

        if world_size > 1:
            dist.barrier()

    if is_main:
        target = model.module if hasattr(model, "module") else model
        torch.save({
            "step": step,
            "model": target.state_dict(),
            "optimizer": opt.state_dict(),
            "args": {**{k: (str(v) if isinstance(v, Path) else v)
                         for k, v in vars(args).items()},
                     "attn_at": list(attn_at)},
        }, args.out_dir / "model_final.pt")
        print(f"\n[done] elapsed={time.time()-t_start:.0f}s, final step={step}", flush=True)

    cleanup_ddp(world_size)


if __name__ == "__main__":
    main()

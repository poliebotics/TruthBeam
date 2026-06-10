"""Phase H — supervised baseline DDP trainer.

torchrun-launchable. Trains PhaseHBaseline on real-vs-not-real (C, E) pairs.
Negatives 50% shuffled-pair / 50% perturbed-XOF (train pool: Type 1, 2, 4, 6).
Held-out perturbations (Type 3, 5) are NEVER seen during training.

Includes:
  - Identity-render parity HARD GATE check at startup (halts on failure)
  - Loss-decrease overfit canary at step 50 (halts on failure — proves code
    correctness without false-positive on slow learning)
  - NaN/Inf detection
  - Periodic checkpoints at 5k, 10k, 25k

Run:
  torchrun --nproc_per_node=8 scripts/phase_h/train_baseline.py \\
    --d2-dir <path>/data/d2 --v10-dir <path>/data/v10 \\
    --out-dir <path>/experiments/phase_h_supervised_baseline \\
    --max-steps 25000 --bs 4 --bf16
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

from phase_h.baseline_model import PhaseHBaseline, count_params  # noqa: E402
from phase_h.baseline_dataset import (  # noqa: E402
    PhaseHBaselineDataset, collate_phase_h,
)
from phase_g.xof_perturb import verify_identity_render_parity, load_chain_log  # noqa: E402


# ---------------- DDP setup ----------------

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


# ---------------- HARD GATE: identity-render parity ----------------

def run_identity_render_parity_gate(d2_dir: Path, v10_dir: Path,
                                    out_dir: Path) -> None:
    """Verify identity-render matches canonical PNG (max abs diff <= 1e-6).
    HALTS the trainer if mismatch. CC verified bit-exact on g1a; this re-runs
    on Lambda to catch any cross-machine surprise."""
    print("[parity] checking identity-render vs canonical PNG...", flush=True)
    samples = [100, 1500, 3000]
    sessions = []
    if d2_dir.exists():
        sessions.append(("D2", d2_dir, samples))
    if v10_dir.exists():
        sessions.append(("V10", v10_dir,
                         [s for s in samples if s < 3700]))
    all_ok = True
    full_details = {}
    for sess, sd, frames in sessions:
        chain = load_chain_log(sd)
        valid_frames = [f for f in frames if f in chain]
        ok, details = verify_identity_render_parity(
            session_dir=sd, chain_log=chain, sample_frames=valid_frames,
            tol=1e-6)
        full_details[sess] = details
        print(f"[parity] {sess}: max_overall={details['max_overall']:.2e} ok={ok}",
              flush=True)
        if not ok:
            all_ok = False
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "identity_render_parity.json").write_text(
        json.dumps(full_details, indent=2))
    if not all_ok:
        raise RuntimeError(
            "[parity HARD GATE FAIL] identity-render does NOT match canonical "
            "PNG. The CNN would learn render-path artifacts instead of chain "
            "coupling. Aborting Phase H baseline training.")
    print("[parity] HARD GATE PASS", flush=True)


# ---------------- Overfit canary at step 50 ----------------

def run_overfit_canary(model: torch.nn.Module, loader: DataLoader,
                       device: torch.device, dtype: torch.dtype,
                       n_steps: int = 50, lr: float = 1e-3) -> tuple[bool, dict]:
    """Pick a fixed minibatch, train on just it for n_steps. Verify loss
    decreases substantially (rules out wiring errors).

    NOT a hard halt for slow learning — only fails if loss DOES NOT DECREASE
    (i.e., the model literally can't fit a tiny batch, indicating a code bug
    in the gradient flow / loss / data shape).
    """
    print(f"[canary] running {n_steps}-step overfit on fixed minibatch...",
          flush=True)
    # Snapshot params so we can restore (training proper starts fresh)
    snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}
    fixed_batch = next(iter(loader))
    x = fixed_batch["x"].to(device, dtype=dtype)
    y = fixed_batch["label"].to(device, dtype=dtype)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for step in range(n_steps):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=dtype):
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits.float(), y.float())
        loss.backward()
        opt.step()
        losses.append(loss.item())

    initial = losses[0]
    final = losses[-1]
    rel_drop = (initial - final) / max(initial, 1e-9)
    # Restore params for real training
    model.load_state_dict(snapshot)
    print(f"[canary] losses[0]={initial:.4f} losses[{n_steps-1}]={final:.4f} "
          f"rel_drop={rel_drop*100:.1f}%", flush=True)
    # Pass criterion: loss drops by at least 30% over 50 steps on a fixed batch.
    # A correctly-wired BCE classifier at logit-space starting near 0 should
    # easily drop further than this. Failure indicates code bug.
    ok = rel_drop > 0.30 and final < initial
    return ok, {"losses": losses, "initial": initial, "final": final, "rel_drop": rel_drop}


# ---------------- main ----------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d2-dir", type=Path, required=True)
    ap.add_argument("--v10-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-steps", type=int, default=25_000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ckpt-steps", nargs="+", type=int,
                    default=[5000, 10000, 15000, 20000, 25000])
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-parity-gate", action="store_true",
                    help="DEV ONLY: skip identity-render parity check")
    ap.add_argument("--skip-overfit-canary", action="store_true",
                    help="DEV ONLY: skip overfit canary at step 50")
    args = ap.parse_args()

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
        (args.out_dir / "config.json").write_text(json.dumps({
            **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "world_size": world_size,
        }, indent=2))

    # ---------- Identity-render parity HARD GATE ----------
    # Coordinated halt: rank 0 runs the check, broadcasts pass/fail. ALL ranks
    # raise together if fail (avoids hang where rank 0 raises before barrier).
    parity_ok = torch.tensor([1], device=device, dtype=torch.int64)
    if is_main:
        if not args.skip_parity_gate:
            try:
                run_identity_render_parity_gate(args.d2_dir, args.v10_dir, args.out_dir)
            except Exception as e:
                print(f"[parity HARD GATE FAIL on rank 0]: {e}", flush=True)
                parity_ok[0] = 0
    if world_size > 1:
        dist.broadcast(parity_ok, src=0)
        dist.barrier()
    if parity_ok.item() == 0:
        raise RuntimeError("[parity HARD GATE FAIL] coordinated halt across ranks")

    # ---------- Build dataset ----------
    session_dirs = {"D2": args.d2_dir}
    if args.v10_dir is not None and args.v10_dir.exists():
        session_dirs["V10"] = args.v10_dir
    train_ds = PhaseHBaselineDataset(
        session_dirs=session_dirs,
        stride=3, epoch_seed=args.seed,
    )
    if is_main:
        print(f"[init] world_size={world_size} dataset_size={len(train_ds)} "
              f"sessions={list(session_dirs.keys())}", flush=True)

    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                                 shuffle=True, drop_last=True) if world_size > 1 else None
    loader = DataLoader(
        train_ds, batch_size=args.bs, sampler=sampler,
        shuffle=(sampler is None), num_workers=args.num_workers,
        collate_fn=collate_phase_h, pin_memory=True, drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    # ---------- Build model ----------
    model = PhaseHBaseline(in_channels=7).to(device)
    n_params = count_params(model)
    if is_main:
        print(f"[init] PhaseHBaseline params={n_params/1e6:.2f}M "
              f"input_channels=7 resolution=768x1024", flush=True)

    if world_size > 1:
        for p in model.parameters():
            dist.broadcast(p.data, src=0)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # ---------- Overfit canary at step 50 ----------
    # Coordinated halt pattern (matching parity gate): rank 0 runs the canary
    # on the unwrapped model.module (which doesn't trigger DDP), broadcasts
    # pass/fail, ALL ranks raise together on fail.
    canary_ok = torch.tensor([1], device=device, dtype=torch.int64)
    canary_details: dict | None = None
    if is_main:
        if not args.skip_overfit_canary:
            target_model = model.module if hasattr(model, "module") else model
            try:
                ok, canary_details = run_overfit_canary(
                    target_model, loader, device, autocast_dtype, n_steps=50)
                (args.out_dir / "overfit_canary.json").write_text(
                    json.dumps(canary_details, indent=2))
                if not ok:
                    print(f"[canary HARD GATE FAIL on rank 0]: "
                          f"initial={canary_details['initial']:.4f} "
                          f"final={canary_details['final']:.4f} "
                          f"rel_drop={canary_details['rel_drop']*100:.1f}%",
                          flush=True)
                    canary_ok[0] = 0
            except Exception as e:
                print(f"[canary HARD GATE FAIL on rank 0]: exception {e}", flush=True)
                canary_ok[0] = 0
    if world_size > 1:
        dist.broadcast(canary_ok, src=0)
        dist.barrier()
    if canary_ok.item() == 0:
        raise RuntimeError("[canary HARD GATE FAIL] coordinated halt across ranks")

    # ---------- Real training loop ----------
    if is_main:
        history_path = args.out_dir / "history.jsonl"
        if history_path.exists():
            history_path.unlink()

    step = 0
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

        x = batch["x"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        # LR cosine schedule
        lr_now = cosine_lr(step, args.max_steps, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr_now

        model.train()
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=autocast_dtype):
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits.float(), y.float())
        # NaN/Inf detection — hard halt
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"[NaN HALT] loss not finite at step {step}: {loss.item()}. "
                f"Halting per spec.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        step += 1
        if step % args.log_every == 0:
            l_avg = reduce_scalar(loss.item())
            # Quick batch acc
            with torch.no_grad():
                preds = (logits.detach() > 0).float()
                acc = (preds == y).float().mean().item()
            acc_avg = reduce_scalar(acc)
            if is_main:
                elapsed = time.time() - t_start
                rec = {"step": step, "t": round(elapsed, 1),
                       "loss": l_avg, "batch_acc": acc_avg, "lr": lr_now}
                with open(args.out_dir / "history.jsonl", "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print(f"[step {step}] loss={l_avg:.4f} acc={acc_avg:.3f} "
                      f"lr={lr_now:.2e} t={elapsed:.0f}s", flush=True)

        if step in args.ckpt_steps and is_main:
            target = model.module if hasattr(model, "module") else model
            ckpt_path = args.out_dir / "checkpoints" / f"step_{step:08d}.pt"
            torch.save({"step": step, "model": target.state_dict(),
                        "optimizer": opt.state_dict(),
                        "args": {k: (str(v) if isinstance(v, Path) else v)
                                 for k, v in vars(args).items()}},
                       ckpt_path)
            print(f"  [ckpt] saved {ckpt_path}", flush=True)

        if world_size > 1:
            dist.barrier()

    if is_main:
        target = model.module if hasattr(model, "module") else model
        torch.save({"step": step, "model": target.state_dict(),
                    "optimizer": opt.state_dict(),
                    "args": {k: (str(v) if isinstance(v, Path) else v)
                             for k, v in vars(args).items()}},
                   args.out_dir / "model_final.pt")
        print(f"\n[done] elapsed={time.time()-t_start:.0f}s, final step={step}",
              flush=True)
        # Touch sentinel for orchestration
        (args.out_dir / "PHASE_H_TRAINING_DONE").touch()

    cleanup_ddp(world_size)


if __name__ == "__main__":
    main()

"""Phase F-A FULL training run — DDP-enabled.

Builds on the mini-experiment infrastructure (`train_phase_f_a_mini.py`)
but adds:
  - torch.distributed (NCCL) via torchrun
  - DistributedSampler for training data
  - Multi-session training (D2 + V10 train_normal union)
  - Periodic checkpointing (every 5000 steps)
  - Eval cycles every 5000 steps (causality probe + held-out binder MSE)
  - Resume-from-checkpoint
  - Default 100,000 steps × DDP=8 ≈ 12 hr at native CFA resolution

Architecture: ControlNet editor (`editor_controlnet.py`).
Loss: mse_plus_hinge_wrongs (the formulation that won the Case B mini).

Run:
  cd /path/to/poliebotics_phase_b/poliebotics_phase_b
  source .venv_a100/bin/activate
  torchrun --nproc_per_node=8 scripts/phase_f/train_phase_f_a_full.py \
    --d2-dir /path/to/poliebotics_phase_b/data/d2 \
    --v10-dir /path/to/poliebotics_phase_b/data/v10 \
    --out-dir /path/to/poliebotics_phase_b/experiments/phase_f/f_a_full_v1 \
    --max-steps 100000 --bs 2 --bf16

Checkpoints land at <out>/checkpoints/step_<n>.pt; resume with --resume <path>.
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
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler

torch.set_num_threads(4)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "phase_e"))
sys.path.insert(0, str(ROOT / "scripts" / "phase_f"))

from data.emission_dataset import load_emission_at  # noqa: E402
from phase_f.dataset_temporal_pairs import (  # noqa: E402
    TemporalPairDataset, collate_temporal_pairs, split_rows,
)
from phase_f.editor_controlnet import EditorControlNet  # noqa: E402
from phase_f.editor_losses import charbonnier, grad_loss  # noqa: E402
from models.emission_predictor import EmissionPredictor  # noqa: E402
from train_phase_f_a_mini import (  # noqa: E402
    SURROGATE_BINDERS, load_binder, downsample_for_binder,
    binder_loss, causality_probe, psnr01,
)


# ---------- DDP setup ----------

def setup_ddp() -> tuple[int, int, int, bool]:
    """Initialize NCCL process group from torchrun env vars.

    Returns: (rank, world_size, local_rank, is_main).
    If torchrun env not set (single-process run), returns (0, 1, 0, True).
    """
    if "RANK" not in os.environ:
        return 0, 1, 0, True
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    is_main = (rank == 0)
    return rank, world_size, local_rank, is_main


def cleanup_ddp(world_size: int) -> None:
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def reduce_scalar(x: float, op=dist.ReduceOp.AVG) -> float:
    """All-reduce a scalar across DDP ranks (no-op if single-process)."""
    if not (dist.is_available() and dist.is_initialized()):
        return x
    t = torch.tensor([x], device=torch.cuda.current_device())
    dist.all_reduce(t, op=op)
    return t.item()


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d2-dir", type=Path, required=True)
    ap.add_argument("--v10-dir", type=Path, default=None,
                    help="Optional V10 session dir for multi-session training.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-steps", type=int, default=100_000)
    ap.add_argument("--bs", type=int, default=2, help="Per-GPU batch size.")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--coef-recon", type=float, default=1.0)
    ap.add_argument("--coef-grad", type=float, default=0.1)
    ap.add_argument("--coef-binder", type=float, default=5.0)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--n-hard-wrongs", type=int, default=6)
    ap.add_argument("--wrong-pool-size", type=int, default=256)
    ap.add_argument("--binder-loss-type",
                    choices=["hinge", "mse_plus_hinge_wrongs"],
                    default="mse_plus_hinge_wrongs")
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--probe-every", type=int, default=5000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--resume", type=Path, default=None,
                    help="Path to a step_<n>.pt checkpoint to resume from.")
    ap.add_argument("--exp001c-ckpt-warm-start", type=str,
                    default="/path/to/poliebotics_phase_b/experiments/exp001c/checkpoints/ep027.pt")
    ap.add_argument("--hint-mode", choices=["v1_control", "v1_5_treatment"],
                    default="v1_5_treatment",
                    help="v1.5 ablation: 'v1_control' = zero E(t-1)+Δ in hint; "
                         "'v1_5_treatment' = full hint with real E_source. Default = treatment "
                         "(matches existing v1 behavior).")
    ap.add_argument("--max-train-rows", type=int, default=None,
                    help="Cap on D2 train rows used (subsets the head of split_rows('D2','train')). "
                         "Used by the v1.5 ablation to train on D2 [0, 500).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for editor init randomness; same seed across control/treatment "
                         "guarantees identical hint-encoder initial weights.")
    ap.add_argument("--no-v10", action="store_true",
                    help="Skip V10 train data even if --v10-dir provided.")
    args = ap.parse_args()
    # Set seed deterministically for editor init.
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

    # Datasets ------------------------------------------------------------
    train_d2_rows = split_rows("D2", "train")
    if args.max_train_rows is not None:
        train_d2_rows = train_d2_rows[:args.max_train_rows]
    train_d2 = TemporalPairDataset(
        session_dir=args.d2_dir, rows=train_d2_rows,
        k_choices=[1, 2, 3], augment=False, seed=0,
    )
    train_datasets = [train_d2]
    if (args.v10_dir is not None and args.v10_dir.exists()
            and not args.no_v10):
        train_v10_rows = split_rows("V10", "train")
        train_v10 = TemporalPairDataset(
            session_dir=args.v10_dir, rows=train_v10_rows,
            k_choices=[1, 2, 3], augment=False, seed=1,
        )
        train_datasets.append(train_v10)
    if len(train_datasets) == 1:
        train_ds = train_datasets[0]
    else:
        train_ds = ConcatDataset(train_datasets)
    val_ds = TemporalPairDataset(
        session_dir=args.d2_dir, rows=split_rows("D2", "val"),
        k_choices=[1], augment=False, seed=0,
    )

    if is_main:
        print(f"[init] world_size={world_size} train n={len(train_ds)} val n={len(val_ds)}",
              flush=True)

    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                                  shuffle=True, drop_last=True) if world_size > 1 else None
    train_loader = DataLoader(
        train_ds, batch_size=args.bs, sampler=sampler,
        shuffle=(sampler is None), num_workers=args.num_workers,
        collate_fn=collate_temporal_pairs, pin_memory=True, drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    # Editor --------------------------------------------------------------
    editor = EditorControlNet(
        capture_h=2300, capture_w=2660,
        emission_h=1080, emission_w=1920,
        init_mode="exp001c-warm-start",
        hint_use_source=(args.hint_mode == "v1_5_treatment"),
    ).to(device)
    if args.exp001c_ckpt_warm_start and Path(args.exp001c_ckpt_warm_start).exists():
        editor.load_warm_start(Path(args.exp001c_ckpt_warm_start))
    # Zero-init E(t-1) + Δ channels of the hint encoder's first conv. Required
    # for the v1.5 ablation: with this zero-init, control and treatment runs
    # produce identical step-0 outputs regardless of whether E_source is fed.
    editor.zero_init_source_channels()
    n_params = sum(p.numel() for p in editor.parameters())
    if is_main:
        print(f"[init] editor=controlnet params={n_params/1e6:.1f}M", flush=True)

    # DDP wrap (after warm-start so all ranks load identical weights)
    if world_size > 1:
        # Broadcast initial weights from rank 0 to all ranks for identical start
        for p in editor.parameters():
            dist.broadcast(p.data, src=0)
        editor = DDP(editor, device_ids=[local_rank], find_unused_parameters=False)

    # Surrogate binders (frozen, replicated on each rank) ------------------
    binders = []
    for spec in SURROGATE_BINDERS:
        try:
            b = load_binder(spec, device)
            binders.append((spec, b))
            if is_main:
                print(f"[init] loaded binder: {spec['name']}", flush=True)
        except Exception as e:
            if is_main:
                print(f"[WARN] could not load binder {spec['name']}: {e}", flush=True)
    if not binders and is_main:
        print("[WARN] no surrogate binders loaded — running with L_binder=0", flush=True)

    # Hard-wrongs pool (per-rank random subset for diversity across ranks)
    wrong_pool: torch.Tensor | None = None
    if args.n_hard_wrongs > 0 and binders:
        rng = np.random.RandomState(123 + rank)
        d2_train_rows = split_rows("D2", "train")
        pool_rows = sorted(rng.choice(d2_train_rows,
                                      size=min(args.wrong_pool_size, len(d2_train_rows)),
                                      replace=False).tolist())
        if is_main:
            print(f"[init] rank {rank}: loading hard-wrongs pool ({len(pool_rows)} emissions)...",
                  flush=True)
        pool_emissions = []
        for r in pool_rows:
            try:
                em = load_emission_at(args.d2_dir / "derived" / "Emissions" / f"tile_{r:06d}.png",
                                      1080, 1920)
                pool_emissions.append(em)
            except Exception as e:
                if is_main:
                    print(f"  skip row {r}: {e}", flush=True)
        wrong_pool = torch.stack(pool_emissions).to(device)
        del pool_emissions
        pool_mb = wrong_pool.element_size() * wrong_pool.numel() / (1024 ** 2)
        if is_main:
            print(f"[init] wrong_pool shape={tuple(wrong_pool.shape)}  mem={pool_mb:.0f} MiB",
                  flush=True)

    # Optimizer ----------------------------------------------------------
    opt = torch.optim.AdamW(editor.parameters(), lr=args.lr, weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=(autocast_dtype == torch.float16))

    # Resume ------------------------------------------------------------
    start_step = 0
    if args.resume is not None and args.resume.exists():
        if is_main:
            print(f"[init] resume from {args.resume}", flush=True)
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        target_editor = editor.module if hasattr(editor, "module") else editor
        target_editor.load_state_dict(ck["editor"])
        opt.load_state_dict(ck["optimizer"])
        start_step = int(ck.get("step", 0))

    # Train loop --------------------------------------------------------
    if is_main:
        history_path = args.out_dir / "history.jsonl"
        if start_step == 0 and history_path.exists():
            history_path.unlink()
    train_iter = iter(train_loader)
    step = start_step
    t_start = time.time()

    if world_size > 1 and sampler is not None:
        sampler.set_epoch(0)

    while step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            if world_size > 1 and sampler is not None:
                sampler.set_epoch(step // max(1, len(train_ds) // (args.bs * world_size)))
            train_iter = iter(train_loader)
            batch = next(train_iter)

        C_s = batch["C_source"].to(device, non_blocking=True)
        E_s = batch["E_source"].to(device, non_blocking=True)
        E_t = batch["E_target"].to(device, non_blocking=True)
        C_t = batch["C_target"].to(device, non_blocking=True)

        # Sample hard-wrongs for this step
        wrongs_batch = None
        if wrong_pool is not None and args.n_hard_wrongs > 0:
            B = C_s.shape[0]
            idx = torch.randint(0, wrong_pool.shape[0],
                                (B, args.n_hard_wrongs), device=device)
            wrongs_batch = wrong_pool[idx]

        editor.train()
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=autocast_dtype):
            C_pred = editor(C_s, E_s, E_t)
            L_recon = charbonnier(C_pred, C_t)
            L_grad = grad_loss(C_pred, C_t)
            if binders:
                L_binder, binder_diag = binder_loss(
                    binders, C_pred, E_t, E_s, autocast_dtype,
                    margin=args.margin, wrongs=wrongs_batch,
                    loss_type=args.binder_loss_type,
                )
            else:
                L_binder = torch.zeros((), device=device)
                binder_diag = {}
            loss = (args.coef_recon * L_recon + args.coef_grad * L_grad
                    + args.coef_binder * L_binder)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(editor.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(editor.parameters(), 1.0)
            opt.step()

        step += 1
        if step % args.log_every == 0:
            l_recon = reduce_scalar(L_recon.item())
            l_grad = reduce_scalar(L_grad.item())
            l_binder = reduce_scalar(L_binder.item())
            l_total = reduce_scalar(loss.item())
            if is_main:
                elapsed = time.time() - t_start
                rec = {
                    "step": step, "t": round(elapsed, 1),
                    "L_recon": l_recon, "L_grad": l_grad,
                    "L_binder": l_binder, "loss": l_total,
                }
                with open(args.out_dir / "history.jsonl", "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print(f"[step {step}] L_recon={l_recon:.4f}  L_grad={l_grad:.4f}  "
                      f"L_binder={l_binder:.4f}  loss={l_total:.4f}  "
                      f"t={elapsed:.0f}s", flush=True)

        if step % args.probe_every == 0 and is_main:
            inner = editor.module if hasattr(editor, "module") else editor
            inner.eval()
            probe = causality_probe(
                editor=inner, val_ds=val_ds, d2_dir=args.d2_dir,
                n_sources=8, n_targets=32,
                autocast_dtype=autocast_dtype, device=device, seed=7,
            )
            with open(args.out_dir / "probe_history.jsonl", "a") as fh:
                fh.write(json.dumps({"step": step, **probe}) + "\n")
            print(f"\n=== causality probe @ step {step} ===\n"
                  f"  diversity_mean={probe['diversity_mean']:.5f}  "
                  f"cfa_delta_from_source={probe['cfa_delta_from_source_mean']:.5f}",
                  flush=True)
            inner.train()

        if step % args.ckpt_every == 0 and is_main:
            target_editor = editor.module if hasattr(editor, "module") else editor
            ckpt = {
                "step": step,
                "editor": target_editor.state_dict(),
                "optimizer": opt.state_dict(),
                "args": vars(args),
            }
            ckpt_path = args.out_dir / "checkpoints" / f"step_{step:08d}.pt"
            torch.save(ckpt, ckpt_path)
            print(f"  [ckpt] saved {ckpt_path}", flush=True)

        if world_size > 1:
            dist.barrier()

    # Final save (only rank 0)
    if is_main:
        target_editor = editor.module if hasattr(editor, "module") else editor
        ckpt = {
            "step": step,
            "editor": target_editor.state_dict(),
            "optimizer": opt.state_dict(),
            "args": vars(args),
        }
        torch.save(ckpt, args.out_dir / "editor_final.pt")
        print(f"\n[done] elapsed={time.time()-t_start:.0f}s, final step={step}", flush=True)

    cleanup_ddp(world_size)


if __name__ == "__main__":
    main()

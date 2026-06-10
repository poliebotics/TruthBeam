"""Unified fresh binder training launcher (F-A v2 prep).

Loads a YAML config selecting one of the 4 architecture families and trains
the binder per Phase E convention (L1+SSIM loss, AdamW cosine, augmentation).

Usage:
    python scripts/fa_v2/train_fresh_binder.py --config scripts/fa_v2/configs/A1.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.emission_dataset import EmissionDataset  # noqa: E402
from torch.utils.data import DataLoader, ConcatDataset
from models.fresh import FRESH_BINDER_REGISTRY  # noqa: E402


# Phase E split convention (extended with V10 for fresh binders)
TRAIN_RANGES = {
    "D2":  (0, 4194),
    "V10": (0, 1280),
}
VAL_RANGES = {
    "D2":  (4194, 5392),
    "V10": (1280, 1500),
}


def cosine_lr(step: int, max_steps: int, base_lr: float, warmup: int = 500,
              min_lr: float = 1e-6) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(1.0, max(0.0, progress))
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cos


def ssim_simple(p: torch.Tensor, t: torch.Tensor,
                window: int = 11, eps: float = 1e-6) -> torch.Tensor:
    """Simple SSIM (per-channel mean over batch). p, t: (B, C, H, W) in [0, 1]."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    pad = window // 2
    # Use F.avg_pool2d as a box filter for the mean (good enough for loss term)
    mu_p = F.avg_pool2d(p, window, stride=1, padding=pad)
    mu_t = F.avg_pool2d(t, window, stride=1, padding=pad)
    var_p = F.avg_pool2d(p * p, window, stride=1, padding=pad) - mu_p ** 2
    var_t = F.avg_pool2d(t * t, window, stride=1, padding=pad) - mu_t ** 2
    cov  = F.avg_pool2d(p * t, window, stride=1, padding=pad) - mu_p * mu_t
    num = (2 * mu_p * mu_t + C1) * (2 * cov + C2)
    den = (mu_p ** 2 + mu_t ** 2 + C1) * (var_p + var_t + C2)
    return (num / (den + eps)).mean()


def loss_fn(pred: torch.Tensor, target: torch.Tensor,
            l1_w: float = 0.85, ssim_w: float = 0.15) -> torch.Tensor:
    l1 = (pred - target).abs().mean()
    ssim_v = ssim_simple(pred, target)
    return l1_w * l1 + ssim_w * (1.0 - ssim_v)


def augment(cap: torch.Tensor, em: torch.Tensor, rng: torch.Generator
            ) -> tuple[torch.Tensor, torch.Tensor]:
    """50% horizontal flip only.

    Codex audit MED 2026-05-03: rotation/scale augmentation was dropped because
    cap (1150x1330 / 2300x2660) and em (1080x1920) have different aspect ratios,
    and applying the same normalized-coords affine breaks the C↔E correspondence
    (the projector calibration is NOT a pure 2D affine between the two spaces).
    Phase E e1-recipe used no augmentation either; HFlip is the only safe
    additional transform here (preserves correspondence assuming mirror-symmetric
    rig setup, which is the case for our cloth/projector geometry).

    cap: (B, 4, H_c, W_c); em: (B, 3, H_e, W_e). Both flipped together.
    """
    if torch.rand((), generator=rng).item() < 0.5:
        cap = torch.flip(cap, dims=(-1,))
        em  = torch.flip(em, dims=(-1,))
    return cap, em


def build_dataset(session_dirs: dict, ranges: dict, capture_h: int, capture_w: int,
                  augment_flag: bool = False) -> ConcatDataset:
    """Build a ConcatDataset of all session subsets in ranges."""
    parts = []
    for sess, (rs, re) in ranges.items():
        if sess not in session_dirs: continue
        ds = EmissionDataset(
            session_dir=Path(session_dirs[sess]),
            row_start=rs, row_end=re,
            capture_h=capture_h, capture_w=capture_w,
            emission_h=1080, emission_w=1920,
            session_id=sess, augment=augment_flag,
        )
        parts.append(ds)
    if not parts:
        raise SystemExit(f"no datasets built from session_dirs={session_dirs} ranges={ranges}")
    return ConcatDataset(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True,
                    help="Path to per-binder YAML config")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Override output dir from config")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Override max_steps from config (e.g. 100 for smoke test)")
    ap.add_argument("--device", type=str, default="cuda",
                    help="cuda or cpu (cuda recommended; smoke tests can use cpu)")
    ap.add_argument("--resume", type=Path, default=None,
                    help="Resume from a checkpoint .pt file. Loads model + "
                    "optimizer state, sets starting step. Combined with "
                    "--max-steps to extend training (e.g. resume from 25k ckpt "
                    "and run --max-steps 50000 to reach 50k).")
    args = ap.parse_args()

    if not args.config.exists():
        raise SystemExit(f"config not found: {args.config}")
    cfg = yaml.safe_load(args.config.read_text())

    # Resolve overrides
    out_dir = args.out_dir or Path(cfg["out_dir"])
    max_steps = args.max_steps if args.max_steps is not None else cfg["max_steps"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)
    (out_dir / "config_resolved.json").write_text(json.dumps(
        {**cfg, "max_steps_resolved": max_steps, "out_dir_resolved": str(out_dir)},
        indent=2))

    # Reproducibility
    seed = cfg["seed"]
    torch.manual_seed(seed)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    # Device + dtype
    device = torch.device(args.device)
    autocast_dtype = torch.bfloat16 if cfg.get("bf16", True) else torch.float16

    # Build model
    family = cfg["family"]   # one of "A", "B", "C", "D"
    if family not in FRESH_BINDER_REGISTRY:
        raise SystemExit(f"unknown family {family!r}")
    model_cls = FRESH_BINDER_REGISTRY[family]
    model_kwargs = cfg.get("model_kwargs", {})
    model = model_cls(**model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[init] family={family} binder_id={cfg['binder_id']} "
          f"params={n_params/1e6:.2f}M cap={cfg['capture_h']}x{cfg['capture_w']}",
          flush=True)

    # Build datasets
    session_dirs = {k: Path(v).expanduser() for k, v in cfg["session_dirs"].items()}
    train_ds = build_dataset(session_dirs, TRAIN_RANGES,
                              cfg["capture_h"], cfg["capture_w"], augment_flag=False)
    val_ds = build_dataset(session_dirs, VAL_RANGES,
                            cfg["capture_h"], cfg["capture_w"], augment_flag=False)
    print(f"[init] train_ds={len(train_ds)} val_ds={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg["bs"], shuffle=True,
                               num_workers=cfg.get("num_workers", 2), pin_memory=True,
                               drop_last=True, persistent_workers=cfg.get("num_workers", 2) > 0)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)

    # Training loop
    history_path = out_dir / "history.jsonl"

    step = 0
    if args.resume is not None:
        if not args.resume.exists():
            raise SystemExit(f"--resume ckpt not found: {args.resume}")
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=True)
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        step = int(ck.get("step", 0))
        print(f"[resume] loaded {args.resume}; starting at step {step} "
              f"(max_steps={max_steps})", flush=True)
        # Append to existing history rather than wiping
    else:
        if history_path.exists(): history_path.unlink()
    t_start = time.time()
    train_iter = iter(train_loader)
    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        cap = batch["capture"].to(device, non_blocking=True)
        em  = batch["emission"].to(device, non_blocking=True)
        if cfg.get("augment", True):
            cap, em = augment(cap, em, rng)

        # LR schedule
        lr_now = cosine_lr(step, max_steps, cfg["lr"])
        for g in opt.param_groups: g["lr"] = lr_now

        model.train()
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu",
                                 dtype=autocast_dtype, enabled=(device.type == "cuda")):
            pred = model(cap)
            loss = loss_fn(pred.float(), em.float(),
                           cfg.get("loss_l1_w", 0.85), cfg.get("loss_ssim_w", 0.15))
        if not torch.isfinite(loss):
            raise RuntimeError(f"[NaN HALT] loss not finite at step {step}: {loss.item()}")
        loss.backward()
        # Codex audit HIGH 2026-05-03: error_if_nonfinite=True so grad NaN/Inf
        # halts immediately rather than silently corrupting weights.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                       error_if_nonfinite=True)
        opt.step()

        step += 1
        if step % cfg.get("log_every", 50) == 0 or step == max_steps:
            elapsed = time.time() - t_start
            rec = {"step": step, "t": round(elapsed, 1),
                   "loss": float(loss.item()), "lr": lr_now}
            with open(history_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"[step {step}/{max_steps}] loss={loss.item():.4f} lr={lr_now:.2e} "
                  f"t={elapsed:.0f}s", flush=True)

        # Periodic checkpoint
        if step in cfg.get("ckpt_steps", []):
            ckpt_path = out_dir / "checkpoints" / f"step_{step:08d}.pt"
            torch.save({"step": step, "model": model.state_dict(),
                        "optimizer": opt.state_dict(), "args": cfg},
                       ckpt_path)
            print(f"  [ckpt] {ckpt_path}", flush=True)

    # Final checkpoint
    final_path = out_dir / "model_final.pt"
    torch.save({"step": step, "model": model.state_dict(),
                "optimizer": opt.state_dict(), "args": cfg}, final_path)
    elapsed = time.time() - t_start
    print(f"[done] step={step} elapsed={elapsed:.0f}s → {final_path}")
    (out_dir / "TRAINING_DONE").touch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

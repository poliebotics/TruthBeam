"""exp001c training: emission RGB prediction (encoder + U-Net decoder).

Modes:
  --smoke=forward
  --smoke=overfit  one batch, 100 steps; PSNR >= 20 dB pass criterion
  (default)        full training
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.emission_dataset import EmissionDataset  # noqa: E402
from data.native_bayer_dataset import NativeBayerDataset  # noqa: E402
from eval.emission_metrics import emission_metrics, psnr  # noqa: E402
from losses.emission_loss import emission_loss  # noqa: E402
from models.emission_predictor import EmissionPredictor  # noqa: E402
from models.emission_predictor_large import EmissionPredictorLarge  # noqa: E402

_MODEL_CLASSES = {
    "EmissionPredictor": EmissionPredictor,
    "EmissionPredictorLarge": EmissionPredictorLarge,
}


def _build_dataset_emi(cfg, row_start: int, row_end: int):
    name = cfg["data"].get("dataset", "EmissionDataset")
    if name == "NativeBayerDataset":
        return NativeBayerDataset(
            session_dir=Path(cfg["data"]["d2_dir"]),
            row_start=row_start,
            row_end=row_end,
            targets=tuple(cfg["data"].get("targets", ("emission",))),
            emission_h=cfg["data"]["emission_h"],
            emission_w=cfg["data"]["emission_w"],
            session_id="d2",
        )
    if name == "EmissionDataset":
        return EmissionDataset(
            session_dir=Path(cfg["data"]["d2_dir"]),
            row_start=row_start,
            row_end=row_end,
            capture_h=cfg["data"]["capture_h"],
            capture_w=cfg["data"]["capture_w"],
            emission_h=cfg["data"]["emission_h"],
            emission_w=cfg["data"]["emission_w"],
            session_id="d2",
            augment=cfg["data"].get("augment", False),
            seed=cfg.get("seed", 0),
        )
    raise ValueError(f"unknown dataset class: {name}")


def _build_model_emi(cfg) -> torch.nn.Module:
    name = cfg.get("model", {}).get("class", "EmissionPredictor")
    cls = _MODEL_CLASSES.get(name)
    if cls is None:
        raise ValueError(f"unknown model class: {name}")
    return cls(emission_h=cfg["data"]["emission_h"], emission_w=cfg["data"]["emission_w"])


def collate(batch):
    return {
        "capture": torch.stack([b["capture"] for b in batch]),
        "emission": torch.stack([b["emission"] for b in batch]),
        "t": torch.tensor([b["t"] for b in batch]),
        "session_id": [b["session_id"] for b in batch],
    }


def build_train_dataset(cfg):
    return _build_dataset_emi(cfg, cfg["data"]["d2_train_start"], cfg["data"]["d2_train_end"])


def build_val_dataset(cfg):
    return _build_dataset_emi(cfg, cfg["data"]["d2_val_start"], cfg["data"]["d2_val_end"])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    metrics_acc = None
    n = 0
    for batch in loader:
        cap = batch["capture"].to(device, non_blocking=True)
        em = batch["emission"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            pred = model(cap).float()
        m = emission_metrics(pred, em)
        if metrics_acc is None:
            metrics_acc = {k: 0.0 for k in m}
        for k, v in m.items():
            metrics_acc[k] += v * cap.shape[0]
        n += cap.shape[0]
    if n == 0:
        return {"n": 0}
    out = {k: v / n for k, v in metrics_acc.items()}
    out["n"] = n
    return out


def run_smoke_forward(model, batch, device):
    cap = batch["capture"].to(device)
    em = batch["emission"].to(device)
    print(f"[smoke fwd] capture: {cap.shape} {cap.dtype} "
          f"min={cap.min():.3f} max={cap.max():.3f}")
    print(f"[smoke fwd] emission: {em.shape} {em.dtype} "
          f"min={em.min():.3f} max={em.max():.3f}")

    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        pred = model(cap).float()
    print(f"[smoke fwd] pred: {pred.shape} {pred.dtype} "
          f"min={pred.min():.4f} max={pred.max():.4f} "
          f"finite={torch.isfinite(pred).all().item()}")
    assert (pred >= 0).all() and (pred <= 1).all(), "sigmoid out of [0,1]"
    assert pred.shape == em.shape, f"shape mismatch: pred {pred.shape} vs em {em.shape}"

    loss, parts = emission_loss(pred, em)
    print(f"[smoke fwd] cold loss parts: {parts}")
    print(f"[smoke fwd] cold PSNR: {psnr(pred, em):.2f} dB "
          f"(uniform 0.5 prediction would give ~9 dB)")
    assert torch.isfinite(loss), "loss not finite"

    model.train()
    pred = model(cap)
    loss, _ = emission_loss(pred, em)
    loss.backward()
    n_total = sum(1 for _ in model.parameters())
    n_with_grad = sum(p.grad is not None for p in model.parameters())
    n_finite = sum(
        (p.grad is not None and torch.isfinite(p.grad).all().item())
        for p in model.parameters()
    )
    print(f"[smoke fwd] backward: {n_with_grad}/{n_total} params got grads, "
          f"{n_finite}/{n_total} finite")


def run_smoke_overfit(model, batch, device, lr, steps, out_dir,
                      pass_threshold: float = 20.0,
                      fail_threshold: float = 15.0):
    cap = batch["capture"].to(device)
    em = batch["emission"].to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda")
    grad_clip = 1.0

    psnr_history = []
    losses = []
    t0 = time.time()
    model.train()
    print(f"[overfit] mode=fp16/autocast grad_clip={grad_clip} steps={steps}", flush=True)
    for step in range(steps):
        with torch.amp.autocast("cuda", dtype=torch.float16):
            pred = model(cap)
            loss, parts = emission_loss(pred, em)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(opt)
        scaler.update()
        losses.append(parts["total"])
        if step % 10 == 0 or step == steps - 1:
            with torch.no_grad():
                p = pred.float()
                cur_psnr = psnr(p, em)
            psnr_history.append((step, cur_psnr))
            print(f"[overfit step {step}] loss={parts['total']:.5f} "
                  f"charb={parts['charb']:.5f} ms_l1={parts['ms_l1']:.5f} "
                  f"PSNR={cur_psnr:.2f}dB", flush=True)

    # Final
    with torch.no_grad():
        pred = model(cap).float()
        final_psnr = psnr(pred, em)
        peak_psnr = max([p for _, p in psnr_history] + [final_psnr])
    pass_ = peak_psnr >= pass_threshold
    fail_ = peak_psnr < fail_threshold
    print(f"[overfit] FINAL: peak_psnr={peak_psnr:.2f}dB final_psnr={final_psnr:.2f}dB "
          f"PASS={pass_} FAIL_HARD={fail_}  "
          f"({time.time() - t0:.1f}s for {steps} steps)")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"smoke_overfit_{steps}.json").write_text(json.dumps({
        "psnr_history": psnr_history,
        "losses": losses,
        "peak_psnr": peak_psnr,
        "final_psnr": final_psnr,
        "lr": lr,
        "steps": steps,
        "batch_size": int(cap.shape[0]),
        "pass_threshold": pass_threshold,
        "fail_threshold": fail_threshold,
        "passed": bool(pass_),
        "failed_hard": bool(fail_),
    }, indent=2))
    return pass_, fail_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--smoke", choices=["forward", "overfit", ""], default="")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--overfit-steps", type=int, default=100)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out = args.out or Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    device = torch.device("cuda")
    torch.manual_seed(cfg.get("seed", 42))

    bs = cfg["train"]["batch_size"]
    nw = cfg["train"]["num_workers"]

    train_ds = build_train_dataset(cfg)
    print(f"[init] train dataset: {len(train_ds)} rows  "
          f"capture=({cfg['data']['capture_h']},{cfg['data']['capture_w']}) "
          f"emission=({cfg['data']['emission_h']},{cfg['data']['emission_w']})", flush=True)

    if args.smoke:
        loader = DataLoader(
            train_ds, batch_size=bs, shuffle=True,
            num_workers=min(nw, 2), collate_fn=collate, pin_memory=True,
        )
        print(f"[smoke] loading first batch (bs={bs})...", flush=True)
        t0 = time.time()
        batch = next(iter(loader))
        print(f"[smoke] first batch loaded in {time.time() - t0:.1f}s", flush=True)
        model = _build_model_emi(cfg).to(device)
        print(f"[smoke] model class: {type(model).__name__}", flush=True)
        if args.smoke == "forward":
            run_smoke_forward(model, batch, device)
        else:
            passed, failed = run_smoke_overfit(
                model, batch, device, cfg["train"]["lr"],
                steps=args.overfit_steps, out_dir=out,
            )
            sys.exit(0 if passed else (2 if failed else 1))
        return

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=nw,
        collate_fn=collate, pin_memory=True, drop_last=True,
        persistent_workers=(nw > 0),
    )
    val_ds = build_val_dataset(cfg)
    print(f"[init] val dataset: {len(val_ds)} rows", flush=True)
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=nw,
        collate_fn=collate, pin_memory=True,
        persistent_workers=(nw > 0) if len(val_ds) > 0 else False,
    )
    train_eval_size = cfg["train"].get("train_eval_size", 256)
    g = torch.Generator().manual_seed(cfg.get("seed", 42))
    perm = torch.randperm(len(train_ds), generator=g)[:train_eval_size].tolist()
    train_eval_ds = torch.utils.data.Subset(train_ds, perm)
    train_eval_loader = DataLoader(
        train_eval_ds, batch_size=bs, shuffle=False, num_workers=nw,
        collate_fn=collate, pin_memory=True,
        persistent_workers=(nw > 0),
    )
    print(f"[init] train-eval subset: {len(train_eval_ds)} rows", flush=True)

    model = _build_model_emi(cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda")
    epochs = cfg["train"]["epochs"]
    total_steps = epochs * len(train_loader)
    warmup = cfg["train"]["warmup_steps"]
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [
            torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=1e-3, total_iters=max(warmup, 1)),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(total_steps - warmup, 1)),
        ],
        milestones=[warmup],
    )

    writer = SummaryWriter(out / "tb")
    step = 0
    log_every = cfg["train"].get("log_every", 10)
    grad_clip = cfg["train"].get("grad_clip", 1.0)

    for ep in range(epochs):
        model.train()
        ep_t0 = time.time()
        for batch in train_loader:
            if args.max_steps is not None and step >= args.max_steps:
                break
            cap = batch["capture"].to(device, non_blocking=True)
            em = batch["emission"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                pred = model(cap)
                loss, parts = emission_loss(pred, em)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()

            if step % log_every == 0:
                writer.add_scalar("train/total", parts["total"], step)
                writer.add_scalar("train/charb", parts["charb"], step)
                writer.add_scalar("train/ms_l1", parts["ms_l1"], step)
                writer.add_scalar("train/lr", opt.param_groups[0]["lr"], step)
                print(f"[ep {ep} step {step}] total={parts['total']:.5f} "
                      f"charb={parts['charb']:.5f} ms={parts['ms_l1']:.5f} "
                      f"lr={opt.param_groups[0]['lr']:.2e}", flush=True)
            step += 1

        ep_secs = time.time() - ep_t0
        print(f"[ep {ep}] train epoch in {ep_secs:.0f}s "
              f"(steps={len(train_loader)})", flush=True)

        te_metrics = evaluate(model, train_eval_loader, device)
        for k, v in te_metrics.items():
            if isinstance(v, (int, float)):
                writer.add_scalar(f"train_eval/{k}", v, ep)
        print(f"[ep {ep}] train_eval: psnr_full={te_metrics['psnr_full']:.2f} "
              f"psnr_half={te_metrics['psnr_half']:.2f} "
              f"psnr_quarter={te_metrics['psnr_quarter']:.2f} "
              f"l1_all={te_metrics['l1_all']:.4f} n={te_metrics['n']}", flush=True)

        if len(val_ds) > 0:
            val_metrics = evaluate(model, val_loader, device)
            for k, v in val_metrics.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f"val/{k}", v, ep)
            print(f"[ep {ep}] val: psnr_full={val_metrics['psnr_full']:.2f} "
                  f"psnr_half={val_metrics['psnr_half']:.2f} "
                  f"psnr_quarter={val_metrics['psnr_quarter']:.2f} "
                  f"l1_all={val_metrics['l1_all']:.4f} n={val_metrics['n']}", flush=True)

        ckpt = out / "checkpoints" / f"ep{ep:03d}.pt"
        ckpt.parent.mkdir(exist_ok=True, parents=True)
        torch.save({"model": model.state_dict(), "ep": ep, "step": step,
                    "config": cfg}, ckpt)
        if args.max_steps is not None and step >= args.max_steps:
            break

    writer.close()


if __name__ == "__main__":
    main()

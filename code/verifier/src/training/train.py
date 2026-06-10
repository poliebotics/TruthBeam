"""exp001a training: Siamese ConvNeXt-Tiny + multi-octave XOF branch.

Modes:
  --smoke=forward  one batch, forward + backward sanity, finiteness checks
  --smoke=overfit  one batch, 100 steps, loss-drop check
  (default)        full training with periodic eval + checkpoints
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

from data.raw_bayer_dataset import SessionDataset  # noqa: E402
from eval.metrics import margin_stats, topk_retrieval  # noqa: E402
from losses.info_nce import info_nce_loss, vicreg_covariance, vicreg_variance  # noqa: E402
from models.siamese_xof import SiameseXOF  # noqa: E402


def combined_loss(
    z_cap, z_xof, h_cap, h_xof, tau, lambda_var, lambda_cov,
):
    nce = info_nce_loss(z_cap, z_xof, tau)
    var = vicreg_variance(h_cap) + vicreg_variance(h_xof)
    cov = vicreg_covariance(h_cap) + vicreg_covariance(h_xof)
    total = nce + lambda_var * var + lambda_cov * cov
    return total, {"nce": nce.item(), "var": var.item(), "cov": cov.item(),
                   "total": total.item()}


def collate(batch):
    capture = torch.stack([b["capture"] for b in batch])
    octaves = [torch.stack([b[f"xof_oct{i}"] for b in batch]) for i in range(4)]
    t = torch.tensor([b["t"] for b in batch])
    return {
        "capture": capture,
        "octaves": octaves,
        "t": t,
        "session_id": [b["session_id"] for b in batch],
    }


def build_train_dataset(cfg) -> SessionDataset:
    return SessionDataset(
        session_dir=Path(cfg["data"]["d2_dir"]),
        row_start=cfg["data"]["d2_train_start"],
        row_end=cfg["data"]["d2_train_end"],
        session_id="d2",
    )


def build_val_dataset(cfg) -> SessionDataset:
    return SessionDataset(
        session_dir=Path(cfg["data"]["d2_dir"]),
        row_start=cfg["data"]["d2_val_start"],
        row_end=cfg["data"]["d2_val_end"],
        session_id="d2",
    )


def evaluate(model, loader, device, tau):
    model.eval()
    all_zc, all_zx = [], []
    with torch.no_grad():
        for batch in loader:
            cap = batch["capture"].to(device, non_blocking=True)
            octs = [o.to(device, non_blocking=True) for o in batch["octaves"]]
            with torch.amp.autocast("cuda", dtype=torch.float16):
                zc, zx = model(cap, octs)
            all_zc.append(zc.float().cpu())
            all_zx.append(zx.float().cpu())
    if not all_zc:
        return {"top1": 0.0, "n": 0}
    zc = torch.cat(all_zc)
    zx = torch.cat(all_zx)
    sim = zc @ zx.T
    out = {"top1": topk_retrieval(sim, k=1), "n": sim.shape[0]}
    out.update(margin_stats(sim))
    return out


def run_smoke_forward(model, batch, device, tau):
    cap = batch["capture"].to(device)
    octs = [o.to(device) for o in batch["octaves"]]
    print(f"[smoke fwd] capture: {cap.shape} dtype={cap.dtype} "
          f"min={cap.min():.3f} max={cap.max():.3f}")
    for i, o in enumerate(octs):
        print(f"[smoke fwd] octave{i}: {o.shape} dtype={o.dtype}")

    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        z_cap, z_xof = model(cap, octs)
    print(f"[smoke fwd] z_cap: {z_cap.shape} dtype={z_cap.dtype} "
          f"finite={torch.isfinite(z_cap).all().item()}")
    print(f"[smoke fwd] z_xof: {z_xof.shape} dtype={z_xof.dtype} "
          f"finite={torch.isfinite(z_xof).all().item()}")
    print(f"[smoke fwd] z_cap norm: {z_cap.norm(dim=-1).mean():.4f} "
          f"(should be ~1.0)")
    print(f"[smoke fwd] z_xof norm: {z_xof.norm(dim=-1).mean():.4f}")

    loss = info_nce_loss(z_cap.float(), z_xof.float(), tau)
    print(f"[smoke fwd] initial loss (cold model): {loss.item():.4f} "
          f"(random ≈ {2 * (z_cap.shape[0] - 1).bit_length():.2f}; "
          f"ln(N)={torch.tensor(float(z_cap.shape[0])).log().item():.3f})")
    assert torch.isfinite(loss), "loss not finite"

    model.train()
    z_cap, z_xof = model(cap, octs)
    loss = info_nce_loss(z_cap, z_xof, tau)
    loss.backward()
    n_total = sum(1 for _ in model.parameters())
    n_with_grad = sum(p.grad is not None for p in model.parameters())
    n_finite = sum(
        (p.grad is not None and torch.isfinite(p.grad).all().item())
        for p in model.parameters()
    )
    print(f"[smoke fwd] backward: {n_with_grad}/{n_total} params got grads, "
          f"{n_finite}/{n_total} finite")
    return loss.item()


def run_smoke_overfit(model, batch, device, tau, lr, steps, out_dir,
                      lambda_var=1.0, lambda_cov=0.04):
    cap = batch["capture"].to(device)
    octs = [o.to(device) for o in batch["octaves"]]
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda")
    losses = []
    nces = []
    t0 = time.time()
    model.train()
    for step in range(steps):
        with torch.amp.autocast("cuda", dtype=torch.float16):
            z_cap, z_xof, h_cap, h_xof = model.forward_with_features(cap, octs)
            loss, parts = combined_loss(
                z_cap, z_xof, h_cap, h_xof, tau, lambda_var, lambda_cov,
            )
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(parts["total"])
        nces.append(parts["nce"])
        if step % 10 == 0 or step == steps - 1:
            print(f"[overfit step {step}] total={parts['total']:.4f} "
                  f"nce={parts['nce']:.4f} var={parts['var']:.4f} "
                  f"cov={parts['cov']:.4f}", flush=True)
    elapsed = time.time() - t0
    print(f"[overfit] nce: start={nces[0]:.4f} end={nces[-1]:.4f}  "
          f"({elapsed:.1f}s for {steps} steps, "
          f"{elapsed / steps * 1000:.1f}ms/step)")

    out_path = out_dir / "smoke_overfit_losses.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "total_losses": losses,
        "nce_losses": nces,
        "tau": tau,
        "lr": lr,
        "lambda_var": lambda_var,
        "lambda_cov": lambda_cov,
        "steps": steps,
        "batch_size": int(batch["capture"].shape[0]),
        "elapsed_sec": elapsed,
    }, indent=2))
    return losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--smoke", choices=["forward", "overfit", ""], default="")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="cap total training steps (debugging)")
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
    tau = cfg["train"]["temperature"]

    train_ds = build_train_dataset(cfg)
    print(f"[init] train dataset: {len(train_ds)} rows", flush=True)

    lambda_var = cfg["train"].get("vicreg_var_weight", 1.0)
    lambda_cov = cfg["train"].get("vicreg_cov_weight", 0.04)

    if args.smoke:
        # Lighter loader for smoke; only need one batch.
        loader = DataLoader(
            train_ds, batch_size=bs, shuffle=True,
            num_workers=min(nw, 2), collate_fn=collate, pin_memory=True,
        )
        print(f"[smoke] loading first batch (bs={bs})...", flush=True)
        t0 = time.time()
        batch = next(iter(loader))
        print(f"[smoke] first batch loaded in {time.time() - t0:.1f}s", flush=True)

        model = SiameseXOF().to(device)
        if args.smoke == "forward":
            run_smoke_forward(model, batch, device, tau)
        else:
            run_smoke_overfit(
                model, batch, device, tau, cfg["train"]["lr"], steps=100, out_dir=out,
                lambda_var=lambda_var, lambda_cov=lambda_cov,
            )
        return

    # Full training
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

    model = SiameseXOF().to(device)
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
                opt, start_factor=1e-3, total_iters=max(warmup, 1),
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(total_steps - warmup, 1),
            ),
        ],
        milestones=[warmup],
    )

    writer = SummaryWriter(out / "tb")
    step = 0
    log_every = cfg["train"].get("log_every", 10)

    for ep in range(epochs):
        model.train()
        ep_t0 = time.time()
        for batch in train_loader:
            if args.max_steps is not None and step >= args.max_steps:
                break
            cap = batch["capture"].to(device, non_blocking=True)
            octs = [o.to(device, non_blocking=True) for o in batch["octaves"]]
            with torch.amp.autocast("cuda", dtype=torch.float16):
                z_cap, z_xof, h_cap, h_xof = model.forward_with_features(cap, octs)
                loss, parts = combined_loss(
                    z_cap, z_xof, h_cap, h_xof, tau, lambda_var, lambda_cov,
                )
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

            if step % log_every == 0:
                writer.add_scalar("train/total", parts["total"], step)
                writer.add_scalar("train/nce", parts["nce"], step)
                writer.add_scalar("train/var", parts["var"], step)
                writer.add_scalar("train/cov", parts["cov"], step)
                writer.add_scalar("train/lr", opt.param_groups[0]["lr"], step)
                print(f"[ep {ep} step {step}] total={parts['total']:.4f} "
                      f"nce={parts['nce']:.4f} var={parts['var']:.4f} "
                      f"cov={parts['cov']:.4f} "
                      f"lr={opt.param_groups[0]['lr']:.2e}", flush=True)
            step += 1

        ep_secs = time.time() - ep_t0
        print(f"[ep {ep}] train epoch in {ep_secs:.0f}s "
              f"(steps={len(train_loader)})", flush=True)

        if len(val_ds) > 0:
            val_metrics = evaluate(model, val_loader, device, tau)
            for k, v in val_metrics.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f"val/{k}", v, ep)
            print(f"[ep {ep}] val: {val_metrics}", flush=True)

        ckpt = out / "checkpoints" / f"ep{ep:03d}.pt"
        ckpt.parent.mkdir(exist_ok=True, parents=True)
        torch.save({"model": model.state_dict(), "ep": ep, "step": step,
                    "config": cfg}, ckpt)
        if args.max_steps is not None and step >= args.max_steps:
            break

    writer.close()


if __name__ == "__main__":
    main()

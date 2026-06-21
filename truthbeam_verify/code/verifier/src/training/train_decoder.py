"""exp001b training: direct XOF byte prediction (encoder + 4 MLP heads).

Modes:
  --smoke=forward  forward + backward sanity, finiteness checks
  --smoke=overfit  one batch, 100 steps, bit-recovery PASS criterion
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

from data.native_bayer_dataset import NativeBayerDataset  # noqa: E402
from data.raw_bayer_dataset import SessionDataset  # noqa: E402
from eval.xof_metrics import floats_to_bytes, per_octave_metrics  # noqa: E402
from losses.xof_l2 import xof_l2_loss  # noqa: E402
from models.xof_decoder import XOFDecoder  # noqa: E402
from models.xof_decoder_large import XOFDecoderLarge  # noqa: E402

_MODEL_CLASSES = {"XOFDecoder": XOFDecoder, "XOFDecoderLarge": XOFDecoderLarge}


def _build_dataset(cfg, row_start: int, row_end: int):
    name = cfg["data"].get("dataset", "SessionDataset")
    if name == "NativeBayerDataset":
        return NativeBayerDataset(
            session_dir=Path(cfg["data"]["d2_dir"]),
            row_start=row_start,
            row_end=row_end,
            targets=tuple(cfg["data"].get("targets", ("xof",))),
            session_id="d2",
        )
    if name == "SessionDataset":
        kw = {}
        if "capture_h" in cfg["data"]:
            kw["capture_h"] = cfg["data"]["capture_h"]
            kw["capture_w"] = cfg["data"]["capture_w"]
        return SessionDataset(
            session_dir=Path(cfg["data"]["d2_dir"]),
            row_start=row_start,
            row_end=row_end,
            session_id="d2",
            **kw,
        )
    raise ValueError(f"unknown dataset class: {name}")


def _build_model(cfg) -> torch.nn.Module:
    name = cfg.get("model", {}).get("class", "XOFDecoder")
    cls = _MODEL_CLASSES.get(name)
    if cls is None:
        raise ValueError(f"unknown model class: {name}")
    return cls()


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


def build_train_dataset(cfg):
    return _build_dataset(cfg, cfg["data"]["d2_train_start"], cfg["data"]["d2_train_end"])


def build_val_dataset(cfg):
    return _build_dataset(cfg, cfg["data"]["d2_val_start"], cfg["data"]["d2_val_end"])


@torch.no_grad()
def evaluate(model, loader, device):
    """Run model over loader, accumulate per-octave bit/byte/rms metrics."""
    model.eval()
    pred_concat = [[] for _ in range(4)]
    true_concat = [[] for _ in range(4)]
    n = 0
    for batch in loader:
        cap = batch["capture"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            preds = model(cap)
        for i in range(4):
            pred_concat[i].append(floats_to_bytes(preds[i].float()).cpu())
            true_concat[i].append(floats_to_bytes(batch["octaves"][i]).cpu())
        n += cap.shape[0]
    if n == 0:
        return {"n": 0}
    preds_b = [torch.cat(p) for p in pred_concat]
    trues_b = [torch.cat(t) for t in true_concat]
    metrics = per_octave_metrics(preds_b, trues_b)
    metrics["n"] = n
    return metrics


def run_smoke_forward(model, batch, device):
    cap = batch["capture"].to(device)
    octs = [o.to(device) for o in batch["octaves"]]
    print(f"[smoke fwd] capture: {cap.shape} dtype={cap.dtype} "
          f"min={cap.min():.3f} max={cap.max():.3f}")
    for i, o in enumerate(octs):
        print(f"[smoke fwd] target oct{i}: {o.shape} dtype={o.dtype} "
              f"min={o.min():.3f} max={o.max():.3f}")

    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        preds = model(cap)
    for i, p in enumerate(preds):
        print(f"[smoke fwd] pred oct{i}: {p.shape} dtype={p.dtype} "
              f"min={p.min():.4f} max={p.max():.4f} "
              f"finite={torch.isfinite(p).all().item()}")
        assert (p >= 0).all() and (p <= 1).all(), f"sigmoid out of [0,1] in oct{i}"

    loss, parts = xof_l2_loss(
        [p.float() for p in preds],
        [o.float() for o in octs],
    )
    print(f"[smoke fwd] cold-model loss parts: {parts}")
    assert torch.isfinite(loss), "loss not finite"

    # Initial bit recovery — should be ~0.5 with sigmoid + uniform-byte targets
    metrics = per_octave_metrics(
        [floats_to_bytes(p.float()) for p in preds],
        [floats_to_bytes(o) for o in octs],
    )
    print(f"[smoke fwd] cold-model recovery (target ~0.5 random): "
          f"oct0_bit={metrics['oct0_bit']:.4f} oct1_bit={metrics['oct1_bit']:.4f} "
          f"oct2_bit={metrics['oct2_bit']:.4f} oct3_bit={metrics['oct3_bit']:.4f}")

    model.train()
    preds = model(cap)
    loss, _ = xof_l2_loss(preds, octs)
    loss.backward()
    n_total = sum(1 for _ in model.parameters())
    n_with_grad = sum(p.grad is not None for p in model.parameters())
    n_finite = sum(
        (p.grad is not None and torch.isfinite(p.grad).all().item())
        for p in model.parameters()
    )
    print(f"[smoke fwd] backward: {n_with_grad}/{n_total} params got grads, "
          f"{n_finite}/{n_total} finite")
    return parts


def run_smoke_overfit(model, batch, device, lr, steps, out_dir, pass_threshold=0.95,
                      out_name=None, fp32: bool = False):
    cap = batch["capture"].to(device)
    octs = [o.to(device) for o in batch["octaves"]]
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda") if not fp32 else None

    losses = []
    bit_history = {f"oct{i}": [] for i in range(4)}
    t0 = time.time()
    model.train()
    grad_clip = 1.0
    print(f"[overfit] mode={'fp32' if fp32 else 'fp16/autocast'} "
          f"grad_clip={grad_clip}", flush=True)
    for step in range(steps):
        if fp32:
            preds = model(cap)
            loss, parts = xof_l2_loss(preds, octs)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            opt.step()
        else:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                preds = model(cap)
                loss, parts = xof_l2_loss(preds, octs)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(opt)
            scaler.update()
        losses.append(parts["total"])

        if step % 10 == 0 or step == steps - 1:
            with torch.no_grad():
                metrics = per_octave_metrics(
                    [floats_to_bytes(p.float()) for p in preds],
                    [floats_to_bytes(o) for o in octs],
                )
            for i in range(4):
                bit_history[f"oct{i}"].append(metrics[f"oct{i}_bit"])
            print(f"[overfit step {step}] loss={parts['total']:.5f} "
                  f"oct0_bit={metrics['oct0_bit']:.4f} "
                  f"oct1_bit={metrics['oct1_bit']:.4f} "
                  f"oct2_bit={metrics['oct2_bit']:.4f} "
                  f"oct3_bit={metrics['oct3_bit']:.4f}", flush=True)

    elapsed = time.time() - t0
    # Final metrics
    with torch.no_grad():
        preds = model(cap)
        final_metrics = per_octave_metrics(
            [floats_to_bytes(p.float()) for p in preds],
            [floats_to_bytes(o) for o in octs],
        )
    final_min_bit = min(final_metrics[f"oct{i}_bit"] for i in range(4))
    # Track best (peak) min across history to also report a peak-pass.
    peak_min_bit = 0.0
    for i in range(len(bit_history["oct0"])):
        per_oct = [bit_history[f"oct{j}"][i] for j in range(4)]
        peak_min_bit = max(peak_min_bit, min(per_oct))
    pass_ = (final_min_bit >= pass_threshold) or (peak_min_bit >= pass_threshold)
    print(f"[overfit] FINAL: oct0_bit={final_metrics['oct0_bit']:.4f} "
          f"oct1_bit={final_metrics['oct1_bit']:.4f} "
          f"oct2_bit={final_metrics['oct2_bit']:.4f} "
          f"oct3_bit={final_metrics['oct3_bit']:.4f}  "
          f"final_min={final_min_bit:.4f}  peak_min={peak_min_bit:.4f}  "
          f"PASS={pass_}  ({elapsed:.1f}s for {steps} steps)")

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_name or f"smoke_overfit_{steps}.json"
    (out_dir / fname).write_text(json.dumps({
        "losses": losses,
        "bit_history": bit_history,
        "final_metrics": final_metrics,
        "final_min_bit": final_min_bit,
        "peak_min_bit": peak_min_bit,
        "lr": lr,
        "steps": steps,
        "batch_size": int(cap.shape[0]),
        "elapsed_sec": elapsed,
        "pass_threshold": pass_threshold,
        "passed": bool(pass_),
        "fp32": bool(fp32),
    }, indent=2))
    return pass_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--smoke", choices=["forward", "overfit", ""], default="")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--overfit-steps", type=int, default=100,
                    help="Steps for --smoke=overfit")
    ap.add_argument("--fp32", action="store_true",
                    help="Disable autocast/GradScaler for the overfit smoke")
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
    print(f"[init] train dataset: {len(train_ds)} rows", flush=True)

    if args.smoke:
        loader = DataLoader(
            train_ds, batch_size=bs, shuffle=True,
            num_workers=min(nw, 2), collate_fn=collate, pin_memory=True,
        )
        print(f"[smoke] loading first batch (bs={bs})...", flush=True)
        t0 = time.time()
        batch = next(iter(loader))
        print(f"[smoke] first batch loaded in {time.time() - t0:.1f}s", flush=True)

        model = _build_model(cfg).to(device)
        print(f"[smoke] model class: {type(model).__name__}", flush=True)
        if args.smoke == "forward":
            run_smoke_forward(model, batch, device)
        else:
            out_name = None
            if args.fp32:
                out_name = f"smoke_overfit_{args.overfit_steps}_fp32.json"
            passed = run_smoke_overfit(
                model, batch, device, cfg["train"]["lr"],
                steps=args.overfit_steps, out_dir=out, fp32=args.fp32,
                out_name=out_name,
            )
            sys.exit(0 if passed else 1)
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

    # Fixed train-eval subset for the train↔val gap diagnostic.
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

    model = _build_model(cfg).to(device)
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
    train_metric_every = cfg["train"].get("train_metric_every", 200)
    grad_clip = cfg["train"].get("grad_clip", 1.0)

    for ep in range(epochs):
        model.train()
        ep_t0 = time.time()
        for batch in train_loader:
            if args.max_steps is not None and step >= args.max_steps:
                break
            cap = batch["capture"].to(device, non_blocking=True)
            octs = [o.to(device, non_blocking=True) for o in batch["octaves"]]
            with torch.amp.autocast("cuda", dtype=torch.float16):
                preds = model(cap)
                loss, parts = xof_l2_loss(preds, octs)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()

            if step % log_every == 0:
                writer.add_scalar("train/total", parts["total"], step)
                for i in range(4):
                    writer.add_scalar(f"train/l2_oct{i}", parts[f"l2_oct{i}"], step)
                writer.add_scalar("train/lr", opt.param_groups[0]["lr"], step)
                print(f"[ep {ep} step {step}] total={parts['total']:.5f} "
                      f"o0={parts['l2_oct0']:.5f} o1={parts['l2_oct1']:.5f} "
                      f"o2={parts['l2_oct2']:.5f} o3={parts['l2_oct3']:.5f} "
                      f"lr={opt.param_groups[0]['lr']:.2e}", flush=True)

            if step % train_metric_every == 0 and step > 0:
                with torch.no_grad():
                    train_metrics = per_octave_metrics(
                        [floats_to_bytes(p.float()) for p in preds],
                        [floats_to_bytes(o) for o in octs],
                    )
                for k, v in train_metrics.items():
                    writer.add_scalar(f"train_metric/{k}", v, step)
                print(f"[ep {ep} step {step}] train metric: "
                      f"o0_bit={train_metrics['oct0_bit']:.4f} "
                      f"o1_bit={train_metrics['oct1_bit']:.4f} "
                      f"o2_bit={train_metrics['oct2_bit']:.4f} "
                      f"o3_bit={train_metrics['oct3_bit']:.4f} "
                      f"all_bit={train_metrics['all_bit']:.4f}", flush=True)
            step += 1

        ep_secs = time.time() - ep_t0
        print(f"[ep {ep}] train epoch in {ep_secs:.0f}s "
              f"(steps={len(train_loader)})", flush=True)

        # train-eval (fixed subset) — primary diagnostic vs. val
        te_metrics = evaluate(model, train_eval_loader, device)
        for k, v in te_metrics.items():
            if isinstance(v, (int, float)):
                writer.add_scalar(f"train_eval/{k}", v, ep)
        print(f"[ep {ep}] train_eval: o0_bit={te_metrics['oct0_bit']:.4f} "
              f"o1_bit={te_metrics['oct1_bit']:.4f} "
              f"o2_bit={te_metrics['oct2_bit']:.4f} "
              f"o3_bit={te_metrics['oct3_bit']:.4f} "
              f"all_bit={te_metrics['all_bit']:.4f} "
              f"all_byte={te_metrics['all_byte']:.5f} "
              f"all_rms={te_metrics['all_rms']:.2f} n={te_metrics['n']}",
              flush=True)

        if len(val_ds) > 0:
            val_metrics = evaluate(model, val_loader, device)
            for k, v in val_metrics.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f"val/{k}", v, ep)
            print(f"[ep {ep}] val: o0_bit={val_metrics['oct0_bit']:.4f} "
                  f"o1_bit={val_metrics['oct1_bit']:.4f} "
                  f"o2_bit={val_metrics['oct2_bit']:.4f} "
                  f"o3_bit={val_metrics['oct3_bit']:.4f} "
                  f"all_bit={val_metrics['all_bit']:.4f} "
                  f"all_byte={val_metrics['all_byte']:.5f} "
                  f"all_rms={val_metrics['all_rms']:.2f} n={val_metrics['n']}",
                  flush=True)

        ckpt = out / "checkpoints" / f"ep{ep:03d}.pt"
        ckpt.parent.mkdir(exist_ok=True, parents=True)
        torch.save({"model": model.state_dict(), "ep": ep, "step": step,
                    "config": cfg}, ckpt)
        if args.max_steps is not None and step >= args.max_steps:
            break

    writer.close()


if __name__ == "__main__":
    main()

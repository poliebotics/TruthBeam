"""Phase F prep — diffusion verifier scaffolding (trainable, not yet trained).

Trains a CondDDPMUNet binder: capture (4, capH, capW) → emission (3, em_h, em_w)
via DDPM denoising. The binder learns the conditional distribution
p(emission | capture) and at inference time can be queried for either:

  (a) likelihood of an observed emission given a capture (probabilistic
      verifier score)
  (b) posterior mean / mode emission for a given capture (forward-pass
      reconstruction, comparable to the U-Net binders)

This is the SCAFFOLDING — argparse + dataloader + model + train loop +
single-step probe + checkpoint save. Defaults run for `--max-steps 50`
on small CFA so the operator can confirm the wiring is sound before
committing the full ~4 hr 5000-step training budget.

The model architecture comes from `diffusion_verifier_sizing.py`. We
re-import to avoid duplication.

Run for sanity-check (50 steps, ~30s):
  CUDA_VISIBLE_DEVICES=1 python scripts/phase_f/train_diffusion_binder.py \
    --max-steps 50 --bs 2 --bf16 \
    --d2-dir /path/to/poliebotics_phase_b/data/d2 \
    --out /path/to/poliebotics_phase_b/experiments/phase_f_prep/diffusion_binder_smoke

Run for full latent-style training (5000 steps, ~1.6 hr):
  CUDA_VISIBLE_DEVICES=1 python scripts/phase_f/train_diffusion_binder.py \
    --max-steps 5000 --bs 4 --bf16 --em-downsample 2 \
    --d2-dir /path/to/poliebotics_phase_b/data/d2 \
    --out /path/to/poliebotics_phase_b/experiments/phase_f_prep/diffusion_binder_v0
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

torch.set_num_threads(8)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "phase_f"))

from data.emission_dataset import EmissionDataset  # noqa: E402
from diffusion_verifier_sizing import CondDDPMUNet  # noqa: E402


# ---------- DDPM utilities ----------

def linear_beta_schedule(T: int, beta_start: float = 1e-4,
                         beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T)


def build_diffusion_constants(T: int, device: torch.device, dtype: torch.dtype) -> dict:
    betas = linear_beta_schedule(T).to(device=device, dtype=dtype)
    alphas = 1.0 - betas
    alphas_cum = torch.cumprod(alphas, dim=0)
    return {
        "T": T,
        "betas": betas,
        "alphas": alphas,
        "alphas_cum": alphas_cum,
        "sqrt_alphas_cum": torch.sqrt(alphas_cum),
        "sqrt_one_minus_alphas_cum": torch.sqrt(1.0 - alphas_cum),
    }


def q_sample(em0: torch.Tensor, t: torch.Tensor, dc: dict,
             noise: torch.Tensor) -> torch.Tensor:
    """Forward diffusion: em_t = sqrt(alphas_cum_t)·em0 + sqrt(1 - alphas_cum_t)·noise."""
    sa = dc["sqrt_alphas_cum"][t].view(-1, 1, 1, 1)
    so = dc["sqrt_one_minus_alphas_cum"][t].view(-1, 1, 1, 1)
    return sa * em0 + so * noise


# ---------- training ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d2-dir", type=Path, required=True)
    ap.add_argument("--v10-dir", type=Path, default=None,
                    help="V10 session dir (only used if --sessions includes V10).")
    ap.add_argument("--sessions", nargs="+", default=["D2"],
                    choices=["D2", "V10"],
                    help="Sessions to train on. Use {D2}, {V10}, or both.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--em-downsample", type=int, default=2,
                    help="Run DDPM at em_h/d × em_w/d. d=1 native, d=2 latent-style 540x960.")
    ap.add_argument("--capture-h", type=int, default=1150)
    ap.add_argument("--capture-w", type=int, default=1330)
    ap.add_argument("--em-h", type=int, default=1080)
    ap.add_argument("--em-w", type=int, default=1920)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--base-ch", type=int, default=64)
    ap.add_argument("--mults", nargs="+", type=int, default=[1, 2, 4, 4, 8])
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--probe-every", type=int, default=200)
    ap.add_argument("--ckpt-every", type=int, default=1500,
                    help="Save intermediate checkpoint every N steps. "
                         "Prevents catastrophic loss on early termination "
                         "(per 2026-04-29 H2/H3 incident).")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    em_h_eff = args.em_h // args.em_downsample
    em_w_eff = args.em_w // args.em_downsample

    print(f"[init] CondDDPMUNet base_ch={args.base_ch} mults={tuple(args.mults)} "
          f"em_eff={em_h_eff}×{em_w_eff} dtype={dtype}", flush=True)
    model = CondDDPMUNet(
        base_ch=args.base_ch,
        channel_mults=tuple(args.mults),
        attn_at=tuple(i == len(args.mults) - 1 for i in range(len(args.mults))),
    ).to(device, dtype=dtype)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[init] params={n_params/1e6:.1f}M", flush=True)

    dc = build_diffusion_constants(args.T, device, dtype)

    # train_normal slices per `experiments/phase_f_prep/three_role_split.md`.
    # selection_gate_normal + threshold_calibration_normal are OFF-LIMITS here.
    SESSION_RANGES = {"D2": (0, 4592), "V10": (0, 2500)}
    SESSION_DIRS = {"D2": args.d2_dir, "V10": args.v10_dir}
    sub_datasets = []
    for sess in args.sessions:
        if SESSION_DIRS[sess] is None:
            raise SystemExit(f"--sessions includes {sess} but no --{sess.lower()}-dir provided")
        rs, re = SESSION_RANGES[sess]
        ds = EmissionDataset(
            session_dir=SESSION_DIRS[sess], row_start=rs, row_end=re,
            capture_h=args.capture_h, capture_w=args.capture_w,
            emission_h=args.em_h, emission_w=args.em_w,
            session_id=sess, augment=False,
        )
        print(f"[init] session={sess} train n={len(ds)} rows=[{rs},{re})", flush=True)
        sub_datasets.append(ds)
    if len(sub_datasets) == 1:
        train_ds = sub_datasets[0]
    else:
        from torch.utils.data import ConcatDataset
        train_ds = ConcatDataset(sub_datasets)
        print(f"[init] concat total n={len(train_ds)}", flush=True)
    loader = DataLoader(
        train_ds, batch_size=args.bs, shuffle=True, num_workers=2,
        pin_memory=True, drop_last=True, persistent_workers=True,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    history = []
    t0 = time.time()
    train_iter = iter(loader)
    for step in range(args.max_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(loader)
            batch = next(train_iter)
        cap = batch["capture"].to(device, dtype=dtype)  # (B, 4, capH, capW)
        em0 = batch["emission"].to(device, dtype=dtype)  # (B, 3, em_h, em_w)
        # Optionally downsample emission to latent res
        if args.em_downsample > 1:
            em0 = F.interpolate(em0, size=(em_h_eff, em_w_eff),
                                mode="bilinear", align_corners=False)
        B = cap.shape[0]
        t = torch.randint(0, args.T, (B,), device=device).long()
        noise = torch.randn_like(em0)
        em_t = q_sample(em0, t, dc, noise)
        # Predict noise
        pred_noise = model(em_t, cap, t.float())
        loss = F.mse_loss(pred_noise, noise)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 10 == 0:
            elapsed = time.time() - t0
            history.append({"step": step, "loss": loss.item(), "t": round(elapsed, 1)})
            print(f"[step {step}] loss={loss.item():.4f}  t={elapsed:.0f}s", flush=True)

        if step > 0 and step % args.ckpt_every == 0:
            ckpt_path = args.out / f"ckpt_step{step:08d}.pt"
            torch.save({
                "model": model.state_dict(),
                "config": {"base_ch": args.base_ch, "mults": list(args.mults),
                           "em_h_eff": em_h_eff, "em_w_eff": em_w_eff,
                           "T": args.T, "em_downsample": args.em_downsample,
                           "capture_hw": [args.capture_h, args.capture_w]},
                "step": step,
            }, ckpt_path)
            print(f"  [ckpt] saved {ckpt_path}", flush=True)

        if step > 0 and step % args.probe_every == 0:
            # Quick probe: predict emission from a single capture via simple
            # 1-step posterior-mean estimate (NOT a real DDPM sample, just a
            # sanity check that conditioning is wired)
            model.eval()
            with torch.no_grad():
                cap_p = batch["capture"][:1].to(device, dtype=dtype)
                em_p0 = batch["emission"][:1].to(device, dtype=dtype)
                if args.em_downsample > 1:
                    em_p0 = F.interpolate(em_p0, size=(em_h_eff, em_w_eff),
                                          mode="bilinear", align_corners=False)
                t_one = torch.tensor([args.T // 2], device=device).long()
                noise_p = torch.randn_like(em_p0)
                em_t_p = q_sample(em_p0, t_one, dc, noise_p)
                pred_n = model(em_t_p, cap_p, t_one.float())
                # Predicted clean em0_hat = (em_t - sqrt(1-acum)*pred_noise) / sqrt(acum)
                sa = dc["sqrt_alphas_cum"][t_one].view(-1, 1, 1, 1)
                so = dc["sqrt_one_minus_alphas_cum"][t_one].view(-1, 1, 1, 1)
                em_pred = (em_t_p - so * pred_n) / sa
                pred_l1 = (em_pred.float() - em_p0.float()).abs().mean().item()
            print(f"  [probe] em_pred L1 vs em_target = {pred_l1:.4f}", flush=True)
            model.train()

    ckpt_path = args.out / "diffusion_binder.pt"
    torch.save({"model": model.state_dict(),
                "config": {"base_ch": args.base_ch, "mults": list(args.mults),
                           "em_h_eff": em_h_eff, "em_w_eff": em_w_eff,
                           "T": args.T, "em_downsample": args.em_downsample,
                           "capture_hw": [args.capture_h, args.capture_w]}},
               ckpt_path)
    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    summary = {
        "max_steps": args.max_steps,
        "elapsed_sec": round(time.time() - t0, 1),
        "params_m": round(n_params / 1e6, 2),
        "final_loss": history[-1]["loss"] if history else None,
        "config": {
            "capture_hw": [args.capture_h, args.capture_w],
            "em_hw_eff": [em_h_eff, em_w_eff],
            "em_downsample": args.em_downsample,
            "base_ch": args.base_ch,
            "mults": list(args.mults),
            "T": args.T,
            "lr": args.lr,
            "bs": args.bs,
            "dtype": "bf16" if args.bf16 else "fp32",
        },
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    final_loss_str = f"{summary['final_loss']:.4f}" if summary['final_loss'] is not None else "n/a"
    print(f"\n[done] elapsed={time.time()-t0:.0f}s, params={n_params/1e6:.1f}M, "
          f"final_loss={final_loss_str}", flush=True)


if __name__ == "__main__":
    main()

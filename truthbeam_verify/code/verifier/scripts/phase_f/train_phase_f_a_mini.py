"""Phase F-A mini-experiment: empirical test of pose-vs-emission curriculum.

Operator's hypothesis: emission is chain-derived random, so its 5% variance
share is signal that can't be predicted without E_target. Network may learn
pose first (high amplitude), then emission second (lower amplitude but
genuinely needs the conditioning input).

This script trains the editor on UNFILTERED temporal pairs for 3000 steps,
runs an inline causality probe every 250 steps, logs:
  - output diversity per source across 32 E_targets
  - L_binder margin (target preferred over source)
  - 4 visual grids per checkpoint to catch adversarial speckle

Pass: diversity grows monotonically AND outputs look like emission swaps.
Fail: diversity stays at 0 (pose-only) OR speckle.

Single GPU. ~2-3 hours.
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
import yaml
from torch.utils.data import DataLoader

torch.set_num_threads(8)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "phase_e"))

from data.emission_dataset import load_emission_at  # noqa: E402
from phase_f.dataset_temporal_pairs import (  # noqa: E402
    TemporalPairDataset, collate_temporal_pairs, split_rows,
)
from phase_f.editor_model import Editor  # noqa: E402
from phase_f.editor_controlnet import EditorControlNet  # noqa: E402
from phase_f.editor_losses import charbonnier, grad_loss  # noqa: E402
from models.emission_predictor import EmissionPredictor  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False


def psnr01(p: torch.Tensor, t: torch.Tensor) -> float:
    mse = ((p - t) ** 2).mean().item()
    return float("inf") if mse < 1e-12 else 20.0 * math.log10(1.0 / math.sqrt(mse))


# ---------- surrogate binders (NOT including E2 / E3r per audit #5) ----------

SURROGATE_BINDERS = [
    {
        "name": "exp001c",
        "ckpt": "/path/to/poliebotics_phase_b/experiments/exp001c/checkpoints/ep027.pt",
        "capture_h": 1150, "capture_w": 1330,
        "arch": "EmissionPredictor",
    },
    {
        "name": "E1-bf16",
        "ckpt": "/path/to/poliebotics_phase_b/experiments/phase_e/e1/checkpoints/best_by_psnr.pt",
        "capture_h": 1150, "capture_w": 1330,
        "arch": "EmissionPredictor",
    },
]


def load_binder(spec: dict, device: torch.device) -> torch.nn.Module:
    if spec["arch"] != "EmissionPredictor":
        raise NotImplementedError(spec["arch"])
    m = EmissionPredictor(emission_h=1080, emission_w=1920, pretrained=False).to(device)
    ck = torch.load(spec["ckpt"], map_location=device, weights_only=False)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    m.load_state_dict(state, strict=False)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def downsample_for_binder(cfa_pred: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Editor outputs at native (4, 2300, 2660); binders trained at smaller res need INTER_AREA-equivalent."""
    if cfa_pred.shape[-2:] == (target_h, target_w):
        return cfa_pred
    return F.interpolate(cfa_pred, size=(target_h, target_w), mode="area")


def binder_loss(
    binders: list[tuple[dict, torch.nn.Module]],
    C_pred_native: torch.Tensor,
    E_target: torch.Tensor,
    E_source: torch.Tensor,
    autocast_dtype,
    *,
    margin: float = 0.001,
    wrongs: torch.Tensor | None = None,
    loss_type: str = "hinge",
) -> tuple[torch.Tensor, dict]:
    """L_binder. Two formulations available:

    `loss_type="hinge"` (legacy, FiLM v2 + ControlNet v1):
        max(0, l_t + margin - l_s) + max(0, l_t + margin - l_w_i)
        where l_t = MSE(binder(C_pred), E_target), etc.
        Saturates to zero once binder prediction beats source/wrongs by margin.

    `loss_type="mse_plus_hinge_wrongs"` (per operator 2026-04-29):
        l_t (direct MSE) + max(0, l_t + margin - l_w_i)
        Keeps target-prediction pressure even after the binder beats source.
        Hard-wrongs hinge prevents the trivial solution of binder predicting
        any random emission.

    Args:
        margin: hinge margin in MSE units (only used for hinge terms).
        wrongs: optional (B, N_w, 3, em_h, em_w) of hard-wrong emissions.
        loss_type: "hinge" or "mse_plus_hinge_wrongs".
    """
    losses = []
    diagnostics = {
        "per_binder_target_mse": [], "per_binder_source_mse": [],
        "per_binder_margin_vs_source": [], "per_binder_margin_vs_wrongs_mean": [],
    }
    for spec, binder in binders:
        C_in = downsample_for_binder(C_pred_native, spec["capture_h"], spec["capture_w"])
        with torch.amp.autocast("cuda", dtype=autocast_dtype):
            pred_em = binder(C_in).float()
        # MSE to target & source emissions, shape (B,)
        l_t = ((pred_em - E_target) ** 2).mean(dim=(-3, -2, -1))
        l_s = ((pred_em - E_source) ** 2).mean(dim=(-3, -2, -1))
        # Target-prediction pressure: either hinge vs source OR direct MSE.
        if loss_type == "mse_plus_hinge_wrongs":
            target_term = l_t.mean()
        else:  # "hinge"
            target_term = (l_t + margin - l_s).clamp(min=0).mean()
        # Hard-wrongs term is hinge in BOTH formulations (prevents trivial).
        if wrongs is not None and wrongs.numel() > 0:
            l_w = ((pred_em.unsqueeze(1) - wrongs) ** 2).mean(dim=(-3, -2, -1))
            wrongs_term = (l_t.unsqueeze(1) + margin - l_w).clamp(min=0).mean()
            diff_total = target_term + wrongs_term
            diagnostics["per_binder_margin_vs_wrongs_mean"].append(
                (l_w - l_t.unsqueeze(1)).mean().item())
        else:
            diff_total = target_term
            diagnostics["per_binder_margin_vs_wrongs_mean"].append(float("nan"))
        losses.append(diff_total)
        diagnostics["per_binder_target_mse"].append(l_t.mean().item())
        diagnostics["per_binder_source_mse"].append(l_s.mean().item())
        diagnostics["per_binder_margin_vs_source"].append((l_s - l_t).mean().item())
    total = torch.stack(losses).mean() if losses else torch.zeros((), device=C_pred_native.device)
    return total, diagnostics


# ---------- causality probe ----------

@torch.no_grad()
def causality_probe(
    *,
    editor,
    val_ds: TemporalPairDataset,
    d2_dir: Path,
    n_sources: int = 8,
    n_targets: int = 32,
    autocast_dtype,
    device,
    seed: int = 7,
) -> dict:
    """For 8 fixed val sources × 32 different E_targets:
      - generate fakes
      - measure output diversity (std across 32 fakes per source)
      - measure binder L_target / L_source / margin per fake
      - return summary + 4 visual examples
    """
    rng = np.random.RandomState(seed)
    val_rows = val_ds.rows
    sources = sorted(rng.choice(val_rows, size=min(n_sources, len(val_rows)), replace=False).tolist())
    targets = sorted(rng.choice(val_rows, size=min(n_targets, len(val_rows)), replace=False).tolist())

    diversities = []
    cfa_deltas_from_source = []
    binder_target_mses = []
    binder_source_mses = []

    visuals: list[dict] = []  # per source: 4 (target, fake) pairs

    editor.eval()
    for s_idx, src_t in enumerate(sources):
        if src_t not in val_ds.rows:
            continue
        sample = val_ds[val_ds.rows.index(src_t)]
        C_s = sample["C_source"].unsqueeze(0).to(device)
        E_s = sample["E_source"].unsqueeze(0).to(device)
        fakes = []
        for tgt_t in targets:
            E_t_path = d2_dir / "derived/Emissions" / f"tile_{tgt_t:06d}.png"
            if not E_t_path.exists():
                continue
            E_t = load_emission_at(E_t_path, 1080, 1920).unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                C_pred = editor(C_s, E_s, E_t).float().clamp(0, 1)
            fakes.append((tgt_t, C_pred.squeeze(0).cpu()))
            cfa_deltas_from_source.append((C_pred - C_s).abs().mean().item())
        if len(fakes) >= 2:
            stack = torch.stack([f[1] for f in fakes], dim=0)
            diversity = stack.std(dim=0).mean().item()
            diversities.append(diversity)
            # Save visuals: 4 representative target/fake pairs from this source
            if len(visuals) < 2:
                visuals.append({
                    "source_t": src_t,
                    "examples": [{
                        "target_t": int(t),
                        "fake_mean": float(f.mean().item()),
                        "fake_std":  float(f.std().item()),
                    } for (t, f) in fakes[:4]],
                })
    summary = {
        "n_sources": len(diversities),
        "n_targets": len(targets),
        "diversity_mean":  float(np.mean(diversities)) if diversities else 0.0,
        "diversity_min":   float(np.min(diversities)) if diversities else 0.0,
        "diversity_max":   float(np.max(diversities)) if diversities else 0.0,
        "cfa_delta_from_source_mean": float(np.mean(cfa_deltas_from_source)) if cfa_deltas_from_source else 0.0,
        "visuals_per_source": visuals,
    }
    return summary


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "experiments/phase_f_prep/mini_experiment")
    ap.add_argument("--d2-dir", type=Path,
                    default=Path("/path/to/poliebotics_phase_b/poliebotics_phase_b/data/d2"))
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--probe-every", type=int, default=250)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--coef-recon", type=float, default=1.0)
    ap.add_argument("--coef-grad", type=float, default=0.1)
    ap.add_argument("--coef-binder", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=0.001,
                    help="Hinge margin in MSE units for L_binder.")
    ap.add_argument("--n-hard-wrongs", type=int, default=0,
                    help="Number of hard-wrong emissions to contrast E_target against per step. 0 disables.")
    ap.add_argument("--wrong-pool-size", type=int, default=256,
                    help="Size of the precomputed pool of random emissions to sample hard wrongs from.")
    ap.add_argument("--exp001c-ckpt-warm-start", type=str, default=None,
                    help="Path to exp001c ckpt for editor.source_encoder warm-start")
    ap.add_argument("--editor", choices=["film", "controlnet"], default="film",
                    help="Which editor architecture to use.")
    ap.add_argument("--binder-loss-type",
                    choices=["hinge", "mse_plus_hinge_wrongs"],
                    default="hinge",
                    help="L_binder formulation. 'hinge' is legacy; "
                         "'mse_plus_hinge_wrongs' replaces target hinge "
                         "with direct MSE (operator 2026-04-29).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16

    # Datasets
    train_rows = split_rows("D2", "train")
    val_rows = split_rows("D2", "val")
    train_ds = TemporalPairDataset(
        session_dir=args.d2_dir, rows=train_rows,
        k_choices=[1, 2, 3], augment=False, seed=0,
    )
    val_ds = TemporalPairDataset(
        session_dir=args.d2_dir, rows=val_rows,
        k_choices=[1], augment=False, seed=0,
    )
    print(f"[init] train n={len(train_ds)} val n={len(val_ds)}", flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.bs, shuffle=True, num_workers=2,
        collate_fn=collate_temporal_pairs, pin_memory=True, drop_last=True,
        persistent_workers=True,
    )

    # Editor
    if args.editor == "controlnet":
        editor = EditorControlNet(
            capture_h=2300, capture_w=2660,
            emission_h=1080, emission_w=1920,
            init_mode="exp001c-warm-start",
        ).to(device)
    else:
        editor = Editor(
            capture_h=2300, capture_w=2660,
            emission_h=1080, emission_w=1920,
            init_mode="exp001c-warm-start",
        ).to(device)
    if args.exp001c_ckpt_warm_start:
        editor.load_warm_start(Path(args.exp001c_ckpt_warm_start))
    n_params = sum(p.numel() for p in editor.parameters())
    print(f"[init] editor={args.editor} params={n_params/1e6:.1f}M", flush=True)

    # Surrogate binders (frozen)
    binders = []
    for spec in SURROGATE_BINDERS:
        try:
            b = load_binder(spec, device)
            binders.append((spec, b))
            print(f"[init] loaded binder: {spec['name']}", flush=True)
        except Exception as exc:
            print(f"[WARN] could not load binder {spec['name']}: {exc}", flush=True)
    if len(binders) == 0:
        print("[WARN] no surrogate binders loaded — running with L_binder=0 (recon+grad only)", flush=True)

    # Hard-wrongs pool (only built if requested)
    wrong_pool: torch.Tensor | None = None
    if args.n_hard_wrongs > 0 and binders:
        rng = np.random.RandomState(123)
        pool_rows = sorted(rng.choice(train_rows,
                                      size=min(args.wrong_pool_size, len(train_rows)),
                                      replace=False).tolist())
        print(f"[init] loading hard-wrongs pool: {len(pool_rows)} emissions...", flush=True)
        pool_emissions = []
        for r in pool_rows:
            try:
                em = load_emission_at(args.d2_dir / "derived" / "Emissions" / f"tile_{r:06d}.png",
                                      1080, 1920)  # (3, 1080, 1920) float32
                pool_emissions.append(em)
            except Exception as e:
                print(f"  skip row {r}: {e}", flush=True)
        wrong_pool = torch.stack(pool_emissions).to(device)  # (N_POOL, 3, 1080, 1920)
        del pool_emissions
        pool_mem_mb = wrong_pool.element_size() * wrong_pool.numel() / (1024 ** 2)
        print(f"[init] wrong_pool shape={tuple(wrong_pool.shape)}  "
              f"mem={pool_mem_mb:.0f} MiB", flush=True)

    opt = torch.optim.AdamW(editor.parameters(), lr=args.lr, weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=(autocast_dtype == torch.float16))

    history: list[dict] = []
    probe_history: list[dict] = []
    t_start = time.time()
    train_iter = iter(train_loader)
    step = 0
    while step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        C_s = batch["C_source"].to(device, non_blocking=True)
        E_s = batch["E_source"].to(device, non_blocking=True)
        E_t = batch["E_target"].to(device, non_blocking=True)
        C_t = batch["C_target"].to(device, non_blocking=True)

        # Sample hard-wrongs for this step (B, N, 3, em_h, em_w)
        wrongs_batch = None
        if wrong_pool is not None and args.n_hard_wrongs > 0:
            B = C_s.shape[0]
            idx = torch.randint(0, wrong_pool.shape[0],
                                (B, args.n_hard_wrongs), device=device)
            wrongs_batch = wrong_pool[idx]  # (B, N, 3, em_h, em_w)

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

        if step % 25 == 0:
            history.append({
                "step": step, "L_recon": L_recon.item(), "L_grad": L_grad.item(),
                "L_binder": L_binder.item(), "loss": loss.item(),
                **{f"binder_diag_{k}": v for k, v in binder_diag.items()},
                "t": round(time.time() - t_start, 1),
            })
            print(f"[step {step}] L_recon={L_recon.item():.4f}  L_grad={L_grad.item():.4f}  "
                  f"L_binder={L_binder.item():.4f}  loss={loss.item():.4f}  "
                  f"t={time.time()-t_start:.0f}s", flush=True)

        if step > 0 and step % args.probe_every == 0:
            print(f"\n=== causality probe @ step {step} ===", flush=True)
            probe = causality_probe(
                editor=editor, val_ds=val_ds, d2_dir=args.d2_dir,
                n_sources=8, n_targets=32, autocast_dtype=autocast_dtype, device=device,
            )
            probe["step"] = step
            probe_history.append(probe)
            print(f"  diversity_mean={probe['diversity_mean']:.5f}  "
                  f"cfa_delta_from_source={probe['cfa_delta_from_source_mean']:.5f}", flush=True)
            (args.out_dir / "probe_history.jsonl").open("a").write(json.dumps(probe) + "\n")

        step += 1

    # Final probe + ckpt
    final_probe = causality_probe(
        editor=editor, val_ds=val_ds, d2_dir=args.d2_dir,
        n_sources=8, n_targets=32, autocast_dtype=autocast_dtype, device=device,
    )
    final_probe["step"] = step
    probe_history.append(final_probe)
    (args.out_dir / "probe_history.jsonl").open("a").write(json.dumps(final_probe) + "\n")

    ckpt_path = args.out_dir / "editor_final.pt"
    torch.save({"model": editor.state_dict(), "step": step}, ckpt_path)
    (args.out_dir / "loss_history.jsonl").write_text("\n".join(json.dumps(h) for h in history) + "\n")
    print(f"[done] saved {ckpt_path}", flush=True)

    # Plot diversity curve
    if HAVE_PLT and probe_history:
        steps = [p["step"] for p in probe_history]
        divs = [p["diversity_mean"] for p in probe_history]
        cfa_deltas = [p["cfa_delta_from_source_mean"] for p in probe_history]
        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        l1 = ax1.plot(steps, divs, "o-", color="C0", label="output diversity (across 32 targets)")
        ax1.set_xlabel("training step")
        ax1.set_ylabel("std(C_pred) across targets per source")
        ax1.axhline(0.01, color="gray", linestyle=":", alpha=0.6, label="passing threshold (0.01)")
        ax2 = ax1.twinx()
        l2 = ax2.plot(steps, cfa_deltas, "s-", color="C3", label="cfa_delta from source")
        ax2.set_ylabel("|C_pred − C_source| mean", color="C3")
        ax2.tick_params(axis="y", labelcolor="C3")
        ax1.legend(loc="upper left")
        ax2.legend(loc="lower right")
        ax1.set_title("Phase F-A mini-experiment: causality vs training step")
        fig.tight_layout()
        fig.savefig(args.out_dir / "diversity_curve.png", dpi=110)
        plt.close(fig)
        print(f"[done] saved diversity_curve.png", flush=True)

    # Final summary
    verdict = "PASS" if final_probe["diversity_mean"] > 0.01 else "FAIL"
    print(f"\n=== VERDICT: {verdict} ===", flush=True)
    print(f"  final diversity_mean = {final_probe['diversity_mean']:.5f}", flush=True)
    print(f"  final cfa_delta_from_source = {final_probe['cfa_delta_from_source_mean']:.5f}", flush=True)
    summary = {
        "verdict": verdict,
        "max_steps": args.max_steps,
        "elapsed_sec": round(time.time() - t_start, 1),
        "final_probe": final_probe,
        "n_loss_history_entries": len(history),
        "n_probe_history_entries": len(probe_history),
        "interpretation": (
            "PASS: editor uses E_target — diversity grew over training. F-A as designed."
            if verdict == "PASS"
            else "FAIL: editor pose-warps without using E_target. Need pose-filtering or different supervision."
        ),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

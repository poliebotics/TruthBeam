"""Adversarial probes against an emission binder.

Four probes (per operator priority order after P0b outcome A):
  1. splice         — substitute predictions at random rows; measure top-1 drop
  2. gradient_overlay — L_inf ≤ 0.05 input perturbation that maximizes
                         mismatch (50/100/500 iterations); does it fool the binder?
  3. cross_session_sub — swap entire captures with V10; does the D2-trained
                         binder retrieve V10 row content?
  4. shifted_history — score prediction at row t vs emissions at row t±k for
                       k ∈ {1, 5, 10, 50, 100}; tests temporal binding precision

Run:
  python scripts/phase_e/adversarial_probes.py \
    --ckpt experiments/phase_e/e1/checkpoints/best_by_psnr.pt \
    --config scripts/phase_e/configs/e1.yaml \
    --probes splice,gradient_overlay,cross_session_sub,shifted_history \
    --out experiments/phase_e/probes_e1
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

torch.set_num_threads(8)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "phase_e"))

from data.emission_dataset import EmissionDataset, load_emission_at  # noqa: E402
from data.raw_bayer_dataset import load_chain_log  # noqa: E402
from train_phase_e import build_model  # noqa: E402


def psnr(pred, target):
    mse = ((pred - target) ** 2).mean().item()
    return float("inf") if mse < 1e-12 else 20.0 * math.log10(1.0 / math.sqrt(mse))


@torch.no_grad()
def predict_one(model, sample, autocast_dtype, device):
    cap = sample["capture"].unsqueeze(0).to(device)
    with torch.amp.autocast("cuda", dtype=autocast_dtype):
        p = model(cap).float()
    return p.squeeze(0).clamp(0, 1).cpu()


# ---------- probe 1: splice ----------

def probe_splice(model, ds, val_rows, autocast_dtype, device, n_eval=100, seed=42):
    """For each sample i, splice in another sample j's prediction (j≠i).
    Verifier should NOT match spliced prediction with row i's emission target.
    Measures top-1 retrieval rate of (spliced prediction → original target).
    Random chance = 0; perfect-attack = 1 (attack succeeds = matches).
    Lower is better for the binder."""
    rng = np.random.RandomState(seed)
    n_eval = min(n_eval, len(val_rows))
    rows = sorted(rng.choice(val_rows, size=n_eval, replace=False).tolist())
    # Precompute predictions
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for t in rows:
        if t not in ds.rows: continue
        sample = ds[ds.rows.index(t)]
        preds.append(predict_one(model, sample, autocast_dtype, device))
        targets.append(sample["emission"].clamp(0, 1))
    n = len(preds)
    # Spliced: for each i, pick random j≠i, compute (pred_j vs target_i).
    # Compare to matched (pred_i vs target_i) — if matched still wins, splice failed.
    splice_matched_wins = 0
    psnr_pred_at_i_vs_target_i = []
    psnr_pred_at_j_vs_target_i = []
    for i in range(n):
        j = (i + rng.randint(1, n)) % n  # always != i
        m_psnr = psnr(preds[i], targets[i])
        sp_psnr = psnr(preds[j], targets[i])
        psnr_pred_at_i_vs_target_i.append(m_psnr)
        psnr_pred_at_j_vs_target_i.append(sp_psnr)
        if m_psnr > sp_psnr:
            splice_matched_wins += 1
    return {
        "n": n,
        "matched_psnr_mean": float(np.mean(psnr_pred_at_i_vs_target_i)),
        "spliced_psnr_mean": float(np.mean(psnr_pred_at_j_vs_target_i)),
        "gap_db": float(np.mean(psnr_pred_at_i_vs_target_i)) - float(np.mean(psnr_pred_at_j_vs_target_i)),
        "splice_matched_wins_rate": splice_matched_wins / n if n else 0.0,
        "splice_attack_success_rate": (n - splice_matched_wins) / n if n else 0.0,
        "interpretation": "matched-wins-rate close to 1.0 means binder distinguishes spliced predictions",
    }


# ---------- probe 2: gradient overlay ----------

def probe_gradient_overlay(model, ds, val_rows, autocast_dtype, device,
                           n_eval=20, eps_linf=0.05, iters_list=(50, 100, 500), seed=42):
    """L_inf ≤ eps_linf input perturbation that maximizes pred-vs-target distance.
    Pre-attack PSNR vs post-attack PSNR for each iter count.
    Interpretation: smaller PSNR drop = more robust binder."""
    rng = np.random.RandomState(seed)
    n_eval = min(n_eval, len(val_rows))
    rows = sorted(rng.choice(val_rows, size=n_eval, replace=False).tolist())
    out_per_iters: dict[int, dict] = {}
    pre_psnrs: list[float] = []
    for it_count in iters_list:
        post_psnrs: list[float] = []
        # Reset for each setting
        for t in rows:
            if t not in ds.rows: continue
            sample = ds[ds.rows.index(t)]
            cap0 = sample["capture"].unsqueeze(0).to(device)
            em = sample["emission"].clamp(0, 1).to(device)
            # Pre-attack
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=autocast_dtype):
                p0 = model(cap0).float().clamp(0, 1)
            pre_p = psnr(p0.squeeze(0).cpu(), em.cpu())
            if it_count == iters_list[0]:
                pre_psnrs.append(pre_p)
            # Adversarial perturbation: PGD on -emission_loss
            delta = torch.zeros_like(cap0, requires_grad=True)
            alpha = max(eps_linf / max(it_count, 1) * 4, 1e-3)
            for step in range(it_count):
                # Use fp32 for adversarial gradient (autocast can hurt precision)
                cap_adv = (cap0 + delta).clamp(0, 1)
                pred = model(cap_adv).float()
                # MAXIMIZE L1 distance to target
                loss_attack = -(pred - em).abs().mean()
                grad = torch.autograd.grad(loss_attack, delta, retain_graph=False)[0]
                with torch.no_grad():
                    delta = (delta - alpha * grad.sign()).clamp(-eps_linf, eps_linf)
                    # ensure cap_adv stays in [0,1]
                    delta = (cap0 + delta).clamp(0, 1).sub_(cap0)
                delta = delta.detach().requires_grad_(True)
            with torch.no_grad():
                cap_adv = (cap0 + delta).clamp(0, 1)
                p_adv = model(cap_adv).float().clamp(0, 1)
            post_p = psnr(p_adv.squeeze(0).cpu(), em.cpu())
            post_psnrs.append(post_p)
        pre_arr = np.array(pre_psnrs)
        post_arr = np.array(post_psnrs)
        out_per_iters[it_count] = {
            "iters": it_count,
            "eps_linf": eps_linf,
            "pre_psnr_mean": float(pre_arr.mean()),
            "post_psnr_mean": float(post_arr.mean()),
            "psnr_drop_db": float((pre_arr - post_arr).mean()),
            "n_eval": len(post_psnrs),
        }
    return out_per_iters


# ---------- probe 3: cross-session substitution ----------

def probe_cross_session_sub(model, ds_d2, val_rows_d2, v10_dir: Path,
                            autocast_dtype, device, capture_h, capture_w, emission_h, emission_w,
                            n_eval=50, seed=42):
    """Predict on a V10 capture using D2-trained binder; score the prediction
    against V10's emission at that row, vs random V10 rows.
    Tests cross-session generalization beyond just candidate-set leakage."""
    from data.emission_dataset import EmissionDataset as ED
    chain = load_chain_log(v10_dir / "chain_log.csv") if (v10_dir / "chain_log.csv").exists() else {}
    if not chain:
        return {"error": "v10 chain not available"}
    v10_rows = sorted([t for t in chain
                       if (v10_dir / "Recordings" / f"frame_{t:06d}.raw").exists()
                       and (v10_dir / "derived" / "Emissions" / f"tile_{t:06d}.png").exists()])
    if not v10_rows:
        return {"error": "no v10 frames available"}
    rng = np.random.RandomState(seed)
    pick = sorted(rng.choice(v10_rows, size=min(n_eval, len(v10_rows)), replace=False).tolist())
    ds_v10 = ED(session_dir=v10_dir,
                row_start=min(pick), row_end=max(pick) + 1,
                capture_h=capture_h, capture_w=capture_w,
                emission_h=emission_h, emission_w=emission_w,
                session_id="V10", augment=False)
    matched_psnrs = []
    mm_psnrs = []
    top1 = 0
    for t in pick:
        if t not in ds_v10.rows: continue
        sample = ds_v10[ds_v10.rows.index(t)]
        pred = predict_one(model, sample, autocast_dtype, device)
        em = sample["emission"].clamp(0, 1)
        m = psnr(pred, em)
        # 32 random non-matching V10 candidates
        cand = [r for r in v10_rows if r != t]
        cand_rows = rng.choice(cand, size=min(32, len(cand)), replace=False)
        max_mm = -math.inf
        cand_psnrs = []
        for c in cand_rows:
            ce = load_emission_at(v10_dir / "derived" / "Emissions" / f"tile_{int(c):06d}.png",
                                   emission_h, emission_w).clamp(0, 1)
            cp = psnr(pred, ce)
            cand_psnrs.append(cp)
            if cp > max_mm: max_mm = cp
        matched_psnrs.append(m)
        mm_psnrs.extend(cand_psnrs)
        if m > max_mm: top1 += 1
    n = len(matched_psnrs)
    return {
        "n": n,
        "matched_psnr_mean": float(np.mean(matched_psnrs)) if matched_psnrs else float("nan"),
        "mismatched_psnr_mean": float(np.mean(mm_psnrs)) if mm_psnrs else float("nan"),
        "gap_db": float(np.mean(matched_psnrs) - np.mean(mm_psnrs)) if matched_psnrs and mm_psnrs else float("nan"),
        "top1_retrieval_v10": top1 / n if n else 0.0,
        "interpretation": "top-1 close to 1.0 = D2 binder transfers to V10; close to chance (1/33≈0.030) = no transfer",
    }


# ---------- probe 4: shifted history ----------

def probe_shifted_history(model, ds, val_rows, autocast_dtype, device,
                          d2_dir: Path, emission_h: int, emission_w: int,
                          shifts=(1, 5, 10, 50, 100), n_eval=100, seed=42):
    """For row t, prediction P_t scored against emission(t±k) for k in shifts.
    PSNR_at_shift = mean( PSNR(P_t, em_{t±k}) ).
    If binder is content-bound (independent fBm per row), all shifts ≥ 1 should
    look like random mismatches (low PSNR). If binder leaks temporal structure,
    small-k shifts will show inflated PSNR vs distant shifts."""
    rng = np.random.RandomState(seed)
    n_eval = min(n_eval, len(val_rows))
    rows = sorted(rng.choice(val_rows, size=n_eval, replace=False).tolist())
    matched: list[float] = []
    by_shift: dict[int, list[float]] = {k: [] for k in shifts}
    by_shift_neg: dict[int, list[float]] = {k: [] for k in shifts}
    for t in rows:
        if t not in ds.rows: continue
        sample = ds[ds.rows.index(t)]
        pred = predict_one(model, sample, autocast_dtype, device)
        em = sample["emission"].clamp(0, 1)
        matched.append(psnr(pred, em))
        for k in shifts:
            for sign in (+1, -1):
                t2 = t + sign * k
                p = d2_dir / "derived" / "Emissions" / f"tile_{t2:06d}.png"
                if not p.exists(): continue
                em2 = load_emission_at(p, emission_h, emission_w).clamp(0, 1)
                pp = psnr(pred, em2)
                by_shift[k].append(pp)
    return {
        "n": len(matched),
        "matched_psnr_mean": float(np.mean(matched)) if matched else float("nan"),
        "shift_psnrs": {str(k): {
            "mean": float(np.mean(v)) if v else float("nan"),
            "std":  float(np.std(v))  if v else float("nan"),
            "n": len(v),
        } for k, v in by_shift.items()},
        "interpretation": "if shift_psnr ≈ matched_psnr at small k but drops at large k, binder leaks temporal structure; if all shifts are uniformly low, binder is content-bound (good)",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--probes", default="splice,gradient_overlay,cross_session_sub,shifted_history")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output directory for probe results")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--n-eval-splice", type=int, default=100)
    ap.add_argument("--n-eval-grad", type=int, default=20)
    ap.add_argument("--n-eval-cross", type=int, default=50)
    ap.add_argument("--n-eval-shift", type=int, default=100)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = torch.device("cuda")
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16

    model = build_model(cfg, device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[probes] ckpt={args.ckpt} arch={cfg['model']['arch']}", flush=True)

    val_start = cfg["data"]["d2_val_start"]
    val_end = cfg["data"]["d2_val_end"]
    val_rows = list(range(val_start, val_end))
    d2_dir = Path(cfg["data"]["d2_dir"])
    v10_dir = Path(cfg["data"]["v10_dir"])
    capture_h = cfg["data"]["capture_h"]
    capture_w = cfg["data"]["capture_w"]
    emission_h = cfg["data"]["emission_h"]
    emission_w = cfg["data"]["emission_w"]

    ds = EmissionDataset(session_dir=d2_dir, row_start=val_start, row_end=val_end,
                          capture_h=capture_h, capture_w=capture_w,
                          emission_h=emission_h, emission_w=emission_w,
                          session_id="D2", augment=False)

    probes = args.probes.split(",")
    out: dict = {"ckpt": str(args.ckpt), "config": str(args.config), "probes": {}}
    args.out.mkdir(parents=True, exist_ok=True)

    if "splice" in probes:
        print("\n[probe 1/4: splice]", flush=True)
        t0 = time.time()
        out["probes"]["splice"] = probe_splice(model, ds, val_rows, autocast_dtype, device,
                                                n_eval=args.n_eval_splice)
        out["probes"]["splice"]["elapsed_sec"] = round(time.time() - t0, 1)
        r = out["probes"]["splice"]
        print(f"  matched_PSNR={r['matched_psnr_mean']:.2f} spliced_PSNR={r['spliced_psnr_mean']:.2f} "
              f"gap={r['gap_db']:.2f} matched-wins-rate={r['splice_matched_wins_rate']:.3f} "
              f"({r['elapsed_sec']:.0f}s)", flush=True)

    if "gradient_overlay" in probes:
        print("\n[probe 2/4: gradient_overlay]", flush=True)
        t0 = time.time()
        out["probes"]["gradient_overlay"] = probe_gradient_overlay(
            model, ds, val_rows, autocast_dtype, device, n_eval=args.n_eval_grad)
        out["probes"]["gradient_overlay"]["elapsed_sec"] = round(time.time() - t0, 1)
        for it, r in out["probes"]["gradient_overlay"].items():
            if isinstance(r, dict) and "iters" in r:
                print(f"  iters={r['iters']} eps={r['eps_linf']} pre={r['pre_psnr_mean']:.2f} "
                      f"post={r['post_psnr_mean']:.2f} drop={r['psnr_drop_db']:.2f}", flush=True)

    if "cross_session_sub" in probes:
        print("\n[probe 3/4: cross_session_sub]", flush=True)
        t0 = time.time()
        out["probes"]["cross_session_sub"] = probe_cross_session_sub(
            model, ds, val_rows, v10_dir, autocast_dtype, device,
            capture_h, capture_w, emission_h, emission_w, n_eval=args.n_eval_cross)
        out["probes"]["cross_session_sub"]["elapsed_sec"] = round(time.time() - t0, 1)
        r = out["probes"]["cross_session_sub"]
        if "matched_psnr_mean" in r:
            print(f"  V10 matched_PSNR={r['matched_psnr_mean']:.2f} mismatched_PSNR={r['mismatched_psnr_mean']:.2f} "
                  f"gap={r['gap_db']:.2f} top1={r['top1_retrieval_v10']:.3f} "
                  f"({r['elapsed_sec']:.0f}s)", flush=True)
        else:
            print(f"  ERROR: {r}", flush=True)

    if "shifted_history" in probes:
        print("\n[probe 4/4: shifted_history]", flush=True)
        t0 = time.time()
        out["probes"]["shifted_history"] = probe_shifted_history(
            model, ds, val_rows, autocast_dtype, device,
            d2_dir, emission_h, emission_w, n_eval=args.n_eval_shift)
        out["probes"]["shifted_history"]["elapsed_sec"] = round(time.time() - t0, 1)
        r = out["probes"]["shifted_history"]
        print(f"  matched_PSNR={r['matched_psnr_mean']:.2f}", flush=True)
        for k, v in r["shift_psnrs"].items():
            print(f"  shift k=±{k}: PSNR={v['mean']:.2f} ± {v['std']:.2f} (n={v['n']})", flush=True)

    out_path = args.out / "probes.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

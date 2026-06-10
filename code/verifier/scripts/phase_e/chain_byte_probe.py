"""Chain-byte sensitivity probe.

For each (binder ckpt, D2-val row t):
  - C_t   = real capture for row t
  - X_t   = clean per-channel XOF expanded streams (3 × 43,110 bytes) for row t,
            derived from chain_log[t+1] (S_{t+1}_hex)
  - E_t   = render_from_streams(X_t)  (matched ground truth, bit-exact)
  - P_t   = binder(C_t)               (binder's prediction)

Sweep noise level p in {0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0}
and seed s in {0, 1, 2, 3, 4}:
  - X_t'(p,s) = noise_streams_byte_replace(X_t, p, seed=s ⊕ row)
  - E_t'(p,s) = render_from_streams(X_t'(p,s))
  - matched_score = PSNR(P_t, E_t)
  - noised_score  = PSNR(P_t, E_t'(p,s))
  - margin        = matched_score − noised_score
  - attack_success = (noised_score >= matched_score − 1e-6)

Outputs:
  <out>/chain_byte_sensitivity.json    — per-evaluation + aggregated_per_p
  <out>/chain_byte_sensitivity_curve.png   — attack-success vs p (log scale)
  <out>/chain_byte_margin_distribution.png — violin plot per p
  <out>/chain_byte_threshold_summary.txt   — text summary

Run:
  python scripts/phase_e/chain_byte_probe.py \
    --ckpt experiments/phase_e/e1/checkpoints/best_by_psnr.pt \
    --config scripts/phase_e/configs/e1.yaml \
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
import yaml

torch.set_num_threads(8)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "phase_e"))

from data.emission_dataset import EmissionDataset  # noqa: E402
from data.raw_bayer_dataset import load_chain_log  # noqa: E402
from bitexact_renderer import (  # noqa: E402
    expand_seeds_to_streams,
    noise_streams_byte_replace,
    render_from_streams,
)
from train_phase_e import build_model  # noqa: E402


NOISE_LEVELS = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
SEEDS = (0, 1, 2, 3, 4)


def psnr(pred01: torch.Tensor, target01: torch.Tensor) -> float:
    mse = ((pred01 - target01) ** 2).mean().item()
    if mse < 1e-12:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def bootstrap_ci_rate(hits: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap CI for a binary rate."""
    rng = np.random.RandomState(seed)
    n = len(hits)
    if n == 0: return (float("nan"), float("nan"))
    rates = np.array([rng.choice(hits, size=n, replace=True).mean() for _ in range(n_boot)])
    return (float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-rows", type=int, default=200, help="number of D2-val rows")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--render-device", default="cpu",
                    help="cpu or cuda — pure int math, both produce bit-exact output. "
                         "cpu avoids contention with binder forward on the same GPU.")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = torch.device(args.device)
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16

    print(f"[probe] ckpt={args.ckpt} arch={cfg['model']['arch']}", flush=True)
    print(f"[probe] capture_hw={cfg['data']['capture_h']}x{cfg['data']['capture_w']}  "
          f"emission_hw={cfg['data']['emission_h']}x{cfg['data']['emission_w']}", flush=True)
    print(f"[probe] noise_levels={NOISE_LEVELS}  seeds={SEEDS}", flush=True)

    # Build binder
    model = build_model(cfg, device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()

    # Sample D2-val rows
    val_start, val_end = cfg["data"]["d2_val_start"], cfg["data"]["d2_val_end"]
    val_rows = list(range(val_start, val_end))
    rng = np.random.RandomState(42)
    rows = sorted(rng.choice(val_rows, size=min(args.n_rows, len(val_rows)), replace=False).tolist())

    d2_dir = Path(cfg["data"]["d2_dir"])
    chain = load_chain_log(d2_dir / "chain_log.csv")

    # Build dataset for capture loading
    ds = EmissionDataset(session_dir=d2_dir,
                         row_start=val_start, row_end=val_end,
                         capture_h=cfg["data"]["capture_h"], capture_w=cfg["data"]["capture_w"],
                         emission_h=cfg["data"]["emission_h"], emission_w=cfg["data"]["emission_w"],
                         session_id="D2", augment=False)

    # Precompute predictions + clean emissions per row (so the noise sweep just renders + scores)
    print(f"[probe] precomputing {len(rows)} predictions and clean emissions...", flush=True)
    t0 = time.time()
    pred_cache: dict[int, torch.Tensor] = {}     # row → (3, em_H, em_W) float32 [0,1]
    clean_em_cache: dict[int, torch.Tensor] = {} # row → (3, em_H, em_W) float32 [0,1]
    streams_cache: dict[int, tuple[bytes, bytes, bytes]] = {}
    for i, t in enumerate(rows):
        if t not in ds.rows:
            continue
        # exp001c's EmissionDataset pairs capture[t] with `tile_{t}.png` whose
        # content = render(chain[t]) per determinism check. The binder learned
        # capture[t] → render(chain[t]), so target_chain_row = t for the matched
        # ground truth. (Phase D's PackedCFADataset uses t+1; that's a different
        # alignment convention.)
        target_chain_row = t
        if target_chain_row not in chain:
            continue
        sample = ds[ds.rows.index(t)]
        cap = sample["capture"].unsqueeze(0).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=autocast_dtype):
            p_t = model(cap).float().clamp(0, 1).squeeze(0).cpu()
        pred_cache[t] = p_t

        # Render clean emission from chain bytes (bit-exact equiv to stored tile per check 3)
        s_next = bytes.fromhex(chain[target_chain_row])
        streams = expand_seeds_to_streams(s_next)
        streams_cache[t] = streams
        em_uint8 = render_from_streams(*streams, device=args.render_device)
        # Resize emission to (emission_h, emission_w) if different from native (1080×1920)
        em_em_h = cfg["data"]["emission_h"]; em_em_w = cfg["data"]["emission_w"]
        if em_uint8.shape[1:] != (em_em_h, em_em_w):
            import torch.nn.functional as F
            em_uint8 = F.interpolate(em_uint8.float().unsqueeze(0),
                                      size=(em_em_h, em_em_w), mode="area").squeeze(0).round().clamp(0, 255).to(torch.uint8)
        clean_em_cache[t] = em_uint8.float().div(255.0).cpu()
        if (i + 1) % 50 == 0:
            print(f"  precompute {i+1}/{len(rows)} elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"[probe] precompute done; cached {len(pred_cache)} rows  elapsed={time.time()-t0:.0f}s", flush=True)

    # Sweep
    per_eval: list[dict] = []
    aggregated_per_p: dict = {}
    print(f"[probe] starting sweep ({len(pred_cache)} rows × {len(NOISE_LEVELS)} p × {len(SEEDS)} seeds = {len(pred_cache) * len(NOISE_LEVELS) * len(SEEDS)} evals)", flush=True)
    t0 = time.time()
    em_em_h = cfg["data"]["emission_h"]; em_em_w = cfg["data"]["emission_w"]
    import torch.nn.functional as F  # noqa: E402

    n_done = 0
    n_total = len(pred_cache) * len(NOISE_LEVELS) * len(SEEDS)
    for p_idx, p in enumerate(NOISE_LEVELS):
        margins: list[float] = []
        attack_successes: list[int] = []
        per_p: list[dict] = []
        for t in pred_cache:
            P_t = pred_cache[t]
            E_t = clean_em_cache[t]
            matched_score = psnr(P_t, E_t)
            for s in SEEDS:
                # Use seed=s × prime XOR row so the noise pattern is unique per (row, seed)
                effective_seed = (s * 7919) ^ t
                if p == 0.0:
                    noised_em = E_t  # by construction
                else:
                    noised_streams = noise_streams_byte_replace(streams_cache[t], p=p, seed=effective_seed)
                    noised_uint8 = render_from_streams(*noised_streams, device=args.render_device)
                    if noised_uint8.shape[1:] != (em_em_h, em_em_w):
                        noised_uint8 = F.interpolate(noised_uint8.float().unsqueeze(0),
                                                      size=(em_em_h, em_em_w), mode="area").squeeze(0).round().clamp(0, 255).to(torch.uint8)
                    noised_em = noised_uint8.float().div(255.0).cpu()
                noised_score = psnr(P_t, noised_em)
                margin = matched_score - noised_score
                attack_success = bool(noised_score >= matched_score - 1e-6)
                rec = {"row": t, "p": p, "seed": s,
                       "matched_score": matched_score, "noised_score": noised_score,
                       "margin": margin, "attack_success": attack_success}
                per_p.append(rec)
                margins.append(margin)
                attack_successes.append(int(attack_success))
                n_done += 1
        margins_arr = np.array(margins)
        successes_arr = np.array(attack_successes)
        ci_lo, ci_hi = bootstrap_ci_rate(successes_arr, n_boot=1000, seed=0)
        aggregated_per_p[str(p)] = {
            "n_evaluations": int(len(margins)),
            "mean_margin": float(margins_arr.mean()),
            "std_margin":  float(margins_arr.std()),
            "median_margin": float(np.median(margins_arr)),
            "p5_margin":  float(np.percentile(margins_arr, 5)),
            "p95_margin": float(np.percentile(margins_arr, 95)),
            "attack_success_rate": float(successes_arr.mean()),
            "attack_success_ci_95": [ci_lo, ci_hi],
        }
        per_eval.extend(per_p)
        elapsed = time.time() - t0
        print(f"  p={p:.4f}  n={len(margins)}  margin_mean={margins_arr.mean():.3f} "
              f"attack_success={successes_arr.mean():.3f} CI=[{ci_lo:.3f},{ci_hi:.3f}]  "
              f"({n_done}/{n_total}, {elapsed:.0f}s)", flush=True)

    # Threshold p where attack success rate crosses 5% / 50%
    def threshold_at(target: float) -> float:
        ps = sorted([float(k) for k in aggregated_per_p])
        for p_low, p_hi in zip(ps[:-1], ps[1:]):
            r_low = aggregated_per_p[str(p_low)]["attack_success_rate"]
            r_hi = aggregated_per_p[str(p_hi)]["attack_success_rate"]
            if r_low > target >= r_hi:
                # linear interpolate in log-p (but not for 0)
                if p_low == 0:
                    return p_hi
                lp_lo, lp_hi = math.log10(p_low), math.log10(p_hi)
                frac = (r_low - target) / (r_low - r_hi) if r_low != r_hi else 0.5
                return float(10 ** (lp_lo + frac * (lp_hi - lp_lo)))
        return float("nan")

    thresh_5 = threshold_at(0.05)
    thresh_50 = threshold_at(0.50)

    out = {
        "checkpoint": str(args.ckpt),
        "config": str(args.config),
        "arch": cfg["model"]["arch"],
        "evaluated_rows": list(pred_cache.keys()),
        "noise_levels": list(NOISE_LEVELS),
        "seeds": list(SEEDS),
        "per_evaluation": per_eval,
        "aggregated_per_p": aggregated_per_p,
        "threshold_p_at_5pct_attack_success": thresh_5,
        "threshold_p_at_50pct_attack_success": thresh_50,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "chain_byte_sensitivity.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[probe] threshold p @ 5% attack success: {thresh_5:.5f}", flush=True)
    print(f"[probe] threshold p @ 50% attack success: {thresh_50:.5f}", flush=True)

    # Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Curve
        ps = sorted([float(k) for k in aggregated_per_p])
        rates = [aggregated_per_p[str(p)]["attack_success_rate"] for p in ps]
        cis = [aggregated_per_p[str(p)]["attack_success_ci_95"] for p in ps]
        ci_lo_arr = [c[0] for c in cis]; ci_hi_arr = [c[1] for c in cis]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        # Replace 0 with a small positive for log scale
        ps_for_plot = [max(p, 1e-5) for p in ps]
        ax.plot(ps_for_plot, rates, "o-", color="C0")
        ax.fill_between(ps_for_plot, ci_lo_arr, ci_hi_arr, alpha=0.2, color="C0")
        ax.axhline(0.05, color="gray", linestyle=":", alpha=0.6, label="5% attack success")
        ax.axhline(0.50, color="gray", linestyle="--", alpha=0.6, label="50% attack success")
        ax.set_xscale("log")
        ax.set_xlabel("noise level p (byte-replacement probability)")
        ax.set_ylabel("attack success rate")
        ax.set_title(f"Chain-byte sensitivity — {cfg.get('experiment_id', 'binder')}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out / "chain_byte_sensitivity_curve.png", dpi=110)
        plt.close(fig)

        # Margin distribution (violin)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        violin_data = [[r["margin"] for r in per_eval if r["p"] == p] for p in NOISE_LEVELS]
        positions = list(range(len(NOISE_LEVELS)))
        ax.violinplot(violin_data, positions=positions, showmeans=True, showmedians=False)
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{p:g}" for p in NOISE_LEVELS])
        ax.set_xlabel("noise level p")
        ax.set_ylabel("margin = matched_score − noised_score (dB)")
        ax.set_title(f"Margin distribution — {cfg.get('experiment_id', 'binder')}")
        ax.axhline(0, color="red", linestyle=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(args.out / "chain_byte_margin_distribution.png", dpi=110)
        plt.close(fig)
    except Exception as exc:
        print(f"[WARN] plot failed: {exc}", flush=True)

    # Text summary
    summary_lines = [
        f"Chain-byte sensitivity probe — {cfg.get('experiment_id', 'binder')}",
        f"Checkpoint: {args.ckpt}",
        f"Architecture: {cfg['model']['arch']}",
        f"Evaluations: {len(pred_cache)} rows × {len(NOISE_LEVELS)} p × {len(SEEDS)} seeds = {n_done}",
        "",
        f"Threshold p @ 5% attack success:  {thresh_5:.5f}",
        f"Threshold p @ 50% attack success: {thresh_50:.5f}",
        "",
        "Per-p attack success rate:",
    ]
    for p in NOISE_LEVELS:
        a = aggregated_per_p[str(p)]
        summary_lines.append(
            f"  p={p:7.4f}  n={a['n_evaluations']:5d}  attack_success={a['attack_success_rate']:.3f} "
            f"CI=[{a['attack_success_ci_95'][0]:.3f},{a['attack_success_ci_95'][1]:.3f}]  "
            f"margin_mean={a['mean_margin']:+.3f} dB  median={a['median_margin']:+.3f}"
        )
    (args.out / "chain_byte_threshold_summary.txt").write_text("\n".join(summary_lines))
    print("\n".join(summary_lines), flush=True)
    print(f"\n[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

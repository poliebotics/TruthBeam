"""Closure Task 3: A4 emission candidate ranking with 32 mismatched candidates.

Extends a4_posthoc_psnr.py:
  - 32 random non-matching candidates per frame (vs 10 in original)
  - Top-1 retrieval rate among 32+1
  - Per-frame matched + mismatched PSNR distributions
  - PNG histograms (matched vs mismatched) per split
  - V10 stratified by time bin

Run:
  python scripts/closure/closure_a4_emission_ranking.py \
    --ckpt experiments/exp001h_a4/checkpoints/final_step.pt \
    --config configs/exp001h_a4.yaml \
    --stats-dir cache/normalization_stats \
    --out experiments/closure_package/a4_emission_ranking.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.packed_cfa_dataset import PackedCFADataset  # noqa: E402
from models.emission_predictor_v2 import EmissionPredictorV2  # noqa: E402
from preprocessing.normalization import load_stats  # noqa: E402
from utils.config_loader import load_and_validate  # noqa: E402


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = ((pred - target) ** 2).mean().item()
    if mse < 1e-12:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def evaluate_split(*, model, session_dir, rows, offset, stats, cache_root,
                   session_id, device, autocast_dtype, n_candidates=32, seed=0,
                   verbose=True):
    ds = PackedCFADataset(
        session_dir=session_dir, rows=rows, offset=offset,
        normalization_stats=stats, cache_root=cache_root,
        session_id=session_id, with_xof=False, with_emission=True,
    )
    if len(ds) == 0:
        return {"n": 0, "error": "no rows"}

    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    rows_actual: list[int] = []
    matched_psnrs: list[float] = []

    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for i in range(len(ds)):
            sample = ds[i]
            cap = sample["capture_norm"].unsqueeze(0).to(device)
            target = sample["emission"].to(device)
            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                pred = model(cap).float()
            pred = pred.squeeze(0).clamp(0, 1).cpu()
            target_clamped = target.clamp(0, 1).cpu()
            p = psnr(pred, target_clamped)
            matched_psnrs.append(p)
            preds.append(pred)
            targets.append(target_clamped)
            rows_actual.append(int(sample["t"]))
            if verbose and (i + 1) % 100 == 0:
                print(f"  matched eval: {i+1}/{len(ds)} (avg psnr {np.mean(matched_psnrs):.2f})", flush=True)

    n = len(preds)
    rng = torch.Generator().manual_seed(seed)
    cross_pair_psnrs: list[float] = []
    top1_correct = 0
    per_frame_records = []
    for i in range(n):
        # Sample n_candidates distinct non-matching indices
        candidate_js = []
        seen = {i}
        while len(candidate_js) < min(n_candidates, n - 1):
            j = int(torch.randint(0, n, (1,), generator=rng).item())
            if j in seen:
                continue
            seen.add(j)
            candidate_js.append(j)
        cand_psnrs = [psnr(preds[i], targets[j]) for j in candidate_js]
        cross_pair_psnrs.extend(cand_psnrs)
        top1 = matched_psnrs[i] > max(cand_psnrs) if cand_psnrs else True
        if top1:
            top1_correct += 1
        per_frame_records.append({
            "row": rows_actual[i],
            "matched_psnr": matched_psnrs[i],
            "max_mismatched_psnr": float(max(cand_psnrs)) if cand_psnrs else float("nan"),
            "mean_mismatched_psnr": float(np.mean(cand_psnrs)) if cand_psnrs else float("nan"),
            "top1": bool(top1),
        })

    return {
        "n": n,
        "n_candidates": n_candidates,
        "elapsed_sec": round(time.time() - t0, 1),
        "matched_psnr_mean": float(np.mean(matched_psnrs)),
        "matched_psnr_median": float(np.median(matched_psnrs)),
        "matched_psnr_p5": float(np.quantile(matched_psnrs, 0.05)),
        "matched_psnr_p95": float(np.quantile(matched_psnrs, 0.95)),
        "matched_psnr_min": float(np.min(matched_psnrs)),
        "matched_psnr_max": float(np.max(matched_psnrs)),
        "matched_psnr_std": float(np.std(matched_psnrs)),
        "cross_pair_psnr_mean": float(np.mean(cross_pair_psnrs)),
        "cross_pair_psnr_median": float(np.median(cross_pair_psnrs)),
        "cross_pair_psnr_std": float(np.std(cross_pair_psnrs)),
        "matched_minus_cross_pair_db": float(np.mean(matched_psnrs)) - float(np.mean(cross_pair_psnrs)),
        "top1_retrieval_rate": top1_correct / n,
        "per_frame": per_frame_records,
        "matched_psnrs_raw": matched_psnrs,
        "cross_pair_psnrs_raw": cross_pair_psnrs,
    }


def save_histograms(out_dir: Path, splits_results: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable: {exc}; skipping plots", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, res in splits_results.items():
        if "matched_psnrs_raw" not in res or res.get("n", 0) == 0:
            continue
        m = res["matched_psnrs_raw"]
        x = res["cross_pair_psnrs_raw"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(m, bins=40, alpha=0.55, label=f"matched (n={len(m)})", color="C0")
        ax.hist(x, bins=40, alpha=0.55, label=f"mismatched (n={len(x)})", color="C3")
        ax.set_xlabel("PSNR (dB)")
        ax.set_ylabel("count")
        ax.set_title(f"{split_name}: matched vs mismatched PSNR")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"hist_{split_name}.png", dpi=120)
        plt.close(fig)
        print(f"  wrote {out_dir / f'hist_{split_name}.png'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--stats-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-candidates", type=int, default=32)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--max-rows-per-split", type=int, default=None)
    args = ap.parse_args()

    cfg = load_and_validate(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16

    model = EmissionPredictorV2(
        emission_h=int(cfg["data"].get("emission_h", 1080)),
        emission_w=int(cfg["data"].get("emission_w", 1920)),
        encoder_size=cfg["model"].get("encoder_size", "tiny"),
        pretrained=False,
        fpn_out_channels=int(cfg["model"].get("fpn_out_channels", 256)),
    ).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ck["model"] if "model" in ck else ck
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[ckpt] {args.ckpt}: missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    cache_root = Path(cfg["data"]["cache_root"])
    d2_dir = Path(cfg["data"]["d2_dir"])
    v10_dir = Path(cfg["data"].get("v10_dir", str(d2_dir.parent / "v10")))
    d2_stats = load_stats(args.stats_dir / "d2_train_stats.json")
    v10_stats = load_stats(args.stats_dir / "v10_train_stats.json")
    offset = int(cfg["offset"]["value"])

    splits = {
        "D2_val_full":     ("D2", d2_dir, d2_stats,
                            list(range(int(cfg["data"]["d2_val_start"]), int(cfg["data"]["d2_val_end"])))),
        "V10_val_full":    ("V10", v10_dir, v10_stats,
                            list(range(int(cfg["data"]["v10_val_start"]), int(cfg["data"]["v10_val_end"])))),
        "V10_early":       ("V10", v10_dir, v10_stats,
                            list(range(int(cfg["data"]["v10_early_start"]), int(cfg["data"]["v10_early_end"])))),
        "V10_mid":         ("V10", v10_dir, v10_stats,
                            list(range(int(cfg["data"]["v10_mid_start"]), int(cfg["data"]["v10_mid_end"])))),
        "V10_late":        ("V10", v10_dir, v10_stats,
                            list(range(int(cfg["data"]["v10_late_start"]), int(cfg["data"]["v10_late_end"])))),
    }

    out: dict = {
        "ckpt": str(args.ckpt),
        "n_candidates": args.n_candidates,
        "splits": {},
    }
    splits_for_plot = {}
    for sn, (sess, sd, st, rows) in splits.items():
        if args.max_rows_per_split:
            rows = rows[:args.max_rows_per_split]
        print(f"=== split {sn} ({sess}, n={len(rows)}) ===", flush=True)
        result = evaluate_split(
            model=model, session_dir=sd, rows=rows, offset=offset,
            stats=st, cache_root=cache_root, session_id=sess, device=device,
            autocast_dtype=autocast_dtype, n_candidates=args.n_candidates,
        )
        splits_for_plot[sn] = result
        # Strip raw arrays from JSON (they're saved into hist plots)
        result_lean = {k: v for k, v in result.items()
                       if k not in ("matched_psnrs_raw", "cross_pair_psnrs_raw", "per_frame")}
        # keep first 50 per_frame for sanity
        if "per_frame" in result and result.get("n", 0) > 0:
            result_lean["per_frame_sample"] = result["per_frame"][:50]
        out["splits"][sn] = result_lean
        print(f"  matched mean PSNR: {result.get('matched_psnr_mean')}")
        print(f"  cross-pair mean PSNR: {result.get('cross_pair_psnr_mean')}")
        print(f"  matched − cross-pair gap (dB): {result.get('matched_minus_cross_pair_db')}")
        print(f"  top1 retrieval (1-of-{args.n_candidates+1}): {result.get('top1_retrieval_rate')}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}", flush=True)
    save_histograms(args.out.parent / "histograms", splits_for_plot)


if __name__ == "__main__":
    main()

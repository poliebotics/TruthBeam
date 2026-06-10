"""Closure Task 4c: train-vs-val byte recovery for the exact-RGB model.

Loads the Task 4 model (trained on rendered RGB → XOF) and computes
per-octave matched score + byte exact match on:
  - TRAIN split [0, 4592)
  - VAL report split [5392, 5992)
Plus per-octave Window-FMR@95 + tau95 + top-1 on VAL.

If train→val gap small + val strong: generator is fine, optics is killer.
If train→val gap large: model memorized train chain rows.

Run:
  python scripts/closure/closure_task4c_train_eval.py \
    --ckpt experiments/closure_package/exact_rgb_xof/checkpoints/final_step.pt \
    --d2-dir <data> \
    --out experiments/closure_package/exact_rgb_xof/train_vs_val.json
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

torch.set_num_threads(8)
torch.set_num_interop_threads(2)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "closure"))

from closure_task4_exact_rgb import ExactRGBDataset, ExactRGBDecoder  # noqa: E402
from data.raw_bayer_dataset import load_chain_log  # noqa: E402
from data.packed_cfa_dataset import xof_octaves_centered_from_hex  # noqa: E402
from eval.candidate_ranking import (  # noqa: E402
    assemble_candidate_set, compute_tau95, per_family_window_fmr,
    score_candidate_xof,
)


SCORE_VARIANTS = ("oct0_only", "oct0_plus_oct1",
                  "oct0_plus_oct1_plus_oct2", "all_octaves")


def evaluate_byte_recovery(*, model, val_ds, device, autocast_dtype, n_eval=256, label=""):
    """Per-octave matched score + byte exact match on a dataset."""
    n = min(n_eval, len(val_ds))
    matched = {v: [] for v in SCORE_VARIANTS}
    bit_recovery = {i: [] for i in range(4)}
    matched_per_octave = {i: [] for i in range(4)}
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        for i in range(n):
            sample = val_ds[i]
            x = sample["rgb"].unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                preds = model(x)
            preds_cpu = [p.float().squeeze(0).cpu() if p is not None else None for p in preds]
            true_octs = [sample[f"xof_oct{j}"] for j in range(4)]
            for v in SCORE_VARIANTS:
                aligned = [true_octs[j] if preds_cpu[j] is not None else None for j in range(4)]
                matched[v].append(score_candidate_xof(preds_cpu, aligned, v))
            for j in range(4):
                if preds_cpu[j] is None:
                    continue
                pred_bytes = (preds_cpu[j] * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
                true_bytes = (true_octs[j] * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
                bit_recovery[j].append(((pred_bytes == true_bytes).float().mean()).item())
                # Per-octave matched score (just SmoothL1 of pred vs target for that octave)
                import torch.nn.functional as F
                matched_per_octave[j].append(F.smooth_l1_loss(preds_cpu[j], true_octs[j]).item())
            if (i + 1) % 50 == 0:
                el = time.time() - t0
                print(f"  [{label}] frame {i+1}/{n} elapsed={el:.0f}s", flush=True)
    return {
        "label": label,
        "n": n,
        "elapsed_sec": round(time.time() - t0, 1),
        "matched_score": {v: float(np.mean(matched[v])) for v in SCORE_VARIANTS},
        "byte_match_per_octave": {f"oct{j}": float(np.mean(bit_recovery[j])) if bit_recovery[j] else None for j in range(4)},
        "matched_per_octave": {f"oct{j}": float(np.mean(matched_per_octave[j])) if matched_per_octave[j] else None for j in range(4)},
    }


def evaluate_fmr_per_octave(*, model, val_ds, chain, device, autocast_dtype,
                            d2_chain_rows, n_eval=300):
    """Per-octave Window-FMR@95 + tau95 + top-1 on val."""
    n = min(n_eval, len(val_ds))
    matched = {v: [] for v in SCORE_VARIANTS}
    negatives = {v: [] for v in SCORE_VARIANTS}
    top1 = {v: 0 for v in SCORE_VARIANTS}
    octaves_cache: dict[int, list[torch.Tensor]] = {}

    def get_octs(row):
        cached = octaves_cache.get(row)
        if cached is not None:
            return cached
        if row not in chain:
            return None
        octs = list(xof_octaves_centered_from_hex(chain[row]))
        octaves_cache[row] = octs
        return octs

    t0 = time.time()
    model.eval()
    with torch.no_grad():
        for idx in range(n):
            sample = val_ds[idx]
            x = sample["rgb"].unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                preds = model(x)
            preds_cpu = [p.float().squeeze(0).cpu() if p is not None else None for p in preds]
            t = sample["t"]
            cs = assemble_candidate_set(
                capture_session="D2", capture_row=t, offset=1,
                same_session_chain_rows=d2_chain_rows,
                other_session_chain_rows=d2_chain_rows,
                other_session_id="D2_alt",
                d_search_max=32, n_same_session_random=192,
                n_cross_session_random=192, seed=42 ^ t,
            )
            per_var_neg = {v: {} for v in SCORE_VARIANTS}
            per_var_matched = {v: 0.0 for v in SCORE_VARIANTS}
            per_var_min_neg = {v: math.inf for v in SCORE_VARIANTS}
            for c in cs.candidates:
                octs = get_octs(c.row)
                if octs is None:
                    continue
                for v in SCORE_VARIANTS:
                    aligned = [octs[i] if preds_cpu[i] is not None else None for i in range(4)]
                    s = score_candidate_xof(preds_cpu, aligned, v)
                    if c.family == "matched":
                        per_var_matched[v] = s
                    else:
                        per_var_neg[v].setdefault(c.family, []).append(s)
                        if s < per_var_min_neg[v]:
                            per_var_min_neg[v] = s
            for v in SCORE_VARIANTS:
                matched[v].append(per_var_matched[v])
                negatives[v].append(per_var_neg[v])
                if per_var_matched[v] < per_var_min_neg[v]:
                    top1[v] += 1
            if (idx + 1) % 50 == 0:
                el = time.time() - t0
                print(f"  [val FMR] frame {idx+1}/{n} elapsed={el:.0f}s top1[all]={top1['all_octaves']/(idx+1):.3f}", flush=True)

    actual_n = len(matched[SCORE_VARIANTS[0]])
    tau95 = {v: compute_tau95(matched[v]) for v in SCORE_VARIANTS}
    fmr = {}
    for v in SCORE_VARIANTS:
        fmr[v] = per_family_window_fmr(
            matched_scores_by_frame=matched[v],
            negative_scores_by_frame_by_family=negatives[v],
            tau95=tau95[v],
        )
    return {
        "n_frames": actual_n,
        "tau95": tau95,
        "window_fmr": fmr,
        "top1": {v: (top1[v] / actual_n) for v in SCORE_VARIANTS},
        "matched_score_mean": {v: float(np.mean(matched[v])) for v in SCORE_VARIANTS},
        "elapsed_sec": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--d2-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--n-train", type=int, default=256)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--train-start", type=int, default=0)
    ap.add_argument("--train-end", type=int, default=4592)
    ap.add_argument("--val-report-start", type=int, default=5392)
    ap.add_argument("--val-report-end", type=int, default=5992)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16

    chain = load_chain_log(args.d2_dir / "chain_log.csv")
    train_ds = ExactRGBDataset(chain, list(range(args.train_start, args.train_end)))
    val_ds = ExactRGBDataset(chain, list(range(args.val_report_start, args.val_report_end)))
    print(f"[init] train={len(train_ds)} val_report={len(val_ds)}", flush=True)

    model = ExactRGBDecoder().to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ck["model"] if "model" in ck else ck
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    print(f"[ckpt] loaded {args.ckpt}", flush=True)

    print("=== TRAIN-half byte recovery ===", flush=True)
    train_res = evaluate_byte_recovery(
        model=model, val_ds=train_ds, device=device, autocast_dtype=autocast_dtype,
        n_eval=args.n_train, label="train")

    print("=== VAL-report byte recovery (sanity vs Task 4 final_report) ===", flush=True)
    val_res = evaluate_byte_recovery(
        model=model, val_ds=val_ds, device=device, autocast_dtype=autocast_dtype,
        n_eval=args.n_val, label="val_report")

    print("=== VAL-report Window-FMR per-octave ===", flush=True)
    d2_chain_rows = sorted([t for t in chain if (t + 1) in chain])
    val_fmr = evaluate_fmr_per_octave(
        model=model, val_ds=val_ds, chain=chain, device=device,
        autocast_dtype=autocast_dtype, d2_chain_rows=d2_chain_rows,
        n_eval=args.n_val)

    out = {
        "ckpt": str(args.ckpt),
        "train": train_res,
        "val_report": val_res,
        "val_report_fmr": val_fmr,
        "interpretation_help": {
            "matched_score_null_per_variant": {
                "oct0_only": 1/6, "oct0_plus_oct1": 2/6,
                "oct0_plus_oct1_plus_oct2": 2.5/6, "all_octaves": 2.75/6,
            },
            "byte_match_chance": 1/256,
            "train_val_byte_match_gap_oct0": train_res["byte_match_per_octave"]["oct0"] - val_res["byte_match_per_octave"]["oct0"],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[done] wrote {args.out}", flush=True)
    print(f"\n=== Summary ===", flush=True)
    print(f"  train byte recovery: oct0={train_res['byte_match_per_octave']['oct0']:.3f}  oct1={train_res['byte_match_per_octave']['oct1']:.3f}  oct2={train_res['byte_match_per_octave']['oct2']:.3f}  oct3={train_res['byte_match_per_octave']['oct3']:.3f}", flush=True)
    print(f"  val   byte recovery: oct0={val_res['byte_match_per_octave']['oct0']:.3f}  oct1={val_res['byte_match_per_octave']['oct1']:.3f}  oct2={val_res['byte_match_per_octave']['oct2']:.3f}  oct3={val_res['byte_match_per_octave']['oct3']:.3f}", flush=True)
    print(f"  train_val_gap (oct0): {out['interpretation_help']['train_val_byte_match_gap_oct0']:+.4f}", flush=True)
    print(f"  val FMR worst-family (all_octaves): {val_fmr['window_fmr']['all_octaves']['worst_family_value']:.3f}", flush=True)
    print(f"  val top1 (all_octaves): {val_fmr['top1']['all_octaves']:.3f}", flush=True)


if __name__ == "__main__":
    main()

"""Closure Task 4b: candidate ranking + Window-FMR@95 for the exact-RGB model.

Loads the Task 4 model (trained on rendered emission RGB → XOF) and runs
the same candidate-ranking eval as Task 1, so we can directly compare
the optical-channel result against the upper bound.

Run:
  python scripts/closure/closure_task4b_exact_rgb_fmr.py \
    --ckpt experiments/closure_package/exact_rgb_xof/checkpoints/final_step.pt \
    --d2-dir <data> --stats-dir <stats> \
    --out experiments/closure_package/exact_rgb_xof/fmr_report.json
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


def evaluate_fmr_split(*, model, val_ds, chain, device, autocast_dtype,
                      d2_chain_rows, n_eval=300, d_search_max=32,
                      n_same=192, n_cross=192, seed=42, verbose=True):
    """Note: V10 isn't in the exact-RGB pipeline; cross_session candidates
    sample from the SAME D2 pool but distant rows (acts as cross-session proxy
    since V10 chain isn't rendered here)."""
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
            target_chain_row = sample["target_chain_row"]
            cs = assemble_candidate_set(
                capture_session="D2", capture_row=t, offset=1,
                same_session_chain_rows=d2_chain_rows,
                # Use D2 rows for "other_session" too (same pool, but the
                # candidate-ranking still labels them differently). For this
                # test the family separation isn't critical — what matters is
                # the matched vs random comparison.
                other_session_chain_rows=d2_chain_rows,
                other_session_id="D2_alt",
                d_search_max=d_search_max,
                n_same_session_random=n_same,
                n_cross_session_random=n_cross,
                seed=42 ^ t,
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
            if verbose and ((idx + 1) <= 5 or (idx + 1) % 25 == 0):
                el = time.time() - t0
                print(f"  frame {idx+1}/{n} elapsed={el:.0f}s "
                      f"top1[all]={top1['all_octaves']/(idx+1):.3f}", flush=True)

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
        "matched_score_p5": {v: float(np.quantile(matched[v], 0.05)) for v in SCORE_VARIANTS},
        "matched_score_p95": {v: float(np.quantile(matched[v], 0.95)) for v in SCORE_VARIANTS},
        "elapsed_sec": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--d2-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--val-report-start", type=int, default=5392)
    ap.add_argument("--val-report-end", type=int, default=5992)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16

    chain = load_chain_log(args.d2_dir / "chain_log.csv")
    val_ds = ExactRGBDataset(chain, list(range(args.val_report_start, args.val_report_end)))
    print(f"[init] val_report n={len(val_ds)}", flush=True)

    model = ExactRGBDecoder().to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ck["model"] if "model" in ck else ck
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    print(f"[ckpt] loaded {args.ckpt}", flush=True)

    # All chain rows that have valid emission renderings (every row except last)
    d2_chain_rows = sorted([t for t in chain if (t + 1) in chain])

    result = evaluate_fmr_split(
        model=model, val_ds=val_ds, chain=chain, device=device,
        autocast_dtype=autocast_dtype, d2_chain_rows=d2_chain_rows,
        n_eval=args.n_eval,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    for v in SCORE_VARIANTS:
        print(f"variant={v} tau95={result['tau95'][v]:.4f} "
              f"FMR={result['window_fmr'][v]['worst_family_value']:.3f} "
              f"top1={result['top1'][v]:.3f} "
              f"matched_mean={result['matched_score_mean'][v]:.4f}", flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

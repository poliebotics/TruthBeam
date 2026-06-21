"""Closure tasks 1 & 2: single-process XOF eval with mode swaps.

Modes:
    trained — load final_step.pt and run report-half eval (Task 1)
    zero    — replace forward with constant-zero in centered byte space
    oracle  — replace forward with exact target XOF (matched candidate)
              optionally with Gaussian noise std (--noise-std)

Output schema (per --mode):
    Task 1 (trained):  final_report.json with {split: {tau95, window_fmr, top1, n_frames}}
                        per split (d2_val_report, d2_val_calib for context, plus v10
                        splits where applicable). tau95 source: val first half (calib).
    Task 2 (zero/oracle): {split: {tau95, window_fmr, top1, n_frames}} where tau95 is
                        computed from this run's own positives.

Run:
    python scripts/closure/closure_xof_eval.py --config configs/exp001h_a1.yaml \
        --ckpt experiments/exp001h_a1/checkpoints/final_step.pt \
        --stats-dir cache/normalization_stats --mode trained --bf16 \
        --out experiments/closure_package/exp001h_a1/final_report.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

# Limit intra-op threads — multiple closure processes share the host.
# Without this they each grab all cores and stall.
import os as _os
_n_threads = int(_os.environ.get("CLOSURE_TORCH_THREADS", "8"))
torch.set_num_threads(_n_threads)
torch.set_num_interop_threads(2)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.packed_cfa_dataset import PackedCFADataset, xof_octaves_centered_from_hex  # noqa: E402
from data.raw_bayer_dataset import load_chain_log  # noqa: E402
from eval.candidate_ranking import (  # noqa: E402
    NEAR_SHIFT_OFFSETS,
    assemble_candidate_set,
    compute_tau95,
    per_family_window_fmr,
    score_candidate_xof,
    score_variants_for_experiment,
)
from models.xof_decoder_v2 import XOFDecoderV2  # noqa: E402
from preprocessing.normalization import load_stats  # noqa: E402
from utils.config_loader import load_and_validate  # noqa: E402


def _v10_dir(cfg) -> Path:
    return Path(cfg["data"].get("v10_dir", str(Path(cfg["data"]["d2_dir"]).parent / "v10")))


def _strict_d2_only(cfg) -> bool:
    return bool(cfg.get("normalization", {}).get("strict_d2_only", False))


def _build_dataset(cfg, session: str, rows: list[int], stats: dict) -> PackedCFADataset:
    session_dir = Path(cfg["data"]["d2_dir"]) if session == "D2" else _v10_dir(cfg)
    return PackedCFADataset(
        session_dir=session_dir,
        rows=rows,
        offset=int(cfg["offset"]["value"]),
        normalization_stats=stats,
        cache_root=Path(cfg["data"]["cache_root"]),
        session_id=session,
        black_level=int(cfg["data"].get("black_level", 0)),
        with_xof=True,
        with_emission=False,
    )


def _chain_rows_with_emission(session_dir: Path) -> list[int]:
    chain = load_chain_log(session_dir / "chain_log.csv")
    emi_dir = session_dir / "derived" / "Emissions"
    out = []
    for t in chain:
        if (emi_dir / f"tile_{t:06d}.png").exists():
            out.append(t)
    return out


def build_model_for_eval(cfg, device):
    return XOFDecoderV2(
        encoder_size=cfg["model"].get("encoder_size", "tiny"),
        pretrained=False,
        fpn_out_channels=int(cfg["model"].get("fpn_out_channels", 256)),
        head_hidden=int(cfg["model"].get("head_hidden", 128)),
        enabled_octaves=tuple(cfg["model"].get("enabled_octaves", (0, 1, 2, 3))),
        use_stn=bool(cfg["model"].get("use_stn", False)),
    ).to(device)


def load_ckpt(model, ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck["model"] if "model" in ck else ck
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    return {"missing": len(missing), "unexpected": len(unexpected),
            "missing_keys": missing[:5], "unexpected_keys": unexpected[:5]}


def predict(mode, model, cap, sample, autocast_dtype, noise_std, gen):
    """Returns list of 4 (octave) tensors on CPU, fp32, shape (3, H, W)."""
    enabled_octaves = tuple(sample.get("enabled_octaves", (0, 1, 2, 3)))
    if mode == "trained":
        with torch.amp.autocast("cuda", dtype=autocast_dtype):
            preds, _info = model(cap)
        return [p.float().squeeze(0).cpu() if p is not None else None for p in preds]
    if mode == "zero":
        # constant zero in centered byte space, only for enabled octaves
        out = []
        for i in range(4):
            if i not in enabled_octaves:
                out.append(None); continue
            t = sample[f"xof_oct{i}"]
            if isinstance(t, torch.Tensor):
                shape = t.shape
            else:
                shape = (3, 0, 0)
            out.append(torch.zeros(*shape, dtype=torch.float32))
        return out
    if mode == "oracle":
        out = []
        for i in range(4):
            if i not in enabled_octaves:
                out.append(None); continue
            t = sample[f"xof_oct{i}"].clone().float()
            if noise_std > 0:
                t = t + noise_std * torch.randn(t.shape, generator=gen, dtype=torch.float32)
            out.append(t)
        return out
    raise ValueError(f"unknown mode {mode!r}")


def run_eval_split(*, model, dataset, cfg, device, other_session_chain_rows,
                   other_session_id, same_session_chain, other_session_chain,
                   d2_chain, v10_chain,
                   score_variants, n_eval_frames, autocast_dtype, mode, noise_std,
                   verbose=True, max_eval_seconds=None):
    """Single-process candidate-ranking eval over one dataset split.

    Returns matched_scores_per_variant, negatives_per_variant, top1_per_variant,
    n_frames, tau95_per_variant, candidate_log (lightweight).
    """
    n = min(n_eval_frames, len(dataset))
    indices = list(range(n))
    matched_scores = {v: [] for v in score_variants}
    negatives = {v: [] for v in score_variants}
    top1_correct = {v: 0 for v in score_variants}
    candidate_count = []
    same_chain_rows = list(same_session_chain.keys())

    enabled_octaves = tuple(cfg["model"].get("enabled_octaves", (0, 1, 2, 3)))
    gen = torch.Generator().manual_seed(int(cfg.get("seed", 42)))

    # Memoize XOF octaves per (session, row) — blake3.digest(43110) is the
    # bottleneck if recomputed per candidate per frame. Cache across the full
    # eval so each row's centered bytes are computed exactly once.
    octaves_cache: dict[tuple[str, int], list[torch.Tensor]] = {}

    def get_octaves(session: str, row: int):
        key = (session, row)
        cached = octaves_cache.get(key)
        if cached is not None:
            return cached
        chain_for_session = d2_chain if session == "D2" else v10_chain
        hex_str = chain_for_session.get(row)
        if hex_str is None:
            return None
        octs = list(xof_octaves_centered_from_hex(hex_str))
        octaves_cache[key] = octs
        return octs

    t0 = time.time()
    aborted_early = False
    for idx in indices:
        sample = dataset[idx]
        sample["enabled_octaves"] = enabled_octaves
        cap = sample["capture_norm"].unsqueeze(0).to(device)
        preds_cpu = predict(mode, model, cap, sample, autocast_dtype, noise_std, gen)

        cs = assemble_candidate_set(
            capture_session=sample["session_id"],
            capture_row=int(sample["t"]),
            offset=int(cfg["offset"]["value"]),
            same_session_chain_rows=same_chain_rows,
            other_session_chain_rows=other_session_chain_rows,
            other_session_id=other_session_id,
            d_search_max=int(cfg["candidate_ranking"]["d_search_max"]),
            n_same_session_random=int(cfg["candidate_ranking"].get("n_same_session_random", 192)),
            n_cross_session_random=int(cfg["candidate_ranking"].get("n_cross_session_random", 192)),
            seed=int(cfg.get("seed", 42)) ^ int(sample["t"]),
        )
        candidate_count.append(len(cs.candidates))

        per_var_neg = {v: {} for v in score_variants}
        per_var_matched = {v: 0.0 for v in score_variants}
        per_var_min_neg = {v: math.inf for v in score_variants}
        for c in cs.candidates:
            t_octs = get_octaves(c.session, c.row)
            if t_octs is None:
                continue
            for v in score_variants:
                aligned = []
                for i in range(4):
                    if preds_cpu[i] is None or t_octs[i] is None:
                        aligned.append(None)
                    else:
                        aligned.append(t_octs[i])
                s = score_candidate_xof(preds_cpu, aligned, v)
                if c.family == "matched":
                    per_var_matched[v] = s
                else:
                    per_var_neg[v].setdefault(c.family, []).append(s)
                    if s < per_var_min_neg[v]:
                        per_var_min_neg[v] = s
        for v in score_variants:
            matched_scores[v].append(per_var_matched[v])
            negatives[v].append(per_var_neg[v])
            if per_var_matched[v] < per_var_min_neg[v]:
                top1_correct[v] += 1
        if verbose and ((idx + 1) <= 5 or (idx + 1) % 25 == 0):
            elapsed = time.time() - t0
            print(f"  [{mode}] frame {idx+1}/{n} elapsed={elapsed:.0f}s "
                  f"avg_cands={np.mean(candidate_count):.0f} "
                  f"cache={len(octaves_cache)} "
                  f"top1[{score_variants[-1]}]={top1_correct[score_variants[-1]]/(idx+1):.3f}",
                  flush=True)
        if max_eval_seconds is not None and (time.time() - t0) > max_eval_seconds:
            print(f"  [WARN] max_eval_seconds={max_eval_seconds} hit at frame {idx+1}/{n}; aborting split", flush=True)
            aborted_early = True
            break

    actual_n = len(matched_scores[score_variants[0]])
    tau95 = {v: compute_tau95(matched_scores[v]) for v in score_variants}
    fmr = {}
    for v in score_variants:
        fmr[v] = per_family_window_fmr(
            matched_scores_by_frame=matched_scores[v],
            negative_scores_by_frame_by_family=negatives[v],
            tau95=tau95[v],
        )
    return {
        "n_frames": actual_n,
        "n_requested": n,
        "aborted_early": aborted_early,
        "elapsed_sec": round(time.time() - t0, 1),
        "matched_scores_per_variant": matched_scores,
        "negatives_per_variant": negatives,
        "tau95": tau95,
        "window_fmr": fmr,
        "top1": {v: (top1_correct[v] / actual_n if actual_n > 0 else float("nan")) for v in score_variants},
        "matched_mean": {v: float(np.mean(matched_scores[v])) if actual_n > 0 else float("nan") for v in score_variants},
        "matched_median": {v: float(np.median(matched_scores[v])) if actual_n > 0 else float("nan") for v in score_variants},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="Required for --mode trained; ignored for zero/oracle.")
    ap.add_argument("--stats-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--mode", choices=["trained", "zero", "oracle"], required=True)
    ap.add_argument("--noise-std", type=float, default=0.0,
                    help="Oracle Gaussian noise std (centered-byte space; 0 = noiseless)")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--max-frames", type=int, default=600,
                    help="Eval up to this many frames per split.")
    ap.add_argument("--max-eval-seconds-per-split", type=int, default=1200,
                    help="Per-split soft wall-clock cap (default 20 min).")
    ap.add_argument("--splits", type=str, default="all",
                    help="Comma-separated split names or 'all'/'report'/'calib'.")
    args = ap.parse_args()

    cfg = load_and_validate(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16
    is_a6 = _strict_d2_only(cfg)

    # Datasets
    d2_stats = load_stats(args.stats_dir / "d2_train_stats.json")
    v10_stats = d2_stats if is_a6 else load_stats(args.stats_dir / "v10_train_stats.json")
    splits_dict: dict = {
        "d2_val_calib":  ("D2", d2_stats,
                           list(range(int(cfg["data"]["d2_val_calib_start"]),
                                       int(cfg["data"]["d2_val_calib_end"])))),
        "d2_val_report": ("D2", d2_stats,
                           list(range(int(cfg["data"]["d2_val_report_start"]),
                                       int(cfg["data"]["d2_val_report_end"])))),
    }
    if is_a6:
        for nm, ka, kb in [("v10_val_report", "v10_val_report_start", "v10_val_report_end"),
                           ("v10_early", "v10_early_start", "v10_early_end"),
                           ("v10_mid", "v10_mid_start", "v10_mid_end"),
                           ("v10_late", "v10_late_start", "v10_late_end")]:
            splits_dict[nm] = ("V10", v10_stats,
                                list(range(int(cfg["data"][ka]), int(cfg["data"][kb]))))
    else:
        splits_dict["v10_val_calib"] = ("V10", v10_stats,
            list(range(int(cfg["data"]["v10_val_calib_start"]),
                        int(cfg["data"]["v10_val_calib_end"]))))
        splits_dict["v10_val_report"] = ("V10", v10_stats,
            list(range(int(cfg["data"]["v10_val_report_start"]),
                        int(cfg["data"]["v10_val_report_end"]))))

    if args.splits == "report":
        keep = ["d2_val_report", "v10_val_report"]
        if is_a6:
            keep += ["v10_early", "v10_mid", "v10_late"]
        splits_dict = {k: v for k, v in splits_dict.items() if k in keep}
    elif args.splits == "calib":
        splits_dict = {k: v for k, v in splits_dict.items() if k.endswith("_calib")}
    elif args.splits != "all":
        keep = set(args.splits.split(","))
        splits_dict = {k: v for k, v in splits_dict.items() if k in keep}

    # Chain logs
    d2_chain = load_chain_log(Path(cfg["data"]["d2_dir"]) / "chain_log.csv")
    v10_chain = load_chain_log(_v10_dir(cfg) / "chain_log.csv")
    d2_chain_rows = _chain_rows_with_emission(Path(cfg["data"]["d2_dir"]))
    v10_chain_rows = _chain_rows_with_emission(_v10_dir(cfg))

    # Model
    model = build_model_for_eval(cfg, device)
    if args.mode == "trained":
        if not args.ckpt:
            sys.exit("--ckpt required for --mode trained")
        info = load_ckpt(model, args.ckpt)
        print(f"[ckpt] loaded {args.ckpt}: missing={info['missing']} unexpected={info['unexpected']}", flush=True)
    model.eval()

    score_variants = score_variants_for_experiment(args.config.stem)

    out: dict = {
        "mode": args.mode,
        "noise_std": args.noise_std,
        "config": str(args.config),
        "ckpt": str(args.ckpt) if args.ckpt else None,
        "splits": {},
    }
    for split_name, (sess, stats, rows) in splits_dict.items():
        ds = _build_dataset(cfg, sess, rows, stats)
        print(f"=== split {split_name} ({sess}, n_dataset={len(ds)}, n_eval={min(args.max_frames, len(ds))}) ===", flush=True)
        if len(ds) == 0:
            out["splits"][split_name] = {"n_frames": 0, "skipped": "empty"}
            continue
        result = run_eval_split(
            model=model, dataset=ds, cfg=cfg, device=device,
            other_session_chain_rows=v10_chain_rows if sess == "D2" else d2_chain_rows,
            other_session_id="V10" if sess == "D2" else "D2",
            same_session_chain=d2_chain if sess == "D2" else v10_chain,
            other_session_chain=v10_chain if sess == "D2" else d2_chain,
            d2_chain=d2_chain, v10_chain=v10_chain,
            score_variants=score_variants,
            n_eval_frames=args.max_frames,
            autocast_dtype=autocast_dtype, mode=args.mode, noise_std=args.noise_std,
            max_eval_seconds=args.max_eval_seconds_per_split,
        )
        # Drop heavy fields from output (keep summaries)
        result_out = {k: v for k, v in result.items()
                      if k not in ("matched_scores_per_variant", "negatives_per_variant")}
        # Save per-frame matched scores compactly (helps interpret distributions)
        result_out["matched_scores_summary"] = {
            v: {
                "min": float(np.min(result["matched_scores_per_variant"][v])) if result["n_frames"] > 0 else float("nan"),
                "p5": float(np.quantile(result["matched_scores_per_variant"][v], 0.05)) if result["n_frames"] > 0 else float("nan"),
                "median": float(np.median(result["matched_scores_per_variant"][v])) if result["n_frames"] > 0 else float("nan"),
                "p95": float(np.quantile(result["matched_scores_per_variant"][v], 0.95)) if result["n_frames"] > 0 else float("nan"),
                "max": float(np.max(result["matched_scores_per_variant"][v])) if result["n_frames"] > 0 else float("nan"),
                "mean": float(np.mean(result["matched_scores_per_variant"][v])) if result["n_frames"] > 0 else float("nan"),
            } for v in score_variants
        }
        out["splits"][split_name] = result_out
        for v in score_variants:
            print(f"  variant={v} tau95={result['tau95'][v]:.4f} "
                  f"worst_FMR={result['window_fmr'][v]['worst_family_value']:.3f} "
                  f"top1={result['top1'][v]:.3f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

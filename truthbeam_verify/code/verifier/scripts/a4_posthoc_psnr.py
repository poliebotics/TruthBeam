"""Post-hoc held-out PSNR + cross-pair PSNR for A4 (Tiny + emission).

A4's training loop didn't run periodic emission validation (the trainer's
val cadence is XOF-only). This script loads the final_step.pt checkpoint and
computes:

  - matched PSNR per row (D2-val + V10-val + V10 bins early/mid/late)
  - cross-pair PSNR (each prediction vs a random non-matching emission tile)
  - per-channel L1
  - stratified V10 PSNR by time bin

Output: <out_dir>/post_hoc_psnr.json with per-split summaries.

Run on the A100 (or any GPU with ≥10 GB):
  python scripts/a4_posthoc_psnr.py \
    --ckpt /path/to/poliebotics_phase_b/experiments/exp001h_a4/checkpoints/final_step.pt \
    --config configs/exp001h_a4.yaml \
    --stats-dir /path/to/poliebotics_phase_b/cache/normalization_stats \
    --out /path/to/poliebotics_phase_b/experiments/exp001h_a4/post_hoc_psnr.json
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.packed_cfa_dataset import PackedCFADataset  # noqa: E402
from data.raw_bayer_dataset import load_chain_log  # noqa: E402
from models.emission_predictor_v2 import EmissionPredictorV2  # noqa: E402
from preprocessing.normalization import load_stats  # noqa: E402
from utils.config_loader import load_and_validate  # noqa: E402


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR in dB (inputs in [0, 1])."""
    mse = ((pred - target) ** 2).mean().item()
    if mse < 1e-12:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def per_channel_l1(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    diff = (pred - target).abs()
    return {f"l1_{c}": diff[:, i].mean().item() for i, c in enumerate("rgb")}


def evaluate_rows(
    *,
    model: torch.nn.Module,
    session_dir: Path,
    rows: list[int],
    offset: int,
    stats: dict,
    cache_root: Path,
    session_id: str,
    device: torch.device,
    n_cross_pair: int = 10,
    seed: int = 0,
) -> dict:
    """Evaluate matched + cross-pair PSNR over `rows`."""
    ds = PackedCFADataset(
        session_dir=session_dir,
        rows=rows,
        offset=offset,
        normalization_stats=stats,
        cache_root=cache_root,
        session_id=session_id,
        with_xof=False,
        with_emission=True,
    )
    if len(ds) == 0:
        return {"n": 0, "error": "no rows after dataset filtering"}

    matched_psnrs: list[float] = []
    matched_l1: list[float] = []
    per_row_records: list[dict] = []

    # Cache predictions + emissions in CPU memory for cross-pair scoring
    preds_cpu: list[torch.Tensor] = []
    targets_cpu: list[torch.Tensor] = []
    rows_actual: list[int] = []

    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for i in range(len(ds)):
            sample = ds[i]
            cap = sample["capture_norm"].unsqueeze(0).to(device)
            target = sample["emission"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(cap).float()
            pred = pred.squeeze(0).clamp(0, 1)
            target_clamped = target.clamp(0, 1)
            p = psnr(pred, target_clamped)
            matched_psnrs.append(p)
            matched_l1.append((pred - target_clamped).abs().mean().item())
            preds_cpu.append(pred.cpu())
            targets_cpu.append(target_clamped.cpu())
            rows_actual.append(sample["t"])
            per_row_records.append({"row": int(sample["t"]), "psnr": p})
            if (i + 1) % 50 == 0:
                print(f"  matched eval: {i+1}/{len(ds)} (avg psnr {sum(matched_psnrs)/len(matched_psnrs):.2f})", flush=True)

    elapsed = time.time() - t0

    # Cross-pair: for each prediction, score against `n_cross_pair` random
    # NON-MATCHING targets sampled deterministically from this same split.
    rng = torch.Generator().manual_seed(seed)
    cross_pair_psnrs: list[float] = []
    n = len(preds_cpu)
    for i in range(n):
        for k in range(n_cross_pair):
            j = int(torch.randint(0, n, (1,), generator=rng).item())
            attempts = 0
            while j == i and attempts < 20:
                j = int(torch.randint(0, n, (1,), generator=rng).item())
                attempts += 1
            if j == i:
                continue
            cross_pair_psnrs.append(psnr(preds_cpu[i], targets_cpu[j]))

    return {
        "n": n,
        "matched_psnr_mean": sum(matched_psnrs) / n,
        "matched_psnr_min": min(matched_psnrs),
        "matched_psnr_max": max(matched_psnrs),
        "matched_psnr_std": float(torch.tensor(matched_psnrs).std().item()),
        "matched_l1_mean": sum(matched_l1) / n,
        "cross_pair_psnr_mean": (sum(cross_pair_psnrs) / len(cross_pair_psnrs)) if cross_pair_psnrs else float("nan"),
        "cross_pair_psnr_std": float(torch.tensor(cross_pair_psnrs).std().item()) if cross_pair_psnrs else float("nan"),
        "n_cross_pair_samples": len(cross_pair_psnrs),
        "matched_minus_cross_pair_db": (sum(matched_psnrs) / n) - ((sum(cross_pair_psnrs) / len(cross_pair_psnrs)) if cross_pair_psnrs else 0.0),
        "elapsed_sec": round(elapsed, 1),
        "per_row_psnr": per_row_records,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--stats-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-rows-per-split", type=int, default=None,
                    help="Cap rows per split (debug; default: all rows).")
    args = ap.parse_args()

    cfg = load_and_validate(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model + load checkpoint
    model = EmissionPredictorV2(
        emission_h=int(cfg["data"].get("emission_h", 1080)),
        emission_w=int(cfg["data"].get("emission_w", 1920)),
        encoder_size=cfg["model"].get("encoder_size", "tiny"),
        pretrained=False,
        fpn_out_channels=int(cfg["model"].get("fpn_out_channels", 256)),
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    # Strip "module." prefix from DDP state dict if present
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[ckpt] loaded {args.ckpt}: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:
        print(f"[ckpt] missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"[ckpt] unexpected keys (first 5): {unexpected[:5]}")

    cache_root = Path(cfg["data"]["cache_root"])
    d2_dir = Path(cfg["data"]["d2_dir"])
    v10_dir = Path(cfg["data"].get("v10_dir", str(d2_dir.parent / "v10")))
    d2_stats = load_stats(args.stats_dir / "d2_train_stats.json")
    v10_stats = load_stats(args.stats_dir / "v10_train_stats.json")
    offset = int(cfg["offset"]["value"])

    splits = {
        "D2_val_full":     ("D2",  d2_dir, d2_stats, list(range(int(cfg["data"]["d2_val_start"]), int(cfg["data"]["d2_val_end"])))),
        "D2_val_calib":    ("D2",  d2_dir, d2_stats, list(range(int(cfg["data"]["d2_val_calib_start"]), int(cfg["data"]["d2_val_calib_end"])))),
        "D2_val_report":   ("D2",  d2_dir, d2_stats, list(range(int(cfg["data"]["d2_val_report_start"]), int(cfg["data"]["d2_val_report_end"])))),
        "V10_val_full":    ("V10", v10_dir, v10_stats, list(range(int(cfg["data"]["v10_val_start"]), int(cfg["data"]["v10_val_end"])))),
        "V10_val_calib":   ("V10", v10_dir, v10_stats, list(range(int(cfg["data"]["v10_val_calib_start"]), int(cfg["data"]["v10_val_calib_end"])))),
        "V10_val_report":  ("V10", v10_dir, v10_stats, list(range(int(cfg["data"]["v10_val_report_start"]), int(cfg["data"]["v10_val_report_end"])))),
        "V10_early":       ("V10", v10_dir, v10_stats, list(range(int(cfg["data"]["v10_early_start"]), int(cfg["data"]["v10_early_end"])))),
        "V10_mid":         ("V10", v10_dir, v10_stats, list(range(int(cfg["data"]["v10_mid_start"]), int(cfg["data"]["v10_mid_end"])))),
        "V10_late":        ("V10", v10_dir, v10_stats, list(range(int(cfg["data"]["v10_late_start"]), int(cfg["data"]["v10_late_end"])))),
    }

    out: dict = {
        "ckpt": str(args.ckpt),
        "config": str(args.config),
        "offset": offset,
        "splits": {},
    }
    for split_name, (sess, sess_dir, stats, rows) in splits.items():
        if args.max_rows_per_split:
            rows = rows[:args.max_rows_per_split]
        print(f"=== split {split_name} ({sess}, n_requested={len(rows)}) ===", flush=True)
        result = evaluate_rows(
            model=model,
            session_dir=sess_dir,
            rows=rows,
            offset=offset,
            stats=stats,
            cache_root=cache_root,
            session_id=sess,
            device=device,
        )
        # Don't dump full per_row_psnr for huge splits — keep just first 50 rows
        if "per_row_psnr" in result and len(result["per_row_psnr"]) > 50:
            result["per_row_psnr_sample"] = result["per_row_psnr"][:50]
            del result["per_row_psnr"]
        out["splits"][split_name] = result
        print(f"  matched mean PSNR: {result.get('matched_psnr_mean', 'n/a')}")
        print(f"  cross-pair mean PSNR: {result.get('cross_pair_psnr_mean', 'n/a')}")
        print(f"  matched − cross-pair gap (dB): {result.get('matched_minus_cross_pair_db', 'n/a')}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

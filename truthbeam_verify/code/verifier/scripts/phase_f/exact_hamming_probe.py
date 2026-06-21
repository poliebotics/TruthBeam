"""Phase F #12 — exact-Hamming chain-byte probe.

Variant of chain_byte_probe: instead of probabilistic byte replacement,
flip EXACTLY k random BITS in the 3 × 43,110-byte XOF expanded streams.
Measure binder discrimination at each k ∈ {0, 1, 2, 4, 8, 16, 32, 64, 128}.

Output: per-binder threshold k@5% attack-success + per-row score arrays.
Cross-binder score correlation matrix.

Run on 5 representative binders:
  exp001c (canonical), E1 (U-Net), E3r (FPN), E2 (high-res), E1-fp16-sg

Run:
  python scripts/phase_f/exact_hamming_probe.py \
    --binders exp001c E1 E3r E2 E1_fp16_sg \
    --n-rows 100 --out experiments/phase_f_prep/exact_hamming
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "phase_e"))

from data.emission_dataset import EmissionDataset, load_capture_at, load_emission_at  # noqa: E402
from data.packed_cfa_dataset import PackedCFADataset  # noqa: E402
from data.raw_bayer_dataset import load_chain_log  # noqa: E402
from models.emission_predictor import EmissionPredictor  # noqa: E402
from preprocessing.normalization import load_stats  # noqa: E402
try:
    from models.emission_predictor_v2 import EmissionPredictorV2
except Exception:
    EmissionPredictorV2 = None
from bitexact_renderer import (  # noqa: E402
    expand_seeds_to_streams, render_from_streams, TOTAL_BYTES_PER_CHANNEL,
)


HAMMING_K_VALUES = (0, 1, 2, 4, 8, 16, 32, 64, 128)


# 5 representative binder specs
BINDER_SPECS = {
    "exp001c": {
        "ckpt": "/path/to/poliebotics_phase_b/experiments/exp001c/checkpoints/ep027.pt",
        "arch": "EmissionPredictor", "capture_h": 1150, "capture_w": 1330,
    },
    "E1": {
        "ckpt": "/path/to/poliebotics_phase_b/experiments/phase_e/e1/checkpoints/best_by_psnr.pt",
        "arch": "EmissionPredictor", "capture_h": 1150, "capture_w": 1330,
    },
    "E3r": {
        "ckpt": "/path/to/poliebotics_phase_b/experiments/phase_e/e3r/checkpoints/best_by_psnr.pt",
        "arch": "EmissionPredictorV2", "capture_h": 1150, "capture_w": 1330,
    },
    "E2": {
        "ckpt": "/path/to/poliebotics_phase_b/experiments/phase_e/e2/checkpoints/best_by_psnr.pt",
        "arch": "EmissionPredictor", "capture_h": 2300, "capture_w": 2660,
    },
    "E1_fp16_sg": {
        "ckpt": "/path/to/poliebotics_phase_b/experiments/phase_e/e1_fp16_sg/checkpoints/best_by_psnr.pt",
        "arch": "EmissionPredictor", "capture_h": 1150, "capture_w": 1330,
    },
}


def psnr01(p: torch.Tensor, t: torch.Tensor) -> float:
    mse = ((p - t) ** 2).mean().item()
    return float("inf") if mse < 1e-12 else 20.0 * math.log10(1.0 / math.sqrt(mse))


def flip_exact_k_bits(streams: tuple[bytes, bytes, bytes], k: int, seed: int) -> tuple[bytes, bytes, bytes]:
    """Flip EXACTLY k bits at random positions across the 3 channel streams."""
    if k == 0:
        return streams
    rng = np.random.default_rng(seed)
    total_bits = 3 * TOTAL_BYTES_PER_CHANNEL * 8
    bit_positions = rng.choice(total_bits, size=k, replace=False)
    arrs = [np.frombuffer(s, dtype=np.uint8).copy() for s in streams]
    for bp in bit_positions:
        ch = bp // (TOTAL_BYTES_PER_CHANNEL * 8)
        within = bp % (TOTAL_BYTES_PER_CHANNEL * 8)
        byte_idx = within // 8
        bit_idx = within % 8
        arrs[ch][byte_idx] ^= (1 << bit_idx)
    return tuple(a.tobytes() for a in arrs)


def load_binder(name: str, spec: dict, device: torch.device):
    if spec["arch"] == "EmissionPredictor":
        m = EmissionPredictor(emission_h=1080, emission_w=1920, pretrained=False).to(device)
    elif spec["arch"] == "EmissionPredictorV2":
        m = EmissionPredictorV2(emission_h=1080, emission_w=1920, pretrained=False).to(device)
    else:
        raise ValueError(spec["arch"])
    ck = torch.load(spec["ckpt"], map_location=device, weights_only=False)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    m.load_state_dict(state, strict=False)
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binders", nargs="+", default=list(BINDER_SPECS.keys()))
    ap.add_argument("--d2-dir", type=Path, default=Path("/path/to/poliebotics_phase_b/poliebotics_phase_b/data/d2"))
    ap.add_argument("--n-rows", type=int, default=100)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16

    chain = load_chain_log(args.d2_dir / "chain_log.csv")
    rng = np.random.RandomState(7)
    val_rows = list(range(5394, 5992))
    rows = sorted(rng.choice(val_rows, size=min(args.n_rows, len(val_rows)), replace=False).tolist())
    print(f"[init] sampled {len(rows)} rows", flush=True)

    # Pre-render: for each row, get clean streams + clean emission (= rendered emission for matched bytes)
    rows_data: list[dict] = []
    for t in rows:
        if t not in chain:
            continue
        streams = expand_seeds_to_streams(bytes.fromhex(chain[t]))
        clean_em_uint8 = render_from_streams(*streams, device="cpu")
        rows_data.append({
            "t": t, "streams": streams,
            "clean_em": clean_em_uint8.float().div(255.0),
        })
    print(f"[init] precomputed {len(rows_data)} clean emissions", flush=True)

    per_binder: dict[str, dict] = {}
    binder_score_arrays: dict[str, dict[int, list[float]]] = {b: {k: [] for k in HAMMING_K_VALUES} for b in args.binders}

    for binder_name in args.binders:
        if binder_name not in BINDER_SPECS:
            print(f"[skip] unknown binder {binder_name}", flush=True)
            continue
        spec = BINDER_SPECS[binder_name]
        print(f"\n=== binder: {binder_name} ===", flush=True)
        binder = load_binder(binder_name, spec, device)

        # Predict on real captures
        ds_capture_h = spec["capture_h"]; ds_capture_w = spec["capture_w"]
        d2_ds = EmissionDataset(
            session_dir=args.d2_dir, row_start=min(r["t"] for r in rows_data),
            row_end=max(r["t"] for r in rows_data) + 1,
            capture_h=ds_capture_h, capture_w=ds_capture_w,
            emission_h=1080, emission_w=1920, session_id="D2", augment=False,
        )

        per_k_attack: dict[int, list[bool]] = {k: [] for k in HAMMING_K_VALUES}
        per_k_margin: dict[int, list[float]] = {k: [] for k in HAMMING_K_VALUES}
        t0 = time.time()
        for ri, rd in enumerate(rows_data):
            t = rd["t"]
            if t not in d2_ds.rows:
                continue
            sample = d2_ds[d2_ds.rows.index(t)]
            cap = sample["capture"].unsqueeze(0).to(device)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=autocast_dtype):
                pred = binder(cap).float().clamp(0, 1).squeeze(0).cpu()
            matched_psnr = psnr01(pred, rd["clean_em"])
            binder_score_arrays[binder_name][0].append(matched_psnr)
            for k in HAMMING_K_VALUES:
                if k == 0:
                    # k=0 sanity: noised == matched, attack succeeds (identity)
                    per_k_attack[0].append(True)
                    per_k_margin[0].append(0.0)
                    continue
                for s in range(args.n_seeds):
                    seed = (k * 7919) ^ s ^ t
                    streams_noised = flip_exact_k_bits(rd["streams"], k, seed)
                    em_noised = render_from_streams(*streams_noised, device="cpu").float().div(255.0)
                    noised_psnr = psnr01(pred, em_noised)
                    margin = matched_psnr - noised_psnr
                    per_k_attack[k].append(noised_psnr >= matched_psnr - 1e-6)
                    per_k_margin[k].append(margin)
                    binder_score_arrays[binder_name][k].append(noised_psnr)
            if (ri + 1) % 25 == 0:
                el = time.time() - t0
                print(f"  row {ri+1}/{len(rows_data)} elapsed={el:.0f}s", flush=True)

        # Aggregate
        per_k_summary = {}
        for k in HAMMING_K_VALUES:
            attack_rate = sum(per_k_attack[k]) / max(len(per_k_attack[k]), 1)
            per_k_summary[k] = {
                "n_evaluations": len(per_k_attack[k]),
                "attack_success_rate": attack_rate,
                "margin_mean": float(np.mean(per_k_margin[k])) if per_k_margin[k] else 0.0,
                "margin_p5":  float(np.percentile(per_k_margin[k], 5)) if per_k_margin[k] else 0.0,
                "margin_p95": float(np.percentile(per_k_margin[k], 95)) if per_k_margin[k] else 0.0,
            }
        # Threshold k @ 5%: smallest k where attack < 5%
        threshold_k = None
        for k in sorted(HAMMING_K_VALUES):
            if k == 0: continue
            if per_k_summary[k]["attack_success_rate"] < 0.05:
                threshold_k = k
                break
        per_binder[binder_name] = {
            "spec": spec,
            "per_k": per_k_summary,
            "threshold_k_at_5pct_attack_success": threshold_k,
            "elapsed_sec": round(time.time() - t0, 1),
        }
        print(f"  threshold k@5%: {threshold_k}", flush=True)
        for k in HAMMING_K_VALUES:
            r = per_k_summary[k]
            print(f"    k={k:3d}: attack={r['attack_success_rate']:.3f} margin_mean={r['margin_mean']:+.3f}",
                  flush=True)
        del binder
        torch.cuda.empty_cache()

    # Score-correlation matrix at noiseless (k=0)
    correlation = {}
    binders_done = list(per_binder.keys())
    for b1 in binders_done:
        correlation[b1] = {}
        for b2 in binders_done:
            a1 = np.array(binder_score_arrays[b1][0])
            a2 = np.array(binder_score_arrays[b2][0])
            n = min(len(a1), len(a2))
            if n >= 2:
                c = np.corrcoef(a1[:n], a2[:n])[0, 1]
            else:
                c = float("nan")
            correlation[b1][b2] = float(c)

    out = {
        "n_rows": len(rows_data),
        "n_seeds": args.n_seeds,
        "k_values": list(HAMMING_K_VALUES),
        "per_binder": per_binder,
        "matched_psnr_correlation_across_binders": correlation,
    }
    (args.out / "exact_hamming.json").write_text(json.dumps(out, indent=2))
    print(f"\n[done] wrote {args.out}/exact_hamming.json", flush=True)


if __name__ == "__main__":
    main()

"""Closure Task 5: Synthetic corruption sweep on exact emitted RGB.

Reuses the model trained in Task 4 (exact RGB → XOF). Applies progressively
heavier blur / noise / downsampling / gamma to inputs and measures XOF
recovery degradation. Identifies the regime where bytes become unrecoverable.

Run:
  python scripts/closure/closure_task5_corruption_sweep.py \
    --ckpt experiments/closure_package/exact_rgb_xof/checkpoints/final_step.pt \
    --d2-dir /path/to/poliebotics_phase_b/poliebotics_phase_b/data/d2 \
    --out experiments/closure_package/blur_noise_sweep.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# Reuse Task 4 helpers
sys.path.insert(0, str(ROOT / "scripts" / "closure"))
from closure_task4_exact_rgb import (  # noqa: E402
    ExactRGBDataset, ExactRGBDecoder, render_emission_rgb,
)
from data.raw_bayer_dataset import load_chain_log  # noqa: E402
from data.packed_cfa_dataset import xof_octaves_centered_from_hex  # noqa: E402
from eval.candidate_ranking import (  # noqa: E402
    assemble_candidate_set, compute_tau95, per_family_window_fmr,
    score_candidate_xof, score_variants_for_experiment,
)


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    radius = max(1, int(round(3 * sigma)))
    k = 2 * radius + 1
    coords = torch.arange(k, dtype=torch.float32) - radius
    g1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g1d = g1d / g1d.sum()
    g2d = (g1d.unsqueeze(0) * g1d.unsqueeze(1)).to(x.device)
    g2d = g2d.view(1, 1, k, k).expand(3, 1, k, k)
    return F.conv2d(x.unsqueeze(0), g2d, padding=radius, groups=3).squeeze(0)


def apply_corruption(x: torch.Tensor, *, blur_sigma=0.0, noise_std=0.0,
                     downsample=1, gamma=1.0, gen=None) -> torch.Tensor:
    if blur_sigma > 0:
        x = gaussian_blur(x, blur_sigma)
    if downsample > 1:
        h, w = x.shape[-2:]
        new_h, new_w = max(1, h // downsample), max(1, w // downsample)
        x = F.interpolate(x.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
    if noise_std > 0:
        if gen is None:
            x = x + noise_std * torch.randn_like(x)
        else:
            x = x + noise_std * torch.randn(x.shape, generator=gen)
    if gamma != 1.0:
        x = x.clamp(0, 1).pow(1.0 / gamma)
    return x.clamp(0, 1)


def evaluate_corruption(model, val_ds, device, autocast_dtype, *,
                        blur_sigma=0.0, noise_std=0.0, downsample=1, gamma=1.0,
                        n_eval=128, seed=42):
    model.eval()
    n = min(n_eval, len(val_ds))
    score_variants = ("oct0_only", "oct0_plus_oct1",
                      "oct0_plus_oct1_plus_oct2", "all_octaves")
    matched_scores = {v: [] for v in score_variants}
    bit_recovery = {i: [] for i in range(4)}
    gen = torch.Generator().manual_seed(seed)
    t0 = time.time()
    with torch.no_grad():
        for i in range(n):
            sample = val_ds[i]
            x = apply_corruption(sample["rgb"].clone(),
                                  blur_sigma=blur_sigma, noise_std=noise_std,
                                  downsample=downsample, gamma=gamma, gen=gen).to(device)
            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                preds = model(x.unsqueeze(0))
            preds_cpu = [p.float().squeeze(0).cpu() if p is not None else None for p in preds]
            true_octs = [sample[f"xof_oct{j}"] for j in range(4)]
            for v in score_variants:
                aligned = [true_octs[j] if preds_cpu[j] is not None else None for j in range(4)]
                matched_scores[v].append(score_candidate_xof(preds_cpu, aligned, v))
            for j in range(4):
                if preds_cpu[j] is None:
                    continue
                pred_bytes = (preds_cpu[j] * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
                true_bytes = (true_octs[j] * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
                bit_recovery[j].append(((pred_bytes == true_bytes).float().mean()).item())
    return {
        "blur_sigma": blur_sigma, "noise_std": noise_std,
        "downsample": downsample, "gamma": gamma,
        "matched_score_mean": {v: float(np.mean(matched_scores[v])) for v in score_variants},
        "byte_match_per_octave": {f"oct{j}": float(np.mean(bit_recovery[j])) if bit_recovery[j] else None
                                   for j in range(4)},
        "n_eval": n,
        "elapsed_sec": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--d2-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--n-eval-per-setting", type=int, default=128)
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
    model.load_state_dict(state, strict=False)
    print(f"[ckpt] loaded {args.ckpt}", flush=True)

    settings: list[dict] = []
    # 1D sweeps
    for sigma in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
        settings.append({"blur_sigma": sigma})
    for std in (0.01, 0.05, 0.1):
        settings.append({"noise_std": std})
    for ds in (2, 4, 8):
        settings.append({"downsample": ds})
    for g in (1.5, 2.2, 3.0):
        settings.append({"gamma": g})
    # Combinations: blur×noise grid
    for sigma in (1.0, 2.0):
        for std in (0.01, 0.05):
            settings.append({"blur_sigma": sigma, "noise_std": std})

    all_results = []
    for s in settings:
        print(f"[sweep] {s}", flush=True)
        res = evaluate_corruption(model, val_ds, device, autocast_dtype,
                                    n_eval=args.n_eval_per_setting, **s)
        print(f"  matched_score(all)={res['matched_score_mean']['all_octaves']:.4f} "
              f"byte_match=oct0:{res['byte_match_per_octave']['oct0']:.3f} "
              f"oct1:{res['byte_match_per_octave']['oct1']:.3f} "
              f"oct2:{res['byte_match_per_octave']['oct2']:.3f} "
              f"oct3:{res['byte_match_per_octave']['oct3']:.3f}", flush=True)
        all_results.append(res)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "ckpt": str(args.ckpt),
        "n_eval_per_setting": args.n_eval_per_setting,
        "results": all_results,
    }, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

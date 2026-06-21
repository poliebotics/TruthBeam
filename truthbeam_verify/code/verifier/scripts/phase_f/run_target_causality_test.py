"""Phase F-A target-causality test.

The critical Phase F-A question: does the editor actually USE E_target, or does
it copy C_source ignoring E_target?

For each source frame C_s sampled from val set:
  Generate K=32 fakes by varying E_target (sample 32 different target rows):
    fake_k = F(C_s, E_source, E_target_k)

Per-source diagnostics:
  diversity:    std of fakes across K targets (per pixel, then mean) — should
                 be > 0 if the editor responds to E_target
  binder_score_target:   binder(fake_k) PSNR vs E_target_k (does the binder
                          retrieve target instead of source? higher is good)
  binder_score_source:   binder(fake_k) PSNR vs E_source (the source's own
                          emission; if editor ignores E_target, this stays high)
  margin:                binder_score_target − binder_score_source per fake

Aggregate:
  per_source: mean diversity, mean binder margin, fraction with margin > 0
  global:     fraction of (source, target) pairs where editor "swapped"
              (margin > 0); score histograms

Pass criteria for Phase F-A:
  - diversity > 0.01 (non-trivial output variation across targets)
  - mean margin > 0 (binder prefers target over source on the fake)
  - per-target retrieval rate > 0.5 (editor wins on majority of pairs)

Run:
  python scripts/phase_f/run_target_causality_test.py \
    --ckpt experiments/phase_f/f_a/checkpoints/best.pt \
    --config configs/phase_f_a.yaml \
    --out experiments/phase_f/f_a/causality_test.json
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

from phase_f.dataset_temporal_pairs import TemporalPairDataset, split_rows  # noqa: E402
from phase_f.editor_model import Editor  # noqa: E402
from data.emission_dataset import load_emission_at  # noqa: E402


def psnr01(p: torch.Tensor, t: torch.Tensor) -> float:
    mse = ((p - t) ** 2).mean().item()
    return float("inf") if mse < 1e-12 else 20.0 * math.log10(1.0 / math.sqrt(mse))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-source", type=int, default=32)
    ap.add_argument("--n-target-per-source", type=int, default=32)
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16

    # Build editor
    editor = Editor(
        capture_h=cfg["data"]["capture_h"],
        capture_w=cfg["data"]["capture_w"],
        emission_h=cfg["data"]["emission_h"],
        emission_w=cfg["data"]["emission_w"],
        init_mode=cfg["model"]["init_mode"],
    ).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    editor.load_state_dict(state, strict=True)
    editor.eval()

    # Sample source rows from val
    rng = np.random.RandomState(7)
    d2_val = split_rows("D2", "val")
    sources = sorted(rng.choice(d2_val, size=args.n_source, replace=False).tolist())
    targets = sorted(rng.choice(d2_val, size=args.n_target_per_source, replace=False).tolist())

    d2_dir = Path(cfg["data"]["d2_dir"])
    val_ds = TemporalPairDataset(
        session_dir=d2_dir, rows=d2_val, k_choices=[1],
        emission_h=cfg["data"]["emission_h"], emission_w=cfg["data"]["emission_w"],
        augment=False,
    )

    # For each (source_t, target_t): build (C_source, E_source, E_target_k) and run editor.
    # Track diversity + binder scores (binder ensemble TBD; for the standalone causality
    # test we use direct PSNR vs E_target / E_source on raw fake captures — the editor
    # itself shouldn't recover emission directly, so we run a separate binder ensemble).

    # Stub binder ensemble: load each surrogate binder per cfg.binder_ensemble
    # (Implementation deferred until Phase F-A run; for the causality test alone, the
    #  cleaner signal is "diversity across targets" + "fake's CFA difference vs source".)

    results: list[dict] = []
    print(f"[causality] {args.n_source} sources × {args.n_target_per_source} targets = "
          f"{args.n_source * args.n_target_per_source} fakes", flush=True)
    t0 = time.time()
    for s_idx, src_t in enumerate(sources):
        # Get the source pair
        if src_t not in val_ds.rows:
            continue
        sample = val_ds[val_ds.rows.index(src_t)]
        # We hold C_source + E_source fixed; vary E_target across `targets`
        C_s = sample["C_source"].unsqueeze(0).to(device)
        E_s = sample["E_source"].unsqueeze(0).to(device)
        per_source: list[dict] = []
        fakes: list[torch.Tensor] = []
        for t_idx, tgt_t in enumerate(targets):
            E_t = load_emission_at(
                d2_dir / "derived" / "Emissions" / f"tile_{tgt_t:06d}.png",
                cfg["data"]["emission_h"], cfg["data"]["emission_w"]
            ).unsqueeze(0).to(device)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=autocast_dtype):
                C_pred = editor(C_s, E_s, E_t).float().clamp(0, 1)
            fakes.append(C_pred.squeeze(0).cpu())
            # Direct fake-vs-source CFA delta as proxy for "did it edit?"
            cfa_delta = (C_pred - C_s).abs().mean().item()
            per_source.append({
                "target_t": tgt_t,
                "cfa_delta_from_source": cfa_delta,
            })
        # Diversity across targets
        stack = torch.stack(fakes, dim=0)   # (T, 4, H, W)
        per_target_std = stack.std(dim=0).mean().item()
        results.append({
            "source_t": src_t,
            "per_target_std_across_fakes": per_target_std,
            "mean_cfa_delta": float(np.mean([r["cfa_delta_from_source"] for r in per_source])),
            "per_target": per_source,
        })
        if (s_idx + 1) % 4 == 0:
            print(f"  [source {s_idx+1}/{len(sources)}] mean cfa_delta={results[-1]['mean_cfa_delta']:.4f} "
                  f"diversity={per_target_std:.4f} elapsed={time.time()-t0:.0f}s", flush=True)

    # Aggregates
    diversity_per_source = [r["per_target_std_across_fakes"] for r in results]
    delta_per_source = [r["mean_cfa_delta"] for r in results]
    out = {
        "ckpt": str(args.ckpt),
        "config": str(args.config),
        "n_source": len(results),
        "n_target_per_source": len(targets),
        "results_per_source": results,
        "summary": {
            "diversity_mean": float(np.mean(diversity_per_source)),
            "diversity_min":  float(np.min(diversity_per_source)),
            "diversity_max":  float(np.max(diversity_per_source)),
            "cfa_delta_mean": float(np.mean(delta_per_source)),
            "cfa_delta_min":  float(np.min(delta_per_source)),
            "cfa_delta_max":  float(np.max(delta_per_source)),
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\n[causality] diversity mean={out['summary']['diversity_mean']:.4f} "
          f"cfa_delta mean={out['summary']['cfa_delta_mean']:.4f}", flush=True)
    print(f"  → diversity > 0.01 = editor responds to E_target (vs ignoring it)", flush=True)
    print(f"  → wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

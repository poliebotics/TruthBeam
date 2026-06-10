"""F-A diagnostic 5: multi-source erasure.

Question: if we feed the editor 8 different source frames with the SAME
target, do all 8 outputs decode to the same target signature, or do they
retain source-specific features?

For each binder, compute the binder's predicted emission for each (source,
shared_target) combination. The collection of N predictions:
  - If editor erases source signature: all N predictions are similar to
    each other and to E_target → low across-source variance, high mean PSNR
    to E_target.
  - If editor retains source signature: predictions vary by source → high
    across-source variance.

Metrics per (target, binder):
  - mean_psnr_to_target: average PSNR of binder predictions vs E_target
  - across_source_psnr_std: std of those PSNRs across N sources
  - pairwise_pred_psnr_mean_db: average PSNR between pairs of predictions
    (high if all predictions look similar = target signature dominates)

Run on Lambda CPU.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(8)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "phase_e"))
sys.path.insert(0, str(ROOT / "scripts" / "phase_f"))

from data.emission_dataset import load_capture_at, load_emission_at  # noqa: E402
from compute_phase_e_thresholds import get_binder_specs, load_binder, psnr01  # noqa: E402
from binder_score_matrix import (  # noqa: E402
    TRAINING_BINDER_FAMILY, HELD_OUT_BINDER_FAMILY,
    load_editor, downsample_for_binder,
)


# 8 source frames spread across D2 selection_gate_normal (similar poses since D2 is yoga, controlled)
DEFAULT_SOURCES = [4900, 4960, 5020, 5080, 5140, 5200, 5260, 5320]
# 3 target rows from same val pool
DEFAULT_TARGETS = [4802, 4973, 5211]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--editor-ckpt", type=Path, required=True)
    ap.add_argument("--d2-dir", type=Path, required=True)
    ap.add_argument("--experiments-root", type=Path,
                    default=Path("/path/to/poliebotics_phase_b/experiments"))
    ap.add_argument("--source-rows", type=int, nargs="+", default=DEFAULT_SOURCES)
    ap.add_argument("--target-rows", type=int, nargs="+", default=DEFAULT_TARGETS)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dtype = torch.float32

    t0 = time.time()
    editor = load_editor(args.editor_ckpt, device)

    print(f"[init] {len(args.source_rows)} sources × {len(args.target_rows)} targets = "
          f"{len(args.source_rows)*len(args.target_rows)} forwards", flush=True)

    # Generate outputs grouped by target
    outputs_by_target = {}  # target_t → list of (source_t, C_pred)
    target_emissions = {}
    for t_t in args.target_rows:
        et = load_emission_at(args.d2_dir / "derived" / "Emissions" / f"tile_{t_t:06d}.png",
                              1080, 1920)
        target_emissions[t_t] = et
        outputs_by_target[t_t] = []
        et_dev = et.unsqueeze(0).to(device)
        for s_t in args.source_rows:
            cs = load_capture_at(args.d2_dir / "Recordings" / f"frame_{s_t:06d}.raw",
                                 2300, 2660).unsqueeze(0).to(device)
            es = load_emission_at(args.d2_dir / "derived" / "Emissions" / f"tile_{s_t:06d}.png",
                                  1080, 1920).unsqueeze(0).to(device)
            pred = editor(cs, es, et_dev).float().clamp(0, 1)
            outputs_by_target[t_t].append({"source_t": s_t, "C_pred": pred.cpu()})
            print(f"  s={s_t} t={t_t}", flush=True)

    print(f"[editor] {sum(len(v) for v in outputs_by_target.values())} outputs cached  "
          f"elapsed={time.time()-t0:.0f}s", flush=True)
    del editor

    # For each binder, compute predictions and metrics
    binder_specs = get_binder_specs(args.experiments_root)
    matrix = {}
    for binder_name, spec in binder_specs.items():
        if not Path(spec["ckpt"]).exists():
            continue
        t1 = time.time()
        binder = load_binder(spec, device, dtype)

        per_target = {}
        for t_t, outs in outputs_by_target.items():
            preds_em = []
            psnrs_to_target = []
            for out in outs:
                cap_in = downsample_for_binder(out["C_pred"].to(device),
                                                spec["capture_h"], spec["capture_w"])
                pred_em = binder(cap_in).float().clamp(0, 1).squeeze(0).cpu()
                preds_em.append(pred_em)
                psnrs_to_target.append(psnr01(pred_em, target_emissions[t_t]))

            # Pairwise PSNR between predictions (signature-similarity)
            pairwise = []
            for i, j in combinations(range(len(preds_em)), 2):
                pairwise.append(psnr01(preds_em[i], preds_em[j]))

            per_target[t_t] = {
                "n_sources": len(outs),
                "mean_psnr_to_target": float(np.mean(psnrs_to_target)),
                "std_psnr_to_target": float(np.std(psnrs_to_target)),
                "pairwise_pred_psnr_mean_db": float(np.mean(pairwise)) if pairwise else 0.0,
                "pairwise_pred_psnr_std": float(np.std(pairwise)) if pairwise else 0.0,
            }

        family = ("training" if binder_name in TRAINING_BINDER_FAMILY
                  else ("held_out" if binder_name in HELD_OUT_BINDER_FAMILY
                        else "other"))
        matrix[binder_name] = {
            "spec": {k: (str(v) if isinstance(v, Path) else v) for k, v in spec.items()},
            "family": family,
            "per_target": per_target,
            "elapsed_sec": round(time.time() - t1, 1),
        }
        # Aggregate across targets
        means_to_target = [d["mean_psnr_to_target"] for d in per_target.values()]
        means_pairwise = [d["pairwise_pred_psnr_mean_db"] for d in per_target.values()]
        print(f"  {binder_name:<22} family={family:<8} "
              f"to_target={np.mean(means_to_target):.2f} dB  "
              f"pairwise={np.mean(means_pairwise):.2f} dB  ({time.time()-t1:.0f}s)",
              flush=True)
        del binder

    # Family aggregates
    family_summary = {}
    for fam in ("training", "held_out"):
        rows = []
        for name, b in matrix.items():
            if b["family"] != fam:
                continue
            for t_t, d in b["per_target"].items():
                rows.append(d)
        if not rows:
            continue
        family_summary[fam] = {
            "n_evals": len(rows),
            "mean_psnr_to_target_db": float(np.mean([r["mean_psnr_to_target"] for r in rows])),
            "std_psnr_to_target_db": float(np.mean([r["std_psnr_to_target"] for r in rows])),
            "pairwise_pred_psnr_mean_db": float(np.mean([r["pairwise_pred_psnr_mean_db"] for r in rows])),
        }

    out_payload = {
        "editor_ckpt": str(args.editor_ckpt),
        "source_rows": args.source_rows,
        "target_rows": args.target_rows,
        "matrix": matrix,
        "family_summary": family_summary,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (args.out / "multi_source_erasure.json").write_text(json.dumps(out_payload, indent=2))

    md = [
        "# Multi-source erasure diagnostic",
        "",
        f"Checkpoint: `{args.editor_ckpt.name}`",
        f"{len(args.source_rows)} sources × {len(args.target_rows)} shared targets",
        f"  - `mean_psnr_to_target` — average binder→E_target PSNR across sources. **High = target signature dominates.**",
        f"  - `pairwise_pred_psnr_mean_db` — pairwise PSNR between binder predictions for different sources at the same target. **High = predictions are similar = target signature erases source signature.**",
        "",
        "## Family aggregates",
        "",
        "| family | mean_psnr_to_target | std_psnr_to_target | pairwise_pred_psnr |",
        "|---|---:|---:|---:|",
    ]
    for fam, fs in family_summary.items():
        md.append(
            f"| {fam} | {fs['mean_psnr_to_target_db']:.2f} dB | "
            f"{fs['std_psnr_to_target_db']:.2f} dB | "
            f"{fs['pairwise_pred_psnr_mean_db']:.2f} dB |"
        )
    md += [
        "",
        "## Reading the table",
        "",
        "- High `mean_psnr_to_target` = predictions land near E_target on average.",
        "- High `pairwise_pred_psnr` = predictions are similar to each other regardless of source.",
        "- Low `std_psnr_to_target` = predictions are consistently close to target across sources.",
        "",
        "If editor is genuinely SWAPPING projection: predictions cluster around E_target, pairwise high, std low.",
        "If editor retains source signature: predictions vary by source, pairwise low, std high.",
        "",
        f"Elapsed: {out_payload['elapsed_sec']}s",
    ]
    (args.out / "summary.md").write_text("\n".join(md))
    print(f"\n[done] wrote {args.out}/multi_source_erasure.json + summary.md", flush=True)


if __name__ == "__main__":
    main()

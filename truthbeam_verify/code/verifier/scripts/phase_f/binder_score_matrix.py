"""F-A diagnostic: binder score matrix for white-box-specialization gate.

Per CGPT round-7 audit, the F-A stop criterion is now blind-binder transfer.
This script computes the binder score matrix: 14 binders × 24 editor outputs
(4 sources × 6 E_targets, the same Grid-1 set used in visual diagnostics).

Per-(C_pred, binder) row contains:
  target_psnr        — score(binder(C_pred), E_target)
  source_psnr        — score(binder(C_pred), E_source)
  candidate_psnrs    — score(binder(C_pred), candidate_i) for all candidates
  target_rank        — position of E_target within candidate set (1 = best)
  target_gap_db      — target_psnr - 2nd_best_candidate_psnr
  source_rank        — position of E_source within candidate set
  best_wrong_id      — candidate id with highest score (after E_target)

Aggregated by binder family:
  training family: exp001c, e1 (and ablations) — saw similar binders during F-A
  held-out family: e2, e2_fp16, e3r, e3r_*    — never seen by editor

Trajectory at step 5k vs step 10k:
  training_target_rank_mean       — should improve (white-box working)
  held_out_target_rank_mean        — must improve (blind transfer working)
  white_box_specialization_score   — training_improvement - held_out_improvement
                                      (positive = bad, white-box overfit)

Run on Lambda CPU (no F-A interruption):
  CUDA_VISIBLE_DEVICES="" python scripts/phase_f/binder_score_matrix.py \
    --editor-ckpt /path/to/poliebotics_phase_b/checkpoints/step_00005000.pt \
    --d2-dir /path/to/poliebotics_phase_b/data/d2 \
    --v10-dir /path/to/poliebotics_phase_b/data/v10 \
    --out /path/to/poliebotics_phase_b/experiments/phase_f/f_a_full_v1/diagnostics/binder_matrix_step5k

Then again with --editor-ckpt step_00010000.pt → step10k subdir.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
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
from phase_f.editor_controlnet import EditorControlNet  # noqa: E402
from compute_phase_e_thresholds import get_binder_specs, load_binder, psnr01  # noqa: E402


# Same source rows as visual diagnostics — D2 selection_gate_normal, unseen during training
DEFAULT_SOURCE_ROWS = [4900, 5050, 5200, 5350]
# Same target pool as Grid 1
DEFAULT_TARGET_ROWS = [4802, 4869, 4902, 4973, 5076, 5211]
SOURCE_K = 1


# Binder family classification (per `experiments/phase_f_prep/binder_audit.md`)
TRAINING_BINDER_FAMILY = {
    "exp001c", "e1", "e1_bf16_sg", "e1_fp16_ddp", "e1_fp16_sg",
    "e1_lr2e4_sg", "e1_no_pretrained_sg", "e1_no_warmup_sg", "e1_seed42_sg",
}
HELD_OUT_BINDER_FAMILY = {
    "e2", "e2_fp16", "e3r", "e3r_fp16_ddp", "e3r_fp16_sg",
}


def load_editor(ckpt_path: Path, device: torch.device) -> EditorControlNet:
    print(f"[editor] loading {ckpt_path}", flush=True)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck["editor"] if "editor" in ck else ck.get("model", ck)
    state = {k[len("module."):] if k.startswith("module.") else k: v
             for k, v in state.items()}
    editor = EditorControlNet(
        capture_h=2300, capture_w=2660,
        emission_h=1080, emission_w=1920, init_mode="random",
    ).to(device)
    editor.load_state_dict(state, strict=False)
    editor.eval()
    return editor


@torch.no_grad()
def generate_outputs(editor, source_rows, target_rows, d2_dir, device):
    """Return list of dicts: 4 sources × 6 targets = 24 outputs."""
    outputs = []
    print(f"[editor] generating {len(source_rows)*len(target_rows)} outputs...", flush=True)
    for s_t in source_rows:
        cs = load_capture_at(d2_dir / "Recordings" / f"frame_{s_t:06d}.raw",
                             2300, 2660).unsqueeze(0).to(device)
        es = load_emission_at(d2_dir / "derived" / "Emissions" / f"tile_{s_t:06d}.png",
                              1080, 1920).unsqueeze(0).to(device)
        for t_t in target_rows:
            et = load_emission_at(d2_dir / "derived" / "Emissions" / f"tile_{t_t:06d}.png",
                                  1080, 1920).unsqueeze(0).to(device)
            pred = editor(cs, es, et).float().clamp(0, 1)
            outputs.append({
                "source_t": int(s_t),
                "target_t": int(t_t),
                "C_pred": pred.cpu(),
            })
            print(f"  s={s_t} t={t_t}", flush=True)
    return outputs


def build_candidate_set(source_rows, target_rows, d2_dir, v10_dir,
                        n_extra_d2=4, n_extra_v10=2, seed=11):
    """Return list of (id, emission_tensor). Includes:
    - All 6 Grid-1 targets
    - E_source for each of 4 sources
    - n_extra_d2 random distant rows from D2 train_normal (avoid val rows)
    - n_extra_v10 random V10 rows (cross-session)
    """
    rng = np.random.RandomState(seed)
    cand = {}

    # Grid-1 targets
    for t_t in target_rows:
        cand[f"d2_target_{t_t}"] = load_emission_at(
            d2_dir / "derived" / "Emissions" / f"tile_{t_t:06d}.png",
            1080, 1920)

    # E_source for each source
    for s_t in source_rows:
        cand[f"d2_source_{s_t}"] = load_emission_at(
            d2_dir / "derived" / "Emissions" / f"tile_{s_t:06d}.png",
            1080, 1920)

    # Extra D2 distractors (from train_normal, avoiding val range [4792, 5392))
    d2_pool = list(range(0, 4592))
    extras_d2 = sorted(rng.choice(d2_pool, size=n_extra_d2, replace=False).tolist())
    for r in extras_d2:
        cand[f"d2_extra_{r}"] = load_emission_at(
            d2_dir / "derived" / "Emissions" / f"tile_{r:06d}.png",
            1080, 1920)

    # V10 cross-session distractors
    v10_pool = list(range(0, 2500))
    extras_v10 = sorted(rng.choice(v10_pool, size=n_extra_v10, replace=False).tolist())
    for r in extras_v10:
        cand[f"v10_extra_{r}"] = load_emission_at(
            v10_dir / "derived" / "Emissions" / f"tile_{r:06d}.png",
            1080, 1920)

    return cand, extras_d2, extras_v10


def downsample_for_binder(cfa_pred: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    if cfa_pred.shape[-2:] == (target_h, target_w):
        return cfa_pred
    return F.interpolate(cfa_pred, size=(target_h, target_w), mode="area")


@torch.no_grad()
def score_binder(binder, spec, outputs, candidates, device):
    """For each output, return per-binder ranks + gaps."""
    rows = []
    for i, out in enumerate(outputs):
        cap_pred = out["C_pred"].to(device)
        # Downsample to binder's training resolution
        cap_in = downsample_for_binder(cap_pred, spec["capture_h"], spec["capture_w"])
        pred_em = binder(cap_in).float().clamp(0, 1).squeeze(0)  # (3, 1080, 1920)

        # Score against every candidate
        cand_scores = {}
        for cid, em in candidates.items():
            cand_scores[cid] = psnr01(pred_em, em.to(device))

        # The "correct target" id for this output
        target_id = f"d2_target_{out['target_t']}"
        source_id = f"d2_source_{out['source_t']}"
        target_psnr = cand_scores[target_id]
        source_psnr = cand_scores[source_id]

        # Rank target among ALL candidates (including E_source as a wrong)
        sorted_cids = sorted(cand_scores.items(), key=lambda x: -x[1])  # high to low
        cid_rank = {cid: r + 1 for r, (cid, _) in enumerate(sorted_cids)}
        target_rank = cid_rank[target_id]
        source_rank = cid_rank[source_id]

        # 2nd-best (excluding target) for gap
        sorted_excl = [(cid, sc) for cid, sc in sorted_cids if cid != target_id]
        best_wrong_id, best_wrong_psnr = sorted_excl[0]
        target_gap_db = target_psnr - best_wrong_psnr

        rows.append({
            "source_t": out["source_t"], "target_t": out["target_t"],
            "target_psnr": target_psnr, "source_psnr": source_psnr,
            "target_rank": target_rank, "source_rank": source_rank,
            "target_gap_db": target_gap_db, "best_wrong_id": best_wrong_id,
            "best_wrong_psnr": best_wrong_psnr,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--editor-ckpt", type=Path, required=True)
    ap.add_argument("--d2-dir", type=Path, required=True)
    ap.add_argument("--v10-dir", type=Path, required=True)
    ap.add_argument("--experiments-root", type=Path,
                    default=Path("/path/to/poliebotics_phase_b/experiments"))
    ap.add_argument("--source-rows", type=int, nargs="+", default=DEFAULT_SOURCE_ROWS)
    ap.add_argument("--target-rows", type=int, nargs="+", default=DEFAULT_TARGET_ROWS)
    ap.add_argument("--n-extra-d2", type=int, default=4)
    ap.add_argument("--n-extra-v10", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu", help="cpu (no F-A interruption) or cuda")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dtype = torch.float32

    t0 = time.time()
    editor = load_editor(args.editor_ckpt, device)
    outputs = generate_outputs(editor, args.source_rows, args.target_rows,
                               args.d2_dir, device)
    print(f"[editor] {len(outputs)} outputs cached  elapsed={time.time()-t0:.0f}s",
          flush=True)
    del editor  # free memory before loading binders

    candidates, extras_d2, extras_v10 = build_candidate_set(
        args.source_rows, args.target_rows, args.d2_dir, args.v10_dir,
        args.n_extra_d2, args.n_extra_v10,
    )
    print(f"[candidates] {len(candidates)} emissions: "
          f"{len(args.target_rows)} targets + {len(args.source_rows)} sources + "
          f"{args.n_extra_d2} d2_extras ({extras_d2}) + "
          f"{args.n_extra_v10} v10_extras ({extras_v10})", flush=True)

    binder_specs = get_binder_specs(args.experiments_root)
    print(f"[binders] {len(binder_specs)} binders", flush=True)

    matrix = {}
    for binder_name, spec in binder_specs.items():
        if not Path(spec["ckpt"]).exists():
            print(f"  [skip] {binder_name}: ckpt missing", flush=True)
            continue
        t1 = time.time()
        binder = load_binder(spec, device, dtype)
        rows = score_binder(binder, spec, outputs, candidates, device)
        family = ("training" if binder_name in TRAINING_BINDER_FAMILY
                  else ("held_out" if binder_name in HELD_OUT_BINDER_FAMILY
                        else "other"))
        matrix[binder_name] = {
            "spec": {k: (str(v) if isinstance(v, Path) else v) for k, v in spec.items()},
            "family": family,
            "rows": rows,
            "elapsed_sec": round(time.time() - t1, 1),
        }
        # Aggregate per-binder
        target_ranks = [r["target_rank"] for r in rows]
        source_ranks = [r["source_rank"] for r in rows]
        gaps = [r["target_gap_db"] for r in rows]
        print(f"  {binder_name:<22} family={family:<8} "
              f"target_rank_mean={np.mean(target_ranks):.2f} "
              f"target_gap_mean={np.mean(gaps):+.2f} dB "
              f"source_rank_mean={np.mean(source_ranks):.2f}  "
              f"({time.time()-t1:.0f}s)", flush=True)
        del binder

    # Aggregate by family
    family_summary = {}
    for fam in ("training", "held_out"):
        all_rows = [r for name, b in matrix.items() if b["family"] == fam for r in b["rows"]]
        if not all_rows:
            continue
        target_ranks = [r["target_rank"] for r in all_rows]
        source_ranks = [r["source_rank"] for r in all_rows]
        gaps = [r["target_gap_db"] for r in all_rows]
        target_psnrs = [r["target_psnr"] for r in all_rows]
        source_psnrs = [r["source_psnr"] for r in all_rows]
        family_summary[fam] = {
            "n_evals": len(all_rows),
            "n_binders": sum(1 for b in matrix.values() if b["family"] == fam),
            "target_rank_mean": float(np.mean(target_ranks)),
            "target_rank_p50": float(np.median(target_ranks)),
            "source_rank_mean": float(np.mean(source_ranks)),
            "target_gap_mean_db": float(np.mean(gaps)),
            "target_gap_p50_db": float(np.median(gaps)),
            "target_psnr_mean": float(np.mean(target_psnrs)),
            "source_psnr_mean": float(np.mean(source_psnrs)),
            "target_rank_at_1_frac": float(np.mean([r == 1 for r in target_ranks])),
        }

    out_payload = {
        "editor_ckpt": str(args.editor_ckpt),
        "source_rows": args.source_rows,
        "target_rows": args.target_rows,
        "n_outputs": len(outputs),
        "candidate_set_size": len(candidates),
        "candidate_extras_d2": extras_d2,
        "candidate_extras_v10": extras_v10,
        "matrix": matrix,
        "family_summary": family_summary,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (args.out / "binder_score_matrix.json").write_text(json.dumps(out_payload, indent=2))

    # Summary table
    md = [
        f"# Binder score matrix",
        f"",
        f"Editor checkpoint: `{args.editor_ckpt.name}`",
        f"Sources: {args.source_rows}, Targets: {args.target_rows}",
        f"Candidate set: {len(candidates)} emissions ({len(args.target_rows)} targets, {len(args.source_rows)} sources, {args.n_extra_d2} d2 extras, {args.n_extra_v10} v10 extras)",
        f"Total evaluations: {len(matrix) * len(outputs)} = {len(matrix)} binders × {len(outputs)} outputs",
        f"",
        f"## Per-binder summary",
        f"",
        f"| binder | family | target_rank_mean | target_gap_mean_db | source_rank_mean | rank_at_1_frac |",
        f"|---|---|---:|---:|---:|---:|",
    ]
    for name, b in matrix.items():
        rows = b["rows"]
        target_ranks = [r["target_rank"] for r in rows]
        source_ranks = [r["source_rank"] for r in rows]
        gaps = [r["target_gap_db"] for r in rows]
        rank1_frac = float(np.mean([r == 1 for r in target_ranks]))
        md.append(
            f"| {name} | {b['family']} | "
            f"{np.mean(target_ranks):.2f} | "
            f"{np.mean(gaps):+.2f} | "
            f"{np.mean(source_ranks):.2f} | "
            f"{rank1_frac:.3f} |"
        )
    md.append("")
    md.append("## Family aggregates")
    md.append("")
    md.append("| family | n_binders | target_rank_mean | target_gap_mean_db | source_rank_mean | rank_at_1_frac |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for fam, fs in family_summary.items():
        md.append(
            f"| {fam} | {fs['n_binders']} | "
            f"{fs['target_rank_mean']:.2f} | "
            f"{fs['target_gap_mean_db']:+.2f} | "
            f"{fs['source_rank_mean']:.2f} | "
            f"{fs['target_rank_at_1_frac']:.3f} |"
        )
    md.append("")
    md.append("## Interpretation")
    md.append("")
    md.append("Smaller `target_rank_mean` is better (1.0 = always picks the right target).")
    md.append("Larger `target_gap_mean_db` is better (target scores higher than best wrong).")
    md.append("Larger `source_rank_mean` is better (binder doesn't confuse output with E_source).")
    md.append("Higher `rank_at_1_frac` is better (binder picks correct target most often).")
    md.append("")
    md.append("**White-box specialization signal**: if `training` family has clearly better")
    md.append("target_rank than `held_out` AND that gap WIDENS between checkpoints, the editor")
    md.append("is overfitting to surrogate binders rather than learning generalizable target swap.")
    md.append("")
    md.append(f"Elapsed: {out_payload['elapsed_sec']}s")
    (args.out / "summary.md").write_text("\n".join(md))
    print(f"\n[done] wrote {args.out}/binder_score_matrix.json + summary.md", flush=True)


if __name__ == "__main__":
    main()

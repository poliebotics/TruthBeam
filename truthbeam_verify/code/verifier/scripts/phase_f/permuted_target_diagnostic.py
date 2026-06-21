"""F-A diagnostic 4: permuted-target in-distribution scoring.

Question: when we feed the editor a PERMUTED E_target (e.g., from V10 — a
totally different chain), does the model produce an output that binders
decode as that permuted target? If so, the editor is genuinely swapping
projections. If not, the editor is just removing projection (or specializing
to its surrogate ensemble in some other way).

Per (source, permuted_target) pair:
  - Generate C_pred = editor(C_source, E_source, E_perm_target)
  - Score binder(C_pred) against:
    - E_perm_target (the permuted target — if model is swapping, this should rank high)
    - E_source (if model is "removing" projection, source ought to rank high)
    - The natural matched E_target_t = source_t + 1 (where the original projection went)
    - distractors

Output: per-binder rank of E_perm_target vs E_source vs E_natural_target.
A "swapping" model: rank(E_perm_target) > rank(E_source) > rank(E_natural_target)
A "removing" model:  rank(E_source) > rank(E_natural_target) ≈ rank(E_perm_target)

Run on Lambda CPU.
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
import torch.nn.functional as F

torch.set_num_threads(8)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "phase_e"))
sys.path.insert(0, str(ROOT / "scripts" / "phase_f"))

from data.emission_dataset import load_capture_at, load_emission_at  # noqa: E402
from phase_f.editor_controlnet import EditorControlNet  # noqa: E402
from compute_phase_e_thresholds import get_binder_specs, load_binder, psnr01  # noqa: E402
from binder_score_matrix import (  # noqa: E402
    TRAINING_BINDER_FAMILY, HELD_OUT_BINDER_FAMILY,
    load_editor, downsample_for_binder,
)


# Same val sources as Grid 1 / binder_score_matrix
DEFAULT_SOURCE_ROWS = [4900, 5050, 5200, 5350]
# Permuted targets: 4 from V10 (cross-session) + 2 distant D2 train rows
DEFAULT_PERMUTED_V10 = [3093, 3181, 1862, 934]
DEFAULT_PERMUTED_D2_FAR = [200, 1000]
SOURCE_K = 1


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--editor-ckpt", type=Path, required=True)
    ap.add_argument("--d2-dir", type=Path, required=True)
    ap.add_argument("--v10-dir", type=Path, required=True)
    ap.add_argument("--experiments-root", type=Path,
                    default=Path("/path/to/poliebotics_phase_b/experiments"))
    ap.add_argument("--source-rows", type=int, nargs="+", default=DEFAULT_SOURCE_ROWS)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dtype = torch.float32

    t0 = time.time()
    editor = load_editor(args.editor_ckpt, device)

    # Permuted target pool: 4 V10 + 2 distant D2
    permuted = []
    for r in DEFAULT_PERMUTED_V10:
        em = load_emission_at(args.v10_dir / "derived" / "Emissions" / f"tile_{r:06d}.png",
                              1080, 1920)
        permuted.append({"id": f"v10_{r}", "row": r, "session": "V10", "em": em})
    for r in DEFAULT_PERMUTED_D2_FAR:
        em = load_emission_at(args.d2_dir / "derived" / "Emissions" / f"tile_{r:06d}.png",
                              1080, 1920)
        permuted.append({"id": f"d2far_{r}", "row": r, "session": "D2", "em": em})

    print(f"[init] editor loaded; {len(args.source_rows)} sources × "
          f"{len(permuted)} permuted targets = "
          f"{len(args.source_rows)*len(permuted)} editor outputs", flush=True)

    # Generate outputs
    outputs = []
    for s_t in args.source_rows:
        cs = load_capture_at(args.d2_dir / "Recordings" / f"frame_{s_t:06d}.raw",
                             2300, 2660).unsqueeze(0).to(device)
        es = load_emission_at(args.d2_dir / "derived" / "Emissions" / f"tile_{s_t:06d}.png",
                              1080, 1920).unsqueeze(0).to(device)
        natural_t = s_t + SOURCE_K
        em_natural = load_emission_at(
            args.d2_dir / "derived" / "Emissions" / f"tile_{natural_t:06d}.png",
            1080, 1920)

        for perm in permuted:
            ept = perm["em"].unsqueeze(0).to(device)
            pred = editor(cs, es, ept).float().clamp(0, 1)
            outputs.append({
                "source_t": int(s_t),
                "perm_id": perm["id"],
                "perm_em": perm["em"],
                "natural_t": int(natural_t),
                "natural_em": em_natural,
                "source_em": es.squeeze(0).cpu(),
                "C_pred": pred.cpu(),
            })
            print(f"  s={s_t}  perm={perm['id']}", flush=True)

    print(f"[editor] {len(outputs)} outputs cached  elapsed={time.time()-t0:.0f}s",
          flush=True)
    del editor

    # Per binder, score each C_pred against (perm_em, natural_em, source_em + distractors)
    binder_specs = get_binder_specs(args.experiments_root)
    matrix = {}
    for binder_name, spec in binder_specs.items():
        if not Path(spec["ckpt"]).exists():
            continue
        t1 = time.time()
        binder = load_binder(spec, device, dtype)
        rows = []
        for out in outputs:
            cap_in = downsample_for_binder(out["C_pred"].to(device),
                                            spec["capture_h"], spec["capture_w"])
            pred_em = binder(cap_in).float().clamp(0, 1).squeeze(0)
            psnr_perm = psnr01(pred_em, out["perm_em"].to(device))
            psnr_natural = psnr01(pred_em, out["natural_em"].to(device))
            psnr_source = psnr01(pred_em, out["source_em"].to(device))
            # Rank perm vs natural vs source (3-way comparison)
            scores = {"perm": psnr_perm, "natural": psnr_natural, "source": psnr_source}
            sorted_s = sorted(scores.items(), key=lambda x: -x[1])
            ranks = {k: r + 1 for r, (k, _) in enumerate(sorted_s)}
            rows.append({
                "source_t": out["source_t"],
                "perm_id": out["perm_id"],
                "natural_t": out["natural_t"],
                "psnr_perm_target": psnr_perm,
                "psnr_natural_target": psnr_natural,
                "psnr_source": psnr_source,
                "rank_perm": ranks["perm"],
                "rank_natural": ranks["natural"],
                "rank_source": ranks["source"],
                "swap_signal": psnr_perm - psnr_source,  # +ve = decoder reads as permuted
                "removal_signal": psnr_source - psnr_natural,  # +ve = decoder reads as source
            })
        family = ("training" if binder_name in TRAINING_BINDER_FAMILY
                  else ("held_out" if binder_name in HELD_OUT_BINDER_FAMILY
                        else "other"))
        matrix[binder_name] = {
            "spec": {k: (str(v) if isinstance(v, Path) else v) for k, v in spec.items()},
            "family": family,
            "rows": rows,
            "elapsed_sec": round(time.time() - t1, 1),
        }
        rank1_perm = float(np.mean([r["rank_perm"] == 1 for r in rows]))
        swap_mean = float(np.mean([r["swap_signal"] for r in rows]))
        print(f"  {binder_name:<22} family={family:<8} rank1_perm={rank1_perm:.3f} "
              f"swap_signal_mean={swap_mean:+.2f} dB  ({time.time()-t1:.0f}s)",
              flush=True)
        del binder

    # Aggregate by family
    family_summary = {}
    for fam in ("training", "held_out"):
        fam_rows = [r for name, b in matrix.items() if b["family"] == fam for r in b["rows"]]
        if not fam_rows:
            continue
        family_summary[fam] = {
            "n_evals": len(fam_rows),
            "rank1_perm_frac": float(np.mean([r["rank_perm"] == 1 for r in fam_rows])),
            "rank1_natural_frac": float(np.mean([r["rank_natural"] == 1 for r in fam_rows])),
            "rank1_source_frac": float(np.mean([r["rank_source"] == 1 for r in fam_rows])),
            "swap_signal_mean_db": float(np.mean([r["swap_signal"] for r in fam_rows])),
            "removal_signal_mean_db": float(np.mean([r["removal_signal"] for r in fam_rows])),
            "psnr_perm_mean": float(np.mean([r["psnr_perm_target"] for r in fam_rows])),
            "psnr_natural_mean": float(np.mean([r["psnr_natural_target"] for r in fam_rows])),
            "psnr_source_mean": float(np.mean([r["psnr_source"] for r in fam_rows])),
        }

    out_payload = {
        "editor_ckpt": str(args.editor_ckpt),
        "source_rows": args.source_rows,
        "permuted_targets": [{"id": p["id"], "row": p["row"], "session": p["session"]}
                             for p in permuted],
        "n_outputs": len(outputs),
        "matrix": matrix,
        "family_summary": family_summary,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (args.out / "permuted_target.json").write_text(json.dumps(out_payload, indent=2))

    md = [
        "# Permuted-target diagnostic",
        "",
        f"Checkpoint: `{args.editor_ckpt.name}`",
        f"Sources: {args.source_rows}",
        f"Permuted targets: {[p['id'] for p in permuted]}",
        f"Total: {len(matrix) * len(outputs)} evaluations.",
        "",
        "## Family aggregates",
        "",
        "| family | rank1_perm_frac | rank1_natural_frac | rank1_source_frac | swap_signal | removal_signal |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for fam, fs in family_summary.items():
        md.append(
            f"| {fam} | {fs['rank1_perm_frac']:.3f} | {fs['rank1_natural_frac']:.3f} | "
            f"{fs['rank1_source_frac']:.3f} | {fs['swap_signal_mean_db']:+.2f} dB | "
            f"{fs['removal_signal_mean_db']:+.2f} dB |"
        )
    md += [
        "",
        "## Interpretation",
        "",
        "- `rank1_perm_frac` = fraction of (source, perm_target, binder) triples where the binder's prediction is closer to E_perm_target than to E_source or E_natural. **High = editor is swapping.**",
        "- `rank1_natural_frac` = the natural matched-target wins. High = editor is doing the right thing for the OLD target.",
        "- `rank1_source_frac` = E_source wins. High = editor is just removing/keeping projection unchanged.",
        "- `swap_signal_mean_db` = psnr(perm) - psnr(source). +ve = decoder reads more as permuted than source.",
        "- `removal_signal_mean_db` = psnr(source) - psnr(natural). +ve = decoder reads more as source than as natural.",
        "",
        f"Elapsed: {out_payload['elapsed_sec']}s",
    ]
    (args.out / "summary.md").write_text("\n".join(md))
    print(f"\n[done] wrote {args.out}/permuted_target.json + summary.md", flush=True)


if __name__ == "__main__":
    main()

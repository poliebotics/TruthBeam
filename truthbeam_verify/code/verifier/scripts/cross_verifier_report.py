"""Cross-verifier comparison report.

Reads:
    --combined-eval     experiments/stage_0/eval (all 4 ckpts; uses --ckpt-step)
    --d2-verifier-eval  experiments/stage_0_cross_verifier/d2_only_verifier
    --v10-verifier-eval experiments/stage_0_cross_verifier/v10_only_verifier

For a given F-A checkpoint step (default 100000), computes per-verifier:
    AUROC(target vs shuffled) — lower MSE on target → higher AUROC
    AUROC(target vs source)
    AUROC(target vs zero)

Per session and combined.

Headline question: does a LOSO verifier (D2-only or V10-only) still
discriminate F-A v1 fakes on its OUT-OF-DISTRIBUTION session at AUROC ≈ 1?
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


COND_NAMES = ("correct", "shuffled", "source", "zero")
SESSIONS = ("d2", "v10")


def _load_session(eval_root: Path, ckpt_step: int, sess: str) -> dict:
    """Load per-cond means. HALTS on missing NPZ."""
    npz_path = eval_root / f"step_{ckpt_step:08d}" / f"stage0_{sess}_raw.npz"
    if not npz_path.exists():
        raise SystemExit(
            f"[cross_verifier_report] missing NPZ: {npz_path}. "
            "Cross-verifier comparison cannot proceed without all sessions.")
    z = np.load(npz_path, allow_pickle=False)
    return {f"{ctype}_{cond}": z[f"cond_{ctype}_{cond}"].mean(axis=(1, 2))
            for ctype in ("real", "fake")
            for cond in COND_NAMES}


def _aurocs(per_cond: dict, ctx: str = "") -> dict:
    """For each negative cond in (shuffled, source, zero), compute AUROC of
    correct vs that cond. Lower MSE on correct = positive class score.

    Returns {(real|fake)_target_vs_<neg>: auroc}.

    if AUROC fails (e.g. all NaN scores or
    single-class labels), raise instead of writing NaN — the report should
    halt loudly so the operator knows the verifier output was malformed.
    """
    out = {}
    for ctype in ("real", "fake"):
        correct = per_cond[f"{ctype}_correct"]
        for neg in ("shuffled", "source", "zero"):
            other = per_cond[f"{ctype}_{neg}"]
            scores = np.concatenate([correct, other])
            labels = np.concatenate([np.ones(len(correct)), np.zeros(len(other))])
            if not np.isfinite(scores).all():
                raise SystemExit(
                    f"[cross_verifier_report] non-finite scores in "
                    f"{ctx} {ctype}/{neg} — verifier output corrupted.")
            try:
                auc = float(roc_auc_score(labels, -scores))
            except ValueError as ex:
                raise SystemExit(
                    f"[cross_verifier_report] AUROC failed for "
                    f"{ctx} {ctype} target_vs_{neg}: {ex}")
            out[f"{ctype}_target_vs_{neg}"] = auc
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined-eval",     type=Path, required=True)
    ap.add_argument("--d2-verifier-eval",  type=Path, required=True)
    ap.add_argument("--v10-verifier-eval", type=Path, required=True)
    ap.add_argument("--ckpt-step", type=int, default=100000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    # enforce all three eval roots resolve to a
    # consistent base (all on Lambda or all local). Mixed state would silently
    # compare against the wrong combined baseline.
    roots_present = [args.combined_eval.exists(),
                     args.d2_verifier_eval.exists(),
                     args.v10_verifier_eval.exists()]
    if not all(roots_present):
        raise SystemExit(
            f"[cross_verifier_report] one or more eval roots missing: "
            f"combined={args.combined_eval}({roots_present[0]}) "
            f"d2_only={args.d2_verifier_eval}({roots_present[1]}) "
            f"v10_only={args.v10_verifier_eval}({roots_present[2]})")
    verifiers = [
        ("combined",  args.combined_eval),
        ("d2_only",   args.d2_verifier_eval),
        ("v10_only",  args.v10_verifier_eval),
    ]

    results = {}
    for vname, vroot in verifiers:
        results[vname] = {}
        for sess in SESSIONS:
            per_cond = _load_session(vroot, args.ckpt_step, sess)
            results[vname][sess] = _aurocs(per_cond, ctx=f"{vname}/{sess}")

    # Build report
    md = []
    md.append("# F-A v1 cross-verifier comparison report")
    md.append("")
    md.append(f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    md.append(f"F-A checkpoint: step_{args.ckpt_step:08d}")
    md.append("")
    md.append("Each cell shows AUROC of (target=correct E vs negative-cond E) on real C samples / on F-A-fake C samples.")
    md.append("Higher is better (1.0 = perfect discrimination, 0.5 = chance).")
    md.append("")

    for sess in SESSIONS:
        md.append(f"## Session: {sess.upper()}")
        md.append("")
        md.append("| verifier | t vs shuffled (real / fake) | t vs source (real / fake) | t vs zero (real / fake) |")
        md.append("|---|---|---|---|")
        for vname, _ in verifiers:
            block = results[vname].get(sess)
            if block is None:
                md.append(f"| {vname} | — | — | — | (missing eval) |")
                continue
            row = []
            for neg in ("shuffled", "source", "zero"):
                r = block.get(f"real_target_vs_{neg}", float("nan"))
                f = block.get(f"fake_target_vs_{neg}", float("nan"))
                row.append(f"{r:.4f} / {f:.4f}")
            md.append(f"| {vname} | " + " | ".join(row) + " |")
        md.append("")

    md.append("## Interpretation")
    md.append("")
    md.append("- **Combined verifier baseline**: from existing Stage 0 eval; F-A v1 fakes scored under target/shuffled/source/zero by the verifier trained on D2+V10.")
    md.append("- **D2-only verifier on V10**: tests whether a verifier trained only on D2 still discriminates fakes when scoring V10 frames. If AUROC ≈ 1.0, the diffusion-feature signal is session-invariant.")
    md.append("- **V10-only verifier on D2**: symmetric.")
    md.append("- **Real-pair AUROC** (e.g. real target vs real shuffled): tests whether the verifier still recognises (C_real, E_correct) as more probable than (C_real, E_shuffled). Drops in real-pair AUROC out-of-distribution indicate the verifier's coupling signal is partly session-specific.")
    md.append("- **Fake-pair AUROC** is the F-A discrimination metric. A drop here from combined → LOSO indicates F-A v1 is exploiting session-specific verifier weakness.")
    md.append("")

    args.out.write_text("\n".join(md))

    # Also dump raw JSON
    with open(args.out.with_suffix(".json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[cross_verifier_report] wrote {args.out}")


if __name__ == "__main__":
    main()

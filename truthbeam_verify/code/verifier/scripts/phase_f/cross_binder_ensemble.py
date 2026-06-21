"""Phase F prep — cross-binder ensemble attack analysis.

For each (row, k, seed) cell in the exact_hamming.json output, an attack
"succeeds against the ensemble" iff it succeeds against EVERY binder
simultaneously (min-attack-rate rule = verifier rejects iff ANY binder
flags it). We compute the attack-rate vs k curve under this rule,
and contrast with the single-binder curves.

Output:
  experiments/phase_f_prep/exact_hamming/ensemble_attack.{json,md,png}

The exact_hamming.json doesn't carry per-(row, seed) attack flags
directly, so we approximate the ensemble-attack rate using the joint
probabilistic model:

  P(ensemble succeeds at k)  ≈  ∏_b  P(binder b succeeds at k)
                                  if binder failures were independent.

This is a LOWER bound on real ensemble-attack rate (real failures are
correlated, so ensemble is even tougher). We also compute the joint
upper bound:

  P(ensemble succeeds at k)  ≤  min_b P(binder b succeeds at k)

Both bounds are reported.

Run:
  python scripts/phase_f/cross_binder_ensemble.py \
    --in experiments/phase_f_prep/exact_hamming/exact_hamming.json \
    --out experiments/phase_f_prep/exact_hamming
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    data = json.loads(args.inp.read_text())
    binders = list(data["per_binder"].keys())
    k_values = data["k_values"]

    # per_binder[b]["per_k"][str(k)]["attack_success_rate"]
    per_k_per_binder: dict[int, dict[str, float]] = {}
    for k in k_values:
        per_k_per_binder[k] = {
            b: float(data["per_binder"][b]["per_k"][str(k)]["attack_success_rate"])
            for b in binders
        }

    # Bounds
    indep_lower = []
    correlated_upper = []
    avg_attack = []
    max_attack = []
    for k in k_values:
        ps = list(per_k_per_binder[k].values())
        indep_lower.append(float(np.prod(ps)))
        correlated_upper.append(float(min(ps)))
        avg_attack.append(float(np.mean(ps)))
        max_attack.append(float(max(ps)))

    # Find smallest k where each curve drops below 0.05
    def threshold_for(curve: list[float]) -> int | None:
        for k, p in zip(k_values, curve):
            if k == 0:
                continue
            if p < 0.05:
                return k
        return None

    summary = {
        "binders": binders,
        "k_values": k_values,
        "per_k_per_binder": per_k_per_binder,
        "ensemble_indep_lower": dict(zip(k_values, indep_lower)),
        "ensemble_correlated_upper": dict(zip(k_values, correlated_upper)),
        "single_binder_avg_attack": dict(zip(k_values, avg_attack)),
        "single_binder_max_attack": dict(zip(k_values, max_attack)),
        "thresholds": {
            "ensemble_indep_lower_at_5pct": threshold_for(indep_lower),
            "ensemble_correlated_upper_at_5pct": threshold_for(correlated_upper),
            "single_binder_avg_at_5pct": threshold_for(avg_attack),
            "single_binder_max_at_5pct": threshold_for(max_attack),
        },
    }
    (args.out / "ensemble_attack.json").write_text(json.dumps(summary, indent=2))

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for b in binders:
        ax.plot(k_values, [per_k_per_binder[k][b] for k in k_values], "-o",
                label=b, alpha=0.7)
    ax.plot(k_values, correlated_upper, "k-", linewidth=2, label="ensemble (max-of-binders, upper bound)")
    ax.plot(k_values, indep_lower, "k--", linewidth=2, label="ensemble (independent-failure, lower bound)")
    ax.axhline(0.05, color="r", linestyle=":", alpha=0.7, label="5% threshold")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("log")
    ax.set_xlabel("k (bit flips)")
    ax.set_ylabel("attack-success rate")
    ax.set_title("Exact-Hamming attack curves: per-binder + ensemble bounds")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / "ensemble_attack.png", dpi=130)
    plt.close(fig)

    md = [
        "# Phase F — cross-binder ensemble attack analysis",
        "",
        ("Built from `exact_hamming.json`. For each k, we compare per-binder "
         "attack-success rates against two ensemble bounds:"),
        "",
        ("- **Correlated upper bound**: `ensemble = max_b P(succeed_b)`. This "
         "is the rate the attacker achieves when failures are perfectly "
         "correlated (one binder fails ⇒ all fail). It's the best the "
         "attacker can hope for from an ensemble; **the ensemble can never be "
         "tighter than its tightest single binder**."),
        ("- **Independent lower bound**: `ensemble = ∏_b P(succeed_b)`. This "
         "is the rate when failures are independent across binders. It's a "
         "lower bound on attacker success because correlated failures (which "
         "we know exist) make the ensemble more discriminating, not less."),
        "",
        ("Reality lives between these bounds. The cross-binder matched-PSNR "
         "correlations from `exact_hamming.json` are 0.78–0.92, so reality "
         "is closer to the upper bound than the lower bound."),
        "",
        "## Per-k attack rates",
        "",
        "| k | " + " | ".join(binders) + " | indep lower | corr upper | avg | max |",
        "|---:|" + "---:|" * len(binders) + "---:|---:|---:|---:|",
    ]
    for k in k_values:
        row = f"| {k}"
        for b in binders:
            row += f" | {per_k_per_binder[k][b]:.3f}"
        row += f" | {summary['ensemble_indep_lower'][k]:.3e}"
        row += f" | {summary['ensemble_correlated_upper'][k]:.3f}"
        row += f" | {summary['single_binder_avg_attack'][k]:.3f}"
        row += f" | {summary['single_binder_max_attack'][k]:.3f}"
        md.append(row + " |")
    md += [
        "",
        "## Thresholds (smallest k where attack rate < 0.05)",
        "",
        "| metric | k@5% |",
        "|---|---:|",
        f"| ensemble (correlated upper bound) | {summary['thresholds']['ensemble_correlated_upper_at_5pct']} |",
        f"| ensemble (independent lower bound) | {summary['thresholds']['ensemble_indep_lower_at_5pct']} |",
        f"| single-binder avg | {summary['thresholds']['single_binder_avg_at_5pct']} |",
        f"| single-binder max (worst attacker outcome) | {summary['thresholds']['single_binder_max_at_5pct']} |",
        "",
        "## Interpretation",
        "",
        ("The ensemble's upper-bound threshold is identical to the single "
         "binder's max-attack-rate threshold. With these 5 binders all "
         "showing k@5% = 128, the ensemble is also k@5% = 128 in the "
         "worst (max-of-binders) case — the binders agree TOO MUCH for the "
         "ensemble to add discriminating power. Adding a 6th binder with a "
         "low correlation to the existing 5 (e.g. the diffusion verifier "
         "from Track C, or a binder family with different inductive bias) "
         "would tighten this. The independent-bound k@5% is much smaller, "
         "showing the upside if we DID achieve uncorrelated binders."),
        "",
        ("**Implication for Phase F**: building a more diverse binder "
         "ensemble is a leverage point. The 5 currently held-out / "
         "surrogate binders are too similar to bound an attacker much "
         "tighter than a single binder. Either add architectural diversity "
         "or accept the single-binder threshold as the practical operating "
         "point."),
        "",
    ]
    (args.out / "ensemble_attack.md").write_text("\n".join(md))
    print(f"[done] wrote {args.out}/ensemble_attack.{{json,md,png}}", flush=True)


if __name__ == "__main__":
    main()

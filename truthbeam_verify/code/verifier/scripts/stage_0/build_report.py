"""Stage 0 — assemble per-F-A-checkpoint comparison report.

Reads per-checkpoint summary.json files from `<runs-root>/eval/step_NNNNNNNN/`,
produces a paper-friendly comparison report.

Headline questions Stage 0 answers:
  1. Does the Phase G diffusion verifier distinguish F-A's outputs from real
     captures?  →  AUROC real_correct vs fake_correct.
  2. Does F-A v1 reproduce the chain-coupling signal?  →  AUROC fake_correct
     vs fake_shuffled (does the F-A-fake C still benefit from correct E?).
  3. Does the gap between real and F-A outputs change across F-A training?
     →  paired_gap_correct vs F-A checkpoint step.

Usage:
  python scripts/stage_0/build_report.py \\
    --eval-root /path/to/poliebotics_phase_b/experiments/stage_0/eval \\
    --out /path/to/poliebotics_phase_b/experiments/stage_0/report
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_per_checkpoint(eval_root: Path) -> list[tuple[int, dict]]:
    """Return [(step, summary_dict), ...] sorted by step."""
    out = []
    for d in sorted(eval_root.glob("step_*")):
        if not d.is_dir():
            continue
        m = re.match(r"step_(\d+)", d.name)
        if not m:
            continue
        step = int(m.group(1))
        sp = d / "summary.json"
        if sp.exists():
            out.append((step, json.loads(sp.read_text())))
    return out


def fmt_ci(d: dict | None) -> str:
    if d is None: return "—"
    return f"{d['mean']:+.5f} [{d['ci_low']:+.5f}, {d['ci_high']:+.5f}]"


def fmt_mean_ci(d: dict | None) -> str:
    if d is None: return "—"
    return f"{d['mean']:.5f} [{d['ci_low']:.5f}, {d['ci_high']:.5f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out / "figures"
    fig_dir.mkdir(exist_ok=True)

    summaries = load_per_checkpoint(args.eval_root)
    if not summaries:
        print(f"[error] no summary.json files found under {args.eval_root}/step_*")
        return
    steps = [s for s, _ in summaries]
    print(f"[init] found {len(summaries)} F-A checkpoints: {steps}")

    # Build per-session per-step series
    series: dict[str, dict[str, list[float]]] = {
        "D2": {"auroc_rvf": [], "auroc_fcfs": [], "paired_gap_correct": [],
               "paired_gap_zero": [], "paired_gap_source": []},
        "V10": {"auroc_rvf": [], "auroc_fcfs": [], "paired_gap_correct": [],
                "paired_gap_zero": [], "paired_gap_source": []},
    }
    for step, summary in summaries:
        for sess in ("D2", "V10"):
            s = summary.get("sessions", {}).get(sess)
            if s is None:
                for k in series[sess]:
                    series[sess][k].append(float("nan"))
                continue
            au = s.get("auroc", {})
            de = s.get("deltas", {})
            series[sess]["auroc_rvf"].append(
                au.get("real_correct_vs_fake_correct", float("nan")))
            series[sess]["auroc_fcfs"].append(
                au.get("fake_correct_vs_fake_shuffled", float("nan")))
            series[sess]["paired_gap_correct"].append(
                de.get("paired_gap_correct", {}).get("mean", float("nan")))
            series[sess]["paired_gap_zero"].append(
                de.get("paired_gap_zero", {}).get("mean", float("nan")))
            series[sess]["paired_gap_source"].append(
                de.get("paired_gap_source", {}).get("mean", float("nan")))

    # Plot AUROC trajectory across F-A training
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, sess in zip(axes, ("D2", "V10")):
        ax.plot(steps, series[sess]["auroc_rvf"], marker="o",
                label="AUROC real vs fake (correct E)", color="#1f77b4")
        ax.plot(steps, series[sess]["auroc_fcfs"], marker="s",
                label="AUROC fake-correct vs fake-shuffled", color="#d62728")
        ax.axhline(0.5, color="gray", lw=0.5, ls="--", label="chance")
        ax.set_xlabel("F-A v1 training step")
        ax.set_ylabel("AUROC")
        ax.set_title(f"Stage 0 — {sess}")
        ax.set_ylim(0.4, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "auroc_trajectory.png", dpi=140)
    plt.close(fig)

    # Plot paired gap trajectory
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, sess in zip(axes, ("D2", "V10")):
        ax.plot(steps, series[sess]["paired_gap_correct"], marker="o",
                label="paired_gap correct E", color="#1f77b4")
        ax.plot(steps, series[sess]["paired_gap_zero"], marker="s",
                label="paired_gap zero E", color="#888888")
        ax.plot(steps, series[sess]["paired_gap_source"], marker="^",
                label="paired_gap source E", color="#2ca02c")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("F-A v1 training step")
        ax.set_ylabel("paired_gap (MSE_fake - MSE_real)")
        ax.set_title(f"Stage 0 paired gaps — {sess}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "paired_gap_trajectory.png", dpi=140)
    plt.close(fig)

    # Compose markdown
    lines = [
        "# Stage 0 — F-A v1 vs Phase G diffusion verifier",
        "",
        f"Per-checkpoint comparison across {len(summaries)} F-A v1 training stages.",
        "",
        "## Headline questions",
        "1. **Does the Phase G diffusion verifier distinguish F-A outputs from real "
        "captures?** Look at AUROC real_correct vs fake_correct: 0.5 = indistinguishable, "
        "1.0 = perfectly catches F-A.",
        "2. **Does F-A v1 reproduce the chain-coupling signal we found in Phase G?** "
        "Look at AUROC fake_correct vs fake_shuffled: if F-A reproduces the coupling, "
        "this should be near 1.0 (matching Phase G's main result). If F-A makes outputs "
        "that ignore E, this drops to 0.5.",
        "3. **Does F-A get harder or easier to detect across training?** Look at the "
        "paired_gap_correct trajectory.",
        "",
        "## Trajectory plots",
        "",
        "![AUROC trajectory](figures/auroc_trajectory.png)",
        "",
        "![Paired gap trajectory](figures/paired_gap_trajectory.png)",
        "",
        "## Per-checkpoint × per-session metrics",
        "",
    ]
    for sess in ("D2", "V10"):
        lines.append(f"### {sess}")
        lines.append("")
        lines.append("| F-A step | AUROC real vs fake | AUROC fake-correct vs fake-shuffled | paired_gap correct E | paired_gap zero E | paired_gap source E |")
        lines.append("|---|---|---|---|---|---|")
        for step, summary in summaries:
            s = summary.get("sessions", {}).get(sess)
            if s is None:
                lines.append(f"| {step} | (no data) |  |  |  |  |")
                continue
            au = s.get("auroc", {})
            de = s.get("deltas", {})
            lines.append(
                f"| {step} | "
                f"{au.get('real_correct_vs_fake_correct', float('nan')):.3f} | "
                f"{au.get('fake_correct_vs_fake_shuffled', float('nan')):.3f} | "
                f"{fmt_ci(de.get('paired_gap_correct'))} | "
                f"{fmt_ci(de.get('paired_gap_zero'))} | "
                f"{fmt_ci(de.get('paired_gap_source'))} |"
            )
        lines.append("")

    lines.append("## Real-baseline lifts (Case 3 diagnosis per spec)")
    lines.append("")
    lines.append("| F-A step | session | real_zero_lift | fake_zero_lift | real_shuffled_lift | fake_shuffled_lift |")
    lines.append("|---|---|---|---|---|---|")
    for step, summary in summaries:
        for sess in ("D2", "V10"):
            s = summary.get("sessions", {}).get(sess)
            if s is None:
                continue
            de = s.get("deltas", {})
            lines.append(
                f"| {step} | {sess} | "
                f"{fmt_ci(de.get('real_zero_lift'))} | "
                f"{fmt_ci(de.get('fake_zero_lift'))} | "
                f"{fmt_ci(de.get('real_shuffled_lift'))} | "
                f"{fmt_ci(de.get('fake_shuffled_lift'))} |"
            )
    lines.append("")

    lines.append("## Interpretation cues")
    lines.append("")
    lines.append("- If `AUROC real vs fake` → 1.0 across all checkpoints: **F-A's outputs "
                 "are easily caught** by the Phase G diffusion verifier (Case: F-A FAILS).")
    lines.append("- If `AUROC real vs fake` ≈ 0.5: **F-A's outputs are indistinguishable "
                 "from real captures** under Phase G. F-A passes diffusion.")
    lines.append("- If `AUROC fake-correct vs fake-shuffled` → 1.0: F-A's outputs "
                 "reproduce the chain coupling (E correctly affects predicted ε on F-A "
                 "outputs). F-A learned the coupling.")
    lines.append("- If `AUROC fake-correct vs fake-shuffled` ≈ 0.5: F-A's outputs lack "
                 "chain coupling. The model is ignoring E in F-A's space (binder-leak "
                 "failure mode at the diffusion level).")
    lines.append("- A sharp `paired_gap_source < 0` would indicate F-A successfully "
                 "ERASED E_source (the projection in C_source) — its training objective.")
    lines.append("")

    out_md = args.out / "stage_0_report.md"
    out_md.write_text("\n".join(lines))
    print(f"[done] wrote {out_md}")


if __name__ == "__main__":
    main()

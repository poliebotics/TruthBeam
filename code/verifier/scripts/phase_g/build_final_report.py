"""Phase G — assemble the final diagnostic_report.md from the 3 runs' eval outputs.

Reads:
  <runs>/main/eval/summary.json
  <runs>/shuffled/eval/summary.json
  <runs>/synthetic_positive/eval/summary.json
  <runs>/main/history.jsonl
  <runs>/shuffled/history.jsonl
  <runs>/synthetic_positive/history.jsonl

Writes:
  results/diffusion_diagnostic/diagnostic_report.md  — final report with
    decision interpretation, tables, plots
  results/diffusion_diagnostic/figures/              — PNG plots:
    - loss_curves.png  (3 runs overlaid)
    - lag_sweep_d2.png (3 runs)
    - lag_sweep_v10.png (3 runs)
    - delta_summary.png (Δ_wrong with CI per run × session)

Run after all 3 eval summary.json files exist:
  python scripts/phase_g/build_final_report.py \
      --runs-root /path/to/poliebotics_phase_b/experiments/phase_g_diffusion_diagnostic \
      --out results/diffusion_diagnostic
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RUN_NAMES = ("main", "shuffled", "synthetic_positive")
COLORS = {"main": "#1f77b4", "shuffled": "#888888", "synthetic_positive": "#d62728"}


def read_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def read_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def fmt_ci(d: dict | None) -> str:
    if d is None:
        return "—"
    return f"{d['mean']:+.5f} [{d['ci_low']:+.5f}, {d['ci_high']:+.5f}]"


def fmt_mean_ci(d: dict | None) -> str:
    if d is None:
        return "—"
    return f"{d['mean']:.5f} [{d['ci_low']:.5f}, {d['ci_high']:.5f}]"


def autonomous_decision(summaries: dict[str, dict | None]) -> tuple[str, list[str]]:
    """Apply the operator's decision criteria. Returns (label, reasoning)."""
    main = summaries.get("main")
    shuffled = summaries.get("shuffled")
    synth = summaries.get("synthetic_positive")
    reasons = []

    def synth_passes() -> bool:
        if synth is None:
            return False
        for sess in ("D2", "V10"):
            s = synth.get("sessions", {}).get(sess)
            if not s:
                continue
            d_w = s.get("deltas", {}).get("delta_wrong")
            au = s.get("auroc", {}).get("correct_vs_wrong_avg")
            if d_w and d_w["mean"] > 0 and d_w["ci_low"] > 0:
                return True
            if au is not None and au >= 0.65:
                return True
        return False

    def shuffled_near_chance() -> bool:
        if shuffled is None:
            return False
        # Shuffled is "near chance" if:
        #   - AUROC ≤ 0.60 in both sessions (chance with margin), AND
        #   - |Δ_wrong| is small in absolute terms (< 1e-3 — tighter than any
        #     plausible signal) OR small relative to main's Δ_wrong.
        # We DON'T use the "CI excludes zero" check alone because, with
        # near-tie data, the CI can be tight at a value 4-5 orders of
        # magnitude below any signal of interest (e.g. +1e-7 with CI
        # [+5e-8, +1.5e-7]) — float noise, not signal.
        main_dw = None
        if main is not None:
            for sess in ("D2", "V10"):
                ms = main.get("sessions", {}).get(sess)
                if ms is None:
                    continue
                d = ms.get("deltas", {}).get("delta_wrong")
                if d is not None:
                    main_dw = max(main_dw or 0.0, abs(d.get("mean", 0.0)))
        for sess in ("D2", "V10"):
            s = shuffled.get("sessions", {}).get(sess)
            if not s:
                continue
            d_w = s.get("deltas", {}).get("delta_wrong")
            au = s.get("auroc", {}).get("correct_vs_wrong_avg")
            # AUROC must be near chance
            if au is not None and au > 0.60:
                return False
            # Magnitude check: shuffled Δ_wrong shouldn't approach main's
            if d_w is not None and main_dw is not None and main_dw > 0:
                if abs(d_w.get("mean", 0.0)) > 0.10 * main_dw:
                    # shuffled has > 10% of main's effect size → suspect leak
                    return False
            # Absolute magnitude floor
            if d_w is not None and abs(d_w.get("mean", 0.0)) > 1e-3:
                return False
        return True

    def main_strong_positive() -> bool:
        if main is None:
            return False
        for sess in ("D2", "V10"):
            s = main.get("sessions", {}).get(sess)
            if not s:
                continue
            d_w = s.get("deltas", {}).get("delta_wrong")
            au = s.get("auroc", {}).get("correct_vs_wrong_avg")
            cond_a = d_w is not None and d_w["mean"] > 0 and d_w["ci_low"] > 0
            cond_b = au is not None and au >= 0.70
            if cond_a or cond_b:
                # also check lag peak at k=0 or ±1
                lag = s.get("lag_curve", {})
                if lag:
                    means = {int(k): v["mean"] for k, v in lag.items()}
                    # MSE: lower is better, so peak = argmin
                    best_k = min(means, key=lambda k: means[k])
                    if abs(best_k) <= 1:
                        return True
        return False

    def main_clean_negative() -> bool:
        if main is None:
            return False
        for sess in ("D2", "V10"):
            s = main.get("sessions", {}).get(sess)
            if not s:
                continue
            d_w = s.get("deltas", {}).get("delta_wrong")
            au = s.get("auroc", {}).get("correct_vs_wrong_avg")
            # tight CI around 0 + AUROC near 0.5
            if d_w is None or au is None:
                continue
            tight_zero = abs(d_w["mean"]) < 1e-4 and d_w["ci_low"] < 0 < d_w["ci_high"]
            chance_auroc = 0.45 <= au <= 0.55
            if not (tight_zero and chance_auroc):
                return False
        return True

    if not synth_passes():
        reasons.append("synthetic_positive control did NOT pass — pipeline cannot reliably "
                       "detect E-dependent signal even when one is injected by construction. "
                       "The diagnostic is uninterpretable.")
        return "UNINTERPRETABLE", reasons

    if not shuffled_near_chance():
        reasons.append("shuffled control shows advantage — train-set leak suspected. "
                       "Cannot distinguish chain coupling from data statistics.")
        return "UNINTERPRETABLE", reasons

    if main_strong_positive():
        reasons.append("main run shows Δ_wrong > 0 with CI excluding zero (or AUROC ≥ 0.70) "
                       "in at least one held-out session AND lag peak at k≈0.")
        reasons.append("synthetic_positive control passed (pipeline can detect E-dependent signal).")
        reasons.append("shuffled near chance (no train-set leak detected).")
        return "STRONG POSITIVE", reasons

    if main_clean_negative():
        reasons.append("main run shows Δ_wrong ≈ 0 with tight CIs spanning zero AND "
                       "AUROC ≈ 0.5 in both D2 and V10.")
        reasons.append("synthetic_positive control passed (so the absence of signal in main "
                       "is informative, not a pipeline failure).")
        reasons.append("shuffled near chance.")
        return "CLEAN NEGATIVE", reasons

    reasons.append("main run shows neither a strong positive (Δ_wrong CI excludes zero with "
                   "k=0 lag peak) nor a clean negative (tight CI at zero, AUROC=0.5).")
    reasons.append("This is an ambiguous result; the diagnostic cannot distinguish weak-signal "
                   "from noise. Document the numerical findings without a project-level claim.")
    return "AMBIGUOUS", reasons


# -------------- plots --------------

def plot_loss_curves(runs_root: Path, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in RUN_NAMES:
        h = read_history(runs_root / name / "history.jsonl")
        if not h:
            continue
        steps = [r["step"] for r in h]
        loss = [r["loss"] for r in h]
        ax.plot(steps, loss, label=name, color=COLORS[name], lw=1.0)
    ax.set_xlabel("training step")
    ax.set_ylabel("ε MSE loss")
    ax.set_title("Phase G — diffusion training loss curves")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_lag_sweep(summaries: dict[str, dict | None], session: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in RUN_NAMES:
        s = summaries.get(name)
        if s is None:
            continue
        sess = s.get("sessions", {}).get(session)
        if sess is None:
            continue
        lag = sess.get("lag_curve", {})
        if not lag:
            continue
        ks_sorted = sorted(int(k) for k in lag)
        means = [lag[str(k)]["mean"] for k in ks_sorted]
        lows = [lag[str(k)]["ci_low"] for k in ks_sorted]
        highs = [lag[str(k)]["ci_high"] for k in ks_sorted]
        ax.plot(ks_sorted, means, label=name, color=COLORS[name], lw=1.5, marker="o", ms=4)
        ax.fill_between(ks_sorted, lows, highs, color=COLORS[name], alpha=0.15)
    ax.set_xlabel("lag k (rows: E_{r+k} vs C_r)")
    ax.set_ylabel("mean ε MSE")
    ax.set_title(f"Phase G — lag sweep, {session} held-out")
    ax.axvline(0, color="black", lw=0.5, ls="--")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_delta_summary(summaries: dict[str, dict | None], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    rows = []
    for name in RUN_NAMES:
        s = summaries.get(name)
        if s is None:
            continue
        for sess in ("D2", "V10"):
            sess_data = s.get("sessions", {}).get(sess)
            if sess_data is None:
                continue
            d_w = sess_data.get("deltas", {}).get("delta_wrong")
            if d_w is None:
                continue
            rows.append((name, sess, d_w["mean"], d_w["ci_low"], d_w["ci_high"]))
    if not rows:
        plt.close(fig)
        return
    xs = list(range(len(rows)))
    means = [r[2] for r in rows]
    err_low = [r[2] - r[3] for r in rows]
    err_high = [r[4] - r[2] for r in rows]
    colors = [COLORS[r[0]] for r in rows]
    ax.errorbar(xs, means, yerr=[err_low, err_high], fmt='o', ecolor='black',
                capsize=4, ms=8, color="white", markeredgecolor="black")
    for x, c in zip(xs, colors):
        ax.scatter([x], [means[x]], c=c, s=80, zorder=10)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r[0]}\n{r[1]}" for r in rows], rotation=0, fontsize=9)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Δ_wrong = MSE(wrong E) − MSE(correct E)")
    ax.set_title("Phase G — Δ_wrong with 95% block-bootstrap CIs")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# -------------- table builder --------------

def build_table(summaries: dict[str, dict | None], session: str) -> list[str]:
    lines = [
        f"### Evaluation table — {session} held-out (200 frames)",
        "",
        "| run | mean MSE correct | mean MSE wrong (avg) | mean MSE uncond | Δ_wrong (95% CI) | Δ_uncond (95% CI) | AUROC vs wrong | AUROC vs uncond |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in RUN_NAMES:
        s = summaries.get(name)
        if s is None:
            lines.append(f"| {name} | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        sess = s.get("sessions", {}).get(session)
        if sess is None:
            lines.append(f"| {name} | (no {session} eval) | — | — | — | — | — | — |")
            continue
        bc = sess.get("by_condition", {})
        de = sess.get("deltas", {})
        au = sess.get("auroc", {})
        wrong_means = [bc.get(f"wrong_{off:+d}", {}) for off in (-2, +2, -15, +15, +30)]
        wrong_avg_mean = float(np.mean([w.get("mean", float("nan")) for w in wrong_means
                                        if w])) if wrong_means else float("nan")
        lines.append(
            f"| {name} | "
            f"{bc.get('correct', {}).get('mean', float('nan')):.5f} | "
            f"{wrong_avg_mean:.5f} | "
            f"{bc.get('uncond', {}).get('mean', float('nan')):.5f} | "
            f"{fmt_ci(de.get('delta_wrong'))} | "
            f"{fmt_ci(de.get('delta_uncond'))} | "
            f"{au.get('correct_vs_wrong_avg', float('nan')):.3f} | "
            f"{au.get('correct_vs_uncond', float('nan')):.3f} |"
        )
    return lines


# -------------- main --------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="Local results dir (also fig destination).")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Load summaries
    summaries: dict[str, dict | None] = {}
    for name in RUN_NAMES:
        sp = args.runs_root / name / "eval" / "summary.json"
        summaries[name] = read_summary(sp)
        if summaries[name] is None:
            print(f"[warn] missing {sp}")

    # Plots
    plot_loss_curves(args.runs_root, fig_dir / "loss_curves.png")
    plot_lag_sweep(summaries, "D2", fig_dir / "lag_sweep_d2.png")
    plot_lag_sweep(summaries, "V10", fig_dir / "lag_sweep_v10.png")
    plot_delta_summary(summaries, fig_dir / "delta_summary.png")

    # Decision
    decision_label, decision_reasons = autonomous_decision(summaries)

    # Compose markdown
    lines = [
        "# Phase G — diffusion diagnostic report",
        "",
        f"**Decision (autonomous): {decision_label}**",
        "",
    ]
    for r in decision_reasons:
        lines.append(f"- {r}")
    lines.append("")

    # Spec recap (preserved from scaffold)
    scaffold_path = args.out / "diagnostic_report_scaffold.md"
    if scaffold_path.exists():
        lines.append("## (Spec recap, preprocessing, training config — see scaffold)\n")

    lines.append("## Loss curves")
    lines.append("")
    lines.append(f"![loss curves](figures/loss_curves.png)")
    lines.append("")

    # Tables
    for sess in ("D2", "V10"):
        lines.extend(build_table(summaries, sess))
        lines.append("")

    # Lag sweep plots
    lines.append("## Lag sweep (chain alignment check)")
    lines.append("")
    lines.append("D2 held-out:")
    lines.append("")
    lines.append("![D2 lag sweep](figures/lag_sweep_d2.png)")
    lines.append("")
    lines.append("V10 held-out:")
    lines.append("")
    lines.append("![V10 lag sweep](figures/lag_sweep_v10.png)")
    lines.append("")

    # Delta summary
    lines.append("## Δ_wrong summary across all runs and sessions")
    lines.append("")
    lines.append(f"![delta summary](figures/delta_summary.png)")
    lines.append("")

    # Per-timestep tables
    lines.append("## Per-timestep breakdown (primary timesteps)")
    lines.append("")
    for sess in ("D2", "V10"):
        lines.append(f"### {sess}")
        lines.append("")
        lines.append("| run | t | Δ_wrong (95% CI) | Δ_uncond (95% CI) | AUROC vs wrong |")
        lines.append("|---|---|---|---|---|")
        for name in RUN_NAMES:
            s = summaries.get(name)
            if s is None:
                continue
            sess_data = s.get("sessions", {}).get(sess, {})
            for t_label, t_data in sess_data.get("by_timestep", {}).items():
                lines.append(
                    f"| {name} | {t_label} | "
                    f"{fmt_ci(t_data.get('delta_wrong'))} | "
                    f"{fmt_ci(t_data.get('delta_uncond'))} | "
                    f"{t_data.get('auroc_correct_vs_wrong_avg', float('nan')):.3f} |"
                )
        lines.append("")

    # Caveats
    lines.append("## Caveats")
    lines.append("")
    lines.append("- Aspect ratio: cropped frame 1704×2278 has aspect 0.748; resized to "
                 "768×1024 (aspect 0.75). Distortion is small (≈0.3%).")
    lines.append("- V10 may have ambient light differences vs D2; check whether D2 and "
                 "V10 results agree before drawing strong conclusions.")
    lines.append("- Held-out blocks are within the same session as training, so this "
                 "diagnostic measures within-session generalization, NOT cross-session.")
    lines.append("- Synthetic positive control uses isotropic photometric injection "
                 "(blur(E) added to C); a real chain signal may have different spectral "
                 "structure, so synth-pass does not guarantee main-pass.")
    lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Raw eval data: `{args.runs_root}/{{main,shuffled,synthetic_positive}}/eval/eval_*_raw.npz`")
    lines.append(f"- Per-run summaries: same dir, `summary.json`")
    lines.append(f"- Loss histories: `{args.runs_root}/<run>/history.jsonl`")
    lines.append(f"- Final model checkpoints: `{args.runs_root}/<run>/model_final.pt`")
    lines.append(f"- Intermediate checkpoints (every 5k steps): "
                 f"`{args.runs_root}/<run>/checkpoints/step_NNNNNNNN.pt`")

    # Write
    out_md = args.out / "diagnostic_report.md"
    out_md.write_text("\n".join(lines))
    print(f"[done] wrote {out_md}")
    print(f"[done] decision: {decision_label}")
    for r in decision_reasons:
        print(f"  - {r}")


if __name__ == "__main__":
    main()

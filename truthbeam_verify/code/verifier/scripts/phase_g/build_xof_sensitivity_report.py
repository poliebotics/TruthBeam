"""Item 1 — XOF perturbation sensitivity report.

Reads `<eval-dir>/summary.json` produced by `eval_diffusion_diagnostic.py
--extended-perturbations` and produces sensitivity curves + tables.

Headline plots:
  - Type 1 (global bit-flip): MSE vs k on log scale (sensitivity curve).
  - Type 2 (octave-localized): heatmap MSE vs (octave, k).
  - Type 4 (octave swap): MSE per octave (which octaves carry diagnostic signal).
  - Type 5 (channel swap): MSE per RGB channel.
  - Type 6 calibration: should match wrong_+30 within float precision.

Usage:
  python scripts/phase_g/build_xof_sensitivity_report.py \\
    --eval-dir /path/to/poliebotics_phase_b/experiments/item_1/eval \\
    --out /path/to/poliebotics_phase_b/experiments/item_1/report
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


TYPE1_K = (1, 4, 16, 64, 256, 1024, 4096)
TYPE2_OCTAVES = (0, 1, 2, 3)
TYPE2_K = (1, 4, 16, 64)
TYPE3_K = (16, 64, 256)
CHANNEL_NAMES = ("R", "G", "B")


def per_cond_mean(summary: dict, sess: str, label: str) -> dict | None:
    s = summary.get("sessions", {}).get(sess, {})
    return s.get("by_condition", {}).get(label)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out / "figures"
    fig_dir.mkdir(exist_ok=True)

    summary_path = args.eval_dir / "summary.json"
    if not summary_path.exists():
        print(f"[error] {summary_path} missing")
        return
    summary = json.loads(summary_path.read_text())
    sessions = list(summary.get("sessions", {}).keys())
    print(f"[init] sessions: {sessions}")

    # ---------------- Type 1 sensitivity curve ----------------
    fig, axes = plt.subplots(1, len(sessions), figsize=(6 * len(sessions), 4.5),
                             squeeze=False, sharey=True)
    for col, sess in enumerate(sessions):
        ax = axes[0, col]
        correct = per_cond_mean(summary, sess, "correct")
        if correct:
            ax.axhline(correct["mean"], color="black", lw=1.5, ls="--",
                       label=f"correct E baseline ({correct['mean']:.5f})")
        means, errs = [], []
        for k in TYPE1_K:
            d = per_cond_mean(summary, sess, f"xof_t1_global_k{k}")
            if d is None:
                means.append(np.nan); errs.append(np.nan)
            else:
                means.append(d["mean"])
                errs.append((d["ci_high"] - d["ci_low"]) / 2)
        ax.errorbar(TYPE1_K, means, yerr=errs, marker="o", color="#d62728",
                    label="Type 1 (global bit-flip)", capsize=4)
        ax.set_xscale("log")
        ax.set_xlabel("k (bits flipped)")
        ax.set_ylabel("mean ε MSE (frame-aggregated)")
        ax.set_title(f"Type 1 sensitivity — {sess}")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "type1_sensitivity.png", dpi=140)
    plt.close(fig)

    # ---------------- Type 2 heatmap ----------------
    fig, axes = plt.subplots(1, len(sessions), figsize=(6 * len(sessions), 4),
                             squeeze=False)
    for col, sess in enumerate(sessions):
        ax = axes[0, col]
        H = np.full((len(TYPE2_OCTAVES), len(TYPE2_K)), np.nan)
        for i, oct_idx in enumerate(TYPE2_OCTAVES):
            for j, k in enumerate(TYPE2_K):
                d = per_cond_mean(summary, sess, f"xof_t2_oct{oct_idx}_k{k}")
                if d is not None:
                    H[i, j] = d["mean"]
        im = ax.imshow(H, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(TYPE2_K)))
        ax.set_xticklabels([str(k) for k in TYPE2_K])
        ax.set_yticks(range(len(TYPE2_OCTAVES)))
        ax.set_yticklabels([f"oct {o}" for o in TYPE2_OCTAVES])
        ax.set_xlabel("k bits flipped")
        ax.set_ylabel("octave")
        ax.set_title(f"Type 2 heatmap — {sess}")
        for i in range(len(TYPE2_OCTAVES)):
            for j in range(len(TYPE2_K)):
                if not np.isnan(H[i, j]):
                    ax.text(j, i, f"{H[i,j]:.4f}", ha="center", va="center",
                            color="white" if H[i,j] > H[~np.isnan(H)].mean() else "black",
                            fontsize=8)
        plt.colorbar(im, ax=ax, label="mean ε MSE")
    fig.tight_layout()
    fig.savefig(fig_dir / "type2_octave_localized.png", dpi=140)
    plt.close(fig)

    # ---------------- Type 4 / 5 / 6 bars ----------------
    fig, axes = plt.subplots(2, len(sessions), figsize=(6 * len(sessions), 8),
                             squeeze=False)
    for col, sess in enumerate(sessions):
        # Type 4 octave swap
        ax = axes[0, col]
        means_t4 = []
        for oct_idx in (0, 1, 2, 3):
            d = per_cond_mean(summary, sess, f"xof_t4_swap_oct{oct_idx}")
            means_t4.append(d["mean"] if d else np.nan)
        ax.bar([f"oct {o}" for o in (0, 1, 2, 3)], means_t4, color="#1f77b4")
        correct = per_cond_mean(summary, sess, "correct")
        if correct:
            ax.axhline(correct["mean"], color="black", lw=1, ls="--", label="correct E")
        ax.set_ylabel("mean ε MSE")
        ax.set_title(f"Type 4 octave swap — {sess}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")

        # Type 5 channel swap + Type 6 replace
        ax = axes[1, col]
        labels_t56 = [f"chan {c}" for c in CHANNEL_NAMES] + ["replace_general", "calib"]
        means_t56 = []
        for c in (0, 1, 2):
            d = per_cond_mean(summary, sess, f"xof_t5_swap_{CHANNEL_NAMES[c]}")
            means_t56.append(d["mean"] if d else np.nan)
        d = per_cond_mean(summary, sess, "xof_t6_replace_general")
        means_t56.append(d["mean"] if d else np.nan)
        d = per_cond_mean(summary, sess, "xof_t6_calibration_row+30")
        means_t56.append(d["mean"] if d else np.nan)
        ax.bar(labels_t56, means_t56, color=["#2ca02c", "#2ca02c", "#2ca02c", "#d62728", "#ff7f0e"])
        if correct:
            ax.axhline(correct["mean"], color="black", lw=1, ls="--", label="correct E")
        # wrong_+30 reference line for calibration check
        wrong30 = per_cond_mean(summary, sess, "wrong_+30")
        if wrong30:
            ax.axhline(wrong30["mean"], color="orange", lw=1, ls=":",
                       label=f"wrong_+30 ({wrong30['mean']:.5f})")
        ax.set_ylabel("mean ε MSE")
        ax.set_title(f"Type 5/6 swap — {sess}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(fig_dir / "type4_5_6_swaps.png", dpi=140)
    plt.close(fig)

    # ---------------- Type 6 calibration check ----------------
    calib_check = []
    for sess in sessions:
        t6_calib = per_cond_mean(summary, sess, "xof_t6_calibration_row+30")
        wrong30 = per_cond_mean(summary, sess, "wrong_+30")
        if t6_calib and wrong30:
            calib_check.append({
                "session": sess,
                "t6_calib_mean": t6_calib["mean"],
                "wrong_+30_mean": wrong30["mean"],
                "diff": t6_calib["mean"] - wrong30["mean"],
                "match": abs(t6_calib["mean"] - wrong30["mean"]) < 1e-4,
            })

    # ---------------- Compose markdown ----------------
    lines = [
        "# Item 1 — XOF perturbation sensitivity report",
        "",
        "## Plots",
        "",
        "![Type 1 sensitivity curve](figures/type1_sensitivity.png)",
        "",
        "![Type 2 octave-localized heatmap](figures/type2_octave_localized.png)",
        "",
        "![Type 4/5/6 swap effects](figures/type4_5_6_swaps.png)",
        "",
        "## Calibration check (Type 6 vs wrong_+30)",
        "",
        "Should match within float precision since both render to E from chain[row+30].",
        "",
        "| session | Type 6 calibration | wrong_+30 | diff | match |",
        "|---|---|---|---|---|",
    ]
    for c in calib_check:
        lines.append(
            f"| {c['session']} | {c['t6_calib_mean']:.6f} | "
            f"{c['wrong_+30_mean']:.6f} | {c['diff']:+.6e} | "
            f"{'YES' if c['match'] else 'NO'} |"
        )
    lines.append("")

    lines.append("## Sensitivity table (all 35 conditions, both sessions)")
    lines.append("")
    lines.append("| condition | D2 mean MSE | V10 mean MSE |")
    lines.append("|---|---|---|")
    all_conds = []
    for k in TYPE1_K:
        all_conds.append(f"xof_t1_global_k{k}")
    for o in TYPE2_OCTAVES:
        for k in TYPE2_K:
            all_conds.append(f"xof_t2_oct{o}_k{k}")
    for k in TYPE3_K:
        all_conds.append(f"xof_t3_region_k{k}")
    for o in (0, 1, 2, 3):
        all_conds.append(f"xof_t4_swap_oct{o}")
    for c in CHANNEL_NAMES:
        all_conds.append(f"xof_t5_swap_{c}")
    all_conds.append("xof_t6_replace_general")
    all_conds.append("xof_t6_calibration_row+30")
    for cond in all_conds:
        d2 = per_cond_mean(summary, "D2", cond)
        v10 = per_cond_mean(summary, "V10", cond)
        d2_str = f"{d2['mean']:.5f}" if d2 else "—"
        v10_str = f"{v10['mean']:.5f}" if v10 else "—"
        lines.append(f"| {cond} | {d2_str} | {v10_str} |")

    out_md = args.out / "xof_sensitivity_report.md"
    out_md.write_text("\n".join(lines))
    print(f"[done] wrote {out_md}")


if __name__ == "__main__":
    main()

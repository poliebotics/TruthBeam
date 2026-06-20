"""Tier 0 Analysis X3 — Stage 0 four-condition decomposition figure.

visualize what F-A v1 achieves and what it
doesn't. Uses target/zero/shuffled/source MSE for both real and F-A across
all 4 checkpoints (existing data, no new compute).

Sign conventions (operator-fixed):
    All diffusion MSE: lower = more compatible / more real-like.
    target_lift  = m_zero - m_target           (positive: correct E helps)
    shuffle_lift = m_shuffled - m_target        (positive: target E beats shuffled)
    source_gap   = m_source - m_target          (negative: source leakage)

Outputs:
    experiments/paper_analyses/stage_0_decomposition/four_condition_bars.png
    experiments/paper_analyses/stage_0_decomposition/lift_trajectories.png
    experiments/paper_analyses/stage_0_decomposition/decomposition_report.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LOCAL_ROOT = Path("/path/to/poliebotics_phase_b")
DEFAULT_OUT = LOCAL_ROOT / "experiments" / "paper_analyses" / "stage_0_decomposition"
STAGE_0_EVAL = LOCAL_ROOT / "experiments" / "stage_0" / "eval"
CKPT_STEPS = (5000, 25000, 70000, 100000)
SESSIONS = ("d2", "v10")


def load_per_frame_means(ckpt: int, sess: str) -> dict[str, np.ndarray]:
    """Returns {cond_name → (n_frames,) per-frame mean MSE}."""
    npz = STAGE_0_EVAL / f"step_{ckpt:08d}" / f"stage0_{sess}_raw.npz"
    z = np.load(npz, allow_pickle=False)
    out = {}
    for ctype in ("real", "fake"):
        for cond in ("correct", "shuffled", "source", "zero"):
            key = f"cond_{ctype}_{cond}"
            if key in z.files:
                out[f"{ctype}_{cond}"] = z[key].mean(axis=(1, 2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Load all data
    data: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for ckpt in CKPT_STEPS:
        data[ckpt] = {}
        for sess in SESSIONS:
            data[ckpt][sess] = load_per_frame_means(ckpt, sess)

    # ==== Figure 1: bars per condition × ckpt ====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    cond_order = [
        ("real_correct",   "real C\ntarget E"),
        ("real_zero",      "real C\nzero E"),
        ("real_shuffled",  "real C\nshuffled E"),
        ("fake_correct",   "F-A C\ntarget E"),
        ("fake_zero",      "F-A C\nzero E"),
        ("fake_shuffled",  "F-A C\nshuffled E"),
        ("fake_source",    "F-A C\nsource E"),
    ]
    n_cond = len(cond_order)
    n_ckpt = len(CKPT_STEPS)
    width = 0.8 / n_ckpt
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, n_ckpt))
    for ax, sess in zip(axes, SESSIONS):
        for ci, ckpt in enumerate(CKPT_STEPS):
            means = []
            stds  = []
            for k, _ in cond_order:
                arr = data[ckpt][sess].get(k)
                if arr is None or arr.size == 0:
                    means.append(np.nan); stds.append(np.nan)
                else:
                    means.append(float(np.nanmean(arr)))
                    stds.append(float(np.nanstd(arr)))
            x = np.arange(n_cond) + (ci - n_ckpt/2 + 0.5) * width
            ax.bar(x, means, width=width, yerr=stds,
                   label=f"step {ckpt//1000}k", color=colors[ci],
                   capsize=2, alpha=0.85)
        ax.set_xticks(np.arange(n_cond))
        ax.set_xticklabels([lbl for _, lbl in cond_order], fontsize=8)
        ax.set_title(f"Session {sess.upper()}")
        ax.set_ylabel("Per-frame mean MSE")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle("Stage 0 — diffusion-verifier MSE by C/E pairing × F-A v1 checkpoint",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out / "four_condition_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[X3] wrote four_condition_bars.png")

    # ==== Figure 2: lift distributions + trajectories ====
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    # Row 1: per-frame distribution boxplots of lifts at step_100k (the "current" model)
    headline_ckpt = 100000
    for col, (lift_name, label) in enumerate([
        ("target_lift", "target_lift = m_zero − m_target"),
        ("shuffle_lift", "shuffle_lift = m_shuffled − m_target"),
        ("source_gap", "source_gap = m_source − m_target"),
    ]):
        ax = axes[0, col]
        for_real = []
        for_fake = []
        for sess in SESSIONS:
            d = data[headline_ckpt][sess]
            real_zero  = d.get("real_zero")
            real_target = d.get("real_correct")
            real_shuf  = d.get("real_shuffled")
            fake_zero  = d.get("fake_zero")
            fake_target = d.get("fake_correct")
            fake_shuf  = d.get("fake_shuffled")
            fake_src   = d.get("fake_source")
            if lift_name == "target_lift":
                if real_zero is not None and real_target is not None:
                    for_real.append(real_zero - real_target)
                if fake_zero is not None and fake_target is not None:
                    for_fake.append(fake_zero - fake_target)
            elif lift_name == "shuffle_lift":
                if real_shuf is not None and real_target is not None:
                    for_real.append(real_shuf - real_target)
                if fake_shuf is not None and fake_target is not None:
                    for_fake.append(fake_shuf - fake_target)
            elif lift_name == "source_gap":
                # source_gap only meaningful for fake (real has no F-A source-E injection)
                if fake_src is not None and fake_target is not None:
                    for_fake.append(fake_src - fake_target)
        # Combine across sessions
        real_combined = np.concatenate(for_real) if for_real else np.array([])
        fake_combined = np.concatenate(for_fake) if for_fake else np.array([])
        positions = []
        data_box = []
        labels_box = []
        if real_combined.size:
            positions.append(0); data_box.append(real_combined); labels_box.append("real C")
        if fake_combined.size:
            positions.append(1); data_box.append(fake_combined); labels_box.append("F-A C @100k")
        if data_box:
            ax.boxplot(data_box, positions=positions, labels=labels_box, widths=0.6,
                       showfliers=True, medianprops={"color":"red","linewidth":1.5})
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("MSE delta")
        ax.grid(alpha=0.3, axis="y")

    # Row 2: trajectories of lift means across F-A checkpoints
    for col, (lift_name, label) in enumerate([
        ("target_lift", "target_lift trajectory"),
        ("shuffle_lift", "shuffle_lift trajectory"),
        ("source_gap", "source_gap trajectory (F-A only)"),
    ]):
        ax = axes[1, col]
        for sess in SESSIONS:
            real_means = []
            fake_means = []
            for ckpt in CKPT_STEPS:
                d = data[ckpt][sess]
                if lift_name == "target_lift":
                    real_arr = d["real_zero"] - d["real_correct"] if "real_zero" in d and "real_correct" in d else None
                    fake_arr = d["fake_zero"] - d["fake_correct"] if "fake_zero" in d and "fake_correct" in d else None
                elif lift_name == "shuffle_lift":
                    real_arr = d["real_shuffled"] - d["real_correct"] if "real_shuffled" in d and "real_correct" in d else None
                    fake_arr = d["fake_shuffled"] - d["fake_correct"] if "fake_shuffled" in d and "fake_correct" in d else None
                else:  # source_gap
                    real_arr = None
                    fake_arr = d["fake_source"] - d["fake_correct"] if "fake_source" in d and "fake_correct" in d else None
                real_means.append(np.nanmean(real_arr) if real_arr is not None else np.nan)
                fake_means.append(np.nanmean(fake_arr) if fake_arr is not None else np.nan)
            steps_k = [c/1000 for c in CKPT_STEPS]
            if not np.isnan(real_means).all():
                ax.plot(steps_k, real_means, marker="o", label=f"{sess.upper()} real", linestyle="--", alpha=0.8)
            if not np.isnan(fake_means).all():
                ax.plot(steps_k, fake_means, marker="s", label=f"{sess.upper()} F-A")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xlabel("F-A checkpoint (×1k steps)")
        ax.set_ylabel("MSE delta (mean over frames)")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Stage 0 — lift decomposition: F-A v1 vs real C", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out / "lift_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[X3] wrote lift_trajectories.png")

    # ==== Markdown report ====
    md = []
    md.append("# Stage 0 four-condition decomposition")
    md.append("")
    md.append(f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    md.append("")
    md.append("## Sign conventions")
    md.append("")
    md.append("All diffusion MSE: **lower = more compatible / more real-like**.")
    md.append("")
    md.append("- `target_lift = m_zero − m_target` — positive means correct E lowers MSE vs zero E (raw E utility).")
    md.append("- `shuffle_lift = m_shuffled − m_target` — positive means target E beats shuffled E (E-coupling).")
    md.append("- `source_gap = m_source − m_target` — negative means source E is preferred (source leakage).")
    md.append("")
    md.append("## Headline numbers (per-frame mean across both sessions)")
    md.append("")
    md.append("| ckpt | metric | real C | F-A C |")
    md.append("|---|---|---|---|")
    for ckpt in CKPT_STEPS:
        rows_combined: dict[str, list[float]] = {}
        for sess in SESSIONS:
            d = data[ckpt][sess]
            for cond in ("correct", "zero", "shuffled"):
                for ctype in ("real", "fake"):
                    key = f"{ctype}_{cond}"
                    if key in d:
                        rows_combined.setdefault(key, []).extend(d[key].tolist())
            if "fake_source" in d:
                rows_combined.setdefault("fake_source", []).extend(d["fake_source"].tolist())
        def m(key):
            v = rows_combined.get(key, [])
            return f"{np.mean(v):.5f}" if v else "—"
        # target_lift = m_zero - m_target
        rl = (np.mean(rows_combined.get("real_zero", [np.nan])) -
              np.mean(rows_combined.get("real_correct", [np.nan])))
        fl = (np.mean(rows_combined.get("fake_zero", [np.nan])) -
              np.mean(rows_combined.get("fake_correct", [np.nan])))
        md.append(f"| step_{ckpt//1000}k | target_lift | {rl:.5f} | {fl:.5f} |")
        rl = (np.mean(rows_combined.get("real_shuffled", [np.nan])) -
              np.mean(rows_combined.get("real_correct", [np.nan])))
        fl = (np.mean(rows_combined.get("fake_shuffled", [np.nan])) -
              np.mean(rows_combined.get("fake_correct", [np.nan])))
        md.append(f"| step_{ckpt//1000}k | shuffle_lift | {rl:.5f} | {fl:.5f} |")
        fl = (np.mean(rows_combined.get("fake_source", [np.nan])) -
              np.mean(rows_combined.get("fake_correct", [np.nan])))
        md.append(f"| step_{ckpt//1000}k | source_gap | — | {fl:.5f} |")
    md.append("")
    md.append("## Interpretation")
    md.append("")
    md.append("- **target_lift > 0 on F-A side**: F-A produces C_fake whose diffusion MSE *responds* to target E (vs zero E). This is the lower-bar success criterion: F-A's outputs are E-conditional, not pure unconditional images.")
    md.append("- **shuffle_lift > 0 on F-A side**: F-A's C_fake actually *prefers* its target E over a shuffled E in the verifier's space — F-A produces target-E-coupled outputs.")
    md.append("- **source_gap < 0 (negative) on F-A side**: F-A's C_fake is scored *better* under source-E than target-E — i.e., F-A leaks source-frame structure into its C_fake. This is the central F-A v1 failure mode the spec aims to characterize.")
    md.append("")
    (args.out / "decomposition_report.md").write_text("\n".join(md))
    print(f"[X3] wrote decomposition_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

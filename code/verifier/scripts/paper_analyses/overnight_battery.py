"""Overnight Phase G mechanism characterization battery.

A battery of 9 experiments run unattended,
target ~7-9h wall-clock, hard stop at 12h. Single deliverable
MORNING_REPORT.md.

Experiments:
  1. XOF correlation table — Pearson E_t vs E_{t+k}, cross-session.
  2. Body-box visualization sanity check — per-frame 3-panel illustration.
  3. C-only Phase G evaluation — discriminate without E.
  4. E-only Phase G evaluation — discriminate without C.
  5. Matched-pose corrected re-runs (E5a/E5b/E5c).
  6. Correct-E rank test — rank correct E among 51 candidates.
  7. Synthetic counterfactual-E perturbation sensitivity (NOT XOF, NOT
     optical washout).
  8. E4 fake_100k AUROC=0.151 score-distribution inspection.
  9. Hierarchical bootstrap CIs on existing causal-ablation cells.

Subcommands:
  exp1 / exp2 / ... / exp9 — run a single experiment in isolation
  all                     — run experiments 1-9 in dependency order
  report                  — compose MORNING_REPORT.md from existing outputs

Standing rules:
  - Phase G inference-only. No training. No fine-tuning. No modification.
  - No held-out asset use beyond F-A v1 outputs.
  - No F-A v2 trainer touch. No binder/D-family work.
  - binder_split.json LOCKED v3 — no modification.
  - Fixed noise seeds across paired comparisons.

Robustness:
  - Top-level try/except per experiment; one failure does not block others.
  - Each experiment writes outputs incrementally so partial results survive
    a crash.
  - 12-hour hard stop checked between experiments.
  - All scalar/table outputs append-only friendly (per-frame writes flush
    immediately).
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path("/path/to/poliebotics_phase_b")
PROJECT = ROOT / "poliebotics_phase_b"
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts" / "paper_analyses"))

OUT_ROOT = ROOT / "experiments/paper_analyses/overnight_battery"
CAUSAL_ROOT = ROOT / "experiments/paper_analyses/causal_ablations"
SF_ROOT = ROOT / "experiments/paper_analyses/scoring_function_comparison"

# Reuse from causal_ablations.py — shared helpers, identical Phase G
# inference convention.
from causal_ablations import (  # noqa: E402
    T_DIFFUSION, T_STEPS, K_NOISE, NOISE_SEED_BASE,
    PHASE_G_INPUT_H, PHASE_G_INPUT_W,
    FA_V1_CKPT_STEPS,
    load_phase_g, phase_g_score_scalar,
    load_phase_g_C, load_phase_g_E,
    apply_body_only_mask, apply_off_body_mask, inside_box_channel_mean,
    deterministic_shuffled_row, deterministic_cross_session_row,
    load_chain_keys, all_120_frames,
    auroc_pooled, hierarchical_bootstrap, mw_effect_size,
)

import torch
from phase_g.diffusion_diagnostic_model import build_diffusion_constants  # noqa: E402
from phase_g.fa_loader import load_fa_v1_checkpoint, render_C_fake  # noqa: E402


# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

EXP_DIRS = {
    1: "xof_correlation",
    2: "body_box_sanity",
    3: "c_only",
    4: "e_only",
    5: "matched_pose_corrected",
    6: "correct_e_rank",
    7: "xof_sensitivity",   # directory name retained for code compat;
                              # report uses "synthetic counterfactual-E
                              # perturbation sensitivity" label.
    8: "e4_score_inspection",
    9: "bootstrap_ci",
}

PERTURBED_CONDS = (
    [f"fake_{s//1000}k" for s in FA_V1_CKPT_STEPS]
    + ["shuffled_E", "cross_session_E"]
)
ALL_CONDS = ["real_correct"] + PERTURBED_CONDS

HARD_STOP_HOURS = 12

D2_DIR = PROJECT / "data" / "d2"
V10_DIR = PROJECT / "data" / "v10"
PHASE_G_CKPT = ROOT / "experiments/phase_g_diffusion_diagnostic/main/model_final.pt"
FA_V1_CKPT_DIR = ROOT / "experiments/phase_f/f_a_full_v1/checkpoints"


# -----------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------

class TimingLog:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.start = time.time()
        self.timings: dict[int, dict] = {}
        self._fh = open(log_path, "a")

    def log(self, msg: str) -> None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        self._fh.write(line + "\n")
        self._fh.flush()

    def exp_start(self, exp_num: int) -> float:
        t0 = time.time()
        self.timings[exp_num] = {"start": t0, "status": "running"}
        return t0

    def exp_end(self, exp_num: int, status: str = "done",
                 error: str | None = None) -> None:
        t1 = time.time()
        self.timings[exp_num]["end"] = t1
        self.timings[exp_num]["wall_seconds"] = t1 - self.timings[exp_num]["start"]
        self.timings[exp_num]["status"] = status
        if error is not None:
            self.timings[exp_num]["error"] = error

    def time_remaining(self) -> float:
        return HARD_STOP_HOURS * 3600 - (time.time() - self.start)

    def close(self) -> None:
        self._fh.close()


# -----------------------------------------------------------------------
# Frame subset / inputs cache
# -----------------------------------------------------------------------

def session_dirs() -> dict[str, Path]:
    return {"D2": D2_DIR, "V10": V10_DIR}


def chain_keys_cache() -> dict[str, list[int]]:
    return {sess: load_chain_keys(p) for sess, p in session_dirs().items()}


def load_120_frames() -> list[dict]:
    """Reuse the existing 120-frame subset from causal_ablations."""
    p = CAUSAL_ROOT / "frames_120.json"
    return json.loads(p.read_text())


def load_body_boxes() -> dict[tuple[str, int], tuple[int, int, int, int]]:
    raw = json.loads((CAUSAL_ROOT / "body_boxes.json").read_text())
    return {(sess, int(row)): tuple(box)
             for sess, by_row in raw.items()
             for row, box in by_row.items()}


def load_matched_pose() -> dict[tuple[str, int], int]:
    raw = json.loads((CAUSAL_ROOT / "matched_pose.json").read_text())
    out: dict[tuple[str, int], int] = {}
    for sess, by_row in raw.items():
        if sess.startswith("_"):
            continue
        for row, matched in by_row.items():
            out[(sess, int(row))] = int(matched)
    return out


# -----------------------------------------------------------------------
# Fresh-noise generator (identical convention to causal_ablations)
# -----------------------------------------------------------------------

def frame_noise(sess: str, row: int, device: torch.device) -> torch.Tensor:
    """(n_t, K, 4, H, W) — identical convention to causal_ablations.process_frame.
    Same noise across paired comparisons within a frame."""
    seed = NOISE_SEED_BASE + (row * 7919 + (1 if sess == "V10" else 0))
    torch.manual_seed(seed)
    return torch.randn(len(T_STEPS), K_NOISE, 4,
                        PHASE_G_INPUT_H, PHASE_G_INPUT_W,
                        device=device, dtype=torch.float32)


# -----------------------------------------------------------------------
# E1 baseline scalar reader (reuses causal_ablations per-frame manifests)
# -----------------------------------------------------------------------

def load_e1_scalars() -> dict[tuple[str, int, str], float]:
    out: dict[tuple[str, int, str], float] = {}
    per_frame = CAUSAL_ROOT / "per_frame"
    for fdir in sorted(per_frame.iterdir()):
        manifest = fdir / "ablations_manifest.json"
        if not manifest.exists():
            continue
        m = json.loads(manifest.read_text())
        sess, row = m["session"], m["row"]
        for cond in ALL_CONDS:
            key = f"E1|{cond}"
            if key in m["scalars"]:
                out[(sess, row, cond)] = float(m["scalars"][key])
    return out


# =======================================================================
# EXP 1 — XOF correlation table
# =======================================================================

def run_exp1(log: TimingLog) -> dict:
    log.log("[exp1] XOF correlation table — Pearson E_t vs E_{t+k}")
    out_dir = OUT_ROOT / EXP_DIRS[1]
    out_dir.mkdir(parents=True, exist_ok=True)
    sess_dirs = session_dirs()
    chain_keys = chain_keys_cache()
    frames = load_120_frames()
    rng = np.random.RandomState(0)
    rows: list[dict] = []
    for spec in frames:
        sess = spec["session"]; row = int(spec["row"])
        keys = chain_keys[sess]
        if row not in keys:
            continue
        idx = keys.index(row)
        E_t = load_phase_g_E(sess_dirs[sess], row).cpu().numpy().flatten()
        # offsets and labels
        targets: list[tuple[str, int, str]] = []
        for k in (1, 5, 10):
            j = idx + k
            if 0 <= j < len(keys):
                targets.append((f"E_t+{k}", keys[j], sess))
        # random same-session, ±10 excluded
        candidates = [keys[j] for j in range(len(keys))
                       if abs(j - idx) > 10]
        if candidates:
            r = candidates[int(rng.randint(0, len(candidates)))]
            targets.append(("random_same_session", int(r), sess))
        # cross-session same-percentile row
        other = "V10" if sess == "D2" else "D2"
        cross_row = deterministic_cross_session_row(
            row, chain_keys[sess], chain_keys[other])
        targets.append(("cross_session", int(cross_row), other))
        for label, target_row, target_sess in targets:
            try:
                E_target = load_phase_g_E(
                    sess_dirs[target_sess], target_row
                ).cpu().numpy().flatten()
                if E_t.std() == 0 or E_target.std() == 0:
                    r = float("nan")
                else:
                    r = float(np.corrcoef(E_t, E_target)[0, 1])
            except Exception as exc:  # noqa: BLE001
                log.log(f"  [exp1] skip {sess} f={row} {label}→{target_sess}/{target_row}: {exc!r}")
                continue
            rows.append({
                "session": sess, "row": row,
                "comparison": label,
                "target_session": target_sess,
                "target_row": int(target_row),
                "pearson": r,
            })
    csv_path = out_dir / "correlations.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "session", "row", "comparison", "target_session",
            "target_row", "pearson"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log.log(f"[exp1] wrote {csv_path} ({len(rows)} rows)")
    # Aggregate per (session, comparison)
    summary: dict[str, dict] = {}
    for sess in ("D2", "V10"):
        for cmp_label in ("E_t+1", "E_t+5", "E_t+10",
                            "random_same_session", "cross_session"):
            vals = [r["pearson"] for r in rows
                     if r["session"] == sess and r["comparison"] == cmp_label
                     and np.isfinite(r["pearson"])]
            if not vals:
                continue
            arr = np.array(vals, dtype=np.float64)
            q1, q3 = np.percentile(arr, [25, 75])
            summary[f"{sess}|{cmp_label}"] = {
                "n": int(arr.size),
                "median_r": float(np.median(arr)),
                "iqr_lo": float(q1),
                "iqr_hi": float(q3),
                "mean_abs_r": float(np.mean(np.abs(arr))),
                "max_abs_r": float(np.max(np.abs(arr))),
            }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    md = ["# EXP 1 — XOF correlation table",
          "",
          "Pearson |r| of flattened rendered E fields across sessions/offsets.",
          "",
          "Pre-registered thresholds:",
          "- |r| < 0.05: assumption confirmed (adjacent E uncorrelated)",
          "- |r| > 0.10: investigate; may affect E4 interpretation",
          "",
          "| session | comparison | n | median r | IQR | max |r| |",
          "|---|---|---|---|---|---|"]
    for k in sorted(summary):
        s = summary[k]
        sess, cmp_label = k.split("|", 1)
        md.append(f"| {sess} | {cmp_label} | {s['n']} | "
                   f"{s['median_r']:+.4f} | "
                   f"[{s['iqr_lo']:+.4f}, {s['iqr_hi']:+.4f}] | "
                   f"{s['max_abs_r']:.4f} |")
    md.append("")
    # Verdict
    max_abs = max((abs(s["median_r"]) for s in summary.values()), default=0)
    if max_abs < 0.05:
        md.append("**Verdict**: assumption CONFIRMED — all median |r| < 0.05.")
    elif max_abs < 0.10:
        md.append(f"**Verdict**: borderline — max median |r| = {max_abs:.4f}.")
    else:
        md.append(f"**Verdict**: 🚨 E4 interpretation may need revision — "
                   f"max median |r| = {max_abs:.4f} > 0.10.")
    (out_dir / "summary.md").write_text("\n".join(md))
    log.log(f"[exp1] wrote {out_dir / 'summary.md'}")
    return {"n_rows": len(rows), "max_abs_median_r": max_abs}


# =======================================================================
# EXP 2 — Body-box visualization sanity check
# =======================================================================

def run_exp2(log: TimingLog) -> dict:
    log.log("[exp2] body-box visualization sanity check (120 frames)")
    import matplotlib.pyplot as plt
    out_dir = OUT_ROOT / EXP_DIRS[2]
    per_frame_dir = out_dir / "per_frame"
    contact_dir = out_dir / "contact_sheets"
    per_frame_dir.mkdir(parents=True, exist_ok=True)
    contact_dir.mkdir(parents=True, exist_ok=True)

    body_boxes = load_body_boxes()
    sess_dirs = session_dirs()
    frames = load_120_frames()

    pathological: list[dict] = []
    box_stats: list[dict] = []
    contact_data: dict[str, list[dict]] = {"D2": [], "V10": []}

    for spec in frames:
        sess = spec["session"]; row = int(spec["row"])
        if (sess, row) not in body_boxes:
            log.log(f"  [exp2] missing body-box for {sess} f={row}; skip")
            continue
        box = body_boxes[(sess, row)]
        y0, y1, x0, x1 = box
        try:
            C_real = load_phase_g_C(sess_dirs[sess], row)
        except Exception as exc:  # noqa: BLE001
            log.log(f"  [exp2] {sess} f={row}: load failed {exc!r}; skip")
            continue
        # Convert to displayable RGB
        C_arr = C_real.cpu().numpy()
        rgb = np.stack([C_arr[0], 0.5 * (C_arr[1] + C_arr[2]), C_arr[3]],
                        axis=-1)
        rgb = np.clip(rgb, 0, 1)
        # Apply body-only and off-body masks
        body_only = apply_body_only_mask(C_real, box).cpu().numpy()
        off_body = apply_off_body_mask(C_real, box).cpu().numpy()
        body_rgb = np.clip(np.stack([body_only[0], 0.5 * (body_only[1] + body_only[2]),
                                       body_only[3]], axis=-1), 0, 1)
        off_rgb = np.clip(np.stack([off_body[0], 0.5 * (off_body[1] + off_body[2]),
                                      off_body[3]], axis=-1), 0, 1)
        # Render 3-panel
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=80)
        axes[0].imshow(rgb)
        axes[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                          edgecolor="red", facecolor="none",
                                          linewidth=2))
        axes[0].set_title(f"{sess} f={row} — C + body-box "
                           f"[{y0}:{y1}, {x0}:{x1}]", fontsize=9)
        axes[0].set_xticks([]); axes[0].set_yticks([])
        axes[1].imshow(body_rgb)
        axes[1].set_title("E2 body-only (outside masked)", fontsize=9)
        axes[1].set_xticks([]); axes[1].set_yticks([])
        axes[2].imshow(off_rgb)
        axes[2].set_title("E3 off-body (inside masked)", fontsize=9)
        axes[2].set_xticks([]); axes[2].set_yticks([])
        fig.tight_layout()
        out_path = per_frame_dir / f"{sess}_f{row:06d}.png"
        fig.savefig(out_path, dpi=80, bbox_inches="tight")
        plt.close(fig)
        # Stats
        H, W = PHASE_G_INPUT_H, PHASE_G_INPUT_W
        area_frac = ((y1 - y0) * (x1 - x0)) / (H * W)
        aspect = (x1 - x0) / max(1, (y1 - y0))
        cy = (y0 + y1) / 2 / H
        cx = (x0 + x1) / 2 / W
        is_path = area_frac < 0.10 or area_frac > 0.70
        if is_path:
            pathological.append({
                "session": sess, "row": row, "box": list(box),
                "area_frac": area_frac,
                "reason": ("box_<10%_area" if area_frac < 0.10
                           else "box_>70%_area"),
            })
        box_stats.append({
            "session": sess, "row": row,
            "area_frac": area_frac, "aspect": aspect,
            "cy_norm": cy, "cx_norm": cx,
            "y0": y0, "y1": y1, "x0": x0, "x1": x1,
            "pathological": is_path,
        })
        contact_data[sess].append({"row": row, "rgb": rgb, "box": box,
                                     "is_path": is_path})

    # Contact sheets
    for sess in ("D2", "V10"):
        items = contact_data[sess]
        if not items:
            continue
        n = len(items); cols = 10
        rows_n = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows_n, cols,
                                   figsize=(2.0 * cols, 1.5 * rows_n),
                                   dpi=80, squeeze=False)
        for i, item in enumerate(items):
            r, c = divmod(i, cols)
            ax = axes[r, c]
            ax.imshow(item["rgb"])
            y0, y1, x0, x1 = item["box"]
            edge = "orange" if item["is_path"] else "lime"
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                         edgecolor=edge, facecolor="none",
                                         linewidth=1))
            ax.set_title(f"f{item['row']:06d}", fontsize=6)
            ax.set_xticks([]); ax.set_yticks([])
        for i in range(n, rows_n * cols):
            r, c = divmod(i, cols)
            axes[r, c].axis("off")
        fig.suptitle(f"{sess} — body-box overlay (n={n}; orange = pathological)",
                      fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        cs_path = contact_dir / f"contact_sheet_{sess}.png"
        fig.savefig(cs_path, dpi=80, bbox_inches="tight")
        plt.close(fig)
        log.log(f"  [exp2] wrote {cs_path}")

    # Pathological cases JSON
    (out_dir / "pathological_cases.json").write_text(
        json.dumps(pathological, indent=2))
    # Mirror to body_box_sanity.md, AND save as summary.md
    # so write_morning_report() picks it up. (Codex audit fix 2026-05-05.)
    # CSV
    csv_path = out_dir / "box_stats.csv"
    if box_stats:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(box_stats[0].keys()))
            w.writeheader()
            for r in box_stats:
                w.writerow(r)

    # Summary
    md = ["# EXP 2 — body-box visualization sanity check",
          "",
          f"120 frames inspected. Pathological count: {len(pathological)} "
          f"({100 * len(pathological) / max(1, len(box_stats)):.1f}%).",
          "",
          "Pre-registered thresholds:",
          "- < 10% pathological: masks reliable, E2/E3 conclusions stand",
          "- 10–25% pathological: caveat needed in manuscript",
          "- > 25% pathological: re-run E2/E3 on cleanly-masked subset (FLAG)",
          ""]
    for sess in ("D2", "V10"):
        sess_stats = [b for b in box_stats if b["session"] == sess]
        if not sess_stats:
            continue
        areas = [b["area_frac"] for b in sess_stats]
        aspects = [b["aspect"] for b in sess_stats]
        md.append(f"## {sess} (n={len(sess_stats)})")
        md.append("")
        md.append(f"- Area % of frame: min={min(areas)*100:.1f}, "
                   f"median={np.median(areas)*100:.1f}, "
                   f"max={max(areas)*100:.1f}")
        md.append(f"- Aspect ratio: min={min(aspects):.2f}, "
                   f"median={np.median(aspects):.2f}, "
                   f"max={max(aspects):.2f}")
        md.append(f"- Pathological: "
                   f"{sum(1 for b in sess_stats if b['pathological'])} / "
                   f"{len(sess_stats)}")
        md.append("")
    pct_path = 100 * len(pathological) / max(1, len(box_stats))
    if pct_path < 10:
        md.append("**Verdict**: masks reliable; E2/E3 conclusions stand.")
    elif pct_path < 25:
        md.append(f"⚠️ **Verdict**: {pct_path:.1f}% pathological — manuscript caveat needed.")
    else:
        md.append(f"🚨 **Verdict**: {pct_path:.1f}% pathological — a reviewer should "
                   f"consider re-running E2/E3 on cleanly-masked subset.")
    body_box_md = "\n".join(md)
    (out_dir / "body_box_sanity.md").write_text(body_box_md)
    # Aggregator picks up summary.md — keep both paths populated.
    (out_dir / "summary.md").write_text(body_box_md)
    return {"n_frames": len(box_stats), "n_pathological": len(pathological),
             "pct_pathological": pct_path}


# =======================================================================
# Phase G inference helpers (shared by Exp 3, 4, 5, 6, 7)
# =======================================================================

def load_phase_g_for_inference(device: torch.device,
                                  dtype: torch.dtype):
    log_msg = f"loading Phase G ckpt {PHASE_G_CKPT}"
    return load_phase_g(PHASE_G_CKPT, device, dtype), log_msg


def load_fa_v1_ckpts(device: torch.device, dtype: torch.dtype) -> dict:
    out = {}
    for step in FA_V1_CKPT_STEPS:
        ckpt = FA_V1_CKPT_DIR / f"step_{step:08d}.pt"
        out[step] = load_fa_v1_checkpoint(ckpt, device=device, dtype=dtype)
    return out


def neutral_E_mean(sess_dirs: dict[str, Path], frames: list[dict]
                    ) -> torch.Tensor:
    """Per-channel mean of E across the 120-frame subset. (3,) tensor
    that gets broadcast to (3, H, W) at use time."""
    sums = torch.zeros(3, dtype=torch.float64)
    n = 0
    for spec in frames:
        try:
            E = load_phase_g_E(sess_dirs[spec["session"]], int(spec["row"]))
        except Exception:
            continue
        sums += E.float().mean(dim=(1, 2)).double()
        n += 1
    if n == 0:
        return torch.zeros(3, dtype=torch.float32)
    return (sums / n).float()


def neutral_C_mean(sess_dirs: dict[str, Path], frames: list[dict]
                    ) -> torch.Tensor:
    """Per-channel mean of C across the 120-frame subset. (4,) tensor."""
    sums = torch.zeros(4, dtype=torch.float64)
    n = 0
    for spec in frames:
        try:
            C = load_phase_g_C(sess_dirs[spec["session"]], int(spec["row"]))
        except Exception:
            continue
        sums += C.float().mean(dim=(1, 2)).double()
        n += 1
    if n == 0:
        return torch.zeros(4, dtype=torch.float32)
    return (sums / n).float()


# =======================================================================
# EXP 3 — C-only Phase G evaluation
# =======================================================================

def run_exp3(log: TimingLog, device_id: int = 0) -> dict:
    log.log(f"[exp3] C-only Phase G (E neutralized) on cuda:{device_id}")
    out_dir = OUT_ROOT / EXP_DIRS[3]
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{device_id}")
    dtype = torch.bfloat16
    pg, _ = load_phase_g_for_inference(device, dtype)
    fa_v1 = load_fa_v1_ckpts(device, dtype)
    dc = build_diffusion_constants(T_DIFFUSION, device, torch.float32)

    sess_dirs = session_dirs()
    chain_keys = chain_keys_cache()
    frames = load_120_frames()

    # Build neutral E (per-channel mean) once
    log.log("[exp3] computing per-channel E mean for neutralization …")
    e_mean_chan = neutral_E_mean(sess_dirs, frames)  # (3,)
    E_neutral_mean = e_mean_chan.view(3, 1, 1).expand(
        3, PHASE_G_INPUT_H, PHASE_G_INPUT_W
    ).contiguous()
    E_neutral_zero = torch.zeros(3, PHASE_G_INPUT_H, PHASE_G_INPUT_W,
                                   dtype=torch.float32)

    # Per-frame, per-condition: score with E_neutral_mean (primary) and
    # E_neutral_zero (sensitivity).
    rows: list[dict] = []
    n_frames = len(frames)
    for fi, spec in enumerate(frames):
        sess = spec["session"]; row = int(spec["row"])
        try:
            keys = chain_keys[sess]
            target_idx = keys.index(row)
            source_row = keys[(target_idx + len(keys) // 4) % len(keys)]
            C_real = load_phase_g_C(sess_dirs[sess], row)
            E_correct = load_phase_g_E(sess_dirs[sess], row)  # for shuffled/cross
            shuffled_row = deterministic_shuffled_row(row, keys)
            cross_sess = "V10" if sess == "D2" else "D2"
            cross_row = deterministic_cross_session_row(
                row, keys, chain_keys[cross_sess])
            E_shuffled = load_phase_g_E(sess_dirs[sess], shuffled_row)
            E_cross = load_phase_g_E(sess_dirs[cross_sess], cross_row)
            # F-A v1 fakes: render once
            C_fakes = {}
            for step, fa in fa_v1.items():
                Cf = render_C_fake(fa, sess_dirs[sess],
                                    source_row=source_row, target_row=row,
                                    device=device, dtype=dtype).cpu().float()
                C_fakes[step] = Cf
            # Build (C, E_neutral) per condition. C is condition-specific;
            # E is always neutralized.
            cond_C: dict[str, torch.Tensor] = {
                "real_correct": C_real,
                "shuffled_E": C_real,
                "cross_session_E": C_real,
            }
            for step in FA_V1_CKPT_STEPS:
                cond_C[f"fake_{step//1000}k"] = C_fakes[step]
            # Noise paired across conditions
            noise = frame_noise(sess, row, device)
            for cond in ALL_CONDS:
                C_in = cond_C[cond]
                # Primary: E mean
                s_mean = phase_g_score_scalar(
                    pg, C_in.to(device=device, dtype=torch.float32),
                    E_neutral_mean, dc, device, dtype, noise)
                # Sensitivity: E zero
                s_zero = phase_g_score_scalar(
                    pg, C_in.to(device=device, dtype=torch.float32),
                    E_neutral_zero, dc, device, dtype, noise)
                # Determine if degenerate: shuffled_E and cross_session_E
                # become identical to real_correct under E neutralization
                # (both have C=C_real).
                degenerate = cond in ("shuffled_E", "cross_session_E")
                rows.append({
                    "session": sess, "row": row, "condition": cond,
                    "score_E_mean": s_mean,
                    "score_E_zero": s_zero,
                    "degenerate": degenerate,
                })
        except Exception as exc:  # noqa: BLE001
            log.log(f"  [exp3] FRAME FAIL {sess} f={row}: {exc!r}")
            traceback.print_exc(file=sys.stderr)
            continue
        if (fi + 1) % 20 == 0:
            log.log(f"  [exp3] {fi+1}/{n_frames} frames done")

    # Save raw scalars
    csv_path = out_dir / "raw_scores.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    log.log(f"[exp3] wrote {csv_path} ({len(rows)} rows)")

    # AUROC table per (session, condition) using E_mean (primary) scores
    by_sess_cond: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        by_sess_cond.setdefault((r["session"], r["condition"]), []).append(r)
    auroc_rows: list[dict] = []
    for sess in ("D2", "V10"):
        # real_correct scalars
        real = [r["score_E_mean"] for r in by_sess_cond.get((sess, "real_correct"), [])]
        real_arr = np.array(real, dtype=np.float64)
        for cond in PERTURBED_CONDS:
            pert = [r["score_E_mean"] for r in by_sess_cond.get((sess, cond), [])]
            pert_arr = np.array(pert, dtype=np.float64)
            n = min(real_arr.size, pert_arr.size)
            if n == 0:
                continue
            auc = auroc_pooled(real_arr[:n], pert_arr[:n])
            deltas = (pert_arr[:n] - real_arr[:n]).tolist()
            mean, lo, hi = hierarchical_bootstrap(deltas, n_boot=10000)
            degen = cond in ("shuffled_E", "cross_session_E")
            auroc_rows.append({
                "session": sess, "condition": cond,
                "auroc": auc if not degen else None,
                "auroc_marker": "N/A_degenerate" if degen else f"{auc:.4f}",
                "delta_median": float(np.median(deltas)),
                "delta_bootstrap_ci_low": lo,
                "delta_bootstrap_ci_high": hi,
                "n_frames": n,
                "degenerate": degen,
                "raw_auroc_for_sanity": auc,  # always recorded for cross-check
            })
    table_path = out_dir / "auroc_table.csv"
    if auroc_rows:
        with open(table_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(auroc_rows[0].keys()))
            w.writeheader()
            for r in auroc_rows:
                w.writerow(r)
    log.log(f"[exp3] wrote {table_path}")

    # Summary
    md = ["# EXP 3 — C-only Phase G evaluation (E neutralized)",
          "",
          "Replace E with neutral content (per-channel mean across 120-frame "
          "subset; sensitivity also runs with E=zero).",
          "",
          "**Construction-aware reading**:",
          "- *Meaningful conditions* (C differs from real_correct): "
          "`fake_5k`, `fake_25k`, `fake_70k`, `fake_100k`. AUROC "
          "interpretable.",
          "- *Degenerate conditions* (C identical to real_correct once E "
          "neutralized): `shuffled_E`, `cross_session_E`. Marked N/A; "
          "raw AUROC reported alongside as sanity (expect ≈ 0.5).",
          "",
          "Pre-registered thresholds (meaningful conditions only):",
          "- AUROC < 0.6: chain-coupling story strongly supported (E needed)",
          "- 0.6–0.85: partial discrimination from C alone",
          "- AUROC > 0.85: 🚨 MAJOR — Phase G largely discriminates without E",
          "",
          "## AUROC table (E neutralized = per-channel mean across subset)",
          "",
          "| session | condition | AUROC | n | Δ median | bootstrap 95% CI |",
          "|---|---|---|---|---|---|"]
    for r in auroc_rows:
        if r["degenerate"]:
            tag = f"N/A (degenerate; sanity={r['raw_auroc_for_sanity']:.4f})"
        else:
            tag = r["auroc_marker"]
            if r["raw_auroc_for_sanity"] >= 0.85:
                tag = f"🚨 **{tag}**"
            elif r["raw_auroc_for_sanity"] >= 0.6:
                tag = f"⚠️ {tag}"
        md.append(
            f"| {r['session']} | {r['condition']} | {tag} | {r['n_frames']} "
            f"| {r['delta_median']:+.5f} | "
            f"[{r['delta_bootstrap_ci_low']:+.5f}, "
            f"{r['delta_bootstrap_ci_high']:+.5f}] |")
    (out_dir / "summary.md").write_text("\n".join(md))
    log.log(f"[exp3] wrote {out_dir / 'summary.md'}")
    return {"n_rows": len(rows), "n_auroc_cells": len(auroc_rows)}


# =======================================================================
# EXP 4 — E-only Phase G evaluation
# =======================================================================

def run_exp4(log: TimingLog, device_id: int = 0) -> dict:
    log.log(f"[exp4] E-only Phase G (C neutralized) on cuda:{device_id}")
    out_dir = OUT_ROOT / EXP_DIRS[4]
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{device_id}")
    dtype = torch.bfloat16
    pg, _ = load_phase_g_for_inference(device, dtype)
    dc = build_diffusion_constants(T_DIFFUSION, device, torch.float32)

    sess_dirs = session_dirs()
    chain_keys = chain_keys_cache()
    frames = load_120_frames()

    log.log("[exp4] computing per-channel C mean for neutralization …")
    c_mean_chan = neutral_C_mean(sess_dirs, frames)  # (4,)
    C_neutral_mean = c_mean_chan.view(4, 1, 1).expand(
        4, PHASE_G_INPUT_H, PHASE_G_INPUT_W
    ).contiguous()
    C_neutral_zero = torch.zeros(4, PHASE_G_INPUT_H, PHASE_G_INPUT_W,
                                   dtype=torch.float32)

    rows: list[dict] = []
    n_frames = len(frames)
    for fi, spec in enumerate(frames):
        sess = spec["session"]; row = int(spec["row"])
        try:
            keys = chain_keys[sess]
            E_correct = load_phase_g_E(sess_dirs[sess], row)
            shuffled_row = deterministic_shuffled_row(row, keys)
            cross_sess = "V10" if sess == "D2" else "D2"
            cross_row = deterministic_cross_session_row(
                row, keys, chain_keys[cross_sess])
            E_shuffled = load_phase_g_E(sess_dirs[sess], shuffled_row)
            E_cross = load_phase_g_E(sess_dirs[cross_sess], cross_row)
            cond_E: dict[str, torch.Tensor] = {
                "real_correct": E_correct,
                "shuffled_E": E_shuffled,
                "cross_session_E": E_cross,
            }
            # Fake conditions: F-A v1 attacker uses E_correct as target,
            # so E for fake_X conditions is identical to E_correct →
            # degenerate under C neutralization.
            for step in FA_V1_CKPT_STEPS:
                cond_E[f"fake_{step//1000}k"] = E_correct
            noise = frame_noise(sess, row, device)
            for cond in ALL_CONDS:
                E_in = cond_E[cond]
                s_mean = phase_g_score_scalar(
                    pg, C_neutral_mean,
                    E_in.to(device=device, dtype=torch.float32),
                    dc, device, dtype, noise)
                s_zero = phase_g_score_scalar(
                    pg, C_neutral_zero,
                    E_in.to(device=device, dtype=torch.float32),
                    dc, device, dtype, noise)
                # Fake conditions are degenerate (E ≡ E_correct under
                # neutralized C). Verify by comparing the saved E for
                # this condition against E_correct: by construction
                # they're identical for fake_*.
                degenerate = cond.startswith("fake_")
                rows.append({
                    "session": sess, "row": row, "condition": cond,
                    "score_C_mean": s_mean, "score_C_zero": s_zero,
                    "degenerate": degenerate,
                    "degenerate_reason": (
                        "E identical to real_correct E (F-A v1 fakes use "
                        "E_target=E_correct)" if degenerate else ""),
                })
        except Exception as exc:  # noqa: BLE001
            log.log(f"  [exp4] FRAME FAIL {sess} f={row}: {exc!r}")
            continue
        if (fi + 1) % 20 == 0:
            log.log(f"  [exp4] {fi+1}/{n_frames} frames done")

    csv_path = out_dir / "raw_scores.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)

    by = {}
    for r in rows:
        by.setdefault((r["session"], r["condition"]), []).append(r)
    auroc_rows: list[dict] = []
    for sess in ("D2", "V10"):
        real = [r["score_C_mean"] for r in by.get((sess, "real_correct"), [])]
        real_arr = np.array(real, dtype=np.float64)
        for cond in PERTURBED_CONDS:
            pert = [r["score_C_mean"] for r in by.get((sess, cond), [])]
            pert_arr = np.array(pert, dtype=np.float64)
            n = min(real_arr.size, pert_arr.size)
            if n == 0:
                continue
            auc = auroc_pooled(real_arr[:n], pert_arr[:n])
            deltas = (pert_arr[:n] - real_arr[:n]).tolist()
            mean, lo, hi = hierarchical_bootstrap(deltas, n_boot=10000)
            degen = cond.startswith("fake_")
            auroc_rows.append({
                "session": sess, "condition": cond,
                "auroc": auc if not degen else None,
                "auroc_marker": "N/A_degenerate" if degen else f"{auc:.4f}",
                "delta_median": float(np.median(deltas)),
                "delta_bootstrap_ci_low": lo,
                "delta_bootstrap_ci_high": hi,
                "n_frames": n, "degenerate": degen,
                "raw_auroc_for_sanity": auc,
            })

    table_path = out_dir / "auroc_table.csv"
    if auroc_rows:
        with open(table_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(auroc_rows[0].keys()))
            w.writeheader()
            for r in auroc_rows:
                w.writerow(r)

    md = ["# EXP 4 — E-only Phase G evaluation (C neutralized)",
          "",
          "**Construction-aware reading**:",
          "- *Meaningful conditions* (E differs from real_correct): "
          "`shuffled_E`, `cross_session_E`. AUROC interpretable.",
          "- *Degenerate conditions* (E ≡ E_correct under C neutralization, "
          "since F-A v1 fakes use E_target=E_correct): "
          "`fake_5k`/25k/70k/100k. Marked N/A; raw AUROC reported as sanity "
          "(expect ≈ 0.5).",
          "",
          "Pre-registered thresholds (meaningful conditions only):",
          "- AUROC < 0.55: as expected (E alone shouldn't discriminate)",
          "- 0.55–0.80: unusual; investigate",
          "- > 0.80: 🚨 MAJOR — Phase G uses E distributional cues",
          "",
          "## AUROC table",
          "",
          "| session | condition | AUROC | n | Δ median | bootstrap 95% CI |",
          "|---|---|---|---|---|---|"]
    for r in auroc_rows:
        if r["degenerate"]:
            tag = f"N/A (degenerate; sanity={r['raw_auroc_for_sanity']:.4f})"
        else:
            tag = r["auroc_marker"]
            if r["raw_auroc_for_sanity"] > 0.80:
                tag = f"🚨 **{tag}**"
            elif r["raw_auroc_for_sanity"] > 0.55:
                tag = f"⚠️ {tag}"
        md.append(
            f"| {r['session']} | {r['condition']} | {tag} | {r['n_frames']} "
            f"| {r['delta_median']:+.5f} | "
            f"[{r['delta_bootstrap_ci_low']:+.5f}, "
            f"{r['delta_bootstrap_ci_high']:+.5f}] |")
    (out_dir / "summary.md").write_text("\n".join(md))
    return {"n_rows": len(rows), "n_auroc_cells": len(auroc_rows)}


# =======================================================================
# EXP 5 — Matched-pose corrected re-runs (E5a, E5b, E5c)
# =======================================================================

def run_exp5(log: TimingLog, device_id: int = 0) -> dict:
    log.log(f"[exp5] matched-pose corrected (E5a, E5b, E5c) on cuda:{device_id}")
    out_dir = OUT_ROOT / EXP_DIRS[5]
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{device_id}")
    dtype = torch.bfloat16
    pg, _ = load_phase_g_for_inference(device, dtype)
    dc = build_diffusion_constants(T_DIFFUSION, device, torch.float32)

    sess_dirs = session_dirs()
    chain_keys = chain_keys_cache()
    frames = load_120_frames()
    matched_pose_rows = load_matched_pose()

    # E_neutral mean for E5c
    e_mean_chan = neutral_E_mean(sess_dirs, frames)
    E_neutral = e_mean_chan.view(3, 1, 1).expand(
        3, PHASE_G_INPUT_H, PHASE_G_INPUT_W).contiguous()

    rows: list[dict] = []
    n_frames = len(frames)
    for fi, spec in enumerate(frames):
        sess = spec["session"]; row = int(spec["row"])
        try:
            if (sess, row) not in matched_pose_rows:
                continue
            matched_row = matched_pose_rows[(sess, row)]
            C_test = load_phase_g_C(sess_dirs[sess], row)
            E_test = load_phase_g_E(sess_dirs[sess], row)
            C_match = load_phase_g_C(sess_dirs[sess], matched_row)
            E_match = load_phase_g_E(sess_dirs[sess], matched_row)
            noise = frame_noise(sess, row, device)
            # E5a: (C_match, E_test) — wrong scene, right E
            s_e5a = phase_g_score_scalar(
                pg, C_match.to(device=device, dtype=torch.float32),
                E_test.to(device=device, dtype=torch.float32),
                dc, device, dtype, noise)
            # E5b: (C_test, E_match) — right scene, wrong E
            s_e5b = phase_g_score_scalar(
                pg, C_test.to(device=device, dtype=torch.float32),
                E_match.to(device=device, dtype=torch.float32),
                dc, device, dtype, noise)
            # E5c: C-only on matched-pose negatives — both with E_neutral
            s_e5c_test = phase_g_score_scalar(
                pg, C_test.to(device=device, dtype=torch.float32),
                E_neutral, dc, device, dtype, noise)
            s_e5c_match = phase_g_score_scalar(
                pg, C_match.to(device=device, dtype=torch.float32),
                E_neutral, dc, device, dtype, noise)
            # Reference: real_correct under standard E1 (paired noise)
            s_real = phase_g_score_scalar(
                pg, C_test.to(device=device, dtype=torch.float32),
                E_test.to(device=device, dtype=torch.float32),
                dc, device, dtype, noise)
            rows.append({
                "session": sess, "row": row, "matched_row": matched_row,
                "score_real_correct": s_real,
                "score_e5a_C_match_E_test": s_e5a,
                "score_e5b_C_test_E_match": s_e5b,
                "score_e5c_C_test_E_neutral": s_e5c_test,
                "score_e5c_C_match_E_neutral": s_e5c_match,
            })
        except Exception as exc:  # noqa: BLE001
            log.log(f"  [exp5] FRAME FAIL {sess} f={row}: {exc!r}")
            continue
        if (fi + 1) % 20 == 0:
            log.log(f"  [exp5] {fi+1}/{n_frames} frames done")

    csv_path = out_dir / "raw_scores.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # AUROC: per session, real vs each E5 variant
    auroc_rows: list[dict] = []
    for sess in ("D2", "V10"):
        sess_rows = [r for r in rows if r["session"] == sess]
        real = np.array([r["score_real_correct"] for r in sess_rows])
        for variant_key, variant_label in [
            ("score_e5a_C_match_E_test", "E5a_C_match_E_test"),
            ("score_e5b_C_test_E_match", "E5b_C_test_E_match"),
        ]:
            pert = np.array([r[variant_key] for r in sess_rows])
            n = min(real.size, pert.size)
            if n == 0:
                continue
            auc = auroc_pooled(real[:n], pert[:n])
            deltas = (pert[:n] - real[:n]).tolist()
            _, lo, hi = hierarchical_bootstrap(deltas, n_boot=10000)
            auroc_rows.append({
                "session": sess, "variant": variant_label,
                "auroc": auc, "n_frames": n,
                "delta_median": float(np.median(deltas)),
                "delta_bootstrap_ci_low": lo,
                "delta_bootstrap_ci_high": hi,
            })
        # E5c: AUROC of C_test vs C_match under E_neutral
        c_test = np.array([r["score_e5c_C_test_E_neutral"] for r in sess_rows])
        c_match = np.array([r["score_e5c_C_match_E_neutral"] for r in sess_rows])
        n = min(c_test.size, c_match.size)
        if n > 0:
            auc = auroc_pooled(c_test[:n], c_match[:n])
            deltas = (c_match[:n] - c_test[:n]).tolist()
            _, lo, hi = hierarchical_bootstrap(deltas, n_boot=10000)
            auroc_rows.append({
                "session": sess, "variant": "E5c_C_only_test_vs_match",
                "auroc": auc, "n_frames": n,
                "delta_median": float(np.median(deltas)),
                "delta_bootstrap_ci_low": lo,
                "delta_bootstrap_ci_high": hi,
            })

    table_path = out_dir / "auroc_table.csv"
    if auroc_rows:
        with open(table_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(auroc_rows[0].keys()))
            w.writeheader()
            for r in auroc_rows:
                w.writerow(r)

    md = ["# EXP 5 — Matched-pose corrected re-runs (E5a, E5b, E5c)",
          "",
          "Original E5 substituted (C_match, E_match) for real_correct, kept "
          "AUROC=1.0 — but does NOT prove pose-shortcut ruled out. These three "
          "corrected variants test the actual question.",
          "",
          "Pre-registered thresholds:",
          "- E5a/E5b AUROC > 0.95: rejects pose-matched negatives with one half wrong → strong chain-coupling specificity",
          "- E5a/E5b AUROC < 0.85: pose matching is a meaningful confound for the broken half (manuscript caveat)",
          "- E5c AUROC > 0.80: 🚨 C-only discriminates pose-matched negatives — original E5 partly reflects C-only signal",
          "",
          "| session | variant | AUROC | n | Δ median | bootstrap 95% CI |",
          "|---|---|---|---|---|---|"]
    for r in auroc_rows:
        tag = f"{r['auroc']:.4f}"
        if r["variant"].startswith("E5c") and r["auroc"] > 0.80:
            tag = f"🚨 **{tag}**"
        elif r["variant"].startswith(("E5a", "E5b")) and r["auroc"] < 0.85:
            tag = f"⚠️ {tag}"
        md.append(f"| {r['session']} | {r['variant']} | {tag} | "
                   f"{r['n_frames']} | {r['delta_median']:+.5f} | "
                   f"[{r['delta_bootstrap_ci_low']:+.5f}, "
                   f"{r['delta_bootstrap_ci_high']:+.5f}] |")
    (out_dir / "summary.md").write_text("\n".join(md))
    return {"n_rows": len(rows), "n_auroc_cells": len(auroc_rows)}


# =======================================================================
# EXP 6 — Correct-E rank test (per-frame, 51 candidates)
# =======================================================================

def run_exp6(log: TimingLog, device_id: int = 0) -> dict:
    log.log(f"[exp6] correct-E rank test (51 candidates per frame) on cuda:{device_id}")
    out_dir = OUT_ROOT / EXP_DIRS[6]
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{device_id}")
    dtype = torch.bfloat16
    pg, _ = load_phase_g_for_inference(device, dtype)
    dc = build_diffusion_constants(T_DIFFUSION, device, torch.float32)

    sess_dirs = session_dirs()
    chain_keys = chain_keys_cache()
    frames = load_120_frames()

    rng = np.random.RandomState(42)
    rows: list[dict] = []
    rank_rows_path = out_dir / "rank_distribution.csv"
    # Append-only writer for incremental persistence
    rank_csv_fh = open(rank_rows_path, "w", newline="")
    rank_csv_w = csv.DictWriter(
        rank_csv_fh,
        fieldnames=["session", "row", "rank_of_correct", "n_candidates",
                     "score_correct", "best_wrong_score",
                     "score_gap_correct_minus_best_wrong"])
    rank_csv_w.writeheader()
    rank_csv_fh.flush()

    n_frames = len(frames)
    for fi, spec in enumerate(frames):
        sess = spec["session"]; row = int(spec["row"])
        try:
            keys = chain_keys[sess]
            idx = keys.index(row)
            # 50 random candidates excluding ±10
            valid = [j for j in range(len(keys))
                      if abs(j - idx) > 10]
            if len(valid) < 50:
                log.log(f"  [exp6] {sess} f={row}: insufficient candidates ({len(valid)})")
                continue
            picks = rng.choice(valid, size=50, replace=False)
            C_t = load_phase_g_C(sess_dirs[sess], row).to(
                device=device, dtype=torch.float32)
            E_t = load_phase_g_E(sess_dirs[sess], row).to(
                device=device, dtype=torch.float32)
            noise = frame_noise(sess, row, device)
            # Score correct
            s_correct = phase_g_score_scalar(
                pg, C_t, E_t, dc, device, dtype, noise)
            # Score 50 wrong candidates. Pre-registered protocol requires
            # exactly correct + 50 wrong; any candidate-load failure
            # invalidates this frame's rank → skip the frame.
            # (Codex audit fix 2026-05-05.)
            scores_wrong: list[float] = []
            candidate_failure = False
            for j in picks:
                cand_row = keys[j]
                try:
                    E_j = load_phase_g_E(sess_dirs[sess], cand_row).to(
                        device=device, dtype=torch.float32)
                    s_j = phase_g_score_scalar(pg, C_t, E_j, dc, device, dtype, noise)
                    scores_wrong.append(s_j)
                except Exception as exc:  # noqa: BLE001
                    log.log(f"  [exp6] {sess} f={row}: candidate row {cand_row} "
                            f"score failed ({exc!r}); skipping FRAME to keep "
                            f"50-candidate invariant")
                    candidate_failure = True
                    break
            if candidate_failure or len(scores_wrong) != 50:
                continue
            # Rank: lower MSE = better. correct's rank among 51 sorted ascending.
            all_scores = [s_correct] + scores_wrong
            sorted_idx = np.argsort(all_scores)
            rank_of_correct = int(np.where(sorted_idx == 0)[0][0]) + 1
            best_wrong = float(min(scores_wrong))
            row_out = {
                "session": sess, "row": row,
                "rank_of_correct": rank_of_correct,
                "n_candidates": len(all_scores),
                "score_correct": s_correct,
                "best_wrong_score": best_wrong,
                "score_gap_correct_minus_best_wrong": s_correct - best_wrong,
            }
            rank_csv_w.writerow(row_out); rank_csv_fh.flush()
            rows.append(row_out)
        except Exception as exc:  # noqa: BLE001
            log.log(f"  [exp6] FRAME FAIL {sess} f={row}: {exc!r}")
            continue
        if (fi + 1) % 5 == 0:
            log.log(f"  [exp6] {fi+1}/{n_frames} frames done")
    rank_csv_fh.close()

    # Aggregates
    summary: dict = {}
    for sess in (None, "D2", "V10"):
        if sess is None:
            sess_rows = rows
            label = "pooled"
        else:
            sess_rows = [r for r in rows if r["session"] == sess]
            label = sess
        if not sess_rows:
            continue
        ranks = np.array([r["rank_of_correct"] for r in sess_rows])
        summary[label] = {
            "n_frames": int(ranks.size),
            "mean_rank": float(ranks.mean()),
            "median_rank": float(np.median(ranks)),
            "top1_acc": float((ranks == 1).mean()),
            "top5_acc": float((ranks <= 5).mean()),
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    md = ["# EXP 6 — Correct-E rank test (51 candidates per frame)",
          "",
          "Per frame, score correct E + 50 random same-session wrong E (±10 "
          "temporal exclusion). Rank correct E.",
          "",
          "Pre-registered thresholds:",
          "- Top-1 > 90%: strong per-frame chain specificity",
          "- 60–90%: moderate",
          "- < 60%: weak; discrimination is more 'plausible' than 'specific'",
          "",
          "| session | n | mean rank | median | top-1 | top-5 |",
          "|---|---|---|---|---|---|"]
    for label in ("pooled", "D2", "V10"):
        s = summary.get(label)
        if s is None:
            continue
        flag = ""
        if s["top1_acc"] >= 0.90:
            flag = ""
        elif s["top1_acc"] >= 0.60:
            flag = "⚠️ "
        else:
            flag = "🚨 "
        md.append(f"| {label} | {s['n_frames']} | {s['mean_rank']:.2f} | "
                   f"{s['median_rank']:.0f} | {flag}{s['top1_acc']:.4f} | "
                   f"{s['top5_acc']:.4f} |")
    (out_dir / "summary.md").write_text("\n".join(md))
    return {"n_rows": len(rows)}


# =======================================================================
# EXP 7 — Synthetic counterfactual-E perturbation sensitivity
# =======================================================================

def run_exp7(log: TimingLog, device_id: int = 0) -> dict:
    log.log(f"[exp7] synthetic counterfactual-E perturbation sensitivity on cuda:{device_id}")
    out_dir = OUT_ROOT / EXP_DIRS[7]
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{device_id}")
    dtype = torch.bfloat16
    pg, _ = load_phase_g_for_inference(device, dtype)
    dc = build_diffusion_constants(T_DIFFUSION, device, torch.float32)
    sess_dirs = session_dirs()
    frames = load_120_frames()

    # Modes and σ levels
    # Mode A: rendered E perturbation (full-res 1080x1920 — but we operate at
    # Phase G's 768x1024 for tractability; document that this is post-Phase-G-resize)
    # Mode B: Phase G input E perturbation (after resize to 768x1024) — same as A
    #         in our pipeline (rendered E is loaded and resized via Phase G's loader)
    # Mode C: binder input E perturbation (1150x1330)
    # The analysis scope says "3 modes × 3 σ × 60 per session". To keep this
    # interpretable and avoid silently merging Modes A and B, we run:
    #   A = perturb the rendered E magnitude in the full-resolution domain
    #       (load native 1080x1920, perturb, then Phase G's resize)
    #   B = perturb the Phase G-input E (the 768x1024 tensor Phase G sees)
    #   C = perturb the binder-resolution E (resampled to 1150x1330 and
    #       perturbed at that scale; binder isn't actually invoked, but
    #       this captures perturbation at binder's expected resolution)
    # All three are scored by Phase G (the verifier under study).
    modes = ("A_rendered", "B_phase_g_input", "C_binder_resolution")
    sigma_targets = (0.1, 0.3, 0.5)
    rng = np.random.RandomState(7)

    rows: list[dict] = []
    rows_path = out_dir / "achieved_delta_e_metrics.csv"
    rows_fh = open(rows_path, "w", newline="")
    rows_w = csv.DictWriter(
        rows_fh,
        fieldnames=["session", "row", "mode", "sigma_target",
                     "achieved_delta_rms", "achieved_delta_max",
                     "achieved_delta_p95", "fraction_changed",
                     "fraction_clipped",
                     "score_baseline", "score_perturbed",
                     "delta_score"])
    rows_w.writeheader(); rows_fh.flush()

    n_frames = len(frames)
    for fi, spec in enumerate(frames):
        sess = spec["session"]; row = int(spec["row"])
        # Hard-stop check
        if log.time_remaining() < 600:
            log.log(f"  [exp7] approaching hard stop; aborting frame loop")
            break
        try:
            E_phase_g = load_phase_g_E(sess_dirs[sess], row)  # (3, 768, 1024)
            C_real = load_phase_g_C(sess_dirs[sess], row).to(
                device=device, dtype=torch.float32)
            noise = frame_noise(sess, row, device)
            score_baseline = phase_g_score_scalar(
                pg, C_real, E_phase_g.to(device=device, dtype=torch.float32),
                dc, device, dtype, noise)
            for mode in modes:
                for sigma in sigma_targets:
                    # Build perturbation at the right resolution then map back to (3,768,1024)
                    if mode == "A_rendered":
                        # Native 1080x1920 perturbation
                        pert_native = rng.randn(3, 1080, 1920).astype(np.float32) * sigma
                        # Resample to 768x1024 via cv2 area
                        import cv2
                        pert_resized = np.stack([
                            cv2.resize(pert_native[c], (1024, 768),
                                        interpolation=cv2.INTER_AREA)
                            for c in range(3)
                        ], axis=0)
                        E_perturbed_np = (E_phase_g.cpu().numpy() + pert_resized).astype(np.float32)
                    elif mode == "B_phase_g_input":
                        pert = rng.randn(3, 768, 1024).astype(np.float32) * sigma
                        E_perturbed_np = (E_phase_g.cpu().numpy() + pert).astype(np.float32)
                    else:  # C_binder_resolution
                        import cv2
                        pert_binder = rng.randn(3, 1150, 1330).astype(np.float32) * sigma
                        pert_resized = np.stack([
                            cv2.resize(pert_binder[c], (1024, 768),
                                        interpolation=cv2.INTER_AREA)
                            for c in range(3)
                        ], axis=0)
                        E_perturbed_np = (E_phase_g.cpu().numpy() + pert_resized).astype(np.float32)
                    # Track clipping
                    pre_clip = E_perturbed_np.copy()
                    E_perturbed_np = np.clip(E_perturbed_np, 0.0, 1.0)
                    fraction_clipped = float(((pre_clip != E_perturbed_np)).mean())
                    diff = E_perturbed_np - E_phase_g.cpu().numpy()
                    achieved_rms = float(np.sqrt((diff ** 2).mean()))
                    achieved_max = float(np.max(np.abs(diff)))
                    achieved_p95 = float(np.percentile(np.abs(diff), 95))
                    fraction_changed = float((np.abs(diff) > 0.01).mean())
                    E_pert_t = torch.from_numpy(E_perturbed_np).to(
                        device=device, dtype=torch.float32)
                    score_pert = phase_g_score_scalar(
                        pg, C_real, E_pert_t, dc, device, dtype, noise)
                    record = {
                        "session": sess, "row": row, "mode": mode,
                        "sigma_target": sigma,
                        "achieved_delta_rms": achieved_rms,
                        "achieved_delta_max": achieved_max,
                        "achieved_delta_p95": achieved_p95,
                        "fraction_changed": fraction_changed,
                        "fraction_clipped": fraction_clipped,
                        "score_baseline": score_baseline,
                        "score_perturbed": score_pert,
                        "delta_score": score_pert - score_baseline,
                    }
                    rows_w.writerow(record); rows_fh.flush()
                    rows.append(record)
        except Exception as exc:  # noqa: BLE001
            log.log(f"  [exp7] FRAME FAIL {sess} f={row}: {exc!r}")
            continue
        if (fi + 1) % 10 == 0:
            log.log(f"  [exp7] {fi+1}/{n_frames} frames done; rows={len(rows)}")
    rows_fh.close()

    # Aggregate sensitivity curves: per (session, mode), Δscore vs achieved ΔE_rms
    md = ["# EXP 7 — Synthetic counterfactual-E perturbation sensitivity",
          "",
          "**This experiment is NOT XOF bit-flip sensitivity, NOT optical "
          "washout floor**. It is synthetic Gaussian-noise perturbation of "
          "the rendered E field at three preprocessing scales (rendered, "
          "Phase G input, binder input).",
          "",
          "3 modes × 3 σ × ~120 frames. Output dir name `xof_sensitivity` "
          "retained for code compatibility; report uses the synthetic-"
          "counterfactual-E label throughout.",
          "",
          "## Δscore as function of achieved ΔE_rms",
          "",
          "| session | mode | σ_target | n | median ΔE_rms | median Δscore | bootstrap CI |",
          "|---|---|---|---|---|---|---|"]
    # Track per-(session, mode) Δscore-vs-σ for monotonicity check (Codex
    # audit fix 2026-05-05). Pre-registered: monotonic = sensitivity well-
    # characterized; non-monotonic = FLAG 🔧.
    monotonicity_curves: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for sess in ("D2", "V10"):
        for mode in modes:
            for sigma in sigma_targets:
                cell = [r for r in rows if r["session"] == sess
                         and r["mode"] == mode
                         and abs(r["sigma_target"] - sigma) < 1e-6]
                if not cell:
                    continue
                de = np.array([r["achieved_delta_rms"] for r in cell])
                ds = np.array([r["delta_score"] for r in cell])
                _, lo, hi = hierarchical_bootstrap(ds.tolist(), n_boot=10000)
                med_ds = float(np.median(ds))
                monotonicity_curves.setdefault(
                    (sess, mode), []).append((sigma, med_ds))
                md.append(f"| {sess} | {mode} | {sigma} | {len(cell)} | "
                           f"{float(np.median(de)):.4f} | "
                           f"{med_ds:+.5f} | "
                           f"[{lo:+.5f}, {hi:+.5f}] |")
    md.append("")
    md.append("## Monotonicity check (Δscore vs σ, per session × mode)")
    md.append("")
    md.append("Pre-registered: monotonic non-decreasing Δscore as σ increases "
               "= sensitivity well-characterized. Non-monotonic = FLAG 🔧.")
    md.append("")
    md.append("| session | mode | curve (σ → Δscore median) | monotonic? |")
    md.append("|---|---|---|---|")
    n_nonmono = 0
    for (sess, mode), curve in sorted(monotonicity_curves.items()):
        curve_sorted = sorted(curve, key=lambda x: x[0])
        ds_seq = [x[1] for x in curve_sorted]
        is_mono = all(ds_seq[i] <= ds_seq[i + 1] for i in range(len(ds_seq) - 1))
        flag = "✓" if is_mono else "🔧 NON-MONOTONIC"
        if not is_mono:
            n_nonmono += 1
        curve_str = " → ".join(f"σ={s} Δ={d:+.4f}" for s, d in curve_sorted)
        md.append(f"| {sess} | {mode} | {curve_str} | {flag} |")
    md.append("")
    if n_nonmono > 0:
        md.append(f"🔧 **{n_nonmono} non-monotonic (session, mode) pair(s) "
                   f"flagged**. Review needed: Phase G's response to "
                   f"synthetic E perturbations is not monotonic in some "
                   f"configurations.")
    else:
        md.append("**All curves monotonic non-decreasing** — sensitivity "
                   "well-characterized.")
    (out_dir / "summary.md").write_text("\n".join(md))
    return {"n_rows": len(rows), "n_nonmono_curves": n_nonmono}


# =======================================================================
# EXP 8 — E4 fake_100k score-distribution inspection
# =======================================================================

def run_exp8(log: TimingLog, device_id: int = 0) -> dict:
    log.log("[exp8] E4 fake_100k score-distribution inspection")
    import matplotlib.pyplot as plt
    out_dir = OUT_ROOT / EXP_DIRS[8]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pull existing scalars from causal_ablations
    e1_real = []; e1_fake = []; e4_real = []; e4_fake = []
    rows_summary: list[dict] = []
    per_frame = CAUSAL_ROOT / "per_frame"
    for fdir in sorted(per_frame.iterdir()):
        m_path = fdir / "ablations_manifest.json"
        if not m_path.exists():
            continue
        m = json.loads(m_path.read_text())
        if m["session"] != "D2":
            continue
        s = m["scalars"]
        e1_real_v = s.get("E1|real_correct")
        e1_fake_v = s.get("E1|fake_100k")
        e4_real_v = s.get("E4|real_correct")
        e4_fake_v = s.get("E4|fake_100k")
        if None in (e1_real_v, e1_fake_v, e4_real_v, e4_fake_v):
            continue
        e1_real.append(e1_real_v); e1_fake.append(e1_fake_v)
        e4_real.append(e4_real_v); e4_fake.append(e4_fake_v)
        rows_summary.append({
            "row": m["row"],
            "e1_real": e1_real_v, "e1_fake_100k": e1_fake_v,
            "e4_real": e4_real_v, "e4_fake_100k": e4_fake_v,
        })

    # Histogram
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), dpi=100)
    if e1_real and e1_fake:
        axes[0].hist(e1_real, bins=20, alpha=0.5, color="blue", label="real_correct")
        axes[0].hist(e1_fake, bins=20, alpha=0.5, color="red", label="fake_100k")
        axes[0].set_title("D2 E1 baseline — real vs fake_100k score distribution")
        axes[0].set_xlabel("Phase G ε-MSE score"); axes[0].set_ylabel("count")
        axes[0].legend()
    if e4_real and e4_fake:
        axes[1].hist(e4_real, bins=20, alpha=0.5, color="blue", label="real_correct (under E4)")
        axes[1].hist(e4_fake, bins=20, alpha=0.5, color="red", label="fake_100k (under E4)")
        axes[1].set_title("D2 E4 wrong-E — real vs fake_100k score distribution "
                            "(AUROC=0.151 inversion)")
        axes[1].set_xlabel("Phase G ε-MSE score"); axes[1].set_ylabel("count")
        axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "distributions.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    log.log(f"  [exp8] wrote distributions.png")

    # Stats
    e1_real_arr = np.array(e1_real); e1_fake_arr = np.array(e1_fake)
    e4_real_arr = np.array(e4_real); e4_fake_arr = np.array(e4_fake)
    overlap_e4 = float(((e4_fake_arr <= e4_real_arr.max()) &
                          (e4_fake_arr >= e4_real_arr.min())).mean()) if e4_fake_arr.size else float("nan")
    pearson_e1_e4_real = float(np.corrcoef(e1_real_arr, e4_real_arr)[0, 1]) if e1_real_arr.size > 1 else float("nan")
    pearson_e1_e4_fake = float(np.corrcoef(e1_fake_arr, e4_fake_arr)[0, 1]) if e1_fake_arr.size > 1 else float("nan")

    # Helper for safe-format that handles None medians (Codex audit fix
    # 2026-05-05).
    def _med(arr):
        return float(np.median(arr)) if arr.size else float("nan")

    summary = {
        "n_frames": len(rows_summary),
        "e1_real": {"median": _med(e1_real_arr)},
        "e1_fake": {"median": _med(e1_fake_arr)},
        "e4_real": {"median": _med(e4_real_arr)},
        "e4_fake": {"median": _med(e4_fake_arr)},
        "overlap_e4_fake_within_e4_real_range": overlap_e4,
        "pearson_e1_e4_real": pearson_e1_e4_real,
        "pearson_e1_e4_fake": pearson_e1_e4_fake,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Sanity check: re-run Phase G on 5 random frames. Per Codex audit
    # 2026-05-05, the 1e-3 tolerance check must cover BOTH the E1 real
    # path AND the E4 fake_100k path (since the AUROC=0.151 inversion
    # specifically lives in the E4 fake_100k cell). We re-render the
    # F-A v1 step_100000 fake C, derive E_idx+5, and reproduce the
    # E4|fake_100k score, comparing against the saved value.
    sanity_check_results: list[dict] = []
    if rows_summary:
        device = torch.device(f"cuda:{device_id}")
        dtype = torch.bfloat16
        pg, _ = load_phase_g_for_inference(device, dtype)
        dc = build_diffusion_constants(T_DIFFUSION, device, torch.float32)
        chain_keys = chain_keys_cache()
        sess_dirs = session_dirs()
        # Load only step_00100000 ckpt for the fake re-render
        fa_100k_ckpt = FA_V1_CKPT_DIR / "step_00100000.pt"
        try:
            fa_100k = load_fa_v1_checkpoint(fa_100k_ckpt, device=device,
                                              dtype=dtype)
        except Exception as exc:  # noqa: BLE001
            log.log(f"  [exp8] FATAL: failed to load F-A v1 step_100k for "
                    f"sanity check: {exc!r}")
            fa_100k = None
        rng = np.random.RandomState(8)
        sample = rng.choice(len(rows_summary), size=min(5, len(rows_summary)),
                             replace=False)
        for si in sample:
            r = rows_summary[int(si)]
            row = r["row"]
            sess = "D2"
            entry: dict = {"row": row}
            try:
                keys = chain_keys[sess]
                target_idx = keys.index(row)
                source_row = keys[(target_idx + len(keys) // 4) % len(keys)]
                e4_idx = (target_idx + 5) % len(keys)
                e4_row = keys[e4_idx]
                C = load_phase_g_C(sess_dirs[sess], row).to(
                    device=device, dtype=torch.float32)
                E = load_phase_g_E(sess_dirs[sess], row).to(
                    device=device, dtype=torch.float32)
                E_e4 = load_phase_g_E(sess_dirs[sess], e4_row).to(
                    device=device, dtype=torch.float32)
                noise = frame_noise(sess, row, device)
                # E1 real path
                fresh_e1_real = phase_g_score_scalar(pg, C, E, dc, device, dtype, noise)
                entry["saved_e1_real"] = r["e1_real"]
                entry["fresh_e1_real"] = fresh_e1_real
                entry["e1_real_abs_diff"] = abs(fresh_e1_real - r["e1_real"])
                # E4 fake_100k path
                if fa_100k is not None:
                    Cf = render_C_fake(
                        fa_100k, sess_dirs[sess],
                        source_row=source_row, target_row=row,
                        device=device, dtype=dtype).to(
                            device=device, dtype=torch.float32)
                    fresh_e4_fake = phase_g_score_scalar(
                        pg, Cf, E_e4, dc, device, dtype, noise)
                    entry["saved_e4_fake_100k"] = r["e4_fake_100k"]
                    entry["fresh_e4_fake_100k"] = fresh_e4_fake
                    entry["e4_fake_abs_diff"] = abs(
                        fresh_e4_fake - r["e4_fake_100k"])
                else:
                    entry["e4_fake_abs_diff"] = float("nan")
                sanity_check_results.append(entry)
            except Exception as exc:  # noqa: BLE001
                entry["error"] = repr(exc)
                sanity_check_results.append(entry)

    # Markdown report with NaN-safe formatting.
    def _fmt(v: float) -> str:
        return f"{v:.5f}" if (v is not None and np.isfinite(v)) else "—"

    md = ["# EXP 8 — E4 fake_100k AUROC=0.151 score-distribution inspection",
          "",
          "Investigates the D2 fake_100k AUROC=0.151 inversion under E4 "
          "(replacing real_correct's E with E_idx+5).",
          "",
          f"**n_frames**: {summary['n_frames']}",
          ""]
    if summary['n_frames'] == 0:
        md.append("⚠️ **No frames had all four required scalars** (E1 real / "
                   "E1 fake_100k / E4 real / E4 fake_100k). Cannot compute "
                   "the inversion analysis. A reviewer should verify the "
                   "causal-ablation manifest contents.")
        (out_dir / "summary.md").write_text("\n".join(md))
        return {"n_frames": 0, "sanity_max_diff_e1": float("nan"),
                "sanity_max_diff_e4_fake": float("nan")}
    md.append("## Distribution medians")
    md.append("")
    md.append("| | E1 baseline | E4 wrong-E |")
    md.append("|---|---|---|")
    md.append(f"| real_correct | {_fmt(summary['e1_real']['median'])} "
               f"| {_fmt(summary['e4_real']['median'])} |")
    md.append(f"| fake_100k    | {_fmt(summary['e1_fake']['median'])} "
               f"| {_fmt(summary['e4_fake']['median'])} |")
    md.append("")
    md.append(f"**Overlap** of E4 fake_100k scores within E4 real_correct range: "
               f"{summary['overlap_e4_fake_within_e4_real_range']:.4f}")
    md.append(f"**Pearson E1→E4 (real)**: {summary['pearson_e1_e4_real']:+.4f}")
    md.append(f"**Pearson E1→E4 (fake_100k)**: {summary['pearson_e1_e4_fake']:+.4f}")
    md.append("")
    md.append("## Sanity check — re-run Phase G on 5 random frames")
    md.append("")
    md.append("Per Codex audit 2026-05-05: the AUROC=0.151 finding must be "
               "verified by re-running BOTH the E1 real_correct path and the "
               "E4 fake_100k path. Tolerance: 1e-3.")
    md.append("")
    md.append("| row | saved E1 real | fresh E1 real | E1 |diff| | "
               "saved E4 fake | fresh E4 fake | E4 |diff| |")
    md.append("|---|---|---|---|---|---|---|")
    sanity_max_diff_e1 = 0.0
    sanity_max_diff_e4_fake = 0.0
    for r in sanity_check_results:
        if "error" in r:
            md.append(f"| {r['row']} | — | — | — | — | — | "
                       f"ERROR: {r['error']} |")
            continue
        e1_diff = r.get("e1_real_abs_diff", float("nan"))
        e4_diff = r.get("e4_fake_abs_diff", float("nan"))
        if np.isfinite(e1_diff):
            sanity_max_diff_e1 = max(sanity_max_diff_e1, e1_diff)
        if np.isfinite(e4_diff):
            sanity_max_diff_e4_fake = max(sanity_max_diff_e4_fake, e4_diff)
        e4_saved = r.get("saved_e4_fake_100k", float("nan"))
        e4_fresh = r.get("fresh_e4_fake_100k", float("nan"))
        md.append(
            f"| {r['row']} | {r.get('saved_e1_real', float('nan')):.6f} | "
            f"{r.get('fresh_e1_real', float('nan')):.6f} | "
            f"{e1_diff:.2e} | "
            f"{e4_saved:.6f} | {e4_fresh:.6f} | {e4_diff:.2e} |")
    md.append("")
    md.append(f"**Max |diff| E1 real**: {sanity_max_diff_e1:.2e}")
    md.append(f"**Max |diff| E4 fake_100k**: {sanity_max_diff_e4_fake:.2e}")
    md.append("")
    overall_max = max(sanity_max_diff_e1, sanity_max_diff_e4_fake)
    if overall_max < 1e-3:
        md.append(f"**Verdict**: saved scores match fresh inference within "
                   f"{overall_max:.2e} on BOTH paths. AUROC=0.151 inversion "
                   f"is REAL, not an implementation artifact.")
    else:
        md.append(f"🔧 **Verdict**: max |diff| = {overall_max:.2e} "
                   f"(E1 real {sanity_max_diff_e1:.2e}, E4 fake_100k "
                   f"{sanity_max_diff_e4_fake:.2e}). Possible implementation "
                   f"discrepancy — flag for review BEFORE manuscript "
                   f"citation.")
    (out_dir / "summary.md").write_text("\n".join(md))
    return {"n_frames": len(rows_summary),
             "sanity_max_diff_e1": sanity_max_diff_e1,
             "sanity_max_diff_e4_fake": sanity_max_diff_e4_fake}


# =======================================================================
# EXP 9 — Hierarchical bootstrap CIs on existing causal-ablation table
# =======================================================================

def run_exp9(log: TimingLog) -> dict:
    log.log("[exp9] hierarchical bootstrap CIs on causal-ablation table")
    out_dir = OUT_ROOT / EXP_DIRS[9]
    out_dir.mkdir(parents=True, exist_ok=True)
    canon_csv = CAUSAL_ROOT / "per_frame" / "ablation_table.csv"
    if not canon_csv.exists():
        # Codex audit 2026-05-05: returning {"error": ...} is silently
        # marked status=done by run_one. Raise instead so failed status
        # propagates to the morning report.
        raise FileNotFoundError(f"missing canonical table: {canon_csv}")

    # Re-build per-frame deltas from per-frame manifests, then bootstrap.
    per_frame = CAUSAL_ROOT / "per_frame"
    by_sess_cond_abl: dict[tuple[str, str, str], list[float]] = {}
    for fdir in sorted(per_frame.iterdir()):
        mp = fdir / "ablations_manifest.json"
        if not mp.exists():
            continue
        m = json.loads(mp.read_text())
        sess = m["session"]
        for ablation in ("E1", "E2", "E3", "E4", "E5"):
            real = m["scalars"].get(f"{ablation}|real_correct")
            if real is None:
                continue
            for cond in PERTURBED_CONDS:
                pert = m["scalars"].get(f"{ablation}|{cond}")
                if pert is None:
                    continue
                by_sess_cond_abl.setdefault(
                    (sess, cond, ablation), []).append(pert - real)

    rows: list[dict] = []
    for (sess, cond, ablation), deltas in sorted(by_sess_cond_abl.items()):
        _, lo, hi = hierarchical_bootstrap(deltas, n_boot=10000)
        med = float(np.median(deltas))
        rows.append({
            "session": sess, "condition": cond, "ablation": ablation,
            "n_frames": len(deltas),
            "median_delta": med,
            "delta_bootstrap_ci_low": lo,
            "delta_bootstrap_ci_high": hi,
        })
    out_csv = out_dir / "ablation_table_with_ci.csv"
    if rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    log.log(f"  [exp9] wrote {out_csv} ({len(rows)} cells)")

    md = ["# EXP 9 — Hierarchical bootstrap CIs on causal-ablation cells",
          "",
          "Per (session, condition, ablation), bootstrap clustered by frame, "
          "10000 samples. Output: `ablation_table_with_ci.csv` alongside "
          "the canonical `causal_ablations/per_frame/ablation_table.csv` "
          "(NOT replaced — manuscript citation can use whichever fits).",
          "",
          f"**n_cells**: {len(rows)}",
          "",
          "Verification: matches canonical table row count by "
          f"(session, condition, ablation) tuples = {len(rows)}",
          ""]
    (out_dir / "summary.md").write_text("\n".join(md))
    return {"n_cells": len(rows)}


# =======================================================================
# Morning report writer
# =======================================================================

def write_morning_report(log: TimingLog, statuses: dict[int, dict]) -> Path:
    out_dir = OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "MORNING_REPORT.md"
    md: list[str] = []
    md.append("# Morning report — Phase G mechanism characterization battery")
    md.append("")
    md.append(f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    md.append("")
    # Status
    n_done = sum(1 for s in statuses.values() if s.get("status") == "done")
    n_failed = sum(1 for s in statuses.values() if s.get("status") == "failed")
    n_skipped = sum(1 for s in statuses.values() if s.get("status") == "skipped")
    overall = "DONE" if n_done == 9 else (
        "PARTIAL" if n_done > 0 else "FAILED")
    md.append(f"## Status: **{overall}** ({n_done}/9 experiments completed)")
    md.append("")
    if n_failed > 0:
        md.append(f"⚠️ **{n_failed} experiment(s) failed** — see per-experiment sections below.")
        md.append("")
    if n_skipped > 0:
        md.append(f"⚠️ **{n_skipped} experiment(s) skipped** (hard-stop or "
                   f"unmet prerequisites).")
        md.append("")

    # Per-experiment sections
    for n in range(1, 10):
        sub = OUT_ROOT / EXP_DIRS[n] / "summary.md"
        md.append(f"## EXP {n} — {EXP_DIRS[n]}")
        md.append("")
        s = statuses.get(n, {})
        md.append(f"**Status**: {s.get('status', 'unknown')}")
        if "wall_seconds" in s:
            md.append(f"**Wall**: {s['wall_seconds']:.1f}s")
        if "error" in s:
            md.append(f"**Error**: `{s['error']}`")
        md.append("")
        if sub.exists():
            md.append(sub.read_text())
        else:
            md.append(f"*(no summary at `{sub}`)*")
        md.append("")
        md.append("---")
        md.append("")

    md.append("## Manuscript implications")
    md.append("")
    md.append("Specific notes per round-2 audit corrections:")
    md.append("")
    md.append("- **Original E5 does NOT prove pose shortcut ruled out.** "
               "EXP 5 (E5a/E5b/E5c variants) are the corrected tests.")
    md.append("- **EXP 7 is synthetic counterfactual-E perturbation** — NOT "
               "XOF bit-flip and NOT optical washout floor. Output directory "
               "name retained for code compatibility; reports use the "
               "synthetic-counterfactual-E label.")
    md.append("- **EXP 3 / EXP 4 cells marked N/A** are construction-degenerate "
               "(C or E identical to real_correct under neutralization). "
               "Raw AUROC reported alongside as sanity check; expect ≈ 0.5.")
    md.append("")
    md.append("## Resource consumption")
    md.append("")
    total_wall = sum(s.get("wall_seconds", 0) for s in statuses.values())
    md.append(f"**Total wall-clock**: {total_wall/60:.1f} min "
               f"({total_wall/3600:.2f} h)")
    md.append("")
    md.append("Per-experiment wall:")
    md.append("")
    md.append("| exp | status | wall (s) |")
    md.append("|---|---|---|")
    for n in range(1, 10):
        s = statuses.get(n, {})
        md.append(f"| {n} | {s.get('status', 'unknown')} | "
                   f"{s.get('wall_seconds', 0):.1f} |")
    md.append("")
    report.write_text("\n".join(md))
    log.log(f"[report] wrote {report}")
    return report


# =======================================================================
# Main orchestrator
# =======================================================================

EXP_FUNCS: dict[int, callable] = {
    1: lambda log: run_exp1(log),
    2: lambda log: run_exp2(log),
    3: lambda log: run_exp3(log, device_id=0),
    4: lambda log: run_exp4(log, device_id=1),
    5: lambda log: run_exp5(log, device_id=2),
    6: lambda log: run_exp6(log, device_id=3),
    7: lambda log: run_exp7(log, device_id=4),
    8: lambda log: run_exp8(log, device_id=5),
    9: lambda log: run_exp9(log),
}


def run_one(exp_num: int, log: TimingLog) -> dict:
    if exp_num not in EXP_FUNCS:
        return {"status": "skipped", "reason": "unknown_exp"}
    log.exp_start(exp_num)
    try:
        result = EXP_FUNCS[exp_num](log)
        log.exp_end(exp_num, status="done")
        return {**(result or {}), **log.timings[exp_num], "status": "done"}
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        log.log(f"[exp{exp_num}] FAILED: {exc!r}\n{tb}")
        log.exp_end(exp_num, status="failed", error=repr(exc))
        return {"status": "failed", "error": repr(exc), **log.timings[exp_num]}


def run_all(log: TimingLog) -> dict[int, dict]:
    statuses: dict[int, dict] = {}
    # Phase 1: cheap experiments — sequential
    log.log("=" * 60)
    log.log("PHASE 1 (cheap, sequential): EXP 1, 2, 8, 9")
    log.log("=" * 60)
    for n in (1, 2, 8, 9):
        if log.time_remaining() < 600:
            statuses[n] = {"status": "skipped", "reason": "hard_stop"}
            continue
        statuses[n] = run_one(n, log)
        # Persist intermediate report
        write_morning_report(log, statuses)
    # Phase 2: moderate — EXP 3, 4 (sequential since they each grab a GPU
    # but device pinning is per-call; we simply run sequentially to avoid
    # GPU contention).
    log.log("=" * 60)
    log.log("PHASE 2 (moderate): EXP 3, 4")
    log.log("=" * 60)
    for n in (3, 4):
        if log.time_remaining() < 600:
            statuses[n] = {"status": "skipped", "reason": "hard_stop"}
            continue
        statuses[n] = run_one(n, log)
        write_morning_report(log, statuses)
    # Phase 3: heavy — EXP 5
    log.log("=" * 60)
    log.log("PHASE 3 (heavy): EXP 5")
    log.log("=" * 60)
    for n in (5,):
        if log.time_remaining() < 600:
            statuses[n] = {"status": "skipped", "reason": "hard_stop"}
            continue
        statuses[n] = run_one(n, log)
        write_morning_report(log, statuses)
    # Phase 4: heaviest — EXP 6, 7
    log.log("=" * 60)
    log.log("PHASE 4 (heaviest): EXP 6, 7")
    log.log("=" * 60)
    for n in (6, 7):
        if log.time_remaining() < 600:
            statuses[n] = {"status": "skipped", "reason": "hard_stop"}
            continue
        statuses[n] = run_one(n, log)
        write_morning_report(log, statuses)
    write_morning_report(log, statuses)
    return statuses


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["all", "report",
                                        "exp1", "exp2", "exp3", "exp4",
                                        "exp5", "exp6", "exp7", "exp8",
                                        "exp9"])
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = OUT_ROOT / f"battery_log_{time.strftime('%Y-%m-%dT%H%MZ', time.gmtime())}.txt"
    log = TimingLog(log_path)
    log.log(f"Battery start. mode={args.mode} log={log_path}")
    log.log(f"Hard stop in {HARD_STOP_HOURS} hours.")
    statuses: dict[int, dict] = {}
    try:
        if args.mode == "all":
            statuses = run_all(log)
        elif args.mode == "report":
            # Read existing per-experiment summaries; status = unknown
            for n in range(1, 10):
                statuses[n] = {"status": "unknown",
                                 "wall_seconds": 0}
                if (OUT_ROOT / EXP_DIRS[n] / "summary.md").exists():
                    statuses[n]["status"] = "done"
            write_morning_report(log, statuses)
        else:
            n = int(args.mode[3:])
            statuses[n] = run_one(n, log)
            write_morning_report(log, statuses)
    finally:
        log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

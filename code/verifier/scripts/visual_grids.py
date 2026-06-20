"""Visual grid generation — manuscript-ready figures from existing experiment outputs.

Per the project's spec 2026-05-03. CPU/IO only — does NOT touch GPUs (Phase H may
still be training). Reads from existing artifacts only:
    - sessions/d2 + sessions/v10 (.raw + .png)
    - chain_log.csv per session (for byte-stream derivation)
    - experiments/stage_0/eval/step_*/stage0_*_raw.npz (verifier MSE)
    - experiments/cross_session_ablation/{d2_only,v10_only}/eval/summary.json
    - experiments/item_1/eval/summary.json
    - experiments/phase_h_supervised_baseline/eval/* (when available)

Writes to:
    /path/to/poliebotics_phase_b/visual_grids/

Grids:
    1. session_overview_D2.png + session_overview_V10.png  (E + C 6×6 grid each)
    2. perturbation_examples_D2_f100.png                    (3×4 perturbation variants)
    3. fa_v1_training_trajectory.png                        (SKIPPED by default — needs C_fake)
    4. phase_g_verifier_response.png                        (6 frames × 4 conds + MSE labels)
    5. cross_session_{d2_to_v10,v10_to_d2}.png             (cross-session generalization viz)
    6. item_1_sensitivity_curves.png                        (4-panel)
    7. phase_h_baseline_results.png                         (gated on Phase H eval landing)

After all grids: writes MANIFEST.md with sizes + BLAKE3 hashes + source paths.

Run:
    python scripts/visual_grids.py [--include-fa-trajectory] [--phase-h-only] [--out DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Callable, Sequence

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase_g.xof_perturb import (  # noqa: E402
    load_chain_log, render_E_for_phase_h, load_canonical_E_phase_g_resolution,
    _spec_by_label, TRAIN_POOL_LABELS, HELDOUT_POOL_LABELS,
)
from phase_g.diffusion_diagnostic_dataset import (  # noqa: E402
    _crop_and_resize_C, _load_packed_cfa_float01, EVAL_BLOCKS,
)


# ---------------- defaults / paths ----------------

DEFAULT_OUT = Path("/path/to/poliebotics_phase_b/visual_grids")
SESSION_DIRS = {
    "D2": Path("/path/to/poliebotics_phase_b/poliebotics_phase_b/data/d2"),
    "V10": Path("/path/to/poliebotics_phase_b/poliebotics_phase_b/data/v10"),
}
EXPER_ROOT = Path("/path/to/poliebotics_phase_b/experiments")
PHASE_G_ROOT = EXPER_ROOT / "phase_g_diffusion_diagnostic"
STAGE_0_ROOT = EXPER_ROOT / "stage_0"
CROSS_ROOT = EXPER_ROOT / "cross_session_ablation"
ITEM_1_ROOT = EXPER_ROOT / "item_1"
PHASE_H_ROOT = EXPER_ROOT / "phase_h_supervised_baseline"

# Decide between Lambda and local roots. ALL Lambda roots must exist for
# Lambda mode; otherwise ALL must be local. Mixed state is a CRITICAL bug
# (would silently mix mounted/unmounted partial trees and write to the wrong
# place). Fail loudly on mixed.
_LAMBDA_ROOTS = [SESSION_DIRS["D2"], SESSION_DIRS["V10"], EXPER_ROOT]
_LAMBDA_AVAIL = sum(int(p.exists()) for p in _LAMBDA_ROOTS)
if _LAMBDA_AVAIL == len(_LAMBDA_ROOTS):
    pass  # use Lambda paths as defined above
elif _LAMBDA_AVAIL == 0:
    LOCAL_ROOT = Path(__file__).resolve().parents[1]
    SESSION_DIRS = {
        "D2": LOCAL_ROOT / "data" / "d2",
        "V10": LOCAL_ROOT / "data" / "v10",
    }
    EXPER_ROOT = LOCAL_ROOT / "experiments"
    PHASE_G_ROOT = EXPER_ROOT / "phase_g_diffusion_diagnostic"
    STAGE_0_ROOT = EXPER_ROOT / "stage_0"
    CROSS_ROOT = EXPER_ROOT / "cross_session_ablation"
    ITEM_1_ROOT = EXPER_ROOT / "item_1"
    PHASE_H_ROOT = EXPER_ROOT / "phase_h_supervised_baseline"
    DEFAULT_OUT = LOCAL_ROOT / "results" / "visual_grids"
else:
    raise SystemExit(
        f"[visual_grids] mixed root state: {_LAMBDA_AVAIL}/{len(_LAMBDA_ROOTS)} "
        f"Lambda paths exist. This is dangerous (could silently write to wrong "
        f"tree). Either run on Lambda (all paths present) or local (all absent), "
        f"not in between. Lambda paths checked: "
        f"{[str(p) for p in _LAMBDA_ROOTS]}"
    )


# ---------------- helpers ----------------

def _gamma_correct(rgb01: np.ndarray, gamma: float = 1.6) -> np.ndarray:
    """Apply mild gamma so dark scene content is visible. Input/output [0,1]."""
    return np.clip(rgb01, 0, 1) ** (1.0 / gamma)


def _packed_cfa_to_rgb(cfa: np.ndarray | "np.ndarray") -> np.ndarray:
    """Packed CFA (4, H, W) [0,1] → RGB (H, W, 3) uint8 for visualization."""
    if hasattr(cfa, "numpy"):
        cfa = cfa.numpy()
    R, G1, G2, B = cfa[0], cfa[1], cfa[2], cfa[3]
    G = 0.5 * (G1 + G2)
    rgb = np.stack([R, G, B], axis=-1)
    rgb = _gamma_correct(rgb)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def _e_tensor_to_rgb(e: "torch.Tensor | np.ndarray") -> np.ndarray:
    """E tensor (3, H, W) [0,1] → RGB (H, W, 3) uint8."""
    if hasattr(e, "numpy"):
        e = e.numpy()
    arr = e.transpose(1, 2, 0)
    arr = _gamma_correct(arr)
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def _load_C_resized(session_dir: Path, frame_id: int) -> np.ndarray:
    """Load .raw → packed CFA → crop+resize to 768×1024 → debayered RGB uint8."""
    cfa = _crop_and_resize_C(_load_packed_cfa_float01(
        session_dir / "Recordings" / f"frame_{frame_id:06d}.raw"))
    return _packed_cfa_to_rgb(cfa)


def _load_E_resized(session_dir: Path, frame_id: int) -> np.ndarray:
    """Load tile_<frame>.png → resize to 768×1024 via INTER_AREA → RGB uint8."""
    tile = cv2.imread(str(session_dir / "derived" / "Emissions" / f"tile_{frame_id:06d}.png"),
                      cv2.IMREAD_UNCHANGED)
    tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(tile_rgb, (1024, 768), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32) / 255.0
    return (_gamma_correct(arr) * 255).astype(np.uint8)


def _safe_json_load(path: Path) -> dict | None:
    """Read+parse JSON; return None on missing/empty/parse error (skip+message)."""
    try:
        if not path.exists() or path.stat().st_size == 0:
            return None
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as ex:
        print(f"  [WARN] cannot parse JSON at {path}: {ex}; skipping")
        return None


def _safe_npz_load(path: Path):
    """Load NPZ; return None on missing/empty/corrupt (skip+message)."""
    try:
        if not path.exists() or path.stat().st_size == 0:
            return None
        return np.load(path, allow_pickle=False)
    except (ValueError, OSError) as ex:
        print(f"  [WARN] cannot load NPZ at {path}: {ex}; skipping")
        return None


def _blake3_file(path: Path) -> str:
    try:
        from blake3 import blake3 as _blake3
        h = _blake3()
    except ImportError:
        h = hashlib.blake2b(digest_size=32)
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


# ---------------- Grid 1: session overview ----------------

def grid_session_overview(session: str, out_dir: Path, n_per_side: int = 6,
                          seed: int = 42) -> Path | None:
    """6×6 grid of (E top half, C bottom half) per cell."""
    sd = SESSION_DIRS[session]
    if not (sd / "Recordings").exists():
        print(f"[grid 1] SKIP {session}: session dir missing")
        return None
    chain = load_chain_log(sd)
    rng = np.random.RandomState(seed)
    valid = sorted(chain.keys())
    chosen = sorted(rng.choice(valid, size=min(n_per_side * n_per_side, len(valid)),
                               replace=False).tolist())
    print(f"[grid 1] {session}: rendering {len(chosen)} cells...")

    fig, axes = plt.subplots(n_per_side, n_per_side,
                             figsize=(3 * n_per_side, 3.6 * n_per_side))
    for k, frame_id in enumerate(chosen):
        ax = axes[k // n_per_side, k % n_per_side]
        try:
            e_rgb = _load_E_resized(sd, frame_id)
            c_rgb = _load_C_resized(sd, frame_id)
        except Exception as ex:
            print(f"  skip frame {frame_id}: {ex}")
            ax.axis("off")
            continue
        # Stack vertically: E on top, C on bottom
        target_w = 320
        e_disp = cv2.resize(e_rgb, (target_w, int(e_rgb.shape[0] * target_w / e_rgb.shape[1])),
                            interpolation=cv2.INTER_AREA)
        c_disp = cv2.resize(c_rgb, (target_w, int(c_rgb.shape[0] * target_w / c_rgb.shape[1])),
                            interpolation=cv2.INTER_AREA)
        gap = np.full((4, target_w, 3), 30, dtype=np.uint8)
        cell = np.concatenate([e_disp, gap, c_disp], axis=0)
        ax.imshow(cell)
        ax.set_title(f"frame {frame_id}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{session} session overview — emission (top) / capture (bottom)",
                 fontsize=14)
    fig.tight_layout()
    out = out_dir / f"session_overview_{session}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}  ({out.stat().st_size // 1024} KB)")
    return out


# ---------------- Grid 2: perturbation examples ----------------

PERTURBATION_GRID_LABELS = [
    ("identity",            "canonical E"),
    ("xof_t1_global_k1",    "Type 1 k=1"),
    ("xof_t1_global_k64",   "Type 1 k=64"),
    ("xof_t1_global_k4096", "Type 1 k=4096"),
    ("xof_t2_oct0_k64",     "Type 2 oct0 k=64"),
    ("xof_t2_oct2_k64",     "Type 2 oct2 k=64"),
    ("xof_t3_region_k64",   "Type 3 region k=64"),
    ("xof_t4_swap_oct2",    "Type 4 swap oct2"),
    ("xof_t5_swap_R",       "Type 5 swap R"),
    ("xof_t6_replace_general", "Type 6 replace"),
    # Three "non-perturbation" comparators:
    ("__shuffled",          "shuffled (far frame)"),
    ("__source",            "source E (frame−2)"),
]


def grid_perturbation_examples(session: str, frame_id: int, out_dir: Path,
                               donor_lag: int = 200) -> Path | None:
    sd = SESSION_DIRS[session]
    if not (sd / "Recordings").exists():
        print(f"[grid 2] SKIP {session}: session dir missing")
        return None
    chain = load_chain_log(sd)
    if frame_id not in chain:
        print(f"[grid 2] frame {frame_id} not in {session} chain log")
        return None
    print(f"[grid 2] rendering 12 perturbation variants on {session} frame {frame_id}...")

    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    for idx, (label, display_name) in enumerate(PERTURBATION_GRID_LABELS):
        ax = axes[idx // 4, idx % 4]
        try:
            if label == "identity":
                e = render_E_for_phase_h(session, frame_id, "identity", chain, device="cpu")
            elif label == "__shuffled":
                # Far frame within session (lag = donor_lag)
                partner = (frame_id + donor_lag) % max(chain.keys())
                while partner not in chain:
                    partner = (partner + 1) % max(chain.keys())
                e = render_E_for_phase_h(session, partner, "identity", chain, device="cpu")
            elif label == "__source":
                src_frame = frame_id - 2
                if src_frame in chain:
                    e = render_E_for_phase_h(session, src_frame, "identity", chain, device="cpu")
                else:
                    ax.axis("off"); continue
            else:
                spec = _spec_by_label(label)
                if spec.needs_donor():
                    donor = (frame_id + donor_lag) % max(chain.keys())
                    while donor not in chain:
                        donor = (donor + 1) % max(chain.keys())
                    e = render_E_for_phase_h(session, frame_id, label, chain,
                                             donor_chain_log=chain, donor_frame_id=donor,
                                             device="cpu")
                else:
                    e = render_E_for_phase_h(session, frame_id, label, chain, device="cpu")
            rgb = _e_tensor_to_rgb(e)
            ax.imshow(rgb)
            ax.set_title(display_name, fontsize=11)
            ax.axis("off")
        except Exception as ex:
            print(f"  [grid 2] skip {label}: {ex}")
            ax.axis("off")
    fig.suptitle(f"Perturbation examples — {session} frame {frame_id}", fontsize=14)
    fig.tight_layout()
    out = out_dir / f"perturbation_examples_{session}_f{frame_id}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}  ({out.stat().st_size // 1024} KB)")
    return out


# ---------------- Grid 4: Phase G verifier response ----------------

def grid_phase_g_verifier_response(out_dir: Path, n_d2: int = 3, n_v10: int = 3,
                                   ckpt_step: int = 100000) -> Path | None:
    """6 frames × 4 conditions, with MSE values from Stage 0 NPZs.

    Uses Stage 0 step-100k NPZ which has real_correct/shuffled/source/zero conds.
    """
    print(f"[grid 4] Phase G verifier response — {n_d2} D2 + {n_v10} V10 frames...")
    rows_per_session = []
    for session, n in (("D2", n_d2), ("V10", n_v10)):
        sd = SESSION_DIRS[session]
        npz_path = STAGE_0_ROOT / "eval" / f"step_{ckpt_step:08d}" / f"stage0_{session.lower()}_raw.npz"
        z = _safe_npz_load(npz_path)
        if z is None:
            print(f"  [grid 4] SKIP {session}: {npz_path} missing/corrupt")
            continue
        try:
            rows = z["rows"]
        except KeyError:
            print(f"  [grid 4] SKIP {session}: NPZ missing 'rows' key")
            continue
        # Pick n evenly across the array
        step = max(1, len(rows) // n)
        chosen_idx = list(range(0, len(rows), step))[:n]
        for ci in chosen_idx:
            row = int(rows[ci])
            rows_per_session.append((session, sd, row, z, ci))
    if not rows_per_session:
        print("[grid 4] SKIP: no Stage 0 NPZ data")
        return None

    n_rows = len(rows_per_session)
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    cond_keys = [
        ("real_correct", "correct E"),
        ("real_shuffled", "shuffled E"),
        ("real_source", "source E"),
        ("real_zero", "zero E"),
    ]
    for r_idx, (session, sd, row, z, ci) in enumerate(rows_per_session):
        chain = load_chain_log(sd)
        for c_idx, (cond, display) in enumerate(cond_keys):
            ax = axes[r_idx, c_idx]
            try:
                if cond == "real_correct":
                    e = render_E_for_phase_h(session, row, "identity", chain)
                elif cond == "real_shuffled":
                    # Use a far-lag donor for visual; MSE is from the NPZ which
                    # used its own deterministic shuffled partner — these may
                    # differ visually from the score, but the score IS the score.
                    partner = (row + 1000) % max(chain.keys())
                    while partner not in chain:
                        partner = (partner + 1) % max(chain.keys())
                    e = render_E_for_phase_h(session, partner, "identity", chain)
                elif cond == "real_source":
                    src = row - 2
                    if src in chain:
                        e = render_E_for_phase_h(session, src, "identity", chain)
                    else:
                        ax.axis("off"); continue
                else:  # zero
                    import torch as _t
                    e = _t.zeros(3, 768, 1024)
                rgb = _e_tensor_to_rgb(e)
                ax.imshow(rgb)
                # Pull MSE: shape is (n_frames, n_t, K). Average over (t, K) for headline.
                mse = float(z[f"cond_{cond}"][ci].mean())
                ax.set_title(f"{display}\nMSE = {mse:.5f}", fontsize=11)
                ax.axis("off")
            except Exception as ex:
                print(f"  skip {session}/{row}/{cond}: {ex}")
                ax.axis("off")
        # Side label for the row
        axes[r_idx, 0].set_ylabel(f"{session} frame {row}", fontsize=10, rotation=90,
                                   labelpad=20)
    fig.suptitle("Phase G verifier MSE under different E conditions (real captures)",
                 fontsize=14)
    fig.tight_layout()
    out = out_dir / "phase_g_verifier_response.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}  ({out.stat().st_size // 1024} KB)")
    return out


# ---------------- Grid 5: cross-session ----------------

def grid_cross_session(out_dir: Path) -> list[Path]:
    """Two grids: D2-only model on V10 frames, V10-only on D2 frames."""
    print("[grid 5] cross-session response visualization...")
    out_paths = []
    for trained_on, eval_on in (("d2_only", "V10"), ("v10_only", "D2")):
        sd = SESSION_DIRS[eval_on]
        chain = load_chain_log(sd)
        summary_path = CROSS_ROOT / trained_on / "eval" / "summary.json"
        summary = _safe_json_load(summary_path)
        if summary is None:
            print(f"  [grid 5] SKIP {trained_on}→{eval_on}: {summary_path} missing/corrupt")
            continue
        sess_data = summary.get("sessions", {}).get(eval_on)
        if not sess_data:
            print(f"  [grid 5] SKIP {trained_on}→{eval_on}: no {eval_on} in summary")
            continue
        # Pull headline numbers
        au = sess_data.get("auroc", {}).get("correct_vs_wrong_avg", float("nan"))
        d_w = sess_data.get("deltas", {}).get("delta_wrong", {})
        delta_mean = d_w.get("mean", float("nan"))

        # Pick 6 frames evenly from eval_on's eval blocks (first block; deterministic)
        a, b = EVAL_BLOCKS[eval_on][0]
        frames = list(range(a + 30, b - 30))
        chosen = [frames[i * len(frames) // 6] for i in range(6)]

        fig, axes = plt.subplots(3, 6, figsize=(24, 12))
        for k, row in enumerate(chosen):
            try:
                # Top row: C capture (the actual scene)
                c_rgb = _load_C_resized(sd, row)
                axes[0, k].imshow(c_rgb)
                axes[0, k].set_title(f"{eval_on} frame {row}\nC (capture)", fontsize=10)
                axes[0, k].axis("off")
                # Middle row: correct E
                e_correct = render_E_for_phase_h(eval_on, row, "identity", chain)
                axes[1, k].imshow(_e_tensor_to_rgb(e_correct))
                axes[1, k].set_title("E (correct)", fontsize=10)
                axes[1, k].axis("off")
                # Bottom row: shuffled E partner
                partner = (row + 1000) % max(chain.keys())
                while partner not in chain:
                    partner = (partner + 1) % max(chain.keys())
                e_shuffled = render_E_for_phase_h(eval_on, partner, "identity", chain)
                axes[2, k].imshow(_e_tensor_to_rgb(e_shuffled))
                axes[2, k].set_title(f"E (shuffled, partner {partner})", fontsize=10)
                axes[2, k].axis("off")
            except Exception as ex:
                print(f"  skip frame {row}: {ex}")
                for r in range(3):
                    axes[r, k].axis("off")
        fig.suptitle(
            f"Cross-session generalization: {trained_on.upper()} model on {eval_on} held-out\n"
            f"AUROC = {au:.4f}  |  Δ_wrong = {delta_mean:+.6f}",
            fontsize=14)
        fig.tight_layout()
        out = out_dir / f"cross_session_{trained_on}_to_{eval_on.lower()}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out}  ({out.stat().st_size // 1024} KB)")
        out_paths.append(out)
    return out_paths


# ---------------- Grid 6: Item 1 sensitivity curves ----------------

def grid_item_1_sensitivity(out_dir: Path) -> Path | None:
    """Item 1 sensitivity: read mean MSE per condition from raw NPZ, not from
    summary.json's by_condition (which only has baseline conditions, not the
    35 extended perturbations)."""
    eval_dir = ITEM_1_ROOT / "eval"
    # Discover available sessions by which raw NPZs exist
    session_data: dict[str, dict[str, float]] = {}
    for sess_lc in ("d2", "v10"):
        npz_path = eval_dir / f"eval_{sess_lc}_raw.npz"
        z = _safe_npz_load(npz_path)
        if z is None:
            continue
        cond_means = {}
        for k in z.files:
            if k.startswith("cond_"):
                arr = z[k]
                # arr shape: (n_frames, n_timesteps, n_K_noise)
                cond_means[k[len("cond_"):]] = float(np.nanmean(arr))
        if cond_means:
            session_data[sess_lc.upper()] = cond_means
    if not session_data:
        print(f"[grid 6] SKIP: no Item 1 raw NPZs at {eval_dir}")
        return None
    sessions = list(session_data.keys())
    print(f"[grid 6] Item 1 sensitivity — sessions {sessions} "
          f"(reading from raw NPZ, {sum(len(v) for v in session_data.values())} cond/sess pairs)")

    def _get(sess: str, key: str) -> float:
        return session_data[sess].get(key, float("nan"))

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # Panel A: representative perturbation per type
    type_specs = [
        ("Type 1 (global k=64)",   "xof_t1_global_k64"),
        ("Type 2 (oct2 k=64)",     "xof_t2_oct2_k64"),
        ("Type 3 (region k=64)",   "xof_t3_region_k64"),
        ("Type 4 (oct2 swap)",     "xof_t4_swap_oct2"),
        ("Type 5 (R swap)",        "xof_t5_swap_R"),
        ("Type 6 (replace)",       "xof_t6_replace_general"),
    ]
    ax = axes[0, 0]
    type_names = [t[0] for t in type_specs]
    width = 0.35
    x = np.arange(len(type_names))
    for i, sess in enumerate(sessions):
        baseline = _get(sess, "correct")
        deltas = [_get(sess, key) - baseline for _, key in type_specs]
        ax.bar(x + i * width - width / 2, deltas, width, label=sess)
    ax.set_xticks(x)
    ax.set_xticklabels(type_names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Δ MSE vs correct E")
    ax.set_title("Panel A: representative perturbation per type (k=64 where applicable)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")

    # Panel B: Type 1 sensitivity curve (k vs Δ)
    ax = axes[0, 1]
    type1_k = (1, 4, 16, 64, 256, 1024, 4096)
    for sess in sessions:
        baseline = _get(sess, "correct")
        ys = [_get(sess, f"xof_t1_global_k{k}") - baseline for k in type1_k]
        ax.plot(type1_k, ys, marker="o", label=sess)
    ax.set_xscale("log")
    ax.set_xlabel("k (bits flipped, log scale)")
    ax.set_ylabel("Δ MSE vs correct E")
    ax.set_title("Panel B: Type 1 global bit-flip sensitivity")
    ax.legend(); ax.grid(alpha=0.3, which="both")

    # Panel C: Type 2 octave-localized at k=64, per octave
    ax = axes[1, 0]
    octaves = (0, 1, 2, 3)
    for sess in sessions:
        baseline = _get(sess, "correct")
        ys = [_get(sess, f"xof_t2_oct{o}_k64") - baseline for o in octaves]
        ax.plot(octaves, ys, marker="s", label=sess)
    ax.set_xticks(octaves)
    ax.set_xlabel("octave index (0=coarse / 3=fine)")
    ax.set_ylabel("Δ MSE vs correct E")
    ax.set_title("Panel C: Type 2 octave-localized k=64")
    ax.legend(); ax.grid(alpha=0.3)

    # Panel D: lag sweep (in lieu of region-stratified, which wasn't collected)
    ax = axes[1, 1]
    lag_keys = (-30, -15, -5, -2, -1, 0, 1, 2, 5, 15, 30)
    for sess in sessions:
        baseline = _get(sess, "correct")
        ys = [_get(sess, f"lag_{k:+d}") - baseline for k in lag_keys]
        ax.plot(lag_keys, ys, marker=".", label=sess)
    ax.axvline(0, color="black", lw=0.5, ls="--")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("lag k (frames)")
    ax.set_ylabel("Δ MSE vs correct E")
    ax.set_title("Panel D: lag sweep (in lieu of region-stratified — not collected)")
    ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle("Item 1 — XOF perturbation sensitivity curves", fontsize=14)
    fig.tight_layout()
    out = out_dir / "item_1_sensitivity_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}  ({out.stat().st_size // 1024} KB)")
    return out


# ---------------- Grid 7: Phase H baseline results ----------------

def grid_phase_h_results(out_dir: Path) -> Path | None:
    eval_root = PHASE_H_ROOT / "eval" / "final"
    summary_path = eval_root / "summary.json"
    summary = _safe_json_load(summary_path)
    if summary is None:
        print(f"[grid 7] SKIP: {summary_path} missing/corrupt — "
              f"Phase H eval not done yet")
        return None
    print("[grid 7] Phase H baseline results...")
    sessions = list(summary.get("sessions", {}).keys())

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Panel A: AUROC across conditions, in-family vs held-out
    ax = axes[0, 0]
    for sess in sessions:
        metrics = summary["sessions"][sess].get("metrics", {})
        in_family_aurocs = [(c, metrics[c]["auroc"]) for c in TRAIN_POOL_LABELS if c in metrics]
        heldout_aurocs   = [(c, metrics[c]["auroc"]) for c in HELDOUT_POOL_LABELS if c in metrics]
        in_y = [a[1] for a in in_family_aurocs]
        ho_y = [a[1] for a in heldout_aurocs]
        ax.scatter([0] * len(in_y) + [1] * len(ho_y),
                   in_y + ho_y,
                   alpha=0.6, label=sess)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["in-family (Type 1,2,4,6)", "held-out (Type 3,5)"])
    ax.set_ylabel("AUROC")
    ax.axhline(0.5, color="gray", lw=0.5, ls="--", label="chance")
    ax.set_title("Panel A: in-family vs held-out perturbation AUROCs")
    ax.legend(); ax.grid(alpha=0.3)

    # Panel B: shuffled-pair AUROC + comparison to Phase G diffusion verifier (where available)
    ax = axes[0, 1]
    labels, ph_h_auroc, ph_g_auroc = [], [], []
    for sess in sessions:
        m = summary["sessions"][sess].get("metrics", {})
        if "shuffled" not in m: continue
        labels.append(f"{sess} shuffled")
        ph_h_auroc.append(m["shuffled"]["auroc"])
        # Phase G main verifier's shuffled-pair AUROC. Bug fix 2026-05-03:
        # the original path read from "shuffled" mode (a Phase G control where
        # the model was deliberately trained on shuffled pairs and lands at
        # ~0.5 chance), which made the comparison misleading. The MAIN verifier
        # is the actual Phase G result and achieves ~1.0 on shuffled-pair
        # detection, matching Phase H baseline.
        ph_g = json.loads(
            (PHASE_G_ROOT / "main" / "eval" / "summary.json").read_text()
        ).get("sessions", {}).get(sess, {}).get("auroc", {}).get("correct_vs_wrong_avg", float("nan"))
        ph_g_auroc.append(ph_g)
    if labels:
        x = np.arange(len(labels))
        ax.bar(x - 0.2, ph_h_auroc, 0.4, label="Phase H (this work)")
        ax.bar(x + 0.2, ph_g_auroc, 0.4, label="Phase G (diffusion baseline)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.4, 1.05)
        ax.set_ylabel("AUROC")
        ax.set_title("Panel B: shuffled-pair AUROC, Phase H vs Phase G")
        ax.axhline(0.5, color="gray", lw=0.5, ls="--")
        ax.legend(); ax.grid(alpha=0.3, axis="y")

    # Panel C: training curves
    ax = axes[1, 0]
    history_path = PHASE_H_ROOT / "history.jsonl"
    if history_path.exists():
        steps, losses, accs = [], [], []
        for line in history_path.read_text().splitlines():
            line = line.strip()
            if not line: continue
            try:
                r = json.loads(line)
                steps.append(r["step"]); losses.append(r["loss"])
                accs.append(r.get("batch_acc", 0.0))
            except Exception:
                continue
        if steps:
            ax2 = ax.twinx()
            ax.plot(steps, losses, color="tab:blue", lw=0.8, label="loss")
            ax2.plot(steps, accs, color="tab:red", lw=0.8, label="batch acc")
            ax.set_xlabel("step")
            ax.set_ylabel("BCE loss", color="tab:blue")
            ax.set_yscale("log")
            ax2.set_ylabel("batch acc", color="tab:red")
            ax2.set_ylim(0, 1.05)
            ax.set_title("Panel C: training curves")
            ax.grid(alpha=0.3)

    # Panel D: per-Type held-out generalization (avg AUROC by Type)
    ax = axes[1, 1]
    for sess in sessions:
        m = summary["sessions"][sess].get("metrics", {})
        type_aurocs = {f"T{t}": [] for t in (1, 2, 3, 4, 5, 6)}
        for cond, info in m.items():
            if cond.startswith("xof_t"):
                t = int(cond[5])
                type_aurocs[f"T{t}"].append(info["auroc"])
        type_names = list(type_aurocs.keys())
        type_means = [np.mean(type_aurocs[t]) if type_aurocs[t] else np.nan
                       for t in type_names]
        x = np.arange(len(type_names))
        ax.plot(x, type_means, marker="o", label=sess)
    ax.set_xticks(np.arange(6))
    ax.set_xticklabels([f"Type {t}" for t in (1, 2, 3, 4, 5, 6)])
    # Highlight held-out types
    ax.axvspan(1.5, 2.5, alpha=0.15, color="red", label="held-out")
    ax.axvspan(3.5, 4.5, alpha=0.15, color="red")
    ax.set_ylabel("mean AUROC across conditions")
    ax.set_title("Panel D: per-Type AUROC — held-out shaded")
    ax.set_ylim(0.4, 1.05)
    ax.axhline(0.5, color="gray", lw=0.5, ls="--")
    ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle("Phase H supervised baseline performance", fontsize=14)
    fig.tight_layout()
    out = out_dir / "phase_h_baseline_results.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}  ({out.stat().st_size // 1024} KB)")
    return out


# ---------------- manifest ----------------

def write_manifest(out_dir: Path, paths: list[Path]) -> Path:
    lines = ["# Visual grid MANIFEST", "",
             f"Generated: {os.popen('date -u +%Y-%m-%dT%H:%M:%SZ').read().strip()}", ""]
    lines.append("| filename | size | hash (blake3 or blake2b-256) |")
    lines.append("|---|---|---|")
    for p in paths:
        if p is None or not p.exists():
            continue
        size_kb = p.stat().st_size // 1024
        h = _blake3_file(p)[:32]
        lines.append(f"| {p.name} | {size_kb} KB | `{h}...` |")
    out = out_dir / "MANIFEST.md"
    out.write_text("\n".join(lines))
    print(f"  → {out}")
    return out


# ---------------- main ----------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--include-fa-trajectory", action="store_true",
                    help="Generate Grid 3 (F-A trajectory) — requires C_fake re-run; "
                         "default skips per spec")
    ap.add_argument("--phase-h-only", action="store_true",
                    help="Only generate Phase H grid (Grid 7), skip 1-6.")
    ap.add_argument("--frame-id", type=int, default=100,
                    help="Reference frame for perturbation example grid (Grid 2)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    if args.phase_h_only:
        out = grid_phase_h_results(args.out)
        if out: paths.append(out)
    else:
        # Grid 1: session overviews
        for sess in ("D2", "V10"):
            p = grid_session_overview(sess, args.out)
            if p: paths.append(p)
        # Grid 2: perturbation examples
        p = grid_perturbation_examples("D2", args.frame_id, args.out)
        if p: paths.append(p)
        # Grid 4: Phase G verifier response
        p = grid_phase_g_verifier_response(args.out)
        if p: paths.append(p)
        # Grid 5: cross-session
        paths.extend(grid_cross_session(args.out))
        # Grid 6: Item 1 sensitivity
        p = grid_item_1_sensitivity(args.out)
        if p: paths.append(p)
        # Grid 3: F-A trajectory (skip unless requested)
        if args.include_fa_trajectory:
            print("[grid 3] requested but skipped — C_fake not persisted; "
                  "operator must approve --save-c-fake re-run")
        # Grid 7: Phase H (if eval landed)
        p = grid_phase_h_results(args.out)
        if p: paths.append(p)

    if paths:
        write_manifest(args.out, paths)
    print(f"\n[done] {len(paths)} grids generated under {args.out}")


if __name__ == "__main__":
    main()

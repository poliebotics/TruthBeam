"""Spatial sensitivity heatmap pilot v2 — C-only occlusion of Phase G verifier.

Per the project's expanded spec 2026-05-04. Methodology validation + preliminary
insight; not a paper-ready claim.

Score:
    score(C, E) = Phase G diffusion MSE at t=300, K=1, fixed noise seed.

Two distinct map types per (frame, source) pair:

(a) Compatibility map — single-source:
    delta_score(cell) = score(masked C, E_target) - score(unmasked C, E_target)
    green = positive = masking increases MSE = region supports target-E compatibility
    red   = negative = masking decreases MSE = region hurts compatibility / artifact

(b) Detection contribution map — paired real vs F-A fake at one ckpt:
    detection_contribution(cell) = gap_unmasked - gap_masked
    where gap = score(C_fake, E_target) - score(C_real, E_target)
    red   = positive = masking REDUCES gap = region helps defender / bad for attacker
    green = negative = masking INCREASES gap = region helps attacker

Smoke (operator-specified):
    1. 1 real frame at 16×16 mean-fill
    2. 1 matched F-A v1 step_100k at 16×16 mean-fill
    3. 1 real frame at 16×16 blur-fill (control)

Full pilot:
    5 D2 + 5 V10 target frames, selected by gap quantile (4 high / 4 median /
    2 low). Each frame × {real, F-A v1 @5k, @25k, @70k, @100k} × mean-fill.
    50 compatibility maps + 40 detection_contribution maps + comparison panel.

Halt if cuda:7 needed for fresh binders; do not delay binder training.

Usage:
    python scripts/paper_analyses/spatial_heatmap_pilot.py --mode smoke \
        --device cuda:0 --grid 16 \
        --output-dir <path> --phase-g-ckpt <path> \
        --d2-dir <path> --v10-dir <path> --experiments-root <path>

    python scripts/paper_analyses/spatial_heatmap_pilot.py --mode full \
        --device cuda:0 --grid 16 \
        --fa-v1-ckpt-dir <path> ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from phase_g.diffusion_diagnostic_model import (  # noqa: E402
    DiffusionDiagnosticUNet, build_diffusion_constants, q_sample,
)
from phase_g.diffusion_diagnostic_dataset import (  # noqa: E402
    EVAL_BLOCKS, _crop_and_resize_C, _load_packed_cfa_float01,
)
from phase_g.xof_perturb import load_chain_log, render_E_for_phase_h  # noqa: E402

T_DIFFUSION = 1000
T_PROBE = 300
NOISE_SEED = 12345
PHASE_G_INPUT_H = 768
PHASE_G_INPUT_W = 1024
FA_V1_CKPT_STEPS = (5000, 25000, 70000, 100000)


# ============================================================================
# scoring
# ============================================================================

def score_pair(model: DiffusionDiagnosticUNet,
               C: torch.Tensor, E: torch.Tensor,
               dc: dict, device: torch.device, dtype: torch.dtype) -> float:
    """Phase G MSE for one (C, E) at fixed t=T_PROBE, K=1, fixed noise seed."""
    with torch.no_grad():
        H, W = C.shape[-2:]
        torch.manual_seed(NOISE_SEED)
        noise = torch.randn(1, 4, H, W, device=device, dtype=torch.float32)
        t_tensor = torch.tensor([T_PROBE], device=device, dtype=torch.long)
        t_float = t_tensor.float()
        C_in = C.unsqueeze(0).float().to(device)
        C_t = q_sample(C_in, t_tensor, dc, noise).to(dtype)
        E_in = E.unsqueeze(0).to(device, dtype=dtype)
        with torch.amp.autocast("cuda", dtype=dtype):
            eps_pred = model(C_t, E_in, t_float, force_uncond=False)
        diff = eps_pred.float() - noise.float()
        return float(diff.pow(2).mean().item())


def grid_cells(grid: int, H: int = PHASE_G_INPUT_H,
               W: int = PHASE_G_INPUT_W) -> list[tuple[int, int, int, int]]:
    cells = []
    ys = np.linspace(0, H, grid + 1).round().astype(int)
    xs = np.linspace(0, W, grid + 1).round().astype(int)
    for gy in range(grid):
        for gx in range(grid):
            cells.append((int(ys[gy]), int(ys[gy + 1]),
                          int(xs[gx]), int(xs[gx + 1])))
    return cells


# ============================================================================
# masks
# ============================================================================

def compute_training_mean_C(d2_dir: Path, n_samples: int = 8,
                             seed: int = 7) -> torch.Tensor:
    chain = load_chain_log(d2_dir)
    eval_set: set[int] = set()
    for a, b in EVAL_BLOCKS["D2"]:
        for r in range(a, b):
            eval_set.add(r)
    train_keys = [k for k in sorted(chain.keys()) if k not in eval_set]
    rng = np.random.RandomState(seed)
    chosen = rng.choice(train_keys, size=min(n_samples, len(train_keys)),
                        replace=False).tolist()
    sums = torch.zeros(4, dtype=torch.float64)
    n_pix = 0
    for r in sorted(chosen):
        path = d2_dir / "Recordings" / f"frame_{r:06d}.raw"
        C = _crop_and_resize_C(_load_packed_cfa_float01(path))
        sums += C.double().sum(dim=(1, 2))
        n_pix += C.shape[1] * C.shape[2]
    return (sums / n_pix).float()


def make_mean_fill_C(training_means: torch.Tensor, H: int, W: int) -> torch.Tensor:
    return training_means.view(4, 1, 1).expand(4, H, W).clone()


def build_blur_full_C(C: torch.Tensor, sigma: float = 20.0) -> torch.Tensor:
    out = torch.empty_like(C)
    for ch in range(C.shape[0]):
        arr = C[ch].cpu().numpy().astype(np.float32)
        blurred = cv2.GaussianBlur(arr, ksize=(0, 0), sigmaX=sigma)
        out[ch] = torch.from_numpy(blurred)
    return out


def apply_mask(C: torch.Tensor, fill: torch.Tensor,
               y0: int, y1: int, x0: int, x1: int) -> torch.Tensor:
    out = C.clone()
    out[..., y0:y1, x0:x1] = fill[..., y0:y1, x0:x1]
    return out


def compute_compatibility_map(model, C: torch.Tensor, E: torch.Tensor,
                              fill: torch.Tensor,
                              cells: list[tuple[int, int, int, int]],
                              dc, device, dtype, baseline=None
                              ) -> tuple[float, np.ndarray]:
    if baseline is None:
        baseline = score_pair(model, C, E, dc, device, dtype)
    grid_h = int(np.sqrt(len(cells)))
    deltas = np.zeros((grid_h, grid_h), dtype=np.float64)
    for idx, (y0, y1, x0, x1) in enumerate(cells):
        masked_C = apply_mask(C, fill, y0, y1, x0, x1)
        s = score_pair(model, masked_C, E, dc, device, dtype)
        gy, gx = idx // grid_h, idx % grid_h
        deltas[gy, gx] = s - baseline
        if idx % 64 == 0:
            print(f"    cell {idx+1}/{len(cells)} delta={s-baseline:+.6f}",
                  flush=True)
    return baseline, deltas


# ============================================================================
# rendering
# ============================================================================

def _render_signed_heatmap(heatmap: np.ndarray, title: str, out_path: Path,
                            cmap: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    vmax = float(np.max(np.abs(heatmap))) if heatmap.size else 1.0
    if vmax == 0:
        vmax = 1.0
    im = ax.imshow(heatmap, cmap=cmap, vmin=-vmax, vmax=+vmax,
                   interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def render_compatibility_png(heatmap: np.ndarray, title: str, out_path: Path) -> None:
    """Compatibility map: positive=green (supports), negative=red (hurts).
    RdYlGn: low→red, high→green — direct mapping is correct."""
    _render_signed_heatmap(heatmap, title, out_path, cmap="RdYlGn")


def render_detection_contribution_png(heatmap: np.ndarray, title: str,
                                       out_path: Path) -> None:
    """Detection contribution: positive=red (helps defender), negative=green (helps attacker).
    RdYlGn_r: low→green, high→red — reversed mapping puts red at high (positive)."""
    _render_signed_heatmap(heatmap, title, out_path, cmap="RdYlGn_r")


# ============================================================================
# data loading
# ============================================================================

def load_real_C(session_dir: Path, frame_id: int) -> torch.Tensor:
    return _crop_and_resize_C(
        _load_packed_cfa_float01(
            session_dir / "Recordings" / f"frame_{frame_id:06d}.raw"
        )
    )


def load_target_E(session: str, frame_id: int,
                  chain_log: dict[int, str], device: str = "cpu") -> torch.Tensor:
    return render_E_for_phase_h(session, frame_id, "identity", chain_log,
                                 device=device)


def load_fa_v1_c_fake_step100k(session: str, frame_id: int,
                               experiments_root: Path) -> torch.Tensor | None:
    """Load saved C_fake from Stage 0 --save-c-fake at step_100000."""
    npz_path = (experiments_root / "stage_0_with_c_fake" / "eval"
                / "step_00100000" / f"stage0_{session.lower()}_raw.npz")
    if not npz_path.exists():
        return None
    z = np.load(npz_path)
    if "rows" not in z or "c_fake_tensor" not in z:
        return None
    rows = z["rows"]
    if frame_id not in rows:
        return None
    idx = int(np.where(rows == frame_id)[0][0])
    return torch.from_numpy(z["c_fake_tensor"][idx].astype(np.float32))


def render_fa_c_fake(ckpt_path: Path, session: str, target_row: int,
                     source_row: int, session_dir: Path,
                     device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Inline F-A v1 inference for a single frame."""
    from phase_g.fa_loader import load_fa_v1_checkpoint, render_C_fake
    fa = load_fa_v1_checkpoint(ckpt_path, device=device, dtype=dtype)
    return render_C_fake(fa, session_dir,
                          source_row=source_row, target_row=target_row,
                          device=device, dtype=dtype).cpu().float()


# ============================================================================
# frame selection by gap quantile
# ============================================================================

def select_frames_by_gap(d2_npz_path: Path, v10_npz_path: Path,
                          d2_fake_npz: Path, v10_fake_npz: Path,
                          n_high: int = 4, n_med: int = 4, n_low: int = 2
                          ) -> dict:
    """For each session, compute frame-level gap = mean(fake_correct - real_correct)
    using Phase G main eval (real) and Stage 0 step_100000 fake_target_correct.
    Pick (n_high high-gap, n_med median-gap, n_low low-gap) frames per session.
    """
    out: dict = {"D2": [], "V10": []}
    for sess, real_npz, fake_npz in [
        ("D2", d2_npz_path, d2_fake_npz),
        ("V10", v10_npz_path, v10_fake_npz),
    ]:
        z_real = np.load(real_npz)
        z_fake = np.load(fake_npz)
        # Phase G main eval npz: cond_correct (n_frames, n_t, K) under real;
        # rows tells us frame_id list.
        # Phase G main eval npz uses key `cond_correct`; Stage 0 saved npz uses
        # `cond_fake_correct` (fake under target E) and `cond_real_correct`.
        # Try both layouts.
        if "rows" not in z_real.files:
            print(f"[select] WARN: {real_npz} missing rows; "
                  f"available={z_real.files}; SKIPPING {sess}")
            continue
        rows_real = z_real["rows"]
        if "cond_correct" in z_real.files:
            cond_real = z_real["cond_correct"]
        elif "cond_real_correct" in z_real.files:
            cond_real = z_real["cond_real_correct"]
        else:
            print(f"[select] WARN: {real_npz} missing cond_correct/"
                  f"cond_real_correct; available={z_real.files}; SKIPPING {sess}")
            continue
        m_real = cond_real.mean(axis=(1, 2))
        # Stage 0 npz key for fake-target: prefer `cond_fake_correct` (Stage 0
        # saved npz convention), fall back to legacy alternatives.
        if "rows" not in z_fake.files:
            print(f"[select] WARN: {fake_npz} missing rows; "
                  f"available={z_fake.files}; SKIPPING {sess}")
            continue
        rows_fake = z_fake["rows"]
        if "cond_fake_correct" in z_fake.files:
            cond_fake = z_fake["cond_fake_correct"]
        elif "fake_target_correct" in z_fake.files:
            cond_fake = z_fake["fake_target_correct"]
        elif "cond_fake_target" in z_fake.files:
            cond_fake = z_fake["cond_fake_target"]
        else:
            print(f"[select] WARN: {fake_npz} missing fake-target cond key; "
                  f"available={z_fake.files}; SKIPPING {sess}")
            continue
        m_fake = cond_fake.mean(axis=(1, 2))
        # Common rows (intersection)
        common = sorted(set(rows_real.tolist()) & set(rows_fake.tolist()))
        if len(common) < (n_high + n_med + n_low):
            print(f"[select] WARN: {sess} only {len(common)} common frames")
        gaps = []
        for r in common:
            i_r = int(np.where(rows_real == r)[0][0])
            i_f = int(np.where(rows_fake == r)[0][0])
            gap = float(m_fake[i_f] - m_real[i_r])
            gaps.append((r, gap))
        gaps.sort(key=lambda x: x[1], reverse=True)  # high gap first
        n = len(gaps)
        if n == 0:
            continue
        # Dedup-aware quantile picking: if intersection is small, ensure
        # high/med/low slices don't overlap. Use a seen-set and shrink bins.
        seen: set[int] = set()
        def _take(slice_pairs, label, max_count):
            picks = []
            for r, g in slice_pairs:
                if r in seen:
                    continue
                seen.add(r)
                picks.append((r, g, label))
                if len(picks) >= max_count:
                    break
            return picks
        high = _take(gaps[:max(n_high * 3, n_high)], "high", n_high)
        med_start = max(0, (n // 2) - (n_med // 2))
        med = _take(gaps[med_start: med_start + max(n_med * 3, n_med)],
                    "med", n_med)
        low = _take(list(reversed(gaps[-max(n_low * 3, n_low):])),
                    "low", n_low)
        out[sess] = high + med + low
    return out


# ============================================================================
# comparison panel
# ============================================================================

def build_comparison_panel(per_frame_data: list[dict], out_path: Path,
                           grid: int, ckpt_steps: tuple[int, ...]) -> None:
    """Two stacked rows per target frame:
        upper: visual C | compatibility(real) | compatibility(F-A 5k) | ... | compatibility(F-A 100k)
        lower: blank   | (blank)              | detection(F-A 5k)     | ... | detection(F-A 100k)
    Cols = 1 (visual) + 1 (real) + 4 (F-A ckpts) = 6.
    Rows = 2 × n_frames.
    """
    n_frames = len(per_frame_data)
    if n_frames == 0:
        return
    n_cols = 2 + len(ckpt_steps)
    fig, axes = plt.subplots(
        2 * n_frames, n_cols,
        figsize=(2.0 * n_cols, 2.2 * 2 * n_frames),
        dpi=110, squeeze=False,
    )
    for fi, fd in enumerate(per_frame_data):
        # Row top: compatibility maps + visual C
        ax_vis = axes[2 * fi, 0]
        ax_vis.imshow(fd["visual_C"], aspect="auto")
        ax_vis.set_title(f"{fd['session']} f={fd['frame_id']}\nvisual C", fontsize=8)
        ax_vis.set_xticks([]); ax_vis.set_yticks([])
        ax_vis_b = axes[2 * fi + 1, 0]
        ax_vis_b.axis("off")

        # Real compatibility
        ax_r = axes[2 * fi, 1]
        comp_real = fd["compat_maps"]["real"]
        vmax = float(np.max(np.abs(comp_real)))
        if vmax == 0: vmax = 1.0
        ax_r.imshow(comp_real, cmap="RdYlGn", vmin=-vmax, vmax=+vmax)
        ax_r.set_title("real\ncompatibility", fontsize=8)
        ax_r.set_xticks([]); ax_r.set_yticks([])
        # No detection_contribution for real (only paired with fakes)
        axes[2 * fi + 1, 1].axis("off")

        # F-A checkpoints
        for ci, step in enumerate(ckpt_steps):
            col = 2 + ci
            comp_f = fd["compat_maps"].get(f"fa_v1_step_{step}")
            det_f = fd["detection_maps"].get(f"fa_v1_step_{step}")
            ax_top = axes[2 * fi, col]
            ax_bot = axes[2 * fi + 1, col]
            if comp_f is not None:
                v = float(np.max(np.abs(comp_f)))
                if v == 0: v = 1.0
                ax_top.imshow(comp_f, cmap="RdYlGn", vmin=-v, vmax=+v)
                ax_top.set_title(f"F-A @ {step//1000}k\ncompatibility", fontsize=8)
            ax_top.set_xticks([]); ax_top.set_yticks([])
            if det_f is not None:
                v = float(np.max(np.abs(det_f)))
                if v == 0: v = 1.0
                ax_bot.imshow(det_f, cmap="RdYlGn_r", vmin=-v, vmax=+v)
                ax_bot.set_title(f"F-A @ {step//1000}k\ndetection contrib", fontsize=8)
            ax_bot.set_xticks([]); ax_bot.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def visualize_C(C: torch.Tensor) -> np.ndarray:
    """Convert (4, H, W) packed CFA float [0,1] to (H, W, 3) RGB-ish thumbnail
    by combining R/G1/G2/B channels — for human reference only."""
    arr = C.cpu().numpy()  # (4, H, W)
    rgb = np.stack([arr[0], (arr[1] + arr[2]) / 2.0, arr[3]], axis=-1)
    rgb = np.clip(rgb, 0, 1)
    return rgb


# ============================================================================
# pilot report
# ============================================================================

def write_pilot_report(out_dir: Path, results: dict) -> None:
    md = ["# Spatial heatmap pilot — report", ""]
    md.append(f"**Mode**: `{results.get('mode')}`")
    md.append(f"**Grid**: {results.get('grid')}×{results.get('grid')}")
    md.append(f"**Phase G probe**: t={results.get('t_probe')}, K=1, fixed seed "
              f"{results.get('noise_seed')}")
    md.append(f"**Training-set mean C** (per channel, normalized [0,1]): "
              f"{results.get('training_means')}")
    md.append("")
    if results.get("frame_selection"):
        md.append("## Frame selection")
        md.append("")
        md.append("Gap = mean(fake_target_step100k MSE) - mean(real_target MSE) "
                  "averaged over (timesteps × K_noise) of Phase G main eval.")
        md.append("")
        for sess, picks in results["frame_selection"].items():
            md.append(f"### {sess}")
            md.append("")
            md.append("| frame_id | gap | quantile |")
            md.append("|---|---|---|")
            for r, g, q in picks:
                md.append(f"| {r} | {g:+.6f} | {q} |")
            md.append("")
    if results.get("smoke_results"):
        md.append("## Smoke results")
        md.append("")
        for k, v in results["smoke_results"].items():
            md.append(f"- {k}: {v}")
        md.append("")
    if results.get("halt_checks"):
        md.append("## Halt-condition checks")
        md.append("")
        for k, v in results["halt_checks"].items():
            md.append(f"- {k}: {v}")
        md.append("")
    if results.get("runtime"):
        md.append(f"**Wall time**: {results['runtime']}")
    md.append("")
    md.append("## Methodology notes")
    md.append("")
    md.append("- C-only occlusion. E_target held fixed.")
    md.append("- Compatibility map: green = positive delta = supports; "
              "red = negative = artifact.")
    md.append("- Detection contribution map: red = positive (defender helped); "
              "green = negative (attacker helped).")
    md.append("- Mean-fill in normalized C-space; blur-fill (sigma=20) only as smoke control.")
    md.append("- Donor for F-A v1 inference: source_row = chain_row[(target_idx + N//4) "
              "mod N], where N is the number of valid chain rows for the session. "
              "Deterministic; same source for all 4 F-A checkpoints.")
    md.append("- Frame-selection gap formula: per-frame `m_fake_step100k - m_real`, "
              "averaged over (timesteps × K_noise) of Phase G main eval. Selection "
              "uses Stage 0 saved fake-target scores at step_100000.")
    md.append("- Pilot, NOT a paper-ready claim.")
    (out_dir / "pilot_report.md").write_text("\n".join(md))


# ============================================================================
# smoke runner
# ============================================================================

def run_smoke(args, model, dc, training_means, cells, log) -> None:
    chain_d2 = load_chain_log(args.d2_dir)
    candidates = sorted(chain_d2.keys())
    chosen_frame = None
    for r in candidates:
        if load_fa_v1_c_fake_step100k("D2", r, args.experiments_root) is not None:
            chosen_frame = r
            break
    if chosen_frame is None:
        log("FATAL: no D2 frame has saved C_fake @ step_100000")
        sys.exit(1)
    log(f"[smoke] D2 frame={chosen_frame}")

    chain = chain_d2
    sess = "D2"
    E = load_target_E(sess, chosen_frame, chain, device="cpu")
    C_real = load_real_C(args.d2_dir, chosen_frame)
    C_fake = load_fa_v1_c_fake_step100k(sess, chosen_frame, args.experiments_root)
    if C_fake is None:
        log("FATAL: load_fa_v1_c_fake_step100k returned None unexpectedly")
        sys.exit(2)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    fill_mean = make_mean_fill_C(training_means, C_real.shape[1], C_real.shape[2])
    fill_blur_real = build_blur_full_C(C_real, sigma=20.0)

    # 3 smoke tests per operator
    runs = [
        ("real", "mean_fill", C_real, fill_mean),
        ("fa_v1_step_100000", "mean_fill", C_fake, fill_mean),
        ("real", "blur_fill", C_real, fill_blur_real),
    ]
    smoke_results: dict = {}
    heatmaps: dict = {}
    for source, fill_method, C, fill in runs:
        t0 = time.time()
        baseline, deltas = compute_compatibility_map(
            model, C, E, fill, cells, dc, device, dtype,
        )
        wall = time.time() - t0
        heatmaps[(source, fill_method)] = (baseline, deltas)
        log(f"[smoke] {source}/{fill_method}: baseline={baseline:.6f} "
            f"min={deltas.min():+.6f} max={deltas.max():+.6f} wall={wall:.1f}s")
        png = args.output_dir / (
            f"compatibility_delta_score_{sess}_f{chosen_frame:06d}_"
            f"{source}_{fill_method}.png"
        )
        render_compatibility_png(
            deltas,
            title=(f"compatibility · {source} · {sess} f={chosen_frame} · "
                   f"{fill_method} · grid={args.grid}"),
            out_path=png,
        )
        smoke_results[f"{source}/{fill_method}"] = {
            "baseline": float(baseline),
            "delta_min": float(deltas.min()),
            "delta_max": float(deltas.max()),
            "wall_s": float(wall),
        }

    # Detection contribution: paired real ↔ F-A v1 step_100000 (mean_fill)
    base_real, dr = heatmaps[("real", "mean_fill")]
    base_fake, df = heatmaps[("fa_v1_step_100000", "mean_fill")]
    gap_unmasked = base_fake - base_real
    gap_masked = (base_fake + df) - (base_real + dr)
    detection_contribution = gap_unmasked - gap_masked  # NOTE sign flip vs v1
    log(f"[smoke] detection_contribution(mean_fill, fa@100k): "
        f"gap_unmasked={gap_unmasked:+.6f} "
        f"min={detection_contribution.min():+.6f} "
        f"max={detection_contribution.max():+.6f} "
        f"mean={detection_contribution.mean():+.6f}")
    png = args.output_dir / (
        f"detection_contribution_{sess}_f{chosen_frame:06d}_"
        f"fa_v1_step_100000.png"
    )
    render_detection_contribution_png(
        detection_contribution,
        title=(f"detection_contribution · fa_v1@100k vs real · {sess} "
               f"f={chosen_frame} · mean_fill"),
        out_path=png,
    )
    smoke_results["detection_contribution_fa_v1_step_100000"] = {
        "gap_unmasked": float(gap_unmasked),
        "dc_min": float(detection_contribution.min()),
        "dc_max": float(detection_contribution.max()),
        "dc_mean": float(detection_contribution.mean()),
    }

    # Stability: mean-fill ↔ blur-fill correlation on REAL only (the project's spec)
    dm = heatmaps[("real", "mean_fill")][1]
    db = heatmaps[("real", "blur_fill")][1]
    if dm.std() > 0 and db.std() > 0:
        pearson = float(np.corrcoef(dm.flatten(), db.flatten())[0, 1])
    else:
        pearson = float("nan")
    smoke_results["stability_real"] = {
        "pearson_mean_vs_blur": pearson,
        "mean_fill_max_abs": float(np.abs(dm).max()),
        "blur_fill_max_abs": float(np.abs(db).max()),
    }
    log(f"[smoke] stability(real): pearson(mean,blur)={pearson:.4f}")

    # NPZ + report
    save_dict = {"cells": np.array(cells)}
    for (lab, fm), (b, d) in heatmaps.items():
        save_dict[f"compat__{lab}__{fm}"] = d
        save_dict[f"compat_baseline__{lab}__{fm}"] = np.float64(b)
    save_dict["detection_contribution__fa_v1_step_100000__mean_fill"] = detection_contribution
    np.savez(args.output_dir / "heatmap_data.npz", **save_dict)

    halt_checks = {
        "nan": any(np.isnan(d).any() for (_, _), (_, d) in heatmaps.items()),
        "uniform": all((d.max() - d.min()) < 1e-9 for (_, _), (_, d) in heatmaps.items()),
        "method_unstable": np.isnan(pearson) or pearson < 0.3,
    }
    log(f"[smoke] halt_checks: {halt_checks}")

    write_pilot_report(args.output_dir, {
        "mode": "smoke", "grid": args.grid, "t_probe": T_PROBE,
        "noise_seed": NOISE_SEED, "training_means": training_means.tolist(),
        "smoke_results": smoke_results, "halt_checks": halt_checks,
        "frame_selection": None,
    })
    log("[smoke] DONE.")


# ============================================================================
# full runner
# ============================================================================

def run_full(args, model, dc, training_means, cells, log) -> None:
    log(f"[full] WARNING: this is the full pilot. Wall time will be substantial. "
        f"Will halt if cuda:7 needed.")

    # Locate Phase G + Stage 0 npz files for frame selection
    pg_d2 = (args.experiments_root / "phase_g_diffusion_diagnostic" / "main"
             / "eval" / "eval_d2_raw.npz")
    pg_v10 = (args.experiments_root / "phase_g_diffusion_diagnostic" / "main"
              / "eval" / "eval_v10_raw.npz")
    s0_d2 = (args.experiments_root / "stage_0_with_c_fake" / "eval"
             / "step_00100000" / "stage0_d2_raw.npz")
    s0_v10 = (args.experiments_root / "stage_0_with_c_fake" / "eval"
              / "step_00100000" / "stage0_v10_raw.npz")
    for p in (pg_d2, pg_v10, s0_d2, s0_v10):
        if not p.exists():
            log(f"[full] FATAL: missing required npz {p}")
            sys.exit(1)

    selection = select_frames_by_gap(pg_d2, pg_v10, s0_d2, s0_v10,
                                       n_high=args.n_high, n_med=args.n_med,
                                       n_low=args.n_low)
    log(f"[full] frame selection: D2={len(selection['D2'])} V10={len(selection['V10'])}")
    for sess, picks in selection.items():
        for r, g, q in picks:
            log(f"  [{sess}] f={r} gap={g:+.6f} quantile={q}")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    chains = {"D2": load_chain_log(args.d2_dir), "V10": load_chain_log(args.v10_dir)}
    sess_dirs = {"D2": args.d2_dir, "V10": args.v10_dir}

    # F-A inference per checkpoint, per frame, source_row = target_row - 100
    # (deterministic donor; documented in pilot_report.md).
    ckpt_paths = {
        step: args.fa_v1_ckpt_dir / f"step_{step:08d}.pt"
        for step in FA_V1_CKPT_STEPS
    }
    for step, p in ckpt_paths.items():
        if not p.exists():
            log(f"[full] FATAL: missing F-A v1 ckpt {p}")
            sys.exit(2)

    # Helper to pick source_row (donor) per target — deterministic, in train pool
    def pick_source_row(sess: str, target_row: int) -> int:
        chain = chains[sess]
        keys = sorted(chain.keys())
        idx = keys.index(target_row)
        # offset of N/4 forward in chain, modulo
        return keys[(idx + len(keys) // 4) % len(keys)]

    # Pre-render C_fake for all (frame, ckpt). Load each ckpt once.
    log(f"[full] running F-A v1 inference for "
        f"{len(FA_V1_CKPT_STEPS)} ckpts × "
        f"{len(selection['D2']) + len(selection['V10'])} frames…")
    c_fakes: dict[tuple[str, int, int], torch.Tensor] = {}
    for step, ckpt in ckpt_paths.items():
        from phase_g.fa_loader import load_fa_v1_checkpoint, render_C_fake
        log(f"[full]   loading F-A ckpt step_{step:08d}…")
        fa = load_fa_v1_checkpoint(ckpt, device=device, dtype=dtype)
        for sess, picks in selection.items():
            for r, _g, _q in picks:
                src = pick_source_row(sess, r)
                with torch.no_grad():
                    Cf = render_C_fake(fa, sess_dirs[sess],
                                        source_row=src, target_row=r,
                                        device=device, dtype=dtype)
                c_fakes[(sess, r, step)] = Cf.cpu().float()
        del fa
        torch.cuda.empty_cache()
    log(f"[full] F-A inference done — {len(c_fakes)} fakes generated")

    # Compute heatmaps
    fill_mean = make_mean_fill_C(training_means, PHASE_G_INPUT_H, PHASE_G_INPUT_W)
    per_frame_data: list[dict] = []
    npz_save: dict = {"cells": np.array(cells)}

    for sess, picks in selection.items():
        chain = chains[sess]
        for r, gap, quant in picks:
            log(f"[full] === {sess} f={r} (quantile={quant}, gap={gap:+.6f}) ===")
            E = load_target_E(sess, r, chain, device="cpu")
            C_real = load_real_C(sess_dirs[sess], r)
            visual_C = visualize_C(C_real)

            compat_maps: dict[str, np.ndarray] = {}
            base_real, d_real = compute_compatibility_map(
                model, C_real, E, fill_mean, cells, dc, device, dtype,
            )
            compat_maps["real"] = d_real
            png = args.output_dir / (
                f"compatibility_delta_score_{sess}_f{r:06d}_real_mean_fill.png"
            )
            render_compatibility_png(
                d_real, f"compat · real · {sess} f={r} · mean_fill", png)
            npz_save[f"compat__{sess}__f{r}__real"] = d_real
            npz_save[f"compat_baseline__{sess}__f{r}__real"] = np.float64(base_real)

            detection_maps: dict[str, np.ndarray] = {}
            for step in FA_V1_CKPT_STEPS:
                Cf = c_fakes[(sess, r, step)]
                base_f, d_f = compute_compatibility_map(
                    model, Cf, E, fill_mean, cells, dc, device, dtype,
                )
                src_lab = f"fa_v1_step_{step}"
                compat_maps[src_lab] = d_f
                png = args.output_dir / (
                    f"compatibility_delta_score_{sess}_f{r:06d}_{src_lab}_mean_fill.png"
                )
                render_compatibility_png(
                    d_f, f"compat · F-A@{step//1000}k · {sess} f={r}", png)
                npz_save[f"compat__{sess}__f{r}__{src_lab}"] = d_f
                npz_save[f"compat_baseline__{sess}__f{r}__{src_lab}"] = np.float64(base_f)
                # detection_contribution
                gap_unmasked = base_f - base_real
                gap_masked = (base_f + d_f) - (base_real + d_real)
                dc_map = gap_unmasked - gap_masked
                detection_maps[src_lab] = dc_map
                png_dc = args.output_dir / (
                    f"detection_contribution_{sess}_f{r:06d}_{src_lab}.png"
                )
                render_detection_contribution_png(
                    dc_map, f"detection_contrib · F-A@{step//1000}k vs real · {sess} f={r}",
                    png_dc)
                npz_save[f"detection_contribution__{sess}__f{r}__{src_lab}"] = dc_map

            per_frame_data.append({
                "session": sess, "frame_id": r, "quantile": quant, "gap": gap,
                "visual_C": visual_C,
                "compat_maps": compat_maps,
                "detection_maps": detection_maps,
            })

    # comparison panel
    comp_png = args.output_dir / "comparison_panel.png"
    build_comparison_panel(per_frame_data, comp_png, args.grid, FA_V1_CKPT_STEPS)
    log(f"[full] saved comparison panel → {comp_png}")

    np.savez(args.output_dir / "heatmap_data.npz", **npz_save)
    log(f"[full] saved npz")

    write_pilot_report(args.output_dir, {
        "mode": "full", "grid": args.grid, "t_probe": T_PROBE,
        "noise_seed": NOISE_SEED, "training_means": training_means.tolist(),
        "frame_selection": selection,
        "halt_checks": {},
    })
    log("[full] DONE.")


# ============================================================================
# main
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--output-dir", type=Path,
                    default=ROOT / "experiments" / "paper_analyses"
                    / "spatial_heatmaps" / "pilot")
    ap.add_argument("--phase-g-ckpt", type=Path, required=True)
    ap.add_argument("--d2-dir", type=Path, required=True)
    ap.add_argument("--v10-dir", type=Path, required=True)
    ap.add_argument("--experiments-root", type=Path, required=True,
                    help="Root containing stage_0_with_c_fake, phase_g_*, etc.")
    ap.add_argument("--fa-v1-ckpt-dir", type=Path,
                    default=Path("/path/to/poliebotics_phase_b/experiments/"
                                  "phase_f/f_a_full_v1/checkpoints"),
                    help="F-A v1 checkpoint dir; expects step_<NNNNNNNN>.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-high", type=int, default=4,
                    help="full mode: number of high-gap frames per session")
    ap.add_argument("--n-med", type=int, default=4,
                    help="full mode: number of nearest-median-gap frames per session")
    ap.add_argument("--n-low", type=int, default=2,
                    help="full mode: number of low-gap frames per session")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log_path = args.output_dir / "pilot_log.txt"
    log_f = open(log_path, "w")
    def log(msg: str) -> None:
        print(msg, flush=True); log_f.write(msg + "\n"); log_f.flush()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    log(f"[pilot] mode={args.mode} grid={args.grid} device={device} dtype={dtype}")
    log(f"[pilot] T={T_DIFFUSION} t_probe={T_PROBE} noise_seed={NOISE_SEED}")

    log(f"[pilot] loading Phase G ckpt {args.phase_g_ckpt}")
    ck = torch.load(args.phase_g_ckpt, map_location="cpu", weights_only=False)
    state = ck["model"] if "model" in ck else ck
    saved_args = ck.get("args", {}) if isinstance(ck, dict) else {}
    base_ch = saved_args.get("base_ch", 96)
    mults = tuple(saved_args.get("mults", [1, 2, 4, 4]))
    attn_at = saved_args.get("attn_at")
    if attn_at is None:
        attn_at = tuple(i == len(mults) - 1 for i in range(len(mults)))
    else:
        attn_at = tuple(bool(x) for x in attn_at)
    cond_drop_prob = saved_args.get("cond_drop_prob", 0.2)
    log(f"[pilot] model cfg: base_ch={base_ch} mults={mults} attn_at={attn_at}")
    model = DiffusionDiagnosticUNet(
        base_ch=base_ch, channel_mults=mults, attn_at=attn_at,
        cond_drop_prob=cond_drop_prob,
    ).to(device, dtype=dtype)
    model.load_state_dict(state, strict=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[pilot] model loaded — {n_params/1e6:.2f}M params")

    dc = build_diffusion_constants(T_DIFFUSION, device, torch.float32)
    log("[pilot] computing training-set per-channel mean (8 D2 train frames)…")
    training_means = compute_training_mean_C(args.d2_dir, n_samples=8)
    log(f"[pilot] training_means = {training_means.tolist()}")
    cells = grid_cells(args.grid)

    if args.mode == "smoke":
        run_smoke(args, model, dc, training_means, cells, log)
    else:
        run_full(args, model, dc, training_means, cells, log)
    log_f.close()


if __name__ == "__main__":
    main()

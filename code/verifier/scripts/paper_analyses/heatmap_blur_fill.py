"""Track 2B — blur-fill control for the spatial heatmap pilot.

Diagnostic: tests whether saliency reflects E-coupling or generic
    C-content. Pilot occludes C with fixed fill (mean). Blur-fill replaces
    occluded patch with a Gaussian-blurred version of itself.

This is an in-methodology extension of `spatial_heatmap_pilot.py`. We
deliberately DO NOT modify the original pilot script; the original
`run_full` (mean-fill on a 10+10 frame set at grid=16) remains the
canonical pilot. This driver imports the pilot's primitives and runs the
same compatibility + detection-contribution pipeline, but with the fill
tensor constructed as a Gaussian blur of the per-source input C rather
than the global per-channel training mean.

Per-source blur semantics (matches the existing smoke-mode control's
implementation): for each (frame, source) pair, build
`blur = cv2.GaussianBlur(C, sigma)` over the entire input C, then use
`blur[..., y0:y1, x0:x1]` as the fill at masked region. This means the
masked region gets the local low-frequency content of the input — same
geometry, different fill semantics from mean-fill.

Scope (per operator):
    - Full pilot frame set: 10 D2 + 10 V10 (selected by gap quantile,
      default n_high=4 / n_med=4 / n_low=2 — same as the mean-fill pilot).
    - 5 sources per frame: real + F-A v1 @ 5k/25k/70k/100k.
    - blur sigma a CLI parameter; intended values per operator are
      σ ∈ {2, 4, 8} pixels.
    - Same probe-t (300), same noise seed (12345), same Phase G ckpt.
    - One run per σ. Output dir is per-σ to keep them separate.

GPU plan from operator:
    GPU 6: σ=2 then σ=4 SEQUENTIALLY in one screen.
    GPU 7: σ=8.

Usage:

    python scripts/paper_analyses/heatmap_blur_fill.py \
        --device cuda:0 --grid 16 --blur-sigma 2 \
        --output-dir <out>/sigma_2 \
        --phase-g-ckpt <path> --d2-dir <path> --v10-dir <path> \
        --experiments-root <path> \
        [--fa-v1-ckpt-dir <path>] [--n-high 4 --n-med 4 --n-low 2]

NOT a paper-ready claim — methodology validation / preliminary insight
only. Phase G is consumed via forward passes only (existing pilot
methodology); no Phase G code modifications, no retraining, no
checkpoint changes, no held-out binder use.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "paper_analyses"))
sys.path.insert(0, str(ROOT / "src"))

# Primitives are imported from the canonical pilot — no copy/paste.
from spatial_heatmap_pilot import (  # noqa: E402
    FA_V1_CKPT_STEPS, NOISE_SEED,
    T_DIFFUSION, T_PROBE,
    build_blur_full_C,
    build_comparison_panel,
    compute_compatibility_map,
    compute_training_mean_C,
    grid_cells,
    load_real_C,
    load_target_E,
    render_compatibility_png,
    render_detection_contribution_png,
    select_frames_by_gap,
    visualize_C,
    write_pilot_report,
)
from phase_g.diffusion_diagnostic_model import (  # noqa: E402
    DiffusionDiagnosticUNet, build_diffusion_constants,
)
from phase_g.xof_perturb import load_chain_log  # noqa: E402


# ---------------------------------------------------------------------
# fill construction (the only material difference from mean-fill pilot)
# ---------------------------------------------------------------------

def build_blur_fill_for_C(C: torch.Tensor, sigma: float) -> torch.Tensor:
    """Per-source fill: Gaussian-blur the entire input C with the given
    sigma. The masked region in `apply_mask(C, fill, y0, y1, x0, x1)`
    receives `fill[..., y0:y1, x0:x1]`, i.e. the local low-frequency
    content of the input at that region.

    Wraps `spatial_heatmap_pilot.build_blur_full_C` to make the
    semantics explicit and to keep the call site short.
    """
    return build_blur_full_C(C, sigma=float(sigma))


# ---------------------------------------------------------------------
# main runner
# ---------------------------------------------------------------------

def validate_and_select_frames(args, log) -> dict:
    """Pre-GPU step: confirm the required npz files exist, run
    `select_frames_by_gap`, and assert the per-session frame count
    matches `n_high + n_med + n_low`. Exits before any GPU work if a
    session is truncated, so a short selection cannot waste model
    cold-start allocation."""
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
            log(f"[blur_fill] FATAL (pre-GPU): missing required npz {p}")
            sys.exit(1)

    selection = select_frames_by_gap(pg_d2, pg_v10, s0_d2, s0_v10,
                                       n_high=args.n_high, n_med=args.n_med,
                                       n_low=args.n_low)
    expected = args.n_high + args.n_med + args.n_low
    for sess in ("D2", "V10"):
        n_got = len(selection.get(sess, []))
        if n_got != expected:
            log(f"[blur_fill] FATAL (pre-GPU): session {sess} returned "
                f"{n_got} frames, expected {expected} (n_high={args.n_high} "
                f"+ n_med={args.n_med} + n_low={args.n_low}). Operator spec "
                f"requires the full pilot frame set; refusing to proceed "
                f"with a truncated selection. Fix the upstream npz "
                f"(stage_0_with_c_fake / phase_g_diffusion_diagnostic) "
                f"or relax --n-high/--n-med/--n-low explicitly.")
            sys.exit(3)
    log(f"[blur_fill] frame selection: D2={len(selection['D2'])} "
        f"V10={len(selection['V10'])} (expected {expected} per session)")
    for sess, picks in selection.items():
        for r, g, q in picks:
            log(f"  [{sess}] f={r} gap={g:+.6f} quantile={q}")
    return selection


def run_blur_fill(args, model, dc, training_means, cells, log,
                  selection: dict, device: torch.device,
                  dtype: torch.dtype) -> None:
    """GPU phase: requires `selection` produced by
    `validate_and_select_frames`, which has already enforced the
    pre-GPU frame-count guard. `device`/`dtype` are passed in (single
    canonical source) so they cannot drift from the values used to
    place the model.

    The session header (sigma, grid, frame-count budget) was already
    logged by main() before validation; we don't repeat it here.
    """
    chains = {"D2": load_chain_log(args.d2_dir),
              "V10": load_chain_log(args.v10_dir)}
    sess_dirs = {"D2": args.d2_dir, "V10": args.v10_dir}

    # F-A v1 inference per checkpoint, per frame. Mirrors pilot run_full.
    ckpt_paths = {
        step: args.fa_v1_ckpt_dir / f"step_{step:08d}.pt"
        for step in FA_V1_CKPT_STEPS
    }
    for step, p in ckpt_paths.items():
        if not p.exists():
            log(f"[blur_fill] FATAL: missing F-A v1 ckpt {p}")
            sys.exit(2)

    def pick_source_row(sess: str, target_row: int) -> int:
        """Same donor convention as the pilot: deterministic, in-train pool."""
        chain = chains[sess]
        keys = sorted(chain.keys())
        idx = keys.index(target_row)
        return keys[(idx + len(keys) // 4) % len(keys)]

    log(f"[blur_fill] running F-A v1 inference for "
        f"{len(FA_V1_CKPT_STEPS)} ckpts × "
        f"{len(selection['D2']) + len(selection['V10'])} frames…")
    c_fakes: dict[tuple[str, int, int], torch.Tensor] = {}
    for step, ckpt in ckpt_paths.items():
        from phase_g.fa_loader import load_fa_v1_checkpoint, render_C_fake
        log(f"[blur_fill]   loading F-A ckpt step_{step:08d}…")
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
    log(f"[blur_fill] F-A inference done — {len(c_fakes)} fakes generated")

    # The per-source-fill suffix used in filenames + npz keys. Operator's
    # comparison panel expects a clear sigma label.
    fill_label = f"blur_s{int(args.blur_sigma)}_fill"

    per_frame_data: list[dict] = []
    npz_save: dict = {"cells": np.array(cells)}

    for sess, picks in selection.items():
        for r, gap, quant in picks:
            log(f"[blur_fill] === {sess} f={r} (quantile={quant}, "
                f"gap={gap:+.6f}) ===")
            E = load_target_E(sess, r, chains[sess], device="cpu")
            C_real = load_real_C(sess_dirs[sess], r)
            visual_C = visualize_C(C_real)

            # Real: build per-frame blur fill from C_real
            t0 = time.time()
            fill_real = build_blur_fill_for_C(C_real, args.blur_sigma)
            base_real, d_real = compute_compatibility_map(
                model, C_real, E, fill_real, cells, dc, device, dtype,
            )
            log(f"  real fill_built+compat in {time.time() - t0:.1f}s "
                f"baseline={base_real:.6f} "
                f"delta_min={d_real.min():+.6f} delta_max={d_real.max():+.6f}")

            compat_maps: dict[str, np.ndarray] = {"real": d_real}
            png = args.output_dir / (
                f"compatibility_delta_score_{sess}_f{r:06d}_real_"
                f"{fill_label}.png"
            )
            render_compatibility_png(
                d_real,
                f"compat · real · {sess} f={r} · {fill_label}",
                png,
            )
            npz_save[f"compat__{sess}__f{r}__real"] = d_real
            npz_save[f"compat_baseline__{sess}__f{r}__real"] = np.float64(base_real)

            # Each F-A v1 ckpt: build per-source blur fill from C_fake
            detection_maps: dict[str, np.ndarray] = {}
            for step in FA_V1_CKPT_STEPS:
                Cf = c_fakes[(sess, r, step)]
                t1 = time.time()
                fill_f = build_blur_fill_for_C(Cf, args.blur_sigma)
                base_f, d_f = compute_compatibility_map(
                    model, Cf, E, fill_f, cells, dc, device, dtype,
                )
                src_lab = f"fa_v1_step_{step}"
                log(f"  {src_lab} fill_built+compat in {time.time() - t1:.1f}s "
                    f"baseline={base_f:.6f}")
                compat_maps[src_lab] = d_f
                png = args.output_dir / (
                    f"compatibility_delta_score_{sess}_f{r:06d}_{src_lab}_"
                    f"{fill_label}.png"
                )
                render_compatibility_png(
                    d_f,
                    f"compat · F-A@{step//1000}k · {sess} f={r} · {fill_label}",
                    png,
                )
                npz_save[f"compat__{sess}__f{r}__{src_lab}"] = d_f
                npz_save[f"compat_baseline__{sess}__f{r}__{src_lab}"] = (
                    np.float64(base_f)
                )
                # detection_contribution (same convention as pilot run_full)
                gap_unmasked = base_f - base_real
                gap_masked = (base_f + d_f) - (base_real + d_real)
                dc_map = gap_unmasked - gap_masked
                detection_maps[src_lab] = dc_map
                png_dc = args.output_dir / (
                    f"detection_contribution_{sess}_f{r:06d}_{src_lab}_"
                    f"{fill_label}.png"
                )
                render_detection_contribution_png(
                    dc_map,
                    f"detection_contrib · F-A@{step//1000}k vs real · "
                    f"{sess} f={r} · {fill_label}",
                    png_dc,
                )
                npz_save[
                    f"detection_contribution__{sess}__f{r}__{src_lab}"
                ] = dc_map

            per_frame_data.append({
                "session": sess, "frame_id": r, "quantile": quant, "gap": gap,
                "visual_C": visual_C,
                "compat_maps": compat_maps,
                "detection_maps": detection_maps,
            })

    # Reuse pilot's comparison panel — it's purely a layout function.
    comp_png = args.output_dir / "comparison_panel.png"
    build_comparison_panel(per_frame_data, comp_png, args.grid, FA_V1_CKPT_STEPS)
    log(f"[blur_fill] saved comparison panel → {comp_png}")

    np.savez(args.output_dir / "heatmap_data.npz", **npz_save)
    log("[blur_fill] saved npz")

    write_pilot_report(args.output_dir, {
        "mode": f"full_blur_fill_sigma_{args.blur_sigma}",
        "grid": args.grid, "t_probe": T_PROBE, "noise_seed": NOISE_SEED,
        "training_means": training_means.tolist(),
        "frame_selection": selection,
        "halt_checks": {},
    })
    # Operator-facing config dump for traceability.
    (args.output_dir / "blur_fill_config.json").write_text(json.dumps({
        "blur_sigma": args.blur_sigma,
        "grid": args.grid,
        "t_probe": T_PROBE,
        "noise_seed": NOISE_SEED,
        "n_high": args.n_high, "n_med": args.n_med, "n_low": args.n_low,
        "fa_v1_ckpt_steps": list(FA_V1_CKPT_STEPS),
        "fill_label": fill_label,
        "phase_g_ckpt": str(args.phase_g_ckpt),
    }, indent=2))
    log("[blur_fill] DONE.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--grid", type=int, default=16,
                    help="cells per side (default 16; matches pilot)")
    ap.add_argument("--blur-sigma", type=float, required=True,
                    help="Gaussian sigma in pixels for the fill blur")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--phase-g-ckpt", type=Path, required=True)
    ap.add_argument("--d2-dir", type=Path, required=True)
    ap.add_argument("--v10-dir", type=Path, required=True)
    ap.add_argument("--experiments-root", type=Path, required=True)
    ap.add_argument("--fa-v1-ckpt-dir", type=Path,
                    default=Path("/path/to/poliebotics_phase_b/experiments/"
                                  "phase_f/f_a_full_v1/checkpoints"))
    ap.add_argument("--n-high", type=int, default=4)
    ap.add_argument("--n-med", type=int, default=4)
    ap.add_argument("--n-low", type=int, default=2)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log_path = args.output_dir / "blur_fill_log.txt"
    log_f = open(log_path, "w")
    def log(msg: str) -> None:
        print(msg, flush=True); log_f.write(msg + "\n"); log_f.flush()

    log(f"[blur_fill] grid={args.grid}  sigma={args.blur_sigma}  "
        f"device={args.device}  T={T_DIFFUSION}  t_probe={T_PROBE}  "
        f"noise_seed={NOISE_SEED}  "
        f"frames per session: high={args.n_high} med={args.n_med} "
        f"low={args.n_low}")

    # Pre-GPU validation: npz presence + frame count. This MUST run
    # before any model/GPU allocation so a truncated selection cannot
    # waste a cold-start.
    selection = validate_and_select_frames(args, log)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    log(f"[blur_fill] resolved device={device}  dtype={dtype}")

    log(f"[blur_fill] loading Phase G ckpt {args.phase_g_ckpt}")
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
    log(f"[blur_fill] model cfg: base_ch={base_ch}  mults={mults}  "
        f"attn_at={attn_at}")
    model = DiffusionDiagnosticUNet(
        base_ch=base_ch, channel_mults=mults, attn_at=attn_at,
        cond_drop_prob=cond_drop_prob,
    ).to(device, dtype=dtype)
    model.load_state_dict(state, strict=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[blur_fill] model loaded — {n_params/1e6:.2f}M params")

    dc = build_diffusion_constants(T_DIFFUSION, device, torch.float32)
    log("[blur_fill] computing training-set per-channel mean (8 D2 train "
        "frames)…")
    training_means = compute_training_mean_C(args.d2_dir, n_samples=8)
    log(f"[blur_fill] training_means = {training_means.tolist()}")
    cells = grid_cells(args.grid)

    run_blur_fill(args, model, dc, training_means, cells, log, selection,
                  device, dtype)
    log_f.close()


if __name__ == "__main__":
    main()

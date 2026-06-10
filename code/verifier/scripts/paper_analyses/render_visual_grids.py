"""High-resolution visual grids for Phase G discrimination inspection.

Per operator's directive 2026-05-05:

  6 representative frames (3 D2 + 3 V10), one 5×3 grid per frame:

  Row 1 — Inputs (what Phase G operates on):
    Col 1: debayered C at native 768×1024, RGB
    Col 2: rendered E at native 768×1024, viridis
    Col 3: C with E-footprint contour overlay (p50/p75/p90)

  Row 2 — ε-MSE score fields (real_correct vs fake_100k):
    Col 1: ε-MSE on (C, E_real)
    Col 2: ε-MSE on (C_fake_100k, E_target)
    Col 3: signed contrast — ε-MSE(fake_100k) − ε-MSE(real)

  Row 3 — ε-MSE for E-perturbation conditions:
    Col 1: ε-MSE on (C, E_shuffled)
    Col 2: ε-MSE on (C, E_cross_session)
    Col 3: signed contrast — ε-MSE(shuffled) − ε-MSE(real)

  Row 4 — VGG-distance fields:
    Col 1: VGG-distance on real_correct (= 0 baseline)
    Col 2: VGG-distance on fake_100k
    Col 3: signed contrast — VGG(fake_100k) − VGG(real)

  Row 5 — B1-encoder distance fields:
    Col 1: B1-distance on real_correct (= 0 baseline)
    Col 2: B1-distance on fake_100k
    Col 3: signed contrast — B1(fake_100k) − B1(real)

  Resolutions:
    - inputs (Row 1): native 768×1024
    - score fields (Rows 2–5): headline 384×512 (2× downsample),
      supplementary native 768×1024 + smoothed 192×256 (4× downsample)

  Per-row vmax = 95th percentile across magnitude columns 1–2;
  signed-contrast vmax = 95th of |contrast| (symmetric around 0).
  Each panel annotated with vmin/vmax.

Reads:
  - per-frame npy outputs from scoring_function_comparison/<sess>_f<row>/
  - includes 4-channel residual_4ch_mean for each condition
  - footprint mask is computed inline from rendered E (matches prior pilot
    convention; not loaded from disk)

Standing rules: Phase G inference-only. No held-out asset use beyond F-A v1.
No F-A v2 trainer touch. No information feedback to Phase G design.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


SCORING_NATIVE = (PHASE_G_INPUT_H, PHASE_G_INPUT_W) = (768, 1024)
HEADLINE_H, HEADLINE_W = 384, 512   # 2× downsample
SMOOTH_H, SMOOTH_W = 192, 256       # 4× downsample
INPUT_H, INPUT_W = 768, 1024        # inputs always native
FOOTPRINT_PERCENTILES = (50, 75, 90)
FOOTPRINT_LINESTYLES = {50: "dotted", 75: "dashed", 90: "solid"}
FA_V1_CKPT_STEPS = (5000, 25000, 70000, 100000)
PERTURBED_CONDS = (
    [f"fake_{s//1000}k" for s in FA_V1_CKPT_STEPS]
    + ["shuffled_E", "cross_session_E"]
)
ALL_CONDS = ["real_correct"] + PERTURBED_CONDS


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------

def load_npy(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.load(path, allow_pickle=False)


def block_average_2d(arr: np.ndarray, factor: int) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D, got {arr.shape}")
    if factor == 1:
        # Strict native pass-through: no reshape/mean churn.
        return arr
    h, w = arr.shape
    if h % factor != 0 or w % factor != 0:
        raise ValueError(f"shape {arr.shape} not divisible by {factor}")
    h2, w2 = h // factor, w // factor
    return arr.reshape(h2, factor, w2, factor).mean(axis=(1, 3))


def cfa_packed_to_rgb(C_packed: np.ndarray) -> np.ndarray:
    """(4, H, W) packed CFA float [0, 1] → (H, W, 3) uint8 RGB via
    (R, mean(G1, G2), B). Per operator default."""
    if C_packed.shape[0] != 4:
        raise ValueError(f"expected 4 CFA channels, got {C_packed.shape}")
    rgb = np.stack([
        C_packed[0],
        0.5 * (C_packed[1] + C_packed[2]),
        C_packed[3],
    ], axis=-1)  # (H, W, 3)
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return rgb


def E_to_magnitude(E_3ch: np.ndarray) -> np.ndarray:
    """(3, H, W) float [0, 1] → (H, W) magnitude via channel mean."""
    return E_3ch.mean(axis=0)


def load_phase_g_C(session_dir: Path, row: int) -> np.ndarray:
    """Load the Phase G C input at native (4, 768, 1024). Re-uses Phase G's
    canonical crop+resize to ensure pixel alignment with score fields."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from phase_g.diffusion_diagnostic_dataset import (
        _crop_and_resize_C, _load_packed_cfa_float01,
    )
    return _crop_and_resize_C(_load_packed_cfa_float01(
        session_dir / "Recordings" / f"frame_{row:06d}.raw"
    )).numpy().astype(np.float32)


def load_phase_g_E(session_dir: Path, row: int) -> np.ndarray:
    """Load Phase G E input at native (3, 768, 1024)."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from phase_g.diffusion_diagnostic_dataset import (
        _resize_E_to_target, EMISSION_NATIVE_H, EMISSION_NATIVE_W,
    )
    from data.emission_dataset import load_emission_at
    return _resize_E_to_target(load_emission_at(
        session_dir / "derived" / "Emissions" / f"tile_{row:06d}.png",
        EMISSION_NATIVE_H, EMISSION_NATIVE_W,
    )).numpy().astype(np.float32)


# ---------------------------------------------------------------------
# field assembly per frame
# ---------------------------------------------------------------------

def assemble_eps_mse_fields(frame_dir: Path) -> dict[str, np.ndarray]:
    """Read per-condition ε-MSE score fields at native (768, 1024)."""
    out: dict[str, np.ndarray] = {}
    for cond in ALL_CONDS:
        f = load_npy(frame_dir / f"score_eps_mse_{cond}.npy")
        if f is None:
            raise FileNotFoundError(
                f"missing eps_mse for {cond} at {frame_dir}")
        out[cond] = f.astype(np.float32)
    return out


def assemble_vgg_distance_fields(frame_dir: Path) -> dict[str, np.ndarray]:
    """VGG distance per perturbed condition. Native = saved
    `vgg_distance_native_{cond}.npy` at conv4_2's natural resolution
    (~96×128 for 768×1024 input). Real_correct = zeros baseline."""
    out: dict[str, np.ndarray] = {}
    out["real_correct"] = np.zeros((96, 128), dtype=np.float32)
    for cond in PERTURBED_CONDS:
        f = load_npy(frame_dir / f"vgg_distance_native_{cond}.npy")
        if f is None:
            raise FileNotFoundError(
                f"missing vgg_distance for {cond} at {frame_dir}")
        out[cond] = f.astype(np.float32)
    return out


def assemble_b1_distance_fields(frame_dir: Path) -> dict[str, np.ndarray]:
    """B1 encoder distance per perturbed condition. Native ≈ (36, 42)."""
    out: dict[str, np.ndarray] = {}
    # We need to know B1's feature shape to build the zero baseline. Read
    # one perturbed condition first to learn the shape.
    sample = None
    for cond in PERTURBED_CONDS:
        f = load_npy(frame_dir / f"b1_distance_native_{cond}.npy")
        if f is not None:
            sample = f
            break
    if sample is None:
        raise FileNotFoundError(f"no b1_distance_native_*.npy in {frame_dir}")
    out["real_correct"] = np.zeros(sample.shape, dtype=np.float32)
    for cond in PERTURBED_CONDS:
        f = load_npy(frame_dir / f"b1_distance_native_{cond}.npy")
        if f is None:
            raise FileNotFoundError(
                f"missing b1_distance for {cond} at {frame_dir}")
        out[cond] = f.astype(np.float32)
    return out


# ---------------------------------------------------------------------
# downsample for display
# ---------------------------------------------------------------------

def to_display_res(field: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resample a 2-D field to target (H, W). Uses block-mean if the
    target is an integer divisor of the input shape on both axes
    (preserves average); otherwise cv2 resize. Direction-aware
    interpolation: INTER_AREA for downsampling, INTER_LINEAR for
    upsampling — INTER_AREA is poorly defined for upsampling per cv2
    docs."""
    th, tw = target_hw
    h, w = field.shape
    if h % th == 0 and w % tw == 0 and h // th == w // tw:
        return block_average_2d(field, h // th)
    interp = cv2.INTER_AREA if (th < h and tw < w) else cv2.INTER_LINEAR
    return cv2.resize(field.astype(np.float32), (tw, th),
                       interpolation=interp)


# ---------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------

def percentile_vmax(*arrs: np.ndarray, q: float = 95.0) -> float:
    """vmax = q-percentile across all values in the given arrays (positive).
    For magnitude images."""
    if not arrs:
        return 1.0
    cat = np.concatenate([a.flatten() for a in arrs if a is not None])
    cat = cat[np.isfinite(cat)]
    if cat.size == 0:
        return 1.0
    return float(np.percentile(cat, q))


def percentile_abs_vmax(*arrs: np.ndarray, q: float = 95.0) -> float:
    """vmax for symmetric diverging colormap = q-percentile of |values|."""
    if not arrs:
        return 1.0
    cat = np.concatenate([np.abs(a).flatten() for a in arrs if a is not None])
    cat = cat[np.isfinite(cat)]
    if cat.size == 0:
        return 1.0
    return float(np.percentile(cat, q))


def add_panel_label(ax, vmin: float, vmax: float, title: str,
                     col_subtitle: str | None = None) -> None:
    """Annotate a panel with title + actual vmin/vmax. col_subtitle is
    a small label above the title (e.g. column header on row 0)."""
    ax.set_xticks([]); ax.set_yticks([])
    if col_subtitle:
        ax.set_title(col_subtitle, fontsize=9)
    if vmin is not None and vmax is not None:
        ax.text(0.02, -0.04,
                 f"[{vmin:+.4g}, {vmax:+.4g}]",
                 transform=ax.transAxes, fontsize=7,
                 verticalalignment="top",
                 family="monospace")


def add_footprint_contours(ax, E_mag: np.ndarray) -> dict:
    """Draw p50/p75/p90 contours on the current axes. E_mag must already
    be sized to match what `ax` shows (axes use imshow's pixel grid). The
    canonical convention: p50 = dotted, p75 = dashed, p90 = solid.
    Returns {p: threshold_value} for the manifest."""
    H, W = E_mag.shape
    thresholds: dict[int, float] = {}
    for p in FOOTPRINT_PERCENTILES:
        thr = float(np.percentile(E_mag, p))
        thresholds[p] = thr
        ax.contour(E_mag, levels=[thr], colors="white",
                    linestyles=FOOTPRINT_LINESTYLES[p],
                    linewidths=1.0)
    return thresholds


def render_frame_grid(frame_dir: Path, sess: str, row: int,
                       sess_dirs: dict[str, Path],
                       output_dir: Path,
                       res_label: str,
                       score_field_target: tuple[int, int],
                       log) -> dict:
    """Render the 5×3 grid for one frame at the given score-field
    display resolution. Returns the manifest dict for this frame at this
    resolution."""
    import matplotlib.pyplot as plt

    # ----- inputs -----
    C_packed = load_phase_g_C(sess_dirs[sess], row)        # (4, 768, 1024)
    E_3ch = load_phase_g_E(sess_dirs[sess], row)            # (3, 768, 1024)
    C_rgb = cfa_packed_to_rgb(C_packed)                     # (768, 1024, 3) uint8
    E_mag_native = E_to_magnitude(E_3ch)                    # (768, 1024)

    # ----- score fields at native + display res -----
    eps_mse_native = assemble_eps_mse_fields(frame_dir)
    vgg_native = assemble_vgg_distance_fields(frame_dir)
    b1_native = assemble_b1_distance_fields(frame_dir)

    eps_mse_disp = {c: to_display_res(eps_mse_native[c], score_field_target)
                     for c in eps_mse_native}
    vgg_disp = {c: to_display_res(vgg_native[c], score_field_target)
                 for c in vgg_native}
    b1_disp = {c: to_display_res(b1_native[c], score_field_target)
                for c in b1_native}

    # E magnitude resampled for footprint overlay on score-field rows (we
    # use INPUT resolution for Row 1 panels since the inputs there are
    # native; for score-field rows the contour can be omitted to keep the
    # row focused on the score field — operator's spec says contours are
    # on Row 1 Col 3, with optional overlay on score panels. We add them
    # as an OPTIONAL second supplementary variant later if needed.)
    E_mag_input = E_mag_native  # for Row 1 inputs (native res)

    # ----- per-row vmax (95th percentile) -----
    # Row 2 (eps_mse real vs fake_100k)
    r2_mag_vmax = percentile_vmax(
        eps_mse_disp["real_correct"], eps_mse_disp["fake_100k"], q=95.0)
    r2_signed = (eps_mse_disp["fake_100k"] - eps_mse_disp["real_correct"])
    r2_signed_vmax = percentile_abs_vmax(r2_signed, q=95.0)

    # Row 3 (eps_mse shuffled vs cross-session, signed = shuffled - real)
    r3_mag_vmax = percentile_vmax(
        eps_mse_disp["shuffled_E"], eps_mse_disp["cross_session_E"], q=95.0)
    r3_signed = (eps_mse_disp["shuffled_E"] - eps_mse_disp["real_correct"])
    r3_signed_vmax = percentile_abs_vmax(r3_signed, q=95.0)

    # Row 4 (VGG distance real_correct=0 vs fake_100k). Per spec, vmax is
    # 95th percentile across BOTH magnitude columns; including the
    # zero-baseline real field is correct convention even if it cannot
    # raise the vmax.
    r4_mag_vmax = percentile_vmax(
        vgg_disp["real_correct"], vgg_disp["fake_100k"], q=95.0)
    r4_signed = vgg_disp["fake_100k"] - vgg_disp["real_correct"]
    r4_signed_vmax = percentile_abs_vmax(r4_signed, q=95.0)

    # Row 5 (B1 distance) — same convention.
    r5_mag_vmax = percentile_vmax(
        b1_disp["real_correct"], b1_disp["fake_100k"], q=95.0)
    r5_signed = b1_disp["fake_100k"] - b1_disp["real_correct"]
    r5_signed_vmax = percentile_abs_vmax(r5_signed, q=95.0)

    # ----- figure layout -----
    n_rows, n_cols = 5, 3
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.0 * n_cols, 3.0 * n_rows),
        dpi=110, squeeze=False,
    )

    # Row 1, Col 1: debayered C
    ax = axes[0, 0]
    ax.imshow(C_rgb, interpolation="nearest")
    ax.set_ylabel("Row 1 — Inputs (native 768×1024)", fontsize=9)
    add_panel_label(ax, 0, 255, "C debayered (R, mean(G1,G2), B)",
                     col_subtitle="C debayered RGB")

    # Row 1, Col 2: rendered E magnitude
    ax = axes[0, 1]
    ax.imshow(E_mag_input, cmap="viridis", vmin=0.0, vmax=1.0,
               interpolation="nearest")
    add_panel_label(ax, 0.0, 1.0, "rendered E magnitude (chan-mean)",
                     col_subtitle="rendered E magnitude")

    # Row 1, Col 3: C with footprint contours
    ax = axes[0, 2]
    ax.imshow(C_rgb, interpolation="nearest")
    contour_thresholds = add_footprint_contours(ax, E_mag_input)
    add_panel_label(ax, None, None,
                     "p50 dotted / p75 dashed / p90 solid",
                     col_subtitle="C + E footprint contours")

    # Helper to draw a magnitude row (eps_mse / vgg / b1). The
    # signed-contrast label is passed in explicitly because Row 3's
    # contrast is `shuffled_E − real_correct`, NOT `cross_session_E
    # − shuffled_E`. The Col 1/Col 2 labels still come from cond_a /
    # cond_b.
    def draw_mag_row(row_i: int, fields: dict[str, np.ndarray],
                      cond_a: str, cond_b: str, mag_vmax: float,
                      signed: np.ndarray, signed_vmax: float,
                      signed_label: str,
                      ylabel: str,
                      mag_cmap: str = "viridis",
                      mag_vmin: float = 0.0):
        ax_a = axes[row_i, 0]
        ax_a.imshow(fields[cond_a], cmap=mag_cmap, vmin=mag_vmin,
                     vmax=mag_vmax, interpolation="nearest")
        ax_a.set_ylabel(ylabel, fontsize=9)
        add_panel_label(ax_a, mag_vmin, mag_vmax, f"{cond_a}",
                         col_subtitle=f"{cond_a}")
        ax_b = axes[row_i, 1]
        ax_b.imshow(fields[cond_b], cmap=mag_cmap, vmin=mag_vmin,
                     vmax=mag_vmax, interpolation="nearest")
        add_panel_label(ax_b, mag_vmin, mag_vmax, f"{cond_b}",
                         col_subtitle=f"{cond_b}")
        ax_c = axes[row_i, 2]
        ax_c.imshow(signed, cmap="RdBu", vmin=-signed_vmax,
                     vmax=+signed_vmax, interpolation="nearest")
        add_panel_label(ax_c, -signed_vmax, +signed_vmax,
                         signed_label,
                         col_subtitle=f"signed contrast")

    # Row 2: eps_mse real_correct vs fake_100k
    draw_mag_row(1, eps_mse_disp, "real_correct", "fake_100k",
                  r2_mag_vmax, r2_signed, r2_signed_vmax,
                  signed_label="fake_100k − real_correct (signed)",
                  ylabel="Row 2 — ε-MSE: real vs fake_100k")
    # Row 3: eps_mse shuffled_E vs cross_session_E (signed = shuffled − real)
    draw_mag_row(2, eps_mse_disp, "shuffled_E", "cross_session_E",
                  r3_mag_vmax, r3_signed, r3_signed_vmax,
                  signed_label="shuffled_E − real_correct (signed)",
                  ylabel="Row 3 — ε-MSE: shuffled vs cross-session")
    # Row 4: VGG real_correct (zero) vs fake_100k
    draw_mag_row(3, vgg_disp, "real_correct", "fake_100k",
                  r4_mag_vmax, r4_signed, r4_signed_vmax,
                  signed_label="fake_100k − real_correct (signed)",
                  ylabel="Row 4 — VGG distance: real vs fake_100k")
    # Row 5: B1 real_correct (zero) vs fake_100k
    draw_mag_row(4, b1_disp, "real_correct", "fake_100k",
                  r5_mag_vmax, r5_signed, r5_signed_vmax,
                  signed_label="fake_100k − real_correct (signed)",
                  ylabel="Row 5 — B1-encoder distance: real vs fake_100k")

    fig.suptitle(
        f"Phase G discrimination inspection — {sess} f={row} ({res_label})",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out_path = output_dir / f"{sess}_f{row:06d}_grid_{res_label}.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {out_path}")

    return {
        "session": sess,
        "row": row,
        "resolution_label": res_label,
        "score_field_target_hw": list(score_field_target),
        "input_resolution_hw": [INPUT_H, INPUT_W],
        "footprint_thresholds": contour_thresholds,
        "row2_eps_mse_real_vs_fake_100k": {
            "mag_vmax": r2_mag_vmax,
            "signed_abs_vmax": r2_signed_vmax,
        },
        "row3_eps_mse_shuffled_vs_cross_session": {
            "mag_vmax": r3_mag_vmax,
            "signed_abs_vmax_shuffled_minus_real": r3_signed_vmax,
        },
        "row4_vgg_distance": {
            "mag_vmax_fake_100k": r4_mag_vmax,
            "signed_abs_vmax": r4_signed_vmax,
        },
        "row5_b1_distance": {
            "mag_vmax_fake_100k": r5_mag_vmax,
            "signed_abs_vmax": r5_signed_vmax,
        },
        "output_path": str(out_path),
    }


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scoring-dir", type=Path, required=True,
                    help="dir containing per-frame extraction outputs from "
                         "scoring_function_comparison")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--d2-dir", type=Path, required=True)
    ap.add_argument("--v10-dir", type=Path, required=True)
    ap.add_argument("--frames", type=str, required=True,
                    help="comma-separated list of session:row, e.g. "
                         "'D2:1500,D2:3000,V10:1140'")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log_path = args.output_dir / "render_log.txt"
    log_f = open(log_path, "a")
    def log(msg: str) -> None:
        print(msg, flush=True)
        log_f.write(msg + "\n"); log_f.flush()

    sess_dirs = {"D2": args.d2_dir, "V10": args.v10_dir}
    frame_specs: list[tuple[str, int]] = []
    for tok in args.frames.split(","):
        parts = tok.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"frame token {tok!r} must be 'session:row'")
        sess = parts[0].strip()
        row_s = parts[1].strip()
        frame_specs.append((sess, int(row_s)))

    log(f"[render] {len(frame_specs)} frames; output_dir={args.output_dir}")
    manifests: list[dict] = []
    headline_dir = args.output_dir
    supp_dir = args.output_dir / "supplementary"
    supp_dir.mkdir(parents=True, exist_ok=True)

    for sess, row in frame_specs:
        frame_dir = args.scoring_dir / f"{sess}_f{row:06d}"
        if not frame_dir.exists():
            log(f"[render] FATAL: missing {frame_dir}")
            return 2
        log(f"[render] === {sess} f={row} ===")
        # Headline at 384×512
        m_headline = render_frame_grid(
            frame_dir, sess, row, sess_dirs, headline_dir,
            res_label="headline_384x512",
            score_field_target=(HEADLINE_H, HEADLINE_W),
            log=log,
        )
        # Supplementary native 768×1024
        m_native = render_frame_grid(
            frame_dir, sess, row, sess_dirs, supp_dir,
            res_label="native_768x1024",
            score_field_target=(SCORING_NATIVE[0], SCORING_NATIVE[1]),
            log=log,
        )
        # Supplementary smoothed 192×256
        m_smooth = render_frame_grid(
            frame_dir, sess, row, sess_dirs, supp_dir,
            res_label="smoothed_192x256",
            score_field_target=(SMOOTH_H, SMOOTH_W),
            log=log,
        )
        manifests.append({
            "session": sess, "row": row,
            "headline": m_headline,
            "supplementary_native": m_native,
            "supplementary_smoothed": m_smooth,
        })

    # Combined manifest
    out_manifest = {
        "generated_utc": __import__("time").strftime(
            "%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
        "scoring_function_comparison_dir": str(args.scoring_dir),
        "n_frames": len(frame_specs),
        "frames": [{"session": s, "row": r} for s, r in frame_specs],
        "resolution_choices": {
            "inputs": [INPUT_H, INPUT_W],
            "score_fields_native": list(SCORING_NATIVE),
            "score_fields_headline": [HEADLINE_H, HEADLINE_W],
            "score_fields_smoothed": [SMOOTH_H, SMOOTH_W],
        },
        "footprint_percentiles": list(FOOTPRINT_PERCENTILES),
        "footprint_linestyles": FOOTPRINT_LINESTYLES,
        "per_frame": manifests,
        "scoring_functions_in_grid": [
            "eps_mse (Row 2 + Row 3)",
            "vgg_distance (Row 4)",
            "b1_distance (Row 5)",
        ],
        "conditions_in_grid": {
            "row2": ["real_correct", "fake_100k"],
            "row3": ["shuffled_E", "cross_session_E", "(signed: shuffled − real)"],
            "row4": ["real_correct (zero)", "fake_100k"],
            "row5": ["real_correct (zero)", "fake_100k"],
        },
        "normalization": {
            "per_row_magnitude_vmax": "95th percentile across magnitude columns",
            "signed_contrast_vmax": "95th percentile of |signed contrast|, "
                                     "symmetric vmin = -vmax",
            "across_rows": "NOT shared (different scoring-function magnitudes)",
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(out_manifest, indent=2))
    log(f"[render] wrote {args.output_dir / 'manifest.json'}")

    # README
    readme = [
        "# Visual grids — Phase G discrimination inspection",
        "",
        f"Generated: {out_manifest['generated_utc']}",
        f"Frames: {[f'{s}_f{r:06d}' for s, r in frame_specs]}",
        "",
        "## What each grid shows (5×3 layout per frame)",
        "",
        "**Row 1 — Inputs (native 768×1024)**:",
        "- Col 1: debayered C (R, mean(G1,G2), B → uint8 RGB)",
        "- Col 2: rendered E magnitude (channel mean, viridis 0–1)",
        "- Col 3: C with E-footprint contours overlaid",
        "  - p50 dotted / p75 dashed / p90 solid (per-frame thresholds)",
        "",
        "**Row 2 — ε-MSE: real_correct vs fake_100k**:",
        "- Col 1: ε-MSE on (C_real, E_real) — Phase G's score on the real frame",
        "- Col 2: ε-MSE on (C_fake_100k, E_target) — Phase G's score on F-A v1's "
        "100k-step output",
        "- Col 3: signed contrast — fake − real (RdBu, symmetric)",
        "",
        "**Row 3 — ε-MSE: shuffled_E vs cross_session_E**:",
        "- Col 1: ε-MSE on (C_real, E_shuffled) — same C, far-row E from same session",
        "- Col 2: ε-MSE on (C_real, E_cross_session) — same C, same-percentile row "
        "from other session",
        "- Col 3: signed contrast — shuffled − real (RdBu, symmetric)",
        "",
        "**Row 4 — VGG-16 conv4_2 distance: real_correct (zero) vs fake_100k**:",
        "- Col 1: zero baseline (VGG-distance vs itself = 0; rendered as flat panel)",
        "- Col 2: VGG-distance(residual_fake_100k, residual_real)",
        "- Col 3: signed contrast (RdBu) — same as Col 2 since baseline is 0",
        "",
        "**Row 5 — B1-encoder bottleneck distance: real_correct (zero) vs "
        "fake_100k**:",
        "- Col 1: zero baseline",
        "- Col 2: B1-encoder L2 distance (residual_fake_100k vs residual_real "
        "in B1 encoder feature space)",
        "- Col 3: signed contrast (RdBu) — same as Col 2 since baseline is 0",
        "",
        "## Resolution variants",
        "",
        "Per frame, three PNGs are written:",
        "- `<sess>_f<row>_grid_headline_384x512.png` — score fields displayed "
        "at 384×512 (2× downsample of native via 2×2 block-mean).",
        "- `supplementary/<sess>_f<row>_grid_native_768x1024.png` — score "
        "fields at full native resolution.",
        "- `supplementary/<sess>_f<row>_grid_smoothed_192x256.png` — score "
        "fields at 192×256 (4× downsample, 4×4 block-mean) for noise-suppressed view.",
        "",
        "Inputs (Row 1) are always at native 768×1024 regardless of resolution variant.",
        "",
        "## Normalization",
        "",
        "- Per-row vmax for magnitude columns 1–2: 95th percentile across both "
        "fields (vmin = 0).",
        "- Signed-contrast vmax (Col 3): 95th percentile of |signed contrast|, "
        "symmetric (vmin = -vmax).",
        "- Across rows: NOT shared (different scoring functions have different "
        "natural magnitudes — cross-row comparison is qualitative).",
        "- Each panel annotated with its actual [vmin, vmax] underneath.",
        "",
        "## What the grid answers (operator's pre-registered framing)",
        "",
        "1. Where does projection actually land? — Row 1 Col 3 footprint contours.",
        "2. Where is Phase G's prediction error highest under each condition? "
        "— Rows 2–5, Cols 1 and 2.",
        "3. Where does discrimination signal concentrate? — Rows 2–5, Col 3 "
        "(signed contrast).",
        "4. Does discrimination concentrate in, near, or away from illuminated "
        "regions? — visual comparison of Row 1 Col 3 footprint contour with the "
        "Col 3 signed-contrast fields in Rows 2–5.",
        "5. Do scoring functions agree spatially? — compare Rows 2, 4, 5 within a "
        "frame.",
        "",
        "## What the grid does NOT answer",
        "",
        "- Whether Phase G NEEDS projection regions to discriminate (causal ablation, "
        "separate experiment).",
        "- Whether the spatial patterns generalize beyond these 6 frames (population "
        "stats; partially addressed by aggregate scoring-function comparison).",
        "- Whether harder attackers would produce different spatial signatures "
        "(F-A v1 may be too easy a target).",
        "",
        "## Standing-rule note",
        "",
        "This is visualization. No new statistical claims; the manuscript-cited "
        "aggregate numbers come from prior experiments (verifier_heatmaps, "
        "scoring_function_comparison). Phase G inference-only. F-A v1 outputs as-is. "
        "No held-out asset use beyond F-A v1. No information from this experiment "
        "feeds back into Phase G design.",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme))
    log(f"[render] wrote {args.output_dir / 'README.md'}")
    log_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

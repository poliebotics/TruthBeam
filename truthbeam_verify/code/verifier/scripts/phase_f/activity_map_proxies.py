"""Phase F #4 — activity-map proxies.

For each session, compute three activity heatmaps over a sample of frame pairs:

  1. Temporal-activity map: mean |C[t] - C[t-1]| across N pairs (G channel).
     → where pixels actually move/flicker between frames
  2. Emission-activity map: mean |E[t] - E[t-1]| (greyscale).
     → where the projector emits time-varying light
  3. Gradient-activity map: mean Sobel magnitude on capture (G channel).
     → where structural edges live (body, fabric, projector boundary)

These tell us which regions the editor needs to actively model versus which
regions it can copy through unchanged. They also localize the projection
footprint without any prior knowledge of where the projector is aimed.

Output:
  experiments/phase_f_prep/activity_maps/
    temporal_activity_{session}.png
    emission_activity_{session}.png
    gradient_activity_{session}.png
    activity_summary.{json,md}

CPU-only. ~5-10 min per session at N_PAIRS=64.

Run:
  python scripts/phase_f/activity_map_proxies.py \
    --session d2 --n-pairs 64 \
    --out experiments/phase_f_prep/activity_maps
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from phase_f.cfa_roundtrip import bayer_rg8_to_packed_cfa  # noqa: E402


def load_packed_cfa_g(raw_path: Path) -> np.ndarray:
    """Load BayerRG8 raw, return G channel (avg of G1, G2) at half-res."""
    cfa = bayer_rg8_to_packed_cfa(raw_path.read_bytes())  # (4, 2300, 2660) uint8
    g = ((cfa[1].astype(np.float32) + cfa[2].astype(np.float32)) * 0.5)
    return g  # (2300, 2660) float32 in [0, 255]


def load_emission_grey(em_path: Path) -> np.ndarray:
    img = cv2.imread(str(em_path), cv2.IMREAD_COLOR)  # BGR
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grey.astype(np.float32)  # (1080, 1920) float32


def sobel_mag(img: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def normalize_for_save(arr: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    """Scale to [0, 255] uint8, clipping the bright tail at `percentile`."""
    hi = float(np.percentile(arr, percentile))
    lo = float(arr.min())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def colorize_jet(grey_u8: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(grey_u8, cv2.COLORMAP_JET)


def process_session(name: str, recordings: Path, emissions: Path,
                    n_pairs: int, k: int, seed: int,
                    out_dir: Path) -> dict:
    rng = np.random.RandomState(seed)
    raw_files = sorted(recordings.glob("frame_*.raw"))
    em_files = sorted(emissions.glob("tile_*.png"))
    if len(raw_files) < k + 2 or len(em_files) < k + 2:
        return {"error": f"too few frames: raw={len(raw_files)} em={len(em_files)}"}

    n_total = min(len(raw_files), len(em_files))
    valid_t = np.arange(k, n_total)
    if n_pairs > len(valid_t):
        n_pairs = len(valid_t)
    sampled = sorted(rng.choice(valid_t, size=n_pairs, replace=False).tolist())

    # Accumulators (will be float64 to avoid overflow on 64+ frames).
    cap_h, cap_w = 2300, 2660
    em_h, em_w = 1080, 1920

    temporal = np.zeros((cap_h, cap_w), dtype=np.float64)
    emission = np.zeros((em_h, em_w), dtype=np.float64)
    gradient = np.zeros((cap_h, cap_w), dtype=np.float64)
    n_used = 0
    t0 = time.time()
    print(f"[{name}] processing {n_pairs} pairs at k={k}...", flush=True)

    for i, t in enumerate(sampled):
        try:
            cap_t = load_packed_cfa_g(raw_files[t])
            cap_prev = load_packed_cfa_g(raw_files[t - k])
            em_t = load_emission_grey(em_files[t])
            em_prev = load_emission_grey(em_files[t - k])
        except Exception as e:
            print(f"  skip t={t}: {e}", flush=True)
            continue
        temporal += np.abs(cap_t - cap_prev)
        emission += np.abs(em_t - em_prev)
        gradient += sobel_mag(cap_t)
        n_used += 1
        if (i + 1) % 16 == 0:
            print(f"  {i+1}/{n_pairs} elapsed={time.time()-t0:.0f}s", flush=True)

    if n_used == 0:
        return {"error": "no pairs processed"}

    temporal /= n_used
    emission /= n_used
    gradient /= n_used

    out_dir.mkdir(parents=True, exist_ok=True)

    # Save heatmaps (jet-colored)
    cv2.imwrite(str(out_dir / f"temporal_activity_{name}.png"),
                colorize_jet(normalize_for_save(temporal)))
    cv2.imwrite(str(out_dir / f"emission_activity_{name}.png"),
                colorize_jet(normalize_for_save(emission)))
    cv2.imwrite(str(out_dir / f"gradient_activity_{name}.png"),
                colorize_jet(normalize_for_save(gradient)))

    # Per-region statistics: split frame into 4×4 grid of capture-resolution
    # tiles, compute activity per tile so we can tell where energy concentrates.
    def grid_stats(arr: np.ndarray, gh: int = 4, gw: int = 4) -> list[float]:
        h, w = arr.shape
        th, tw = h // gh, w // gw
        out = []
        for r in range(gh):
            for c in range(gw):
                tile = arr[r*th:(r+1)*th, c*tw:(c+1)*tw]
                out.append(float(tile.mean()))
        return out

    summary = {
        "n_pairs": n_used, "k": k,
        "temporal_mean": float(temporal.mean()),
        "temporal_std": float(temporal.std()),
        "temporal_p95": float(np.percentile(temporal, 95)),
        "temporal_grid_4x4": grid_stats(temporal),
        "emission_mean": float(emission.mean()),
        "emission_std": float(emission.std()),
        "emission_p95": float(np.percentile(emission, 95)),
        "emission_grid_4x4": grid_stats(emission),
        "gradient_mean": float(gradient.mean()),
        "gradient_std": float(gradient.std()),
        "gradient_p95": float(np.percentile(gradient, 95)),
        "gradient_grid_4x4": grid_stats(gradient),
        "rows_sampled": sampled,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    print(f"[{name}] done: temporal_mean={summary['temporal_mean']:.2f} "
          f"emission_mean={summary['emission_mean']:.2f} "
          f"gradient_mean={summary['gradient_mean']:.2f}", flush=True)
    return summary


def write_markdown(summary: dict, out_dir: Path) -> None:
    lines = [
        "# Phase F #4 — activity-map proxies",
        "",
        ("Activity heatmaps over sampled temporal pairs. Each heatmap is the "
         "per-pixel mean of |x[t] - x[t-k]| (or Sobel magnitude for the "
         "gradient map), averaged across N_PAIRS sampled rows. Saved as "
         "jet-colormapped PNGs."),
        "",
    ]
    for sess, s in summary["per_session"].items():
        if "error" in s:
            lines += [f"## {sess}", f"\nERROR: {s['error']}\n"]
            continue
        lines += [
            f"## {sess.upper()}",
            "",
            f"- `n_pairs={s['n_pairs']}`, `k={s['k']}`",
            f"- temporal-activity: mean={s['temporal_mean']:.2f}, "
            f"p95={s['temporal_p95']:.2f}, std={s['temporal_std']:.2f}",
            f"- emission-activity: mean={s['emission_mean']:.2f}, "
            f"p95={s['emission_p95']:.2f}, std={s['emission_std']:.2f}",
            f"- gradient-activity: mean={s['gradient_mean']:.2f}, "
            f"p95={s['gradient_p95']:.2f}, std={s['gradient_std']:.2f}",
            "",
            f"### 4×4 grid (row-major, capture-frame coords)",
            "",
            "Temporal:",
            "```",
            *[
                "  " + " ".join(f"{v:6.2f}" for v in s["temporal_grid_4x4"][r*4:(r+1)*4])
                for r in range(4)
            ],
            "```",
            "Emission:",
            "```",
            *[
                "  " + " ".join(f"{v:6.2f}" for v in s["emission_grid_4x4"][r*4:(r+1)*4])
                for r in range(4)
            ],
            "```",
            "Gradient:",
            "```",
            *[
                "  " + " ".join(f"{v:6.2f}" for v in s["gradient_grid_4x4"][r*4:(r+1)*4])
                for r in range(4)
            ],
            "```",
            "",
            f"Heatmap files:",
            f"- `temporal_activity_{sess}.png`",
            f"- `emission_activity_{sess}.png`",
            f"- `gradient_activity_{sess}.png`",
            "",
        ]
    lines += [
        "## Use in F-A design",
        "",
        ("These maps localize where the editor must spend capacity. Regions "
         "with high temporal AND emission activity are the projection "
         "footprint — the editor must change pixels there. Regions with high "
         "gradient but low temporal activity are static structure (body "
         "outline, fabric) — copy-through is safe. Regions with low values "
         "everywhere are background — degenerate, can be zero-padded if a "
         "later F-B/C variant wants to crop."),
        "",
    ]
    (out_dir / "activity_summary.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", default=["d2", "v10"])
    ap.add_argument("--data-root", type=Path, default=Path("/path/to/poliebotics_phase_b/data"))
    ap.add_argument("--n-pairs", type=int, default=64)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"per_session": {}, "n_pairs": args.n_pairs, "k": args.k}
    for sess in args.sessions:
        recordings = args.data_root / sess / "Recordings"
        emissions = args.data_root / sess / "derived" / "Emissions"
        if not recordings.exists():
            summary["per_session"][sess] = {"error": f"missing {recordings}"}
            continue
        if not emissions.exists():
            summary["per_session"][sess] = {"error": f"missing {emissions}"}
            continue
        summary["per_session"][sess] = process_session(
            sess, recordings, emissions, args.n_pairs, args.k, args.seed,
            args.out,
        )

    (args.out / "activity_summary.json").write_text(json.dumps(summary, indent=2))
    write_markdown(summary, args.out)
    print(f"\n[done] wrote {args.out}/activity_summary.{{json,md}}", flush=True)


if __name__ == "__main__":
    main()

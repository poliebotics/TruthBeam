"""Phase F prep — pose-similarity histogram analysis.

For each session × k in K_VALUES, sample N_PAIRS random valid pairs and
compute per pair:
  - pose_similarity: pixel-correlation on the G channel masked by the
                     projection-region mask (the body). Higher = more similar.
  - emission_diff: mean abs diff between rendered emission tiles
                   (chain-derived, expected high entropy).
  - capture_diff: mean abs diff between packed CFA captures.

Output:
  experiments/phase_f_prep/pose_similarity_hist/
    pose_hist_{session}_{k}.png        — pose-similarity histogram per k
    pose_emission_scatter_{session}.png — joint plot per session
    summary.{json,md}

Use:
  - Identifies the low-pose-similarity tail (candidate curriculum filter
    pairs where pose drift is large enough that the editor can't just
    pose-warp).
  - Confirms the emission-diff distribution is k-invariant (consistent
    with chain-derived randomness).

CPU-only. Should complete in ~5-10 min per session.

Run:
  python scripts/phase_f/pose_similarity_histogram.py \
    --sessions d2 v10 --n-pairs 200 \
    --out experiments/phase_f_prep/pose_similarity_hist
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

K_VALUES = (1, 2, 3, 5, 10, 20, 50, 100)


def load_g_channel(raw_path: Path) -> np.ndarray:
    """Returns G channel (avg of G1, G2) at half-res from BayerRG8 raw."""
    cfa = bayer_rg8_to_packed_cfa(raw_path.read_bytes())  # (4, 2300, 2660) uint8
    g = ((cfa[1].astype(np.float32) + cfa[2].astype(np.float32)) * 0.5)
    return g


def load_emission_grey(em_path: Path) -> np.ndarray:
    img = cv2.imread(str(em_path), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def compute_projection_mask(recordings: Path, n_sample: int = 30, seed: int = 0,
                            threshold_pct: float = 75.0) -> np.ndarray:
    """Build a high-variance mask = the body / projection region."""
    rng = np.random.RandomState(seed)
    raw_files = sorted(recordings.glob("frame_*.raw"))
    if len(raw_files) == 0:
        raise FileNotFoundError(f"no raw files in {recordings}")
    sampled = sorted(rng.choice(len(raw_files), size=min(n_sample, len(raw_files)),
                                replace=False).tolist())
    stack = np.stack([load_g_channel(raw_files[i]) for i in sampled], axis=0)
    var = stack.var(axis=0)
    threshold = np.percentile(var, threshold_pct)
    mask = (var >= threshold).astype(np.float32)
    return mask


def masked_correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Pearson correlation over the masked region only."""
    sel = mask > 0
    if sel.sum() == 0:
        return float("nan")
    av = a[sel]
    bv = b[sel]
    av_c = av - av.mean()
    bv_c = bv - bv.mean()
    denom = float(np.sqrt((av_c ** 2).sum() * (bv_c ** 2).sum()) + 1e-12)
    if denom < 1e-12:
        return float("nan")
    return float((av_c * bv_c).sum() / denom)


def process_session(name: str, recordings: Path, emissions: Path,
                    n_pairs: int, k_values: tuple[int, ...], seed: int,
                    out_dir: Path) -> dict:
    raw_files = sorted(recordings.glob("frame_*.raw"))
    em_files = sorted(emissions.glob("tile_*.png"))
    n_total = min(len(raw_files), len(em_files))
    if n_total < max(k_values) + 2:
        return {"error": f"too few frames: {n_total}"}
    print(f"[{name}] computing projection-region mask...", flush=True)
    mask = compute_projection_mask(recordings, n_sample=30, seed=seed)
    print(f"[{name}] mask area = {float(mask.mean()):.3f} of frame", flush=True)

    rng = np.random.RandomState(seed + 1)
    summary: dict = {"name": name, "n_total": n_total, "mask_area": float(mask.mean()),
                     "k_values": list(k_values), "per_k": {}}
    out_dir.mkdir(parents=True, exist_ok=True)

    # Joint scatter accumulator
    all_pose: list[float] = []
    all_emi: list[float] = []
    all_cap: list[float] = []
    all_k: list[int] = []
    t0 = time.time()
    for k in k_values:
        valid = np.arange(k, n_total)
        if n_pairs > len(valid):
            n_pairs_eff = len(valid)
        else:
            n_pairs_eff = n_pairs
        sampled = sorted(rng.choice(valid, size=n_pairs_eff, replace=False).tolist())
        pose_sims: list[float] = []
        emi_diffs: list[float] = []
        cap_diffs: list[float] = []
        for i, t in enumerate(sampled):
            try:
                cap_t = load_g_channel(raw_files[t])
                cap_prev = load_g_channel(raw_files[t - k])
                em_t = load_emission_grey(em_files[t])
                em_prev = load_emission_grey(em_files[t - k])
            except Exception as e:
                print(f"  skip t={t}: {e}", flush=True)
                continue
            pose_sims.append(masked_correlation(cap_t, cap_prev, mask))
            emi_diffs.append(float(np.abs(em_t - em_prev).mean()))
            cap_diffs.append(float(np.abs(cap_t - cap_prev).mean()))

        pose_arr = np.array(pose_sims, dtype=np.float32)
        all_pose.extend(pose_sims)
        all_emi.extend(emi_diffs)
        all_cap.extend(cap_diffs)
        all_k.extend([k] * len(pose_sims))

        # Histogram per k
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(pose_arr[np.isfinite(pose_arr)], bins=40, range=(0.0, 1.0),
                color="C0", edgecolor="white")
        ax.set_xlabel("pose similarity (Pearson G-channel masked)")
        ax.set_ylabel("count")
        ax.set_title(f"{name.upper()} pose-similarity hist  (k={k}, n={len(pose_sims)})")
        ax.axvline(np.median(pose_arr), color="C3", linestyle="--",
                   label=f"median={np.median(pose_arr):.3f}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"pose_hist_{name}_k{k:03d}.png", dpi=110)
        plt.close(fig)

        summary["per_k"][k] = {
            "n_pairs": len(pose_sims),
            "pose_mean": float(np.nanmean(pose_arr)),
            "pose_median": float(np.nanmedian(pose_arr)),
            "pose_p5": float(np.nanpercentile(pose_arr, 5)),
            "pose_p25": float(np.nanpercentile(pose_arr, 25)),
            "pose_p75": float(np.nanpercentile(pose_arr, 75)),
            "pose_p95": float(np.nanpercentile(pose_arr, 95)),
            "emission_diff_mean": float(np.mean(emi_diffs)),
            "emission_diff_std": float(np.std(emi_diffs)),
            "capture_diff_mean": float(np.mean(cap_diffs)),
            "low_pose_count_at_lt0p5": int((pose_arr < 0.5).sum()),
            "low_pose_count_at_lt0p3": int((pose_arr < 0.3).sum()),
        }
        print(f"  k={k:3d}  n={len(pose_sims):3d}  pose_median={summary['per_k'][k]['pose_median']:.3f}  emi={summary['per_k'][k]['emission_diff_mean']:.2f}  cap={summary['per_k'][k]['capture_diff_mean']:.2f}", flush=True)

    # Joint scatter
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    sc0 = ax[0].scatter(all_pose, all_cap, c=all_k, cmap="viridis", s=10, alpha=0.6)
    ax[0].set_xlabel("pose similarity")
    ax[0].set_ylabel("capture diff (mean abs)")
    ax[0].set_title(f"{name.upper()} pose vs capture-diff (color=k)")
    plt.colorbar(sc0, ax=ax[0], label="k")
    sc1 = ax[1].scatter(all_pose, all_emi, c=all_k, cmap="viridis", s=10, alpha=0.6)
    ax[1].set_xlabel("pose similarity")
    ax[1].set_ylabel("emission diff (mean abs)")
    ax[1].set_title(f"{name.upper()} pose vs emission-diff (color=k)")
    plt.colorbar(sc1, ax=ax[1], label="k")
    fig.tight_layout()
    fig.savefig(out_dir / f"pose_emission_scatter_{name}.png", dpi=110)
    plt.close(fig)

    summary["elapsed_sec"] = round(time.time() - t0, 1)
    return summary


def write_markdown(summary: dict, out_dir: Path) -> None:
    lines = [
        "# Phase F prep — pose-similarity histogram analysis",
        "",
        ("For each (t, t-k) pair we compute pose similarity as the Pearson "
         "correlation of the G channel inside the projection-region mask. "
         "This deepens the temporal_pair_stats analysis by surfacing the "
         "**distribution** of pose drift (rather than just the mean), and "
         "identifies the low-pose-similarity tail that could anchor a "
         "curriculum filter for future F-A variants."),
        "",
    ]
    for sess, s in summary["per_session"].items():
        if "error" in s:
            lines += [f"## {sess.upper()}", f"\nERROR: {s['error']}\n"]
            continue
        lines += [
            f"## {sess.upper()}",
            "",
            f"- mask area = {s['mask_area']:.3f} of frame",
            f"- frames available = {s['n_total']}",
            "",
            "| k | n | pose median | pose p5 | pose p25 | pose p95 | emi diff | cap diff | n@<0.5 | n@<0.3 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for k, r in s["per_k"].items():
            lines.append(
                f"| {k} | {r['n_pairs']} | {r['pose_median']:.3f} | "
                f"{r['pose_p5']:.3f} | {r['pose_p25']:.3f} | "
                f"{r['pose_p95']:.3f} | {r['emission_diff_mean']:.2f} | "
                f"{r['capture_diff_mean']:.2f} | "
                f"{r['low_pose_count_at_lt0p5']} | "
                f"{r['low_pose_count_at_lt0p3']} |"
            )
        lines += ["", f"### Files",
                  *[f"- `pose_hist_{sess}_k{k:03d}.png`" for k in s["per_k"].keys()],
                  f"- `pose_emission_scatter_{sess}.png`",
                  ""]
    lines += [
        "## Curriculum-filter implications",
        "",
        ("The fraction of pairs with pose-similarity < 0.5 (or < 0.3) at each "
         "k tells us how much training data survives a curriculum filter that "
         "demands meaningful pose drift. If this fraction is < 5% at small k, "
         "filtering would shrink the curriculum too much; if it's > 20%, "
         "filtering is viable. The scatter plot shows whether emission-diff "
         "is independent of pose (it should be, since emission is "
         "chain-derived random) — if it isn't, pose and emission are "
         "confounded and the editor has a shortcut."),
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", default=["d2", "v10"])
    ap.add_argument("--data-root", type=Path,
                    default=Path("/path/to/poliebotics_phase_b/data"))
    ap.add_argument("--n-pairs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"per_session": {}, "n_pairs": args.n_pairs, "k_values": list(K_VALUES)}
    for sess in args.sessions:
        recordings = args.data_root / sess / "Recordings"
        emissions = args.data_root / sess / "derived" / "Emissions"
        if not recordings.exists() or not emissions.exists():
            summary["per_session"][sess] = {"error": f"missing dirs in {args.data_root}/{sess}"}
            continue
        summary["per_session"][sess] = process_session(
            sess, recordings, emissions, args.n_pairs, K_VALUES, args.seed, args.out,
        )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    write_markdown(summary, args.out)
    print(f"\n[done] wrote {args.out}/summary.{{json,md}}", flush=True)


if __name__ == "__main__":
    main()

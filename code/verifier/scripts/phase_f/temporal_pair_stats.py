"""Phase F prep Task 1 — temporal pair statistics.

For each session × k in K_VALUES:
  Sample N_PAIRS random valid pairs (frame at t-k, frame at t).
  Compute per pair:
    pose_similarity:    SSIM on packed CFA G channel (avg of G1,G2)
                         masked by projection-region mask (high-variance pixels
                         across the session are the projection footprint)
    emission_difference: L2 between emission tiles
    capture_difference:  L2 between packed CFA captures
  Decompose capture_difference via simple linear regression:
    capture_diff = α · emission_diff + β · pose_diff + residual

Output: experiments/phase_f_prep/temporal_pair_stats.json + 3 plots.

Decision criterion:
  - α > β at small k → emission swap dominates → those k usable for F-A
  - β > α at small k → pose drift dominates → need larger k or pose filter

CPU-only (uses skimage SSIM + numpy linear regression). ~1-2 hr for both
sessions × 8 k values × 200 pairs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from phase_f.cfa_roundtrip import bayer_rg8_to_packed_cfa, EXPECTED_BYTES, HALF_H, HALF_W

try:
    from skimage.metrics import structural_similarity as ssim
    HAVE_SSIM = True
except Exception:
    HAVE_SSIM = False
import cv2


K_VALUES = (1, 2, 3, 5, 10, 20, 50, 100)


def load_packed_cfa_uint8(p: Path) -> np.ndarray:
    raw = p.read_bytes()
    return bayer_rg8_to_packed_cfa(raw)  # (4, 2300, 2660) uint8


def load_emission_uint8(p: Path) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # (1080, 1920, 3) uint8


def compute_projection_mask(session_dir: Path, n_sample: int = 50, threshold_pct: float = 75.0,
                            seed: int = 0, half_size: int = 256) -> np.ndarray:
    """Per-pixel std across n_sample random captures' G channel; pixels above
    threshold_pct percentile are treated as 'projection regions'. Returns a
    binary mask (HALF_H, HALF_W) at half-size to keep memory reasonable."""
    rng = np.random.RandomState(seed)
    files = sorted((session_dir / "Recordings").glob("frame_*.raw"))
    picks = rng.choice(len(files), size=min(n_sample, len(files)), replace=False)
    # Stack G channels at downsampled (half_size, half_size) for memory
    target_hw = (half_size, half_size)
    Gs = []
    for i in picks:
        cfa = load_packed_cfa_uint8(files[i])
        G = (cfa[1].astype(np.float32) + cfa[2].astype(np.float32)) / 2.0
        # Downsample with INTER_AREA for std-preserving
        G_small = cv2.resize(G, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_AREA)
        Gs.append(G_small)
    stack = np.stack(Gs, axis=0)
    pix_std = stack.std(axis=0)
    threshold = np.percentile(pix_std, threshold_pct)
    mask_small = pix_std >= threshold
    return mask_small  # (half_size, half_size)


def pose_similarity_ssim(c1: np.ndarray, c2: np.ndarray, proj_mask_small: np.ndarray,
                         half_size: int = 256) -> float:
    """SSIM on the G-channel of two packed CFAs, downsampled to half_size and
    masked AWAY-FROM projection regions (we want pose, not projection)."""
    G1 = (c1[1].astype(np.float32) + c1[2].astype(np.float32)) / 2.0
    G2 = (c2[1].astype(np.float32) + c2[2].astype(np.float32)) / 2.0
    g1s = cv2.resize(G1, (half_size, half_size), interpolation=cv2.INTER_AREA)
    g2s = cv2.resize(G2, (half_size, half_size), interpolation=cv2.INTER_AREA)
    if HAVE_SSIM:
        # Mask away projection regions (use anti-mask: pose lives in the static part)
        non_proj = ~proj_mask_small
        # Apply mask via setting projection-region pixels to mean of non-projection pixels
        m1 = g1s.copy(); m2 = g2s.copy()
        if non_proj.sum() > 0:
            mean1 = g1s[non_proj].mean(); mean2 = g2s[non_proj].mean()
            m1[proj_mask_small] = mean1
            m2[proj_mask_small] = mean2
        return float(ssim(m1, m2, data_range=255.0))
    # fallback: 1 - normalized L2 distance on non-projection regions
    non_proj = ~proj_mask_small
    diff = g1s[non_proj] - g2s[non_proj]
    return float(1.0 - (np.abs(diff).mean() / 255.0))


def emission_l2(e1: np.ndarray, e2: np.ndarray) -> float:
    return float(np.sqrt(((e1.astype(np.float32) - e2.astype(np.float32)) ** 2).mean()))


def capture_l2(c1: np.ndarray, c2: np.ndarray) -> float:
    return float(np.sqrt(((c1.astype(np.float32) - c2.astype(np.float32)) ** 2).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d2-dir", type=Path, required=True)
    ap.add_argument("--v10-dir", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-plot-prefix", type=Path, required=True)
    ap.add_argument("--n-pairs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_plot_prefix.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    sessions = []
    if args.d2_dir.exists():
        sessions.append(("D2", args.d2_dir, list(range(0, 4194))))
    if args.v10_dir.exists():
        sessions.append(("V10", args.v10_dir, list(range(0, 2500))))

    out: dict = {"per_session": {}, "k_values": list(K_VALUES), "n_pairs": args.n_pairs,
                  "have_ssim": HAVE_SSIM}

    for sess_name, sess_dir, train_rows in sessions:
        print(f"\n=== {sess_name} ===", flush=True)
        # Build projection-region mask once
        print(f"[{sess_name}] computing projection-region mask (n=50 captures, downsample to 256x256)", flush=True)
        proj_mask = compute_projection_mask(sess_dir, n_sample=50, threshold_pct=75.0)
        print(f"  proj mask: {proj_mask.sum()}/{proj_mask.size} pixels", flush=True)

        per_k: dict = {}
        for k in K_VALUES:
            valid = [t for t in train_rows if (t - k) >= 0 and t in train_rows]
            if len(valid) < args.n_pairs:
                samples = valid
            else:
                samples = rng.choice(valid, size=args.n_pairs, replace=False).tolist()
            samples = sorted(set(int(t) for t in samples))
            print(f"  k={k}: sampling {len(samples)} pairs", flush=True)

            pose_sims = []
            em_diffs = []
            cap_diffs = []
            t0 = time.time()
            for ti, t in enumerate(samples):
                src_t = t - k
                cap_t_path = sess_dir / "Recordings" / f"frame_{t:06d}.raw"
                cap_s_path = sess_dir / "Recordings" / f"frame_{src_t:06d}.raw"
                em_t_path = sess_dir / "derived" / "Emissions" / f"tile_{t:06d}.png"
                em_s_path = sess_dir / "derived" / "Emissions" / f"tile_{src_t:06d}.png"
                if not (cap_t_path.exists() and cap_s_path.exists()
                        and em_t_path.exists() and em_s_path.exists()):
                    continue
                try:
                    c_t = load_packed_cfa_uint8(cap_t_path)
                    c_s = load_packed_cfa_uint8(cap_s_path)
                    e_t = load_emission_uint8(em_t_path)
                    e_s = load_emission_uint8(em_s_path)
                except Exception as exc:
                    print(f"    skip pair t={t}: {exc}", flush=True)
                    continue
                pose_sims.append(pose_similarity_ssim(c_s, c_t, proj_mask))
                em_diffs.append(emission_l2(e_s, e_t))
                cap_diffs.append(capture_l2(c_s, c_t))
                if (ti + 1) % 50 == 0:
                    print(f"    pair {ti+1}/{len(samples)} elapsed={time.time()-t0:.0f}s", flush=True)

            pose_sims = np.array(pose_sims); em_diffs = np.array(em_diffs); cap_diffs = np.array(cap_diffs)
            if len(pose_sims) >= 5:
                # Linear regression: capture_diff = a*em_diff + b*(1-pose_sim) + intercept
                pose_diff = 1.0 - pose_sims
                X = np.stack([em_diffs, pose_diff, np.ones_like(em_diffs)], axis=1)
                coef, residuals, rank, sv = np.linalg.lstsq(X, cap_diffs, rcond=None)
                alpha, beta, intercept = coef.tolist()
                pred = X @ coef
                ss_res = float(((cap_diffs - pred) ** 2).sum())
                ss_tot = float(((cap_diffs - cap_diffs.mean()) ** 2).sum())
                r2 = 1.0 - ss_res / max(ss_tot, 1e-9)
                # Decompose variance contribution
                var_em_only = float(np.var(alpha * em_diffs))
                var_pose_only = float(np.var(beta * pose_diff))
                total_var = var_em_only + var_pose_only + 1e-9
                em_share = var_em_only / total_var
                pose_share = var_pose_only / total_var
            else:
                alpha = beta = intercept = float("nan")
                r2 = float("nan"); em_share = pose_share = float("nan")

            per_k[str(k)] = {
                "n_pairs": int(len(pose_sims)),
                "pose_similarity_mean": float(pose_sims.mean()) if len(pose_sims) else float("nan"),
                "pose_similarity_p5":   float(np.percentile(pose_sims, 5)) if len(pose_sims) else float("nan"),
                "pose_similarity_p95":  float(np.percentile(pose_sims, 95)) if len(pose_sims) else float("nan"),
                "emission_diff_mean":   float(em_diffs.mean()) if len(em_diffs) else float("nan"),
                "capture_diff_mean":    float(cap_diffs.mean()) if len(cap_diffs) else float("nan"),
                "regression_alpha_emission":      alpha,
                "regression_beta_pose":           beta,
                "regression_intercept":           intercept,
                "regression_r2":                  r2,
                "emission_variance_share":        em_share,
                "pose_variance_share":            pose_share,
                "elapsed_sec": round(time.time() - t0, 1),
            }
            print(f"    k={k}: pose_sim_mean={per_k[str(k)]['pose_similarity_mean']:.3f}  "
                  f"em_diff={per_k[str(k)]['emission_diff_mean']:.2f}  "
                  f"cap_diff={per_k[str(k)]['capture_diff_mean']:.2f}  "
                  f"em_share={em_share:.2f}  pose_share={pose_share:.2f}  R²={r2:.3f}", flush=True)
        out["per_session"][sess_name] = per_k

    # Decision summary
    out["decision"] = {}
    for sess in out["per_session"]:
        decision_per_k = {}
        for k in K_VALUES:
            d = out["per_session"][sess][str(k)]
            em_share = d.get("emission_variance_share", 0.0)
            pose_share = d.get("pose_variance_share", 0.0)
            if em_share > 0.5:
                verdict = "emission-dominant — usable for F-A"
            elif pose_share > 0.5:
                verdict = "pose-dominant — risky for F-A (will learn pose-warp)"
            else:
                verdict = "mixed — both signals present"
            decision_per_k[str(k)] = verdict
        out["decision"][sess] = decision_per_k

    args.out_json.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out_json}", flush=True)

    # Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for sess in out["per_session"]:
            d = out["per_session"][sess]
            ks = sorted(int(k) for k in d)
            pose_means = [d[str(k)]["pose_similarity_mean"] for k in ks]
            em_means = [d[str(k)]["emission_diff_mean"] for k in ks]
            cap_means = [d[str(k)]["capture_diff_mean"] for k in ks]
            em_shares = [d[str(k)]["emission_variance_share"] for k in ks]
            pose_shares = [d[str(k)]["pose_variance_share"] for k in ks]

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            axes[0].plot(ks, pose_means, "o-")
            axes[0].set_xlabel("k (frames)"); axes[0].set_ylabel("pose SSIM"); axes[0].set_title(f"{sess} pose similarity")
            axes[0].set_xscale("log"); axes[0].grid(alpha=0.3)
            axes[1].plot(ks, em_means, "o-", label="emission L2")
            axes[1].plot(ks, cap_means, "s-", label="capture L2")
            axes[1].set_xlabel("k"); axes[1].set_ylabel("L2 distance"); axes[1].set_title(f"{sess} differences")
            axes[1].set_xscale("log"); axes[1].legend(); axes[1].grid(alpha=0.3)
            axes[2].bar([str(k) for k in ks], em_shares, color="C0", label="emission share", alpha=0.7)
            axes[2].bar([str(k) for k in ks], pose_shares, bottom=em_shares, color="C3", label="pose share", alpha=0.7)
            axes[2].set_xlabel("k"); axes[2].set_ylabel("variance share")
            axes[2].set_title(f"{sess}: regression decomposition")
            axes[2].axhline(0.5, color="black", linestyle=":", alpha=0.5)
            axes[2].legend()
            fig.tight_layout()
            fig.savefig(args.out_plot_prefix.with_name(args.out_plot_prefix.name + f"_{sess}.png"), dpi=110)
            plt.close(fig)
        print(f"wrote plots {args.out_plot_prefix}_*.png", flush=True)
    except Exception as exc:
        print(f"[WARN] plot failed: {exc}", flush=True)


if __name__ == "__main__":
    main()

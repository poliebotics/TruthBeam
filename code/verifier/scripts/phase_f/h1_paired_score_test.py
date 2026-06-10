"""H1 paired-score test on locally-warped E negatives.

Question: does the H1 diffusion verifier read capture-emission correspondence,
or just distributional typicality of E?

Methodology (per operator + CGPT round-7):
  1. Held-out (C, E_real) pairs from D2 selection_gate_normal
  2. For each pair, generate locally-warped E negatives at m ∈ {2, 4, 8, 16} px
     using bilinear warps in image space (cleaner coordinate-warp deferred).
  3. For each (C, E_candidate): compute deterministic energy proxy =
     mean denoising loss over K=32 fixed timesteps with shared noise seed.
  4. Score = -energy. Pairwise win rate at each warp magnitude.

CPU only. Estimated 2-4 hr for ~50 pairs.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(8)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "phase_e"))
sys.path.insert(0, str(ROOT / "scripts" / "phase_f"))

from data.emission_dataset import EmissionDataset  # noqa: E402
from diffusion_verifier_sizing import CondDDPMUNet  # noqa: E402
from train_diffusion_binder import build_diffusion_constants, q_sample  # noqa: E402


VAL_RANGE_D2 = (4792, 5392)


def warp_emission(em: np.ndarray, magnitude_px: float, seed: int) -> np.ndarray:
    """Apply local random bilinear warp with given magnitude.

    em: (3, H, W) float32 in [0, 1]
    Returns warped (3, H, W).
    """
    H, W = em.shape[1:]
    rng = np.random.RandomState(seed)
    # Generate random shift field at coarse scale, smooth, scale to magnitude
    coarse_h, coarse_w = H // 32, W // 32
    dy_coarse = rng.randn(coarse_h, coarse_w).astype(np.float32)
    dx_coarse = rng.randn(coarse_h, coarse_w).astype(np.float32)
    # Smooth with Gaussian
    dy_field = cv2.GaussianBlur(cv2.resize(dy_coarse, (W, H)), (31, 31), sigmaX=8)
    dx_field = cv2.GaussianBlur(cv2.resize(dx_coarse, (W, H)), (31, 31), sigmaX=8)
    # Normalize to magnitude
    norms = np.sqrt(dy_field ** 2 + dx_field ** 2)
    target_norm = magnitude_px
    actual_norm_p95 = np.percentile(norms, 95) + 1e-8
    scale = target_norm / actual_norm_p95
    dy_field *= scale
    dx_field *= scale
    # Build remap grid
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    map_y = (yy + dy_field).astype(np.float32)
    map_x = (xx + dx_field).astype(np.float32)
    # Apply per channel
    out = np.zeros_like(em)
    for c in range(3):
        out[c] = cv2.remap(em[c], map_x, map_y, interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
    return out


def load_h1(ckpt_path: Path, device: torch.device, dtype: torch.dtype):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    base_ch = cfg.get("base_ch", 64)
    mults = tuple(cfg.get("mults", [1, 2, 4, 4, 8]))
    em_h_eff = cfg.get("em_h_eff", 540)
    em_w_eff = cfg.get("em_w_eff", 960)
    capture_hw = cfg.get("capture_hw", [1150, 1330])
    T = cfg.get("T", 1000)
    em_downsample = cfg.get("em_downsample", 2)
    print(f"[init] CondDDPMUNet base_ch={base_ch} mults={mults} em_eff={em_h_eff}x{em_w_eff}",
          flush=True)
    model = CondDDPMUNet(
        base_ch=base_ch, channel_mults=mults,
        attn_at=tuple(i == len(mults) - 1 for i in range(len(mults))),
    ).to(device, dtype=dtype)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(state, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    dc = build_diffusion_constants(T, device, dtype)
    return model, dc, em_h_eff, em_w_eff, capture_hw, em_downsample


@torch.no_grad()
def energy_score(model, dc, capture: torch.Tensor, em: torch.Tensor,
                 K: int, seed: int) -> float:
    """Deterministic energy proxy: mean denoising loss over K fixed timesteps.

    Lower energy = better (model assigns higher likelihood to this (C, E) pair).
    """
    rng = torch.Generator(device=capture.device).manual_seed(seed)
    timesteps = torch.linspace(50, dc["T"] - 1, K, device=capture.device).long()
    losses = []
    for t in timesteps:
        t_b = t.view(1)
        noise = torch.randn(em.shape, device=em.device, dtype=em.dtype, generator=rng)
        em_t = q_sample(em, t_b, dc, noise)
        # Capture is shared
        pred_noise = model(em_t, capture, t_b.float())
        loss = F.mse_loss(pred_noise, noise).item()
        losses.append(loss)
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h1-ckpt", type=Path, required=True)
    ap.add_argument("--d2-dir", type=Path, required=True)
    ap.add_argument("--n-pairs", type=int, default=40)
    ap.add_argument("--K", type=int, default=32, help="Number of fixed timesteps for energy proxy")
    ap.add_argument("--magnitudes", nargs="+", type=float, default=[2.0, 4.0, 8.0, 16.0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    dtype = torch.float32

    model, dc, em_h_eff, em_w_eff, capture_hw, em_downsample = load_h1(
        args.h1_ckpt, device, dtype)

    rs, re = VAL_RANGE_D2
    ds = EmissionDataset(
        session_dir=args.d2_dir, row_start=rs, row_end=re,
        capture_h=capture_hw[0], capture_w=capture_hw[1],
        emission_h=1080, emission_w=1920, session_id="D2", augment=False,
    )
    rng = np.random.RandomState(0)
    sample_indices = sorted(rng.choice(len(ds),
                                       size=min(args.n_pairs, len(ds)),
                                       replace=False).tolist())
    print(f"[init] {len(sample_indices)} pairs from D2 val; K={args.K} magnitudes={args.magnitudes}",
          flush=True)

    # Per-magnitude rows: list of {pair_idx, energy_real, energy_warp, won}
    per_mag = {m: [] for m in args.magnitudes}
    t0 = time.time()
    for fi, ds_idx in enumerate(sample_indices):
        sample = ds[ds_idx]
        cap = sample["capture"].unsqueeze(0).to(device, dtype=dtype)  # (1, 4, 1150, 1330)
        em_real_np = sample["emission"].numpy()  # (3, 1080, 1920)

        # Downsample E to H1's working resolution
        em_real_t = sample["emission"].unsqueeze(0)  # (1, 3, 1080, 1920)
        if (em_h_eff, em_w_eff) != (1080, 1920):
            em_real_t = F.interpolate(em_real_t, size=(em_h_eff, em_w_eff),
                                       mode="bilinear", align_corners=False)
        em_real_t = em_real_t.to(device, dtype=dtype)

        # Use shared seed per pair so timesteps + noise sampling identical across cases
        seed_for_pair = (12345 ^ ds_idx) & 0xFFFFFFFF
        e_real = energy_score(model, dc, cap, em_real_t, args.K, seed=seed_for_pair)

        for m in args.magnitudes:
            em_warp_np = warp_emission(em_real_np, magnitude_px=m, seed=seed_for_pair + int(m * 100))
            em_warp_t = torch.from_numpy(em_warp_np).unsqueeze(0)
            if (em_h_eff, em_w_eff) != (1080, 1920):
                em_warp_t = F.interpolate(em_warp_t, size=(em_h_eff, em_w_eff),
                                           mode="bilinear", align_corners=False)
            em_warp_t = em_warp_t.to(device, dtype=dtype)
            e_warp = energy_score(model, dc, cap, em_warp_t, args.K, seed=seed_for_pair)
            per_mag[m].append({
                "pair_idx": ds_idx,
                "energy_real": e_real,
                "energy_warp": e_warp,
                "real_wins": e_real < e_warp,  # lower energy = better
            })

        if (fi + 1) % 5 == 0:
            print(f"  pair {fi+1}/{len(sample_indices)}  elapsed={time.time()-t0:.0f}s",
                  flush=True)

    summary = {}
    for m, rows in per_mag.items():
        wins = [r["real_wins"] for r in rows]
        e_reals = np.array([r["energy_real"] for r in rows])
        e_warps = np.array([r["energy_warp"] for r in rows])
        # AUROC: rank scores combining (-e_real, label=1) and (-e_warp, label=0)
        scores = np.concatenate([-e_reals, -e_warps])
        labels = np.concatenate([np.ones_like(e_reals), np.zeros_like(e_warps)])
        order = np.argsort(scores)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(order) + 1)
        n_pos = labels.sum()
        n_neg = len(labels) - n_pos
        auroc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / max(n_pos * n_neg, 1)
        summary[m] = {
            "n_pairs": len(rows),
            "win_rate": float(np.mean(wins)),
            "energy_real_mean": float(e_reals.mean()),
            "energy_warp_mean": float(e_warps.mean()),
            "energy_gap_mean": float(e_warps.mean() - e_reals.mean()),
            "auroc": float(auroc),
        }

    out = {
        "h1_ckpt": str(args.h1_ckpt),
        "n_pairs": len(sample_indices),
        "K_timesteps": args.K,
        "magnitudes_px": args.magnitudes,
        "summary": summary,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (args.out / "h1_paired_score_test.json").write_text(json.dumps(out, indent=2))

    md = [
        "# H1 paired-score test on warped E negatives",
        "",
        f"H1 checkpoint: `{args.h1_ckpt.name}`",
        f"Pairs analyzed: {len(sample_indices)} from D2 selection_gate_normal",
        f"Energy proxy: mean denoising loss over K={args.K} fixed timesteps (deterministic, shared noise seed per pair)",
        f"Warp magnitudes (px): {args.magnitudes}",
        "",
        "## Per-magnitude results",
        "",
        "| magnitude (px) | n_pairs | win_rate | AUROC | energy(real) mean | energy(warp) mean | energy_gap |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in args.magnitudes:
        s = summary[m]
        md.append(
            f"| {m:.0f} | {s['n_pairs']} | {s['win_rate']:.3f} | "
            f"{s['auroc']:.3f} | {s['energy_real_mean']:.4f} | "
            f"{s['energy_warp_mean']:.4f} | {s['energy_gap_mean']:+.4f} |"
        )
    md += [
        "",
        "## Interpretation gates (per CGPT round-7)",
        "",
        "- Strong correspondence (Layer 4 viable): win rate ≥ 0.80 at 2 px, AUROC ≥ 0.85, monotone score degradation",
        "- Weak correspondence: monotone but shallow, win rate 0.60-0.80",
        "- No correspondence (Layer 4 structurally compromised): flat curve, win rate ≈ 0.50",
        "",
        "**Read this table top-to-bottom**: as warp magnitude grows, win_rate and AUROC SHOULD increase if H1 reads correspondence (real wins more easily over more-warped E). If they stay near 0.5, H1 reads only typicality.",
        "",
        f"Elapsed: {out['elapsed_sec']}s",
    ]
    (args.out / "h1_paired_score_test.md").write_text("\n".join(md))
    print(f"\n[done] wrote {args.out}/h1_paired_score_test.{{json,md}}", flush=True)


if __name__ == "__main__":
    main()

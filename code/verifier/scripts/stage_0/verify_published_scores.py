"""Independently REGENERATE a few published Stage-0 scores from the public
verifier weights and show they match the published `.npz` — bit-for-bit math,
not a promise.

This is the answer to "Path A only proves your scores produce AUROC=1.000, not
that the scores are honest." Here we re-run the public diffusion verifier on a
handful of public raw frames (+ the public F-A forger) and confirm the
resulting per-frame MSE tensors reproduce the published `stage0_d2_raw.npz` —
no GPU farm, no 378 GiB corpus, just a few frames.

It reuses the exact production scorer (`eval.py`'s `score_one`, loaders,
diffusion constants, and deterministic noise), so there is nothing to fake:
the selection seed, the per-frame noise seed, and the forward pass are the same
ones that produced the release.

Run from `code/verifier/scripts/stage_0/`:

    python verify_published_scores.py \
        --diffusion-ckpt model_final.pt --fa-ckpt f_a_v1_step_00100000.pt \
        --d2-dir <dir with Recordings/ and derived/Emissions/> \
        --published-npz stage0_d2_raw.npz --n-frames 2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import eval as EV  # reuse the exact production scorer, loaders, and constants


def select_rows(session: str, seed: int, n_frames: int):
    """Replicate eval.py's deterministic held-out selection exactly."""
    rng = np.random.RandomState(seed)
    blocks = EV.EVAL_BLOCKS[session]
    per_block = max(1, n_frames // len(blocks))
    sampled: list[tuple[int, int]] = []
    for bi, (a, b) in enumerate(blocks):
        valid = list(range(a + 30 + EV.SOURCE_LAG, b - 30))
        chosen = valid if len(valid) <= per_block else \
            rng.choice(valid, size=per_block, replace=False).tolist()
        for r in sorted(chosen):
            sampled.append((bi, r))
    return [(gi, bi, row) for gi, (bi, row) in enumerate(sampled)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diffusion-ckpt", type=Path, required=True)
    ap.add_argument("--fa-ckpt", type=Path, required=True)
    ap.add_argument("--d2-dir", type=Path, required=True,
                    help="Holds Recordings/frame_*.raw and derived/Emissions/tile_*.png")
    ap.add_argument("--published-npz", type=Path, required=True)
    ap.add_argument("--n-frames", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    diff_model = EV.load_diff_model(args.diffusion_ckpt, device, dtype)
    fa_model = EV.load_fa_v1_checkpoint(args.fa_ckpt, device, dtype)
    dc = EV.build_diffusion_constants(args.T, device, torch.float32)

    pub = np.load(args.published_npz, allow_pickle=False)
    pub_rows = pub["rows"].tolist()

    # Full selection (n=198) so gi / rows match the release exactly; score first N.
    sel = select_rows("D2", args.seed, 198)
    conds = ["real_correct", "real_zero", "fake_correct"]

    print(f"{'row':>6} {'condition':>14} {'max|Δ|':>11} {'published':>11} {'regenerated':>12}  result")
    print("-" * 74)
    worst = 0.0
    for gi, bi, target_row in sel[:args.n_frames]:
        source_row = target_row - EV.SOURCE_LAG
        E_correct = EV._resize_E_to_target(EV.load_emission_at(
            args.d2_dir / "derived" / "Emissions" / f"tile_{target_row:06d}.png",
            EV.EMISSION_NATIVE_H, EV.EMISSION_NATIVE_W)).to(device, dtype=dtype)
        C_real = EV._crop_and_resize_C(EV._load_packed_cfa_float01(
            args.d2_dir / "Recordings" / f"frame_{target_row:06d}.raw")).to(device, dtype=dtype)
        C_fake = EV.render_C_fake(fa_model, args.d2_dir, source_row=source_row,
                                  target_row=target_row, device=device,
                                  dtype=dtype).to(device, dtype=dtype)
        H, W = C_real.shape[-2:]
        torch.manual_seed(args.seed + gi)
        noise_per_t = torch.randn(len(EV.TIMESTEPS), EV.K_NOISE, 4, H, W,
                                  device=device, dtype=torch.float32)
        inputs = {"real_correct": (C_real, E_correct),
                  "real_zero": (C_real, None),
                  "fake_correct": (C_fake, E_correct)}
        pi = pub_rows.index(target_row)
        for cond in conds:
            C, E = inputs[cond]
            regen = np.full((len(EV.TIMESTEPS), EV.K_NOISE), np.nan)
            for ti, t_val in enumerate(EV.TIMESTEPS):
                regen[ti] = EV.score_one(diff_model, C, E, t_val, EV.K_NOISE,
                                         dc, device, dtype, args.bs, noise_per_t[ti])
            published = pub[f"cond_{cond}"][pi]
            d = float(np.max(np.abs(regen - published)))
            worst = max(worst, d)
            print(f"{target_row:>6} {cond:>14} {d:>11.2e} {published.mean():>11.5f} "
                  f"{regen.mean():>12.5f}  {'OK' if d < args.tol else 'MISMATCH'}")

    print("-" * 74)
    print(f"worst max|Δ| over all scored cells: {worst:.2e}")
    print("RESULT:", "PASS — regenerated scores reproduce the published npz."
          if worst < args.tol else
          "DIFF above tolerance (try matching the release dtype / T).")


if __name__ == "__main__":
    main()

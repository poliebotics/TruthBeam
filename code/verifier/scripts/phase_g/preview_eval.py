"""Phase G — micro-preview of the main model's correct-vs-wrong gap.

Tiny config: 30 held-out D2 frames, t=300, K=1, 4 conditions
(correct, wrong+15, wrong-15, uncond). Single GPU. ~30 seconds.

Gives an early-signal answer: does the main model produce a measurable
correct-vs-wrong gap at all? If yes, the full eval will show details.
If no, the diagnostic is heading toward CLEAN NEGATIVE or
UNINTERPRETABLE territory.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from phase_g.diffusion_diagnostic_model import (  # noqa: E402
    DiffusionDiagnosticUNet, build_diffusion_constants, q_sample,
)
from phase_g.diffusion_diagnostic_dataset import (  # noqa: E402
    EVAL_BLOCKS, _crop_and_resize_C, _load_packed_cfa_float01,
    _resize_E_to_target, EMISSION_NATIVE_H, EMISSION_NATIVE_W,
)
from data.emission_dataset import load_emission_at  # noqa: E402


def load_model(ckpt_path: Path, device: torch.device, dtype: torch.dtype):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck["model"]
    args = ck.get("args", {})
    base_ch = args.get("base_ch", 96)
    mults = tuple(args.get("mults", (1, 2, 4, 4)))
    attn_at = args.get("attn_at",
                       tuple(i == len(mults) - 1 for i in range(len(mults))))
    attn_at = tuple(bool(x) for x in attn_at)
    m = DiffusionDiagnosticUNet(
        in_ch=4, base_ch=base_ch, channel_mults=mults, attn_at=attn_at,
        cond_drop_prob=0.0, hint_in_ch=11,
    ).to(device, dtype=dtype)
    m.load_state_dict(state)
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--d2-dir", type=Path, required=True)
    ap.add_argument("--n-frames", type=int, default=30)
    ap.add_argument("--t-val", type=int, default=300)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    model = load_model(args.ckpt, device, dtype)
    dc = build_diffusion_constants(args.T, device, torch.float32)

    # Sample frames evenly across D2 eval blocks
    rng = np.random.RandomState(args.seed)
    blocks = EVAL_BLOCKS["D2"]
    per_block = args.n_frames // len(blocks)
    rows = []
    for a, b in blocks:
        valid = list(range(a + 30, b - 30))
        rows.extend(rng.choice(valid, size=per_block, replace=False).tolist())
    rows = sorted(rows)
    print(f"[init] ckpt={args.ckpt}")
    print(f"[init] {len(rows)} D2 frames, t={args.t_val}, K=1")

    def load_C(r):
        return _crop_and_resize_C(_load_packed_cfa_float01(
            args.d2_dir / "Recordings" / f"frame_{r:06d}.raw")).to(device, dtype=dtype)

    def load_E(r):
        return _resize_E_to_target(load_emission_at(
            args.d2_dir / "derived" / "Emissions" / f"tile_{r:06d}.png",
            EMISSION_NATIVE_H, EMISSION_NATIVE_W)).to(device, dtype=dtype)

    mse_correct = []
    mse_wrong_plus = []
    mse_wrong_minus = []
    mse_uncond = []

    t0 = time.time()
    with torch.no_grad():
        for fi, r in enumerate(rows):
            torch.manual_seed(args.seed + r)
            C0 = load_C(r)
            E_correct = load_E(r)
            E_wp = load_E(r + 15)
            E_wm = load_E(r - 15)
            H, W = C0.shape[-2:]
            noise = torch.randn(1, 4, H, W, device=device, dtype=torch.float32)
            t_tensor = torch.tensor([args.t_val], device=device, dtype=torch.long)
            C_t = q_sample(C0.float().unsqueeze(0), t_tensor, dc, noise).to(dtype)
            t_float = t_tensor.float()

            def score(E_in, force_uncond=False):
                with torch.amp.autocast("cuda", dtype=dtype):
                    eps_pred = model(C_t, E_in.unsqueeze(0), t_float,
                                     force_uncond=force_uncond)
                return (eps_pred.float() - noise).pow(2).mean().item()

            mse_correct.append(score(E_correct))
            mse_wrong_plus.append(score(E_wp))
            mse_wrong_minus.append(score(E_wm))
            mse_uncond.append(score(E_correct, force_uncond=True))

            if (fi + 1) % 10 == 0:
                print(f"  {fi+1}/{len(rows)}  elapsed={time.time()-t0:.0f}s")

    c = np.array(mse_correct)
    wp = np.array(mse_wrong_plus)
    wm = np.array(mse_wrong_minus)
    u = np.array(mse_uncond)

    print(f"\n[results]  n={len(c)}  t={args.t_val}")
    print(f"  MSE correct E :  mean={c.mean():.5f}  std={c.std():.5f}")
    print(f"  MSE wrong +15 :  mean={wp.mean():.5f}  std={wp.std():.5f}")
    print(f"  MSE wrong -15 :  mean={wm.mean():.5f}  std={wm.std():.5f}")
    print(f"  MSE uncond    :  mean={u.mean():.5f}  std={u.std():.5f}")
    print(f"\n  Δ_wrong+15  =  {wp.mean() - c.mean():+.6f}")
    print(f"  Δ_wrong-15  =  {wm.mean() - c.mean():+.6f}")
    print(f"  Δ_uncond    =  {u.mean()  - c.mean():+.6f}")
    # Paired test (per-frame): how often is correct < wrong?
    wins_plus = (c < wp).mean()
    wins_minus = (c < wm).mean()
    wins_uncond = (c < u).mean()
    print(f"\n  paired correct < wrong+15: {wins_plus*100:.1f}% of frames")
    print(f"  paired correct < wrong-15: {wins_minus*100:.1f}% of frames")
    print(f"  paired correct < uncond:   {wins_uncond*100:.1f}% of frames")
    print(f"  (50% = chance, 100% = perfect)")
    print(f"\n[done] elapsed={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

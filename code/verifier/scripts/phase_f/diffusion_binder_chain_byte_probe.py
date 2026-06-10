"""Chain-byte sensitivity probe for the diffusion verifier (H1).

Same protocol as `scripts/phase_e/chain_byte_probe.py` but the binder
inference is K-step DDIM sampling instead of single-forward-pass.

Per-row evaluation:
  - C_t  = real capture
  - X_t  = clean XOF streams from chain_log[t+1]
  - E_t  = render_from_streams(X_t) (matched ground truth, bit-exact)
  - P_t  = ddim_sample(model, C_t, K=K_STEPS) (diffusion binder prediction)

For p ∈ NOISE_LEVELS, seed ∈ SEEDS:
  - noised X_t', noised E_t' = render_from_streams(X_t')
  - matched_score = PSNR(P_t, E_t)
  - noised_score = PSNR(P_t, E_t')
  - attack_success = noised_score >= matched_score - 1e-6

Output: same JSON / PNG layout as Phase E probes, in the H1 ckpt's dir.

Run:
  CUDA_VISIBLE_DEVICES=6 python scripts/phase_f/diffusion_binder_chain_byte_probe.py \
    --ckpt /path/to/poliebotics_phase_b/diffusion_binder_h1_full/diffusion_binder.pt \
    --d2-dir /path/to/poliebotics_phase_b/data/d2 \
    --out /path/to/poliebotics_phase_b/diffusion_binder_h1_full/chain_byte_probe \
    --n-rows 100 --K-steps 25 --bf16
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(8)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "phase_e"))
sys.path.insert(0, str(ROOT / "scripts" / "phase_f"))

from data.emission_dataset import EmissionDataset  # noqa: E402
from data.raw_bayer_dataset import load_chain_log  # noqa: E402
from bitexact_renderer import (  # noqa: E402
    expand_seeds_to_streams,
    noise_streams_byte_replace,
    render_from_streams,
)
from diffusion_verifier_sizing import CondDDPMUNet  # noqa: E402
from train_diffusion_binder import linear_beta_schedule, build_diffusion_constants  # noqa: E402


NOISE_LEVELS = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
SEEDS = (0, 1, 2, 3, 4)


def psnr(pred01: torch.Tensor, target01: torch.Tensor) -> float:
    mse = ((pred01 - target01) ** 2).mean().item()
    if mse < 1e-12:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def bootstrap_ci_rate(hits: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    n = len(hits)
    if n == 0:
        return (float("nan"), float("nan"))
    rates = np.array([rng.choice(hits, size=n, replace=True).mean() for _ in range(n_boot)])
    return (float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5)))


def ddim_sample(model: torch.nn.Module, capture: torch.Tensor, dc: dict,
                K: int, em_h: int, em_w: int, dtype: torch.dtype) -> torch.Tensor:
    """K-step DDIM-style deterministic sampling (eta=0).

    Returns predicted clean emission shape (B, 3, em_h, em_w) in [0, 1] range
    after a final clip.
    """
    device = capture.device
    B = capture.shape[0]
    T = dc["T"]
    # Pick K timesteps evenly spaced from T-1 down to 0
    timesteps = torch.linspace(T - 1, 0, K + 1, device=device).long()
    # Start from pure noise
    em = torch.randn(B, 3, em_h, em_w, device=device, dtype=dtype)
    for i in range(K):
        t_now = timesteps[i].view(1).expand(B)
        t_next = timesteps[i + 1].view(1).expand(B)
        sa_now = dc["sqrt_alphas_cum"][t_now].view(B, 1, 1, 1)
        so_now = dc["sqrt_one_minus_alphas_cum"][t_now].view(B, 1, 1, 1)
        # Predict noise
        pred_noise = model(em, capture, t_now.to(dtype))
        # Predicted x0
        x0_pred = (em - so_now * pred_noise) / sa_now
        # Compute next em via DDIM formula (eta=0, deterministic)
        sa_next = dc["sqrt_alphas_cum"][t_next].view(B, 1, 1, 1)
        so_next = dc["sqrt_one_minus_alphas_cum"][t_next].view(B, 1, 1, 1)
        em = sa_next * x0_pred + so_next * pred_noise
    return em.clamp(0, 1)


def load_diffusion_binder(ckpt_path: Path, device: torch.device, dtype: torch.dtype):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
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
        base_ch=base_ch,
        channel_mults=mults,
        attn_at=tuple(i == len(mults) - 1 for i in range(len(mults))),
    ).to(device, dtype=dtype)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(state, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    dc = build_diffusion_constants(T, device, dtype)
    return model, dc, em_h_eff, em_w_eff, capture_hw, em_downsample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--d2-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-rows", type=int, default=100)
    ap.add_argument("--K-steps", type=int, default=25)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--render-device", default="cpu")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    model, dc, em_h_eff, em_w_eff, capture_hw, em_downsample = load_diffusion_binder(
        args.ckpt, device, dtype)

    chain = load_chain_log(args.d2_dir / "chain_log.csv")
    rng = np.random.RandomState(7)
    val_rows = sorted([t for t in range(5394, 5992) if t in chain and (t + 1) in chain])
    rows = sorted(rng.choice(val_rows, size=min(args.n_rows, len(val_rows)),
                             replace=False).tolist())
    print(f"[init] sampled {len(rows)} rows", flush=True)

    # Build EmissionDataset for capture loading
    ds = EmissionDataset(
        session_dir=args.d2_dir, row_start=min(rows), row_end=max(rows) + 1,
        capture_h=capture_hw[0], capture_w=capture_hw[1],
        emission_h=1080, emission_w=1920, session_id="D2", augment=False,
    )

    # Precompute predictions + clean emissions
    print(f"[probe] precomputing {len(rows)} predictions and clean emissions...", flush=True)
    t0 = time.time()
    pred_cache: dict[int, torch.Tensor] = {}
    clean_em_cache: dict[int, torch.Tensor] = {}
    streams_cache: dict[int, tuple[bytes, bytes, bytes]] = {}

    for i, t in enumerate(rows):
        if t not in ds.rows:
            continue
        sample = ds[ds.rows.index(t)]
        cap = sample["capture"].unsqueeze(0).to(device, dtype=dtype)
        # DDIM sample
        with torch.no_grad():
            pred = ddim_sample(model, cap, dc, args.K_steps, em_h_eff, em_w_eff, dtype)
        # Upsample prediction to native 1080×1920 for comparison
        if (em_h_eff, em_w_eff) != (1080, 1920):
            pred = F.interpolate(pred.float(), size=(1080, 1920),
                                  mode="bilinear", align_corners=False)
        pred_cache[t] = pred.float().clamp(0, 1).squeeze(0).cpu()

        # Clean emission via XOF rendering
        target_chain_row = t + 1
        s_next = bytes.fromhex(chain[target_chain_row])
        streams = expand_seeds_to_streams(s_next)
        streams_cache[t] = streams
        em_uint8 = render_from_streams(*streams, device=args.render_device)
        clean_em_cache[t] = em_uint8.float().div(255.0)

        if (i + 1) % 25 == 0:
            print(f"  precompute {i+1}/{len(rows)} elapsed={time.time()-t0:.0f}s", flush=True)

    print(f"[probe] precompute done; cached {len(pred_cache)} rows  elapsed={time.time()-t0:.0f}s",
          flush=True)
    rows = sorted(pred_cache.keys())

    # Sweep
    print(f"[probe] starting sweep ({len(rows)} rows × {len(NOISE_LEVELS)} p × "
          f"{len(SEEDS)} seeds = {len(rows)*len(NOISE_LEVELS)*len(SEEDS)} evals)", flush=True)
    per_row_evals: list[dict] = []
    for p_i, p in enumerate(NOISE_LEVELS):
        t0p = time.time()
        for t in rows:
            for s in SEEDS:
                seed = (s ^ t) & 0xFFFFFFFF
                if p == 0.0:
                    streams_n = streams_cache[t]
                else:
                    streams_n = noise_streams_byte_replace(streams_cache[t], p=p, seed=seed)
                em_n = render_from_streams(*streams_n, device=args.render_device).float().div(255.0)
                matched_score = psnr(pred_cache[t], clean_em_cache[t])
                noised_score = psnr(pred_cache[t], em_n)
                per_row_evals.append({
                    "p": p, "seed": s, "t": t,
                    "matched_psnr": matched_score,
                    "noised_psnr": noised_score,
                    "margin_db": matched_score - noised_score,
                    "attack_success": bool(noised_score >= matched_score - 1e-6),
                })
        n_so_far = (p_i + 1) * len(rows) * len(SEEDS)
        n_total = len(NOISE_LEVELS) * len(rows) * len(SEEDS)
        margins_p = [e["margin_db"] for e in per_row_evals if e["p"] == p]
        attack_p = [e["attack_success"] for e in per_row_evals if e["p"] == p]
        attack_rate = sum(attack_p) / max(len(attack_p), 1)
        ci = bootstrap_ci_rate(np.array(attack_p, dtype=int))
        print(f"  p={p:.4f}  n={len(margins_p)}  margin_mean={np.mean(margins_p):.3f} "
              f"attack_success={attack_rate:.3f} CI=[{ci[0]:.3f},{ci[1]:.3f}]  "
              f"({n_so_far}/{n_total}, {time.time()-t0p:.0f}s)", flush=True)

    # Aggregate per p
    aggregated = {}
    for p in NOISE_LEVELS:
        margins = np.array([e["margin_db"] for e in per_row_evals if e["p"] == p])
        attacks = np.array([e["attack_success"] for e in per_row_evals if e["p"] == p], dtype=int)
        ci = bootstrap_ci_rate(attacks)
        aggregated[str(p)] = {
            "n": int(len(attacks)),
            "attack_success_rate": float(attacks.mean()),
            "attack_success_ci95": list(ci),
            "margin_mean_db": float(margins.mean()),
            "margin_p5_db": float(np.percentile(margins, 5)),
            "margin_p95_db": float(np.percentile(margins, 95)),
        }

    # Threshold p@5%: smallest p where attack < 5%
    threshold_p = None
    for p in sorted(NOISE_LEVELS):
        if p == 0.0:
            continue
        if aggregated[str(p)]["attack_success_rate"] < 0.05:
            threshold_p = p
            break

    out = {
        "ckpt": str(args.ckpt),
        "K_steps": args.K_steps,
        "n_rows": len(rows),
        "noise_levels": list(NOISE_LEVELS),
        "seeds": list(SEEDS),
        "aggregated_per_p": aggregated,
        "threshold_p_at_5pct_attack_success": threshold_p,
    }
    (args.out / "chain_byte_sensitivity.json").write_text(json.dumps(out, indent=2))
    summary_lines = [
        f"H1 diffusion binder chain-byte sensitivity probe",
        f"  ckpt: {args.ckpt}",
        f"  K_steps: {args.K_steps}",
        f"  n_rows: {len(rows)}, seeds: {len(SEEDS)}",
        f"  threshold p @ 5% attack: {threshold_p}",
        f"",
        f"  p=0.000  attack=1.000  margin=0.0 dB",
    ]
    for p in NOISE_LEVELS:
        if p == 0.0:
            continue
        a = aggregated[str(p)]
        summary_lines.append(
            f"  p={p:.4f}  attack={a['attack_success_rate']:.3f}  "
            f"margin_mean={a['margin_mean_db']:+.2f} dB  "
            f"p5/p95=[{a['margin_p5_db']:+.2f},{a['margin_p95_db']:+.2f}]")
    (args.out / "chain_byte_threshold_summary.txt").write_text("\n".join(summary_lines))
    print(f"\n[done] wrote {args.out}/chain_byte_sensitivity.json + summary.txt", flush=True)
    print(f"  threshold p @ 5% = {threshold_p}", flush=True)


if __name__ == "__main__":
    main()

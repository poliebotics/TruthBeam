"""F-A v2 Preflight Test C — synthetic novel-E injection sweep.

METHOD NOTE (2026-05-04 — no upstream spec constrained the injection
mechanics; the design below is a deliberate choice):

Goal: "Test C: synthetic novel-E injection sweep across α and
blur scales."

Design: ablation that probes how robustly a binder detects E
content embedded in C, by synthetically constructing C_synth that injects a
novel-XOF E (rendered into a CFA-pattern lift) into a real C frame at varying
strength α and blur radius σ:

    C_synth(α, σ) = clip(C_real * (1 - α) + α * blur_σ( CFA_lift(E_novel) ), 0, 1)

  where CFA_lift broadcasts the (3, 1080, 1920) RGB E_novel into a (4, H, W)
  packed-CFA tensor by:
      ch_0 = R, ch_1 = G, ch_2 = G, ch_3 = B
  (matching the RGGB packing used by load_capture_at).

After resizing to each binder's capture HW, we compute:
    pred = binder(C_synth)        # (3, 1080, 1920)
    m_synth_target  = ||pred - E_target_real||²
    m_synth_novel   = ||pred - E_novel||²

The (α, σ) sweep tells us:
  - At what α does m_synth_novel begin to drop below m_synth_target?
    (i.e. the binder begins to recover the injected novel E from C_synth.)
  - How does that detection threshold shift with blur σ?

Compute: |α-grid| × |σ-grid| × n_frames × n_binders binder forwards.
With α-grid = 5, σ-grid = 4, n_frames = 20, n_binders = 14: 5,600 forwards.
~1 hr GPU.

Output: experiments/fa_v2_preflight_test_c/
    raw_scores.npz   — sweep[α, σ, frame, binder] of (m_target, m_novel)
    summary.json     — per-binder + aggregate detection threshold
    test_c_report.md — heatmaps + interpretation
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.emission_dataset import load_capture_at, load_emission_at  # noqa: E402
from phase_g.xof_perturb import (  # noqa: E402
    load_chain_log, expand_streams_from_s_t, render_streams_to_tile,
    verify_identity_render_parity,
)
from phase_g.diffusion_diagnostic_dataset import EVAL_BLOCKS  # noqa: E402

from models.emission_predictor import EmissionPredictor  # noqa: E402
try:
    from models.emission_predictor_v2 import EmissionPredictorV2  # noqa: E402
except ImportError:
    EmissionPredictorV2 = None


DEFAULT_OUT = Path("/path/to/poliebotics_phase_b/experiments/fa_v2_preflight_test_c")
DEFAULT_THRESHOLDS = Path(
    "/path/to/poliebotics_phase_b/experiments/phase_e/PHASE_E_THRESHOLDS.json")
SESSION_DIRS = {
    "D2":  Path("/path/to/poliebotics_phase_b/poliebotics_phase_b/data/d2"),
    "V10": Path("/path/to/poliebotics_phase_b/poliebotics_phase_b/data/v10"),
}
EMISSION_H_NATIVE, EMISSION_W_NATIVE = 1080, 1920
CAPTURE_H_NATIVE, CAPTURE_W_NATIVE = 2300, 2660

ALPHA_GRID  = (0.0, 0.05, 0.15, 0.30, 0.60)     # mixing weights
SIGMA_GRID  = (0.0, 5.0, 15.0, 30.0)            # gaussian blur kernel σ (pixels)

# Strict path-mode resolution
_PROBE = [SESSION_DIRS["D2"], SESSION_DIRS["V10"], DEFAULT_THRESHOLDS]
_AVAIL = sum(int(p.exists()) for p in _PROBE)
if _AVAIL == len(_PROBE):
    pass
elif _AVAIL == 0:
    LOCAL_ROOT = Path(__file__).resolve().parents[1]
    SESSION_DIRS = {"D2":  LOCAL_ROOT / "data" / "d2",
                    "V10": LOCAL_ROOT / "data" / "v10"}
    DEFAULT_THRESHOLDS = (LOCAL_ROOT / "experiments" / "phase_e"
                          / "PHASE_E_THRESHOLDS.json")
    DEFAULT_OUT = LOCAL_ROOT / "experiments" / "fa_v2_preflight_test_c"
else:
    raise SystemExit(f"[test_c] mixed root state: {_AVAIL}/{len(_PROBE)}")


def cfa_lift_E(e_rgb: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """(3, 1080, 1920) RGB → (4, target_h, target_w) packed CFA in [0, 1].

    CFA_lift(E_novel)[0] = R (resized), [1] = G, [2] = G, [3] = B.
    Followed by INTER_AREA resize to (target_h, target_w) per channel.
    """
    arr = e_rgb.detach().float().cpu().numpy()
    if arr.shape[1:] != (target_h, target_w):
        out = np.empty((3, target_h, target_w), dtype=np.float32)
        for c in range(3):
            out[c] = cv2.resize(arr[c], (target_w, target_h),
                                 interpolation=cv2.INTER_AREA)
        arr = out
    cfa = np.stack([arr[0], arr[1], arr[1], arr[2]], axis=0)  # (4, H, W)
    return torch.from_numpy(cfa)


def gaussian_blur_2d(t: torch.Tensor, sigma: float) -> torch.Tensor:
    """Per-channel gaussian blur of (C, H, W) tensor, σ in pixels.
    σ = 0.0 → identity (no blur). Implementation via cv2 (cheaper than torch
    on CPU)."""
    if sigma <= 0.0:
        return t
    arr = t.detach().float().cpu().numpy()
    out = np.empty_like(arr)
    # cv2 needs odd ksize; use 6σ rule
    ksize = max(3, int(6 * sigma) | 1)
    for c in range(arr.shape[0]):
        out[c] = cv2.GaussianBlur(arr[c], (ksize, ksize), sigmaX=sigma,
                                   sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    return torch.from_numpy(out)


def synthesize_C(C_real: torch.Tensor, E_novel: torch.Tensor,
                  alpha: float, sigma: float, target_h: int, target_w: int) -> torch.Tensor:
    """C_synth = clip(C_real * (1 - α) + α * blur_σ(CFA_lift(E_novel)), 0, 1).

    Inputs:
      C_real: (4, capture_H, capture_W) [0, 1]
      E_novel: (3, 1080, 1920) [0, 1]
    Returns: (4, target_h, target_w) [0, 1].
    """
    if C_real.shape[-2:] != (target_h, target_w):
        # Resize C_real to binder's capture HW
        arr = C_real.detach().float().cpu().numpy()
        out = np.empty((4, target_h, target_w), dtype=np.float32)
        for c in range(4):
            out[c] = cv2.resize(arr[c], (target_w, target_h),
                                 interpolation=cv2.INTER_AREA)
        C_real = torch.from_numpy(out)
    e_lifted = cfa_lift_E(E_novel, target_h, target_w)
    e_blurred = gaussian_blur_2d(e_lifted, sigma)
    C_synth = C_real * (1.0 - alpha) + alpha * e_blurred
    return C_synth.clamp(0, 1)


def sample_frames(n_total: int, seed: int = 0) -> list[tuple[str, int]]:
    rs = np.random.RandomState(seed)
    n_d2 = n_total // 2
    n_v10 = n_total - n_d2
    out = []
    for sess, n_target in (("D2", n_d2), ("V10", n_v10)):
        sd = SESSION_DIRS[sess]
        chain = load_chain_log(sd)
        blocks = EVAL_BLOCKS[sess]
        valid = []
        for a, b in blocks:
            for r in range(a, b):
                if r in chain and (sd / "Recordings" / f"frame_{r:06d}.raw").exists():
                    valid.append(r)
        if not valid: continue
        picked = rs.choice(valid, size=min(n_target, len(valid)), replace=False)
        for r in sorted(picked.tolist()):
            out.append((sess, int(r)))
    return out


def load_binder(spec: dict, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    arch = spec["arch"]
    if arch == "EmissionPredictor":
        m = EmissionPredictor(emission_h=EMISSION_H_NATIVE, emission_w=EMISSION_W_NATIVE,
                              pretrained=False)
    elif arch == "EmissionPredictorV2":
        if EmissionPredictorV2 is None:
            raise RuntimeError("EmissionPredictorV2 import failed")
        m = EmissionPredictorV2(emission_h=EMISSION_H_NATIVE, emission_w=EMISSION_W_NATIVE,
                                pretrained=False)
    else:
        raise ValueError(arch)
    ckpt = Path(spec["ckpt"])
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    state = {k[len("module."):] if k.startswith("module.") else k: v
             for k, v in state.items()}
    missing, unexpected = m.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"binder ckpt {ckpt.name} mismatch: missing={list(missing)[:3]} "
                           f"unexpected={list(unexpected)[:3]}")
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m.to(device)


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds-json", type=Path, default=DEFAULT_THRESHOLDS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-frames", type=int, default=20)
    ap.add_argument("--n-novel", type=int, default=3,
                    help="# novel S_t hex values (one per frame, but pool size for variety).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--novel-seed-bytes", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--dtype", type=str, default="bf16", choices=("bf16", "fp32"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device("cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" and device.type == "cuda" else torch.float32
    print(f"[test_c] device={device} dtype={dtype}")

    # Parity gate
    print("[test_c] parity gate...")
    for sess, sd in SESSION_DIRS.items():
        chain = load_chain_log(sd)
        if not chain: continue
        samples = [f for f in (100, 1500, 3000) if f in chain][:3]
        if not samples:
            samples = [sorted(chain.keys())[len(chain) // 2]]
        ok, details = verify_identity_render_parity(sd, chain, samples, tol=1e-6)
        print(f"  {sess}: max={details['max_overall']:.2e}  ok={ok}")
        if not ok: return 1

    th = json.loads(args.thresholds_json.read_text())
    binders_meta = th["binders"]
    binder_names = sorted(binders_meta.keys())
    print(f"[test_c] {len(binder_names)} binders")

    # Sample frames
    sampled = sample_frames(args.n_frames, seed=args.seed)
    n_frames = len(sampled)
    print(f"[test_c] {n_frames} frames sampled")

    # Generate novel E (one per frame, repeated if pool < n_frames)
    if args.novel_seed_bytes:
        from blake3 import blake3
        seed_bytes = bytes.fromhex(args.novel_seed_bytes)
        novel_s_t = [blake3(seed_bytes + i.to_bytes(8, "big")).digest(length=32).hex()
                     for i in range(args.n_novel)]
    else:
        novel_s_t = [os.urandom(32).hex() for _ in range(args.n_novel)]
    print(f"[test_c] generated {len(novel_s_t)} novel S_t values")

    # Pre-render the novel E variants
    novel_E_list = []
    for sth in novel_s_t:
        streams = expand_streams_from_s_t(sth)
        tile = render_streams_to_tile(streams, device="cpu")  # (3, 1080, 1920) uint8
        novel_E_list.append(tile.float() / 255.0)

    # Sweep grid
    n_alpha = len(ALPHA_GRID)
    n_sigma = len(SIGMA_GRID)
    n_b = len(binder_names)
    # results: (n_alpha, n_sigma, n_frames, n_b, 2)  — last dim: (m_target, m_novel)
    results = np.full((n_alpha, n_sigma, n_frames, n_b, 2), np.nan, dtype=np.float64)
    print(f"[test_c] sweep shape: {results.shape}; "
          f"total binder forwards = {n_alpha * n_sigma * n_frames * n_b}")

    overall_t0 = time.time()
    for bi, bname in enumerate(binder_names):
        spec = binders_meta[bname]["spec"]
        # halt on binder load failure rather than
        # silently leave NaN result planes.
        binder = load_binder(spec, device, dtype)
        cap_h, cap_w = spec["capture_h"], spec["capture_w"]
        b_t0 = time.time()
        print(f"  [{bi+1}/{n_b}] {bname} cap={cap_h}x{cap_w}", flush=True)
        for fi, (sess, target_row) in enumerate(sampled):
            sd = SESSION_DIRS[sess]
            C_real = load_capture_at(sd / "Recordings" / f"frame_{target_row:06d}.raw",
                                      cap_h, cap_w)
            E_target = load_emission_at(
                sd / "derived" / "Emissions" / f"tile_{target_row:06d}.png",
                EMISSION_H_NATIVE, EMISSION_W_NATIVE)
            E_novel = novel_E_list[fi % len(novel_E_list)]
            for ai, alpha in enumerate(ALPHA_GRID):
                for si, sigma in enumerate(SIGMA_GRID):
                    C_synth = synthesize_C(C_real, E_novel, alpha, sigma, cap_h, cap_w)
                    cap_b = C_synth.unsqueeze(0).to(device, dtype=dtype)
                    with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu",
                                             dtype=dtype, enabled=(dtype != torch.float32)):
                        pred = binder(cap_b).float().clamp(0, 1).squeeze(0).cpu()
                    results[ai, si, fi, bi, 0] = float(((pred - E_target) ** 2).mean())
                    results[ai, si, fi, bi, 1] = float(((pred - E_novel)  ** 2).mean())
        del binder
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"    {bname} done in {time.time()-b_t0:.0f}s")

    print(f"[test_c] full sweep done in {time.time()-overall_t0:.0f}s")

    # Save
    np.savez_compressed(args.out / "raw_scores.npz",
                        results=results,
                        alpha_grid=np.array(ALPHA_GRID),
                        sigma_grid=np.array(SIGMA_GRID),
                        binder_names=np.array(binder_names),
                        sessions=np.array([s for s, _ in sampled]),
                        rows=np.array([r for _, r in sampled]),
                        novel_s_t=np.array(novel_s_t))

    # Aggregate: for each binder, find detection threshold (alpha at which
    # m_synth_novel < m_synth_target, averaged over frames) at each sigma.
    summary = {"alpha_grid": list(ALPHA_GRID), "sigma_grid": list(SIGMA_GRID),
               "binders": {}}
    for bi, bname in enumerate(binder_names):
        m_t = np.nanmean(results[:, :, :, bi, 0], axis=2)  # (n_alpha, n_sigma)
        m_n = np.nanmean(results[:, :, :, bi, 1], axis=2)
        summary["binders"][bname] = {
            "mean_m_target_per_alpha_sigma": m_t.tolist(),
            "mean_m_novel_per_alpha_sigma":  m_n.tolist(),
            # Detection threshold per sigma: smallest alpha at which m_n < m_t
            "alpha_detect_per_sigma": [
                next((float(a) for ai_, a in enumerate(ALPHA_GRID)
                      if m_n[ai_, si] < m_t[ai_, si]), None)
                for si in range(n_sigma)
            ],
        }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    md = []
    md.append("# F-A v2 Preflight Test C — synthetic novel-E injection sweep")
    md.append(f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    md.append(f"α grid: {list(ALPHA_GRID)}")
    md.append(f"σ grid: {list(SIGMA_GRID)}")
    md.append(f"n_frames={n_frames}, n_novel_E={len(novel_E_list)}")
    md.append("")
    md.append("**METHOD NOTE**: injection mechanics are a deliberate design choice; see the script docstring.")
    md.append("")
    md.append("## Detection threshold per binder (α at which mean m_novel drops below mean m_target)")
    md.append("")
    md.append("| binder | σ=0 | σ=5 | σ=15 | σ=30 |")
    md.append("|---|---|---|---|---|")
    for bname in sorted(summary["binders"].keys()):
        v = summary["binders"][bname]
        cells = [f"{a:.2f}" if a is not None else "—"
                 for a in v["alpha_detect_per_sigma"]]
        md.append(f"| {bname} | " + " | ".join(cells) + " |")
    md.append("")
    md.append("## Interpretation")
    md.append("- Lower α-threshold means binder picks up the injected novel-E more sensitively.")
    md.append("- Larger σ should generally raise the threshold (more blur destroys the structure binders learned).")
    md.append("- Cells with '—' indicate no α in the grid triggered detection (binder ignored injection at this σ).")
    md.append("")
    (args.out / "test_c_report.md").write_text("\n".join(md))
    print(f"[test_c] report → {args.out / 'test_c_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

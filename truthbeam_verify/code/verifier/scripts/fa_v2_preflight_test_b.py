"""F-A v2 Preflight Test B — binder(C_fake_v1) vs E_target.

Question: when binders score F-A v1's C_fake (using the existing 13/14 Phase E
binders), does the resulting MSE land near m_real_target (F-A v1 fools the
binders) or near m_shuffled (F-A v1 doesn't fool binders)?

Interpretation:
    m_fake_target ≈ m_real_target → F-A v1 already saturates binder loss.
                                     F-A v2 binder loss training signal will
                                     be weak: there's nothing left to push.
    m_fake_target ≈ m_shuffled    → F-A v1 fakes look like wrong-pair to
                                     binders. F-A v2 binder loss still has
                                     useful gradient.
    in between                    → partial saturation; F-A v2 will get
                                     diminishing returns.

Pipeline:
    For each held-out (session, target_row):
        - C_source, E_source, E_target_real loaded native.
        - C_fake_native = F-A v1(C_source, E_source, E_target_real)
                       at (4, 2300, 2660).
        - For each binder b:
            - Resize C_fake_native to (b.capture_h, b.capture_w).
            - pred = binder(C_fake_resized)  → (3, 1080, 1920)
            - m_fake_target = ||pred - E_target_real||²

Compute: 100 frames × (1 F-A inference + 14 binder forwards) ≈ 1500 forward
passes total. ~1-2 hr on single GPU; ~30 min if batched (we don't batch here
because per-binder resolutions differ).

Reference comparison: this script also re-loads the existing binder eval
results from `experiments/binder_novel_eval/raw_scores.npz` to provide
m_real_target as the baseline. (binder_novel_xof_eval.py must have completed.)

Output: experiments/fa_v2_preflight_test_b/
    raw_scores.npz       — frames × binders → m_fake_target
    summary.json         — per-binder + aggregate stats
    test_b_report.md     — interpretation
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
from phase_g.fa_loader import (  # noqa: E402
    load_fa_v1_checkpoint, load_C_native, load_E_native,
)
from phase_g.xof_perturb import load_chain_log, verify_identity_render_parity  # noqa: E402
from phase_g.diffusion_diagnostic_dataset import EVAL_BLOCKS  # noqa: E402

from models.emission_predictor import EmissionPredictor  # noqa: E402
try:
    from models.emission_predictor_v2 import EmissionPredictorV2  # noqa: E402
except ImportError:
    EmissionPredictorV2 = None


# -------- defaults --------

DEFAULT_OUT = Path("/path/to/poliebotics_phase_b/experiments/fa_v2_preflight_test_b")
DEFAULT_THRESHOLDS = Path(
    "/path/to/poliebotics_phase_b/experiments/phase_e/PHASE_E_THRESHOLDS.json")
DEFAULT_FA_CKPT = Path(
    "/path/to/poliebotics_phase_b/experiments/phase_f/"
    "f_a_full_v1/checkpoints/step_00100000.pt")
DEFAULT_BINDER_REF = Path(
    "/path/to/poliebotics_phase_b/experiments/binder_novel_eval/raw_scores.npz")
SESSION_DIRS = {
    "D2":  Path("/path/to/poliebotics_phase_b/poliebotics_phase_b/data/d2"),
    "V10": Path("/path/to/poliebotics_phase_b/poliebotics_phase_b/data/v10"),
}

EMISSION_H_NATIVE, EMISSION_W_NATIVE = 1080, 1920
SOURCE_LAG = 2

# Strict path-mode resolution
_PROBE = [SESSION_DIRS["D2"], SESSION_DIRS["V10"], DEFAULT_FA_CKPT, DEFAULT_THRESHOLDS]
_AVAIL = sum(int(p.exists()) for p in _PROBE)
if _AVAIL == len(_PROBE):
    pass
elif _AVAIL == 0:
    LOCAL_ROOT = Path(__file__).resolve().parents[1]
    SESSION_DIRS = {"D2":  LOCAL_ROOT / "data" / "d2",
                    "V10": LOCAL_ROOT / "data" / "v10"}
    DEFAULT_FA_CKPT = (LOCAL_ROOT / "experiments" / "phase_f" / "f_a_full_v1"
                       / "checkpoints" / "step_00100000.pt")
    DEFAULT_THRESHOLDS = (LOCAL_ROOT / "experiments" / "phase_e"
                          / "PHASE_E_THRESHOLDS.json")
    DEFAULT_OUT = LOCAL_ROOT / "experiments" / "fa_v2_preflight_test_b"
    DEFAULT_BINDER_REF = (LOCAL_ROOT / "experiments" / "binder_novel_eval"
                          / "raw_scores.npz")
else:
    raise SystemExit(f"[test_b] mixed root state: {_AVAIL}/{len(_PROBE)}")


def sample_frames(n_total: int = 50, seed: int = 0) -> list[tuple[str, int]]:
    """50 frames stratified across D2+V10 held-out blocks (smaller than binder
    eval's 100 since each frame here costs ~15× more compute per binder due to
    extra F-A forward + per-binder resize)."""
    rs = np.random.RandomState(seed)
    n_d2 = n_total // 2
    n_v10 = n_total - n_d2
    out = []
    for sess, n_target in (("D2", n_d2), ("V10", n_v10)):
        sd = SESSION_DIRS[sess]
        chain = load_chain_log(sd)
        blocks = EVAL_BLOCKS[sess]
        per_block = []
        for a, b in blocks:
            valid = []
            for r in range(a, b):
                # Need source row (r - SOURCE_LAG) too
                if (r in chain and (r - SOURCE_LAG) in chain
                        and (sd / "Recordings" / f"frame_{r:06d}.raw").exists()
                        and (sd / "Recordings" / f"frame_{r-SOURCE_LAG:06d}.raw").exists()):
                    valid.append(r)
            per_block.append(valid)
        n_per = [n_target // len(blocks)] * len(blocks)
        for i in range(n_target % len(blocks)):
            n_per[i] += 1
        for n_b, valid in zip(n_per, per_block):
            if not valid: continue
            picked = rs.choice(valid, size=min(n_b, len(valid)), replace=False)
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


def resize_packed_cfa(cfa_native: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """(4, 2300, 2660) → (4, target_h, target_w) via INTER_AREA per channel.
    Matches load_capture_at convention (used by EmissionDataset)."""
    if cfa_native.shape[-2:] == (target_h, target_w):
        return cfa_native
    arr = cfa_native.detach().float().cpu().numpy()
    out = np.empty((4, target_h, target_w), dtype=np.float32)
    for c in range(4):
        out[c] = cv2.resize(arr[c], (target_w, target_h), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(out)


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fa-ckpt", type=Path, default=DEFAULT_FA_CKPT)
    ap.add_argument("--thresholds-json", type=Path, default=DEFAULT_THRESHOLDS)
    ap.add_argument("--binder-eval-ref", type=Path, default=DEFAULT_BINDER_REF)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-frames", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None,
                    help="cuda:N or cpu. Default: CPU (Phase H must finish first for GPU).")
    ap.add_argument("--dtype", type=str, default="bf16", choices=("bf16", "fp32"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" and device.type == "cuda" else torch.float32
    print(f"[test_b] device={device} dtype={dtype}")

    # ---- parity gate ----
    print("[test_b] parity gate...")
    for sess, sd in SESSION_DIRS.items():
        chain = load_chain_log(sd)
        if not chain: continue
        samples = [f for f in (100, 1500, 3000) if f in chain][:3]
        if not samples:
            samples = [sorted(chain.keys())[len(chain) // 2]]
        ok, details = verify_identity_render_parity(sd, chain, samples, tol=1e-6)
        print(f"  {sess}: max={details['max_overall']:.2e}  ok={ok}")
        if not ok: return 1

    # ---- load F-A ----
    print("[test_b] loading F-A v1...")
    fa_model = load_fa_v1_checkpoint(args.fa_ckpt, device, dtype)
    print(f"  loaded ({sum(p.numel() for p in fa_model.parameters())/1e6:.1f}M params)")

    # ---- load binder registry ----
    th = json.loads(args.thresholds_json.read_text())
    binders_meta = th["binders"]
    binder_names = sorted(binders_meta.keys())
    print(f"[test_b] {len(binder_names)} binders: {binder_names}")

    # ---- load reference binder eval (for m_real_target baseline) ----
    ref_means = None  # (binder_name → mean m_real_target)
    if args.binder_eval_ref.exists():
        ref = np.load(args.binder_eval_ref, allow_pickle=False)
        ref_binder_names = list(ref["binder_names"])
        ref_mse = ref["mse_mat"]      # (frames, conds, binders)
        ref_cond_labels = list(ref["cond_labels"])
        ref_target_idx = ref_cond_labels.index("target")
        ref_means = {bname: float(np.nanmean(ref_mse[:, ref_target_idx, bi]))
                     for bi, bname in enumerate(ref_binder_names)}
        print(f"[test_b] loaded m_real_target reference for {len(ref_means)} binders")
    else:
        print(f"[test_b] WARN: reference {args.binder_eval_ref} not found; "
              "no m_real_target baseline (still produces m_fake_target).")

    # ---- sample frames ----
    sampled = sample_frames(args.n_frames, seed=args.seed)
    n_frames = len(sampled)
    print(f"[test_b] {n_frames} frames sampled "
          f"({sum(1 for s,_ in sampled if s == 'D2')} D2 / "
          f"{sum(1 for s,_ in sampled if s == 'V10')} V10)")

    # ---- main loop ----
    n_b = len(binder_names)
    m_fake_target = np.full((n_frames, n_b), np.nan, dtype=np.float64)

    print(f"[test_b] running {n_frames * (1 + n_b)} forwards "
          f"({n_frames} F-A + {n_frames * n_b} binder)...")

    # Cache C_fake at native res per frame so we can re-use across binders
    overall_t0 = time.time()
    C_fake_natives = []
    E_target_natives = []
    for fi, (sess, target_row) in enumerate(sampled):
        sd = SESSION_DIRS[sess]
        source_row = target_row - SOURCE_LAG
        C_s = load_C_native(sd, source_row).to(device, dtype=dtype).unsqueeze(0)
        E_s = load_E_native(sd, source_row).to(device, dtype=dtype).unsqueeze(0)
        E_t = load_E_native(sd, target_row).to(device, dtype=dtype).unsqueeze(0)
        C_pred = fa_model(C_s, E_s, E_t)  # (1, 4, 2300, 2660)
        if tuple(C_pred.shape) != (1, 4, 2300, 2660):
            raise RuntimeError(f"F-A output shape {tuple(C_pred.shape)} unexpected")
        C_fake_natives.append(C_pred.squeeze(0).float().cpu().clamp(0, 1))  # (4,2300,2660)
        E_target_natives.append(load_emission_at(
            sd / "derived" / "Emissions" / f"tile_{target_row:06d}.png",
            EMISSION_H_NATIVE, EMISSION_W_NATIVE))
        if (fi + 1) % 10 == 0:
            print(f"  F-A {fi+1}/{n_frames} elapsed={time.time()-overall_t0:.0f}s",
                  flush=True)
    fa_t = time.time() - overall_t0
    print(f"[test_b] all F-A inferences done in {fa_t:.0f}s")
    del fa_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for bi, bname in enumerate(binder_names):
        spec = binders_meta[bname]["spec"]
        # binder load failure should halt rather
        # than silently produce NaN rows — otherwise downstream summary masks
        # broken binders. Raise to caller.
        binder = load_binder(spec, device, dtype)
        cap_h, cap_w = spec["capture_h"], spec["capture_w"]
        b_t0 = time.time()
        print(f"  [{bi+1}/{n_b}] {bname} cap={cap_h}x{cap_w}", flush=True)
        for fi, (sess, target_row) in enumerate(sampled):
            cfa = resize_packed_cfa(C_fake_natives[fi], cap_h, cap_w)
            cap_b = cfa.unsqueeze(0).to(device, dtype=dtype)
            with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu",
                                     dtype=dtype, enabled=(dtype != torch.float32)):
                pred = binder(cap_b).float().clamp(0, 1).squeeze(0).cpu()
            E_t = E_target_natives[fi]
            m_fake_target[fi, bi] = float(((pred - E_t) ** 2).mean())
        del binder
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"    {bname} done in {time.time()-b_t0:.0f}s  "
              f"mean m_fake_target={np.nanmean(m_fake_target[:, bi]):.5f}")

    # ---- save raw + summary ----
    np.savez_compressed(args.out / "raw_scores.npz",
                        m_fake_target=m_fake_target,
                        binder_names=np.array(binder_names),
                        sessions=np.array([s for s, _ in sampled]),
                        rows=np.array([r for _, r in sampled]))

    summary = {"n_frames": n_frames, "binders": {}}
    for bi, bname in enumerate(binder_names):
        col = m_fake_target[:, bi]
        if np.isnan(col).all():
            summary["binders"][bname] = {"loaded": False}
            continue
        m_fake = float(np.nanmean(col))
        m_real = ref_means.get(bname, None) if ref_means else None
        ratio = (m_fake / m_real) if m_real and m_real > 0 else None
        summary["binders"][bname] = {
            "loaded": True,
            "mean_m_fake_target": m_fake,
            "std_m_fake_target":  float(np.nanstd(col)),
            "mean_m_real_target_ref": m_real,
            "fake_over_real_ratio": ratio,
        }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- report ----
    md = []
    md.append("# F-A v2 Preflight Test B — binder(C_fake_v1) vs E_target")
    md.append("")
    md.append(f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    md.append(f"F-A ckpt: {args.fa_ckpt.name}")
    md.append(f"n_frames: {n_frames}")
    md.append("")
    md.append("| binder | mean m_fake_target | mean m_real_target (ref) | "
              "ratio fake/real | interpretation |")
    md.append("|---|---|---|---|---|")
    for bname in sorted(summary["binders"].keys()):
        v = summary["binders"][bname]
        if not v.get("loaded"):
            md.append(f"| {bname} | — | — | — | (load failed) |")
            continue
        m_fake = v["mean_m_fake_target"]
        m_real = v["mean_m_real_target_ref"]
        ratio = v["fake_over_real_ratio"]
        if ratio is None:
            interp = "no real ref"
        elif ratio < 1.5:
            interp = "F-A fools binder (saturation risk)"
        elif ratio < 5:
            interp = "partial fool (intermediate)"
        else:
            interp = "binder rejects F-A fakes (good gradient for v2)"
        # format conditional precedence — build
        # the strings first, don't try to interleave conditional inside f-string.
        m_real_s = f"{m_real:.5f}" if m_real is not None else "NA"
        ratio_s  = f"{ratio:.2f}"  if ratio  is not None else "NA"
        md.append(f"| {bname} | {m_fake:.5f} | {m_real_s} | {ratio_s} | {interp} |")
    md.append("")
    md.append("## Interpretation guide")
    md.append("- **ratio < 1.5**: F-A v1 already produces C_fake that binder maps near E_target (saturation). F-A v2 binder loss has little headroom on this binder.")
    md.append("- **ratio 1.5-5**: partial saturation. Binder loss has shrinking but useful gradient.")
    md.append("- **ratio > 5**: binders cleanly reject F-A v1 fakes. F-A v2 binder loss has full gradient available.")
    md.append("")
    (args.out / "test_b_report.md").write_text("\n".join(md))
    print(f"[test_b] report → {args.out / 'test_b_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

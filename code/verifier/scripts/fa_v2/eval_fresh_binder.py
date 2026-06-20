"""Fresh binder acceptance evaluator (F-A v2 prep).

Runs all acceptance criteria on a trained fresh-binder ckpt:
    - matched-PSNR ≥ 22 dB on (correct_E vs binder(C_real)) over held-out test split
    - AUROC(correct_E vs ±2/±15/+30 wrong-frame E) ≥ 0.95
    - output range ≥ 0.3 (max-min over channels) — catches collapse
    - no NaN/Inf in any forward pass
    - 5 qualitative spot-check frames saved as PNGs (visual review)

Pass/fail JSON + report.md written to <ckpt_parent>/acceptance/.

Usage:
    python scripts/fa_v2/eval_fresh_binder.py \
        --ckpt experiments/fa_v2/fresh_binders/A1/model_final.pt \
        --d2-dir data/d2 --v10-dir data/v10
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
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.emission_dataset import load_capture_at, load_emission_at  # noqa: E402
from models.fresh import FRESH_BINDER_REGISTRY  # noqa: E402


# Held-out acceptance split (post-training, never seen during training/val)
ACCEPTANCE_RANGES = {
    "D2":  (5394, 5992),
    "V10": (2300, 2950),
}

# Wrong-frame offsets for AUROC
WRONG_OFFSETS = (-2, +2, -15, +15, +30)

# Acceptance thresholds (per operator approval 2026-05-03)
PSNR_THRESHOLD_DB = 22.0
AUROC_THRESHOLD = 0.95
OUTPUT_RANGE_THRESHOLD = 0.30


def psnr01(p: torch.Tensor, t: torch.Tensor) -> float:
    mse = ((p - t) ** 2).mean().item()
    if mse < 1e-12: return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def load_binder(ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    # weights_only=False is needed because cfg dict
    # is in the ckpt; the threat model assumes operator-trusted ckpts only.
    # If running on third-party ckpts, switch to weights_only=True and parse
    # cfg from a sibling JSON file.
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["args"]
    family = cfg["family"]
    model_cls = FRESH_BINDER_REGISTRY[family]
    # override pretrained=False at eval-time
    # reconstruction. The timm encoders' weights are loaded from the ckpt
    # state_dict regardless, so initial pretrain download is wasted compute
    # (and a Lambda offline-mode failure surface). Family D doesn't use this
    # kwarg.
    model_kwargs = dict(cfg.get("model_kwargs", {}))
    if family in ("A", "B", "C"):
        model_kwargs["pretrained"] = False
    m = model_cls(**model_kwargs)
    m.load_state_dict(ck["model"], strict=True)
    m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m.to(device), cfg


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--d2-dir", type=Path, required=True)
    ap.add_argument("--v10-dir", type=Path, default=None)
    ap.add_argument("--n-frames-per-session", type=int, default=50,
                    help="Sub-sample for AUROC (full set for PSNR).")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Default: <ckpt_parent>/acceptance/")
    args = ap.parse_args()

    if not args.ckpt.exists():
        raise SystemExit(f"ckpt not found: {args.ckpt}")
    out_dir = args.out_dir or (args.ckpt.parent / "acceptance")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.bf16 and device.type == "cuda" else torch.float32

    binder, cfg = load_binder(args.ckpt, device)
    cap_h, cap_w = cfg["capture_h"], cfg["capture_w"]
    binder_id = cfg["binder_id"]
    family = cfg["family"]
    print(f"[acc-eval] {binder_id} (family {family}) cap={cap_h}x{cap_w}")

    session_dirs = {"D2": args.d2_dir}
    if args.v10_dir is not None and args.v10_dir.exists():
        session_dirs["V10"] = args.v10_dir

    # ---- 1. matched-PSNR + AUROC across sessions ----
    psnrs_per_sess: dict[str, list[float]] = {}
    aurocs_per_sess: dict[str, dict[str, float]] = {}
    output_ranges: list[float] = []
    nan_count = 0
    qualitative_saved = 0

    rs = np.random.RandomState(0)

    for sess, sd in session_dirs.items():
        if sess not in ACCEPTANCE_RANGES: continue
        a, b = ACCEPTANCE_RANGES[sess]
        rec_dir = sd / "Recordings"
        em_dir  = sd / "derived" / "Emissions"
        # Valid rows: ones with both .raw and tile_*.png on disk
        valid_rows = []
        for r in range(a, b):
            if ((rec_dir / f"frame_{r:06d}.raw").exists()
                    and (em_dir / f"tile_{r:06d}.png").exists()):
                valid_rows.append(r)
        n_valid = len(valid_rows)
        print(f"  {sess}: {n_valid} valid rows in [{a}, {b})")
        if n_valid == 0: continue

        # PSNR over all valid rows (matched-pair)
        psnrs = []
        # AUROC: subset for speed
        n_sub = min(args.n_frames_per_session, n_valid)
        sub_rows = rs.choice(valid_rows, size=n_sub, replace=False)
        # For AUROC: store correct + wrong-* MSEs
        mse_correct = []
        mse_wrong: dict[int, list[float]] = {o: [] for o in WRONG_OFFSETS}

        for r in valid_rows:
            cap = load_capture_at(rec_dir / f"frame_{r:06d}.raw", cap_h, cap_w)
            cap_b = cap.unsqueeze(0).to(device, dtype=dtype)
            with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu",
                                     dtype=dtype, enabled=(dtype != torch.float32)):
                pred = binder(cap_b).float().clamp(0, 1).squeeze(0).cpu()
            if not torch.isfinite(pred).all():
                nan_count += 1
                continue
            output_ranges.append(float(pred.max() - pred.min()))
            em_target = load_emission_at(em_dir / f"tile_{r:06d}.png", 1080, 1920)
            psnrs.append(psnr01(pred, em_target))

            if r in sub_rows:
                m_c = ((pred - em_target) ** 2).mean().item()
                mse_correct.append(m_c)
                for off in WRONG_OFFSETS:
                    rp = r + off
                    if (em_dir / f"tile_{rp:06d}.png").exists():
                        em_w = load_emission_at(em_dir / f"tile_{rp:06d}.png", 1080, 1920)
                        m_w = ((pred - em_w) ** 2).mean().item()
                        mse_wrong[off].append(m_w)

            # Qualitative spot check: first 5 frames
            if qualitative_saved < 5:
                arr = (pred.numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_dir / f"qual_{sess}_row{r:06d}.png"), bgr)
                qualitative_saved += 1
        psnrs_per_sess[sess] = psnrs
        # AUROC: target=correct (lower MSE, label=1), negative=wrong-* (label=0)
        aurocs_per_sess[sess] = {}
        for off, mse_w_list in mse_wrong.items():
            n = min(len(mse_correct), len(mse_w_list))
            if n < 5: continue
            scores = np.concatenate([mse_correct[:n], mse_w_list[:n]])
            labels = np.concatenate([np.ones(n), np.zeros(n)])
            try:
                auc = float(roc_auc_score(labels, -scores))
            except ValueError:
                auc = float("nan")
            aurocs_per_sess[sess][f"correct_vs_wrong_{off:+d}"] = auc

    # ---- aggregate verdict ----
    verdict = {"binder_id": binder_id, "family": family,
                "ckpt": str(args.ckpt), "checks": {}}
    overall_pass = True

    # 1. matched-PSNR
    psnrs_all = []
    for ps in psnrs_per_sess.values():
        psnrs_all.extend([p for p in ps if math.isfinite(p)])
    if psnrs_all:
        psnr_mean = float(np.mean(psnrs_all))
    else:
        psnr_mean = float("nan")
    psnr_pass = math.isfinite(psnr_mean) and psnr_mean >= PSNR_THRESHOLD_DB
    verdict["checks"]["matched_psnr"] = {
        "mean_db": psnr_mean, "threshold_db": PSNR_THRESHOLD_DB,
        "n_frames": len(psnrs_all), "pass": psnr_pass}
    overall_pass = overall_pass and psnr_pass

    # 2. AUROC (require all wrong-offsets to pass on D2; V10 advisory)
    auroc_d2 = aurocs_per_sess.get("D2", {})
    auroc_min_d2 = min(auroc_d2.values()) if auroc_d2 else float("nan")
    auroc_pass = math.isfinite(auroc_min_d2) and auroc_min_d2 >= AUROC_THRESHOLD
    verdict["checks"]["auroc_correct_vs_wrong"] = {
        "min_d2": auroc_min_d2, "all_d2": auroc_d2, "all_v10": aurocs_per_sess.get("V10", {}),
        "threshold": AUROC_THRESHOLD, "pass": auroc_pass}
    overall_pass = overall_pass and auroc_pass

    # 3. output range
    range_min = float(np.min(output_ranges)) if output_ranges else 0.0
    range_pass = range_min >= OUTPUT_RANGE_THRESHOLD
    verdict["checks"]["output_range"] = {
        "min_observed": range_min, "threshold": OUTPUT_RANGE_THRESHOLD, "pass": range_pass}
    overall_pass = overall_pass and range_pass

    # 4. no NaN
    no_nan_pass = nan_count == 0
    verdict["checks"]["no_nan"] = {"n_frames_with_nan": nan_count, "pass": no_nan_pass}
    overall_pass = overall_pass and no_nan_pass

    # 5. qualitative (manual review — flag count saved)
    verdict["checks"]["qualitative_pngs_saved"] = qualitative_saved

    verdict["overall_pass"] = overall_pass
    verdict["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2))

    # Markdown report
    md = [f"# Fresh binder acceptance — {binder_id} (family {family})", ""]
    md.append(f"Generated: {verdict['generated_utc']}")
    md.append(f"Ckpt: `{args.ckpt}`")
    md.append("")
    md.append(f"## Overall: {'✅ PASS' if overall_pass else '❌ FAIL'}")
    md.append("")
    md.append(f"- matched-PSNR mean: {psnr_mean:.2f} dB (threshold ≥ {PSNR_THRESHOLD_DB} dB) — "
              f"{'✅' if psnr_pass else '❌'}")
    md.append(f"- AUROC(correct vs wrong-offset) min on D2: "
              f"{auroc_min_d2:.4f} (threshold ≥ {AUROC_THRESHOLD}) — "
              f"{'✅' if auroc_pass else '❌'}")
    md.append(f"- output range min: {range_min:.3f} (threshold ≥ {OUTPUT_RANGE_THRESHOLD}) — "
              f"{'✅' if range_pass else '❌'}")
    md.append(f"- NaN/Inf in {nan_count} frames — {'✅' if no_nan_pass else '❌'}")
    md.append(f"- qualitative spot-check PNGs: {qualitative_saved} saved (manual review)")
    md.append("")
    md.append("## Per-session AUROC detail")
    md.append("")
    for sess, aurocs in aurocs_per_sess.items():
        md.append(f"### {sess}")
        for k, v in aurocs.items():
            md.append(f"- {k}: {v:.4f}")
        md.append("")
    (out_dir / "report.md").write_text("\n".join(md))
    print(f"[acc-eval] DONE — overall_pass={overall_pass}")
    print(f"[acc-eval] report → {out_dir / 'report.md'}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

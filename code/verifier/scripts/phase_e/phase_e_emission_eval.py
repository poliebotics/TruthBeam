"""Phase E emission evaluation — handles P0, P1 controls, and per-epoch val.

Modes:
  ckpt        — load EmissionPredictor or EmissionPredictorV2 ckpt
  oracle      — prediction = matched ground-truth (passes all sanity checks)
  null_mean   — prediction = train-set mean emission tile (constant, fitted)
  null_grey   — prediction = constant 0.5 grey

Methodologies:
  legacy      — exp001c's 50 random uniformly-spaced val rows × 1 random
                non-matching mismatch each. Matches `cross_pair_psnr.json`.
  family_balanced — Phase D-style 4 negative families (delay_window /
                near_shift / same_session_random / cross_session_random).
                Reports per-family Window-FMR@5 (since PSNR is higher=better,
                tau5 is the 5th percentile of matched PSNR; FMR = fraction
                of frames where any negative ≥ tau5).

Splits supported:
  d2_orig     — Phase B original val [5394, 5992)
  d2_phase_d  — Phase D val [4792, 5992)
  d2_phase_d_calib  — Phase D val first half [4792, 5392)
  d2_phase_d_report — Phase D val second half [5392, 5992)

Run:
  python scripts/phase_e/phase_e_emission_eval.py \
    --mode ckpt --ckpt experiments/exp001c/checkpoints/ep027.pt \
    --arch EmissionPredictor \
    --capture-h 1150 --capture-w 1330 \
    --d2-dir <data> --v10-dir <data> \
    --split d2_orig --methodology legacy,family_balanced \
    --out experiments/phase_e/p0/exp001c_d2_orig.json
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
torch.set_num_interop_threads(2)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.emission_dataset import EmissionDataset, load_capture_at, load_emission_at  # noqa: E402
from data.raw_bayer_dataset import load_chain_log  # noqa: E402
from models.emission_predictor import EmissionPredictor  # noqa: E402
try:
    from models.emission_predictor_v2 import EmissionPredictorV2  # noqa: E402
except Exception:
    EmissionPredictorV2 = None


# ---------- splits ----------

SPLITS = {
    "d2_orig":           (5394, 5992),     # exp001c's original val
    "d2_phase_d":        (4792, 5992),     # Phase D val
    "d2_phase_d_calib":  (4792, 5392),
    "d2_phase_d_report": (5392, 5992),
}

# near_shift offsets per Phase D candidate-ranking convention
NEAR_SHIFT_OFFSETS = (-16, -8, -4, -2, -1, 1, 2, 4, 8, 16)


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = ((pred - target) ** 2).mean().item()
    if mse < 1e-12:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def per_channel_l1(pred: torch.Tensor, target: torch.Tensor) -> dict:
    return {f"l1_{c}": float((pred[i] - target[i]).abs().mean().item())
            for i, c in enumerate("rgb")}


# ---------- model + predict ----------

def build_model(arch: str, capture_h: int, capture_w: int, emission_h: int, emission_w: int, ckpt_path: Path | None):
    if arch == "EmissionPredictor":
        m = EmissionPredictor(emission_h=emission_h, emission_w=emission_w, pretrained=False)
    elif arch == "EmissionPredictorV2":
        if EmissionPredictorV2 is None:
            raise RuntimeError("EmissionPredictorV2 not available")
        m = EmissionPredictorV2(emission_h=emission_h, emission_w=emission_w, pretrained=False)
    else:
        raise ValueError(f"unknown arch {arch}")
    if ckpt_path is not None:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
        missing, unexp = m.load_state_dict(state, strict=False)
        print(f"[ckpt] {ckpt_path}: missing={len(missing)} unexpected={len(unexp)}", flush=True)
    m.eval()
    return m


@torch.no_grad()
def predict(mode: str, model, sample, autocast_dtype, mean_emission_uint8: torch.Tensor | None = None):
    if mode == "ckpt":
        cap = sample["capture"].unsqueeze(0).cuda()
        with torch.amp.autocast("cuda", dtype=autocast_dtype):
            p = model(cap).float()
        return p.squeeze(0).clamp(0, 1).cpu()
    if mode == "oracle":
        return sample["emission"].clamp(0, 1).cpu()
    if mode == "null_mean":
        if mean_emission_uint8 is None:
            raise RuntimeError("null_mean requires precomputed mean_emission")
        return mean_emission_uint8.float().div(255.0).clamp(0, 1)
    if mode == "null_grey":
        em = sample["emission"]
        return torch.full_like(em, 0.5)
    raise ValueError(f"unknown mode {mode}")


# ---------- dataset slicer ----------

def _build_d2_dataset(d2_dir: Path, rows: list[int], capture_h: int, capture_w: int, emission_h: int, emission_w: int):
    return EmissionDataset(
        session_dir=d2_dir,
        row_start=min(rows), row_end=max(rows) + 1,
        capture_h=capture_h, capture_w=capture_w,
        emission_h=emission_h, emission_w=emission_w,
        session_id="D2", augment=False,
    )


def _row_to_emission(d2_dir: Path, t: int, emission_h: int, emission_w: int) -> torch.Tensor | None:
    p = d2_dir / "derived" / "Emissions" / f"tile_{t:06d}.png"
    if not p.exists():
        return None
    return load_emission_at(p, emission_h, emission_w)


# ---------- legacy methodology ----------

def eval_legacy(*, model_or_mode, ckpt_path, capture_h, capture_w, emission_h, emission_w,
                d2_dir: Path, val_rows: list[int], n_eval: int = 50, autocast_dtype=torch.bfloat16,
                mean_emission_uint8: torch.Tensor | None = None, seed: int = 7):
    """exp001c's methodology: n=50 uniformly-spaced val rows; each prediction
    scored against its true matched emission AND one random non-matching
    emission from the same val cohort (deterministic sample).
    """
    rng = np.random.RandomState(seed)
    indices = np.linspace(0, len(val_rows) - 1, n_eval).astype(int)
    rows = [val_rows[i] for i in indices]
    print(f"[legacy] n_eval={n_eval} rows[0:5]={rows[:5]}", flush=True)

    ds = _build_d2_dataset(d2_dir, rows, capture_h, capture_w, emission_h, emission_w)
    matched_psnrs = []
    mismatched_psnrs = []
    per_row = []
    for i, t in enumerate(rows):
        if t not in ds.rows: continue
        local_idx = ds.rows.index(t)
        try:
            sample = ds[local_idx]
        except Exception as exc:
            print(f"[WARN] sample {t} skipped: {exc}", flush=True)
            continue
        pred = predict(model_or_mode if isinstance(model_or_mode, str) else "ckpt",
                       model_or_mode if not isinstance(model_or_mode, str) else None,
                       sample, autocast_dtype, mean_emission_uint8)
        gt = sample["emission"].clamp(0, 1)
        m_psnr = psnr(pred, gt)
        # Random non-matching mismatch from cohort
        cand_pool = [r for r in rows if r != t]
        mm_t = int(rng.choice(cand_pool))
        mm_em = _row_to_emission(d2_dir, mm_t, emission_h, emission_w)
        if mm_em is not None:
            mm_psnr = psnr(pred, mm_em.clamp(0, 1))
        else:
            mm_psnr = float("nan")
        matched_psnrs.append(m_psnr)
        mismatched_psnrs.append(mm_psnr)
        per_row.append({"row": t, "mm_row": mm_t, "matched_psnr": m_psnr, "mismatched_psnr": mm_psnr})
    arr_m = np.array(matched_psnrs)
    arr_x = np.array([p for p in mismatched_psnrs if not (isinstance(p, float) and np.isnan(p))])
    return {
        "n": len(matched_psnrs),
        "matched_psnr_mean": float(arr_m.mean()),
        "matched_psnr_std":  float(arr_m.std()),
        "matched_psnr_min":  float(arr_m.min()),
        "matched_psnr_max":  float(arr_m.max()),
        "mismatched_psnr_mean": float(arr_x.mean()) if arr_x.size else float("nan"),
        "mismatched_psnr_std":  float(arr_x.std())  if arr_x.size else float("nan"),
        "gap_db":              float(arr_m.mean() - arr_x.mean()) if arr_x.size else float("nan"),
        "per_row": per_row,
    }


# ---------- family-balanced methodology ----------

def assemble_emission_candidate_set(*, capture_t: int, val_rows_set: set[int],
                                    cross_rows_set: set[int], d_search_max: int = 32,
                                    n_same: int = 192, n_cross: int = 192, seed: int = 0,
                                    chain_log_offset: int = 1):
    """For emission task: the matched candidate's target_chain_row = capture_t + offset.
    Same family taxonomy as Phase D candidate_ranking. Returns list of
    (target_chain_row, family) tuples."""
    target = capture_t + chain_log_offset
    rng = np.random.RandomState(seed)
    candidates = []
    candidates.append((target, "matched", "D2"))
    used = {("D2", target)}
    # delay_window
    for d in range(-d_search_max, d_search_max + 1):
        if d == 0: continue
        r = target + d
        if r in val_rows_set and ("D2", r) not in used:
            candidates.append((r, "delay_window", "D2"))
            used.add(("D2", r))
    # near_shift: move out of delay_window
    near_rows = {target + d for d in NEAR_SHIFT_OFFSETS}
    out = []
    for c in candidates:
        if c[1] == "delay_window" and c[0] in near_rows:
            out.append((c[0], "near_shift", c[2]))
        else:
            out.append(c)
    candidates = out
    # same_session_random: from val pool, exclude delay window
    delay_keys = {("D2", target + d) for d in range(-d_search_max, d_search_max + 1)}
    pool = sorted([r for r in val_rows_set if ("D2", r) not in delay_keys])
    rng.shuffle(pool)
    n_taken = 0
    for r in pool:
        if ("D2", r) in used: continue
        candidates.append((r, "same_session_random", "D2"))
        used.add(("D2", r))
        n_taken += 1
        if n_taken >= n_same: break
    # cross_session_random: V10 rows (different session)
    cross_pool = sorted(list(cross_rows_set))
    rng.shuffle(cross_pool)
    n_taken = 0
    for r in cross_pool:
        if ("V10", r) in used: continue
        candidates.append((r, "cross_session_random", "V10"))
        used.add(("V10", r))
        n_taken += 1
        if n_taken >= n_cross: break
    return candidates


def eval_family_balanced(*, model_or_mode, ckpt_path, capture_h, capture_w, emission_h, emission_w,
                         d2_dir: Path, v10_dir: Path, val_rows: list[int],
                         n_eval: int = 200, n_same: int = 192, n_cross: int = 192,
                         d_search_max: int = 32, autocast_dtype=torch.bfloat16,
                         mean_emission_uint8: torch.Tensor | None = None, seed: int = 42):
    """Phase D-style family-balanced eval on emission targets.
    Score = matched PSNR is HIGHER=better. tau5 = 5th percentile of matched.
    Window-FMR@5 = fraction of frames with any candidate PSNR ≥ tau5."""
    if n_eval > len(val_rows):
        n_eval = len(val_rows)
    rng = np.random.RandomState(seed)
    rows = sorted(rng.choice(val_rows, size=n_eval, replace=False).tolist())
    print(f"[family-balanced] n_eval={n_eval} rows[0:5]={rows[:5]}", flush=True)

    ds = _build_d2_dataset(d2_dir, rows, capture_h, capture_w, emission_h, emission_w)
    val_rows_set = set(val_rows)

    # V10 cross-session pool: emission-tile rows that exist
    v10_chain = load_chain_log(v10_dir / "chain_log.csv") if (v10_dir / "chain_log.csv").exists() else {}
    v10_emi = v10_dir / "derived" / "Emissions"
    v10_rows = [t for t in v10_chain if (v10_emi / f"tile_{t:06d}.png").exists()]
    v10_rows_set = set(v10_rows)

    matched_psnrs = []
    negatives_per_frame: list[dict[str, list[float]]] = []  # {family: [psnr,...]}
    candidate_psnrs_per_frame: list[dict[int, float]] = []
    top1_count = 0

    em_cache: dict[tuple[str, int], torch.Tensor] = {}
    def get_emission(session: str, t: int):
        key = (session, t)
        if key in em_cache:
            return em_cache[key]
        sd = d2_dir if session == "D2" else v10_dir
        em = _row_to_emission(sd, t, emission_h, emission_w)
        if em is not None:
            em_cache[key] = em.clamp(0, 1)
            return em_cache[key]
        return None

    t0 = time.time()
    for i, t in enumerate(rows):
        if t not in ds.rows: continue
        local_idx = ds.rows.index(t)
        try:
            sample = ds[local_idx]
        except Exception as exc:
            print(f"[WARN] sample {t} skipped: {exc}", flush=True)
            continue
        pred = predict(model_or_mode if isinstance(model_or_mode, str) else "ckpt",
                       model_or_mode if not isinstance(model_or_mode, str) else None,
                       sample, autocast_dtype, mean_emission_uint8)
        gt_matched = sample["emission"].clamp(0, 1)
        m_psnr = psnr(pred, gt_matched)
        matched_psnrs.append(m_psnr)

        cs = assemble_emission_candidate_set(
            capture_t=t, val_rows_set=val_rows_set, cross_rows_set=v10_rows_set,
            d_search_max=d_search_max, n_same=n_same, n_cross=n_cross, seed=seed ^ t,
            chain_log_offset=0,  # for emission task, t -> emission tile at same row
        )
        per_frame_neg: dict[str, list[float]] = {}
        max_neg_psnr = -math.inf
        for r, fam, sess in cs:
            if fam == "matched":
                continue
            em = get_emission(sess, r)
            if em is None:
                continue
            p = psnr(pred, em)
            per_frame_neg.setdefault(fam, []).append(p)
            if p > max_neg_psnr:
                max_neg_psnr = p
        negatives_per_frame.append(per_frame_neg)
        if m_psnr > max_neg_psnr:
            top1_count += 1
        if (i + 1) % 25 == 0:
            print(f"  [fb] frame {i+1}/{len(rows)} elapsed={time.time()-t0:.0f}s "
                  f"matched={np.mean(matched_psnrs):.2f} top1={top1_count/(i+1):.3f}", flush=True)

    n = len(matched_psnrs)
    matched_arr = np.array(matched_psnrs)
    tau5 = float(np.percentile(matched_arr, 5))  # 5th percentile of matched
    families = ["delay_window", "near_shift", "same_session_random", "cross_session_random"]
    per_family_fmr = {}
    for f in families:
        hits = sum(1 for nf in negatives_per_frame if any(p >= tau5 for p in nf.get(f, [])))
        per_family_fmr[f] = hits / n if n else 0.0
    pooled = sum(1 for nf in negatives_per_frame
                 if any(p >= tau5 for fam in nf.values() for p in fam)) / n if n else 0.0
    worst = max(per_family_fmr.values()) if per_family_fmr else 0.0
    macro = float(np.mean(list(per_family_fmr.values()))) if per_family_fmr else 0.0
    return {
        "n": n,
        "tau5": tau5,
        "matched_psnr_mean": float(matched_arr.mean()),
        "matched_psnr_std":  float(matched_arr.std()),
        "matched_psnr_p5":   float(np.percentile(matched_arr, 5)),
        "matched_psnr_p95":  float(np.percentile(matched_arr, 95)),
        "per_family_fmr_at_5": per_family_fmr,
        "worst_family_fmr_at_5": worst,
        "macro_fmr_at_5": macro,
        "pooled_fmr_at_5": pooled,
        "top1_retrieval": top1_count / n if n else 0.0,
        "elapsed_sec": round(time.time() - t0, 1),
    }


def compute_train_set_mean_emission(d2_dir: Path, train_start: int, train_end: int,
                                    emission_h: int, emission_w: int, n_sample: int = 200) -> torch.Tensor:
    rng = np.random.RandomState(0)
    rows = sorted(rng.choice(range(train_start, train_end), size=n_sample, replace=False).tolist())
    accum = torch.zeros((3, emission_h, emission_w), dtype=torch.float64)
    n = 0
    for t in rows:
        em = _row_to_emission(d2_dir, t, emission_h, emission_w)
        if em is None:
            continue
        accum += em.double()
        n += 1
    if n == 0:
        raise RuntimeError("no train emissions found")
    mean = (accum / n).clamp(0, 1)
    return (mean * 255).round().clamp(0, 255).to(torch.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ckpt", "oracle", "null_mean", "null_grey"], required=True)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--arch", choices=["EmissionPredictor", "EmissionPredictorV2"], default="EmissionPredictor")
    ap.add_argument("--capture-h", type=int, required=True)
    ap.add_argument("--capture-w", type=int, required=True)
    ap.add_argument("--emission-h", type=int, default=1080)
    ap.add_argument("--emission-w", type=int, default=1920)
    ap.add_argument("--d2-dir", type=Path, required=True)
    ap.add_argument("--v10-dir", type=Path, required=True)
    ap.add_argument("--split", choices=list(SPLITS.keys()), required=True)
    ap.add_argument("--methodology", default="legacy,family_balanced",
                    help="Comma-separated: legacy, family_balanced")
    ap.add_argument("--n-legacy", type=int, default=50)
    ap.add_argument("--n-fb", type=int, default=200)
    ap.add_argument("--n-same", type=int, default=192)
    ap.add_argument("--n-cross", type=int, default=192)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-start", type=int, default=0)
    ap.add_argument("--train-end", type=int, default=4194)
    args = ap.parse_args()

    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_start, val_end = SPLITS[args.split]
    val_rows = list(range(val_start, val_end))

    model = None
    mean_em = None
    if args.mode == "ckpt":
        if not args.ckpt:
            raise SystemExit("--ckpt required for --mode ckpt")
        model = build_model(args.arch, args.capture_h, args.capture_w, args.emission_h, args.emission_w, args.ckpt).to(device)
    elif args.mode == "null_mean":
        print(f"[init] computing train-set mean emission (rows {args.train_start}..{args.train_end})", flush=True)
        mean_em = compute_train_set_mean_emission(args.d2_dir, args.train_start, args.train_end,
                                                  args.emission_h, args.emission_w)
        print(f"  mean emission shape={tuple(mean_em.shape)} dtype={mean_em.dtype}", flush=True)

    out: dict = {
        "mode": args.mode,
        "ckpt": str(args.ckpt) if args.ckpt else None,
        "arch": args.arch,
        "split": args.split,
        "split_range": [val_start, val_end],
        "n_val_rows": len(val_rows),
        "capture_hw": [args.capture_h, args.capture_w],
        "emission_hw": [args.emission_h, args.emission_w],
        "results": {},
    }

    methodologies = args.methodology.split(",")
    if "legacy" in methodologies:
        print("=" * 60, "\n[legacy methodology]\n", "=" * 60, flush=True)
        out["results"]["legacy"] = eval_legacy(
            model_or_mode=(args.mode if args.mode != "ckpt" else model),
            ckpt_path=args.ckpt,
            capture_h=args.capture_h, capture_w=args.capture_w,
            emission_h=args.emission_h, emission_w=args.emission_w,
            d2_dir=args.d2_dir, val_rows=val_rows, n_eval=args.n_legacy,
            autocast_dtype=autocast_dtype, mean_emission_uint8=mean_em,
        )
        r = out["results"]["legacy"]
        print(f"[legacy] matched={r['matched_psnr_mean']:.2f} ± {r['matched_psnr_std']:.2f}  "
              f"mismatched={r['mismatched_psnr_mean']:.2f}  gap={r['gap_db']:.2f}", flush=True)

    if "family_balanced" in methodologies:
        print("=" * 60, "\n[family-balanced methodology]\n", "=" * 60, flush=True)
        out["results"]["family_balanced"] = eval_family_balanced(
            model_or_mode=(args.mode if args.mode != "ckpt" else model),
            ckpt_path=args.ckpt,
            capture_h=args.capture_h, capture_w=args.capture_w,
            emission_h=args.emission_h, emission_w=args.emission_w,
            d2_dir=args.d2_dir, v10_dir=args.v10_dir, val_rows=val_rows,
            n_eval=args.n_fb, n_same=args.n_same, n_cross=args.n_cross,
            autocast_dtype=autocast_dtype, mean_emission_uint8=mean_em,
        )
        r = out["results"]["family_balanced"]
        print(f"[family-balanced] matched={r['matched_psnr_mean']:.2f} tau5={r['tau5']:.2f}  "
              f"top1={r['top1_retrieval']:.3f}  worst-FMR={r['worst_family_fmr_at_5']:.3f}", flush=True)
        print(f"  per-family: {r['per_family_fmr_at_5']}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

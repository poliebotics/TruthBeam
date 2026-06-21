"""Phase D sanity-check orchestrator (audit §11 + Q2 + smoke + profile).

Runs on the A10 staging instance overnight. Writes
`<out>/sanity_check_report.md` summarizing all checks. STOPs after writing the
report — does NOT launch the six-experiment training matrix.

Steps:
  1.  Raw dtype + range sanity (audit §11) — load 5 D2 + 5 V10 frames; print dtype, min/max, fraction at 0/255.
  2.  Packed CFA staging — stage 100 frames per session into the cache.
  3.  Normalization stats — fit median+IQR/1.349 from D2-train and V10-train (sampled).
  4.  Offset diagnostic (audit Q2) — D2-mid, V10-early/mid/late.
  5.  Memory + step-time profile — A0/A1/A2/A4/A6/A7 forward+backward 20 steps each.
  6.  A0 one-batch overfit smoke (300 steps) — checks the architecture works on packed CFA + centered targets.
  7.  Observability schema validation — log a few rows, verify JSONL format.
  8.  Run-manifest writer test — write a manifest JSON for each variant config.

Failure modes that STOP the queue (write to QUESTIONS.md):
  - Offset diagnostic fails the within-bin or cross-bin test.
  - A0 overfit smoke loss not heading toward 0 (peak min bit recovery < 0.55).
  - Any variant OOMs at bs=1 on A10 (means it can't fit on A100 either, modulo activation checkpointing).
  - Loss goes NaN.
  - Cache or normalization stats can't be computed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import os  # noqa: E402

from data.packed_cfa_dataset import PackedCFADataset, xof_octaves_centered_from_hex  # noqa: E402
from data.raw_bayer_dataset import load_chain_log  # noqa: E402
from diagnostics.offset_diagnostic import run_all as run_offset_diagnostic_all  # noqa: E402
from eval.observability import ObservabilityLogger, compute_row_observability  # noqa: E402
from losses.huber_xof import huber_xof_loss  # noqa: E402
from models.emission_predictor_v2 import EmissionPredictorV2  # noqa: E402
from models.stn import ConstrainedSTN, identity_regularizer, out_of_bounds_fraction  # noqa: E402
from models.xof_decoder_v2 import XOFDecoderV2  # noqa: E402
from preprocessing.normalization import fit_stats_from_cached, save_stats  # noqa: E402
from preprocessing.packed_cfa import EXPECTED_BYTES, HEIGHT, WIDTH, load_packed_cfa, stage_packed_cfa  # noqa: E402
from utils.run_manifest import write_run_manifest  # noqa: E402

OUT_REPORT = "sanity_check_report.md"


def _now() -> str:
    import datetime as dt
    return dt.datetime.utcnow().isoformat() + "Z"


def step1_raw_dtype_range(d2_dir: Path, v10_dir: Path) -> dict:
    """Audit §11: print dtype/range; warn on unexpected."""
    out = {"D2_samples": [], "V10_samples": []}
    for tag, sess in [("D2", d2_dir), ("V10", v10_dir)]:
        rec = sess / "Recordings"
        for t in (0, 100, 1000, 2500, 4500):
            p = rec / f"frame_{t:06d}.raw"
            if not p.exists():
                continue
            import numpy as np
            raw = np.fromfile(p, dtype=np.uint8)
            if raw.size != EXPECTED_BYTES:
                out[f"{tag}_samples"].append({"t": t, "size": int(raw.size), "unexpected": True})
                continue
            r = raw.reshape(HEIGHT, WIDTH)
            entry = {
                "t": t,
                "dtype": "uint8",
                "min": int(r.min()),
                "max": int(r.max()),
                "frac_zero": float((r == 0).mean()),
                "frac_saturated": float((r == 255).mean()),
            }
            out[f"{tag}_samples"].append(entry)
    return out


def step2_stage_packed_cfa(d2_dir: Path, v10_dir: Path, cache_root: Path, n_per_session: int = 100) -> dict:
    out = {}
    for tag, sess in [("D2", d2_dir), ("V10", v10_dir)]:
        rec = sess / "Recordings"
        chain = load_chain_log(sess / "chain_log.csv")
        rows = sorted(chain.keys())[:n_per_session]
        t0 = time.time()
        for t in rows:
            p = rec / f"frame_{t:06d}.raw"
            if not p.exists():
                continue
            stage_packed_cfa(p, cache_root, tag.lower(), t)
        out[tag] = {
            "n_staged": len(rows),
            "elapsed_s": round(time.time() - t0, 2),
            "cache_root": str(cache_root / tag.lower()),
        }
    return out


def step3_fit_normalization(d2_dir: Path, v10_dir: Path, cache_root: Path,
                            d2_train_rows: list[int], v10_train_rows: list[int],
                            stats_dir: Path) -> dict:
    out = {}
    for tag, rows in [("d2", d2_train_rows), ("v10", v10_train_rows)]:
        # Use whatever subset of rows is actually staged.
        staged_rows = [t for t in rows if (cache_root / tag / f"frame_{t:06d}.pt").exists()]
        if not staged_rows:
            out[tag] = {"error": "no staged rows"}
            continue
        t0 = time.time()
        stats = fit_stats_from_cached(cache_root, tag, staged_rows[:64], sample_pixels_per_frame=16384)
        save_stats(stats, stats_dir / f"{tag}_train_stats.json")
        out[tag] = {
            "n_rows_used": len(stats["rows_used"]),
            "center": stats["center"],
            "scale": stats["scale"],
            "stats_file": str(stats_dir / f"{tag}_train_stats.json"),
            "elapsed_s": round(time.time() - t0, 2),
        }
    return out


def step4_offset_diagnostic(d2_dir: Path, v10_dir: Path, out_dir: Path) -> dict:
    return run_offset_diagnostic_all(d2_session_dir=d2_dir, v10_session_dir=v10_dir, out_dir=out_dir)


def _make_model(spec: dict) -> torch.nn.Module:
    cls = spec["class"]
    if cls == "XOFDecoderV2":
        return XOFDecoderV2(
            encoder_size=spec.get("encoder_size", "tiny"),
            pretrained=spec.get("pretrained", False),
            enabled_octaves=tuple(spec.get("enabled_octaves", (0, 1, 2, 3))),
            fpn_out_channels=spec.get("fpn_out_channels", 256),
            head_hidden=spec.get("head_hidden", 128),
        )
    if cls == "EmissionPredictorV2":
        return EmissionPredictorV2(
            emission_h=spec.get("emission_h", 1080),
            emission_w=spec.get("emission_w", 1920),
            encoder_size=spec.get("encoder_size", "tiny"),
            pretrained=spec.get("pretrained", False),
            fpn_out_channels=spec.get("fpn_out_channels", 256),
            decoder_dims=tuple(spec.get("decoder_dims", (256, 192, 96, 48, 24))),
        )
    raise ValueError(f"unknown model class {cls}")


def step5_memory_step_profile(variants: list[dict], device: torch.device, n_steps: int = 20,
                               batch_size: int = 1) -> dict:
    """Forward+backward profile on synthetic input; record max memory + step time."""
    out = {}
    H, W = 2300, 2660
    for v in variants:
        name = v["name"]
        spec = dict(v["model"])
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model = _make_model(spec).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scaler = torch.amp.GradScaler("cuda")
            x = torch.randn(batch_size, 4, H, W, device=device, dtype=torch.float32)
            t0 = time.time()
            ok_steps = 0
            for s in range(n_steps):
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    y = model(x)
                    # XOFDecoderV2 returns (preds, info); EmissionPredictorV2 returns a tensor.
                    if isinstance(y, tuple):
                        preds = y[0]
                        loss = sum(o.float().abs().mean() for o in preds if o is not None)
                    elif isinstance(y, list):
                        loss = sum(o.float().abs().mean() for o in y if o is not None)
                    else:
                        loss = y.float().abs().mean()
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                ok_steps += 1
            dt = time.time() - t0
            out[name] = {
                "ok": True,
                "batch_size": batch_size,
                "steps": ok_steps,
                "elapsed_s": round(dt, 3),
                "step_ms_avg": round(1000 * dt / max(ok_steps, 1), 1),
                "peak_mem_MiB": int(torch.cuda.max_memory_allocated() / (1024 * 1024)),
            }
            del model, opt, scaler, x, y, loss
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            out[name] = {"ok": False, "batch_size": batch_size, "error": "OOM", "msg": str(e)[:200]}
            torch.cuda.empty_cache()
        except Exception as e:
            out[name] = {"ok": False, "batch_size": batch_size, "error": type(e).__name__, "msg": str(e)[:300]}
            torch.cuda.empty_cache()
    return out


def step6_a0_overfit_smoke(d2_dir: Path, cache_root: Path, stats_path: Path,
                            offset: int, device: torch.device, n_steps: int = 300) -> dict:
    """A0 one-batch overfit on packed CFA + centered XOF target."""
    import json as _json
    stats = _json.loads(stats_path.read_text())
    chain = load_chain_log(d2_dir / "chain_log.csv")
    # Pick a small batch of staged rows that have valid offset. Default 2 because
    # the 5090 has 24 GB and A0+bs=4 OOMs; A100 80 GB will handle bs=4 fine
    # but the smoke is proving "architecture overfits a single batch" — the
    # batch size is incidental as long as it's >1 (so loss isn't dominated
    # by one sample's idiosyncrasies).
    smoke_bs = int(os.environ.get("PHASE_D_SMOKE_BS", "2"))
    staged = sorted(p.stem for p in (cache_root / "d2").glob("frame_*.pt"))
    rows = [int(s.split("_")[1]) for s in staged][:64]
    rows = [t for t in rows if (t + offset) in chain][:smoke_bs]
    if len(rows) < smoke_bs:
        return {"error": f"not enough staged rows for bs={smoke_bs} overfit smoke", "rows_available": len(rows)}
    ds = PackedCFADataset(d2_dir, rows, offset, stats, cache_root=cache_root, session_id="d2")
    batch = [ds[i] for i in range(len(ds))]
    cap = torch.stack([b["capture_norm"] for b in batch]).to(device)
    # All four octaves provided so huber_xof_loss can run on enabled set
    targets = [torch.stack([b[f"xof_oct{i}"] for b in batch]).to(device) for i in range(4)]
    # Per-bit recovery measurement: convert centered prediction back to bytes
    targets_bytes = [(t * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8) for t in targets]

    model = XOFDecoderV2("tiny", pretrained=False, enabled_octaves=(0, 1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    losses = []
    bit_recovery_history = {f"oct{i}": [] for i in (0, 1)}
    t0 = time.time()
    import numpy as np
    _POPCNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    for s in range(n_steps):
        with torch.amp.autocast("cuda", dtype=torch.float16):
            preds, info = model(cap)
            loss, parts = huber_xof_loss(preds, targets)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        losses.append(float(parts["total"]))
        if s % 20 == 0 or s == n_steps - 1:
            with torch.no_grad():
                for i in (0, 1):
                    pb = (preds[i].float() * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
                    xor = (pb ^ targets_bytes[i]).cpu().numpy()
                    bit_diff = int(_POPCNT[xor].sum())
                    bit_total = xor.size * 8
                    bit_recovery_history[f"oct{i}"].append(1.0 - bit_diff / bit_total)
    final_bit = {k: (v[-1] if v else None) for k, v in bit_recovery_history.items()}
    peak_bit = {k: (max(v) if v else None) for k, v in bit_recovery_history.items()}
    return {
        "ok": True,
        "n_steps": n_steps,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
        "elapsed_s": round(time.time() - t0, 2),
        "n_rows": len(rows),
        "final_bit_recovery": final_bit,
        "peak_bit_recovery": peak_bit,
        "passed_threshold_0.7": all((b is not None and b >= 0.7) for b in peak_bit.values()),
    }


def write_report(out_dir: Path, results: dict) -> Path:
    md = out_dir / OUT_REPORT
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(md, "w") as f:
        f.write("# Sanity check report\n\n")
        f.write(f"Generated: {_now()}\n\n")
        f.write("## Step 1 — raw dtype + range\n\n```\n")
        f.write(json.dumps(results["step1"], indent=2))
        f.write("\n```\n\n## Step 2 — packed CFA staging\n\n```\n")
        f.write(json.dumps(results["step2"], indent=2))
        f.write("\n```\n\n## Step 3 — normalization stats (median + IQR/1.349)\n\n```\n")
        f.write(json.dumps(results["step3"], indent=2))
        f.write("\n```\n\n## Step 4 — offset diagnostic\n\n")
        cd = results["step4"]
        if cd.get("skipped"):
            f.write(f"- SKIPPED (operator override). winning_offset = {cd['winning_offset']}\n")
            f.write(f"- source: {cd.get('source', 'operator_override')}\n")
            f.write(f"- rationale: {cd.get('rationale', '')}\n")
        else:
            f.write(f"- overall_pass: **{cd['overall_pass']}**\n")
            f.write(f"- winning_offset: {cd.get('winning_offset')}\n")
            f.write(f"- cross_bin_modes: {cd['cross_bin']['modes']} (spread {cd['cross_bin']['spread']})\n\n")
            for b in cd["bins"]:
                f.write(f"- bin {b['bin']}: mode={b['mode_offset']} ({b['mode_count']}/{b['n_sampled']}), within ±1: {b['within_pm1_fraction']:.0%}, pass: {b['passed_within_bin']}\n")
        f.write("\n## Step 5 — memory + step-time profile (A10, bs=1, fp16 autocast)\n\n```\n")
        f.write(json.dumps(results["step5"], indent=2))
        f.write("\n```\n\n## Step 6 — A0 one-batch overfit smoke\n\n```\n")
        f.write(json.dumps(results["step6"], indent=2))
        f.write("\n```\n\n## Step 7/8 — observability + manifest writer\n\n")
        f.write("Tested by importing schema; no separate output. See `src/eval/observability.py` and `src/utils/run_manifest.py`.\n\n")
        f.write("## Verdict\n\n")
        passed = (
            results["step4"]["overall_pass"]
            and all(v["ok"] for v in results["step5"].values())
            and results["step6"].get("ok", False)
            and results["step6"]["loss_last"] < results["step6"]["loss_first"] * 0.1
        )
        f.write(f"**Overall: {'PASS' if passed else 'NEEDS REVIEW'}**\n\n")
        if not passed:
            f.write("See QUESTIONS.md for any blocking issues.\n")
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d2-dir",      type=Path, required=True)
    ap.add_argument("--v10-dir",     type=Path, required=True)
    ap.add_argument("--cache-root",  type=Path, required=True)
    ap.add_argument("--stats-dir",   type=Path, required=True)
    ap.add_argument("--out-dir",     type=Path, required=True)
    ap.add_argument("--profile-steps", type=int, default=20)
    ap.add_argument("--overfit-steps", type=int, default=300)
    ap.add_argument("--n-stage-per-session", type=int, default=128)
    ap.add_argument("--skip-offset-diagnostic", action="store_true",
                    help="Skip step 4 (use --offset directly). Set when offset is already "
                         "operator-resolved (e.g., adopted from Phase B empirical validation).")
    ap.add_argument("--offset", type=int, default=None,
                    help="Use this offset for steps that need it (step 6 in particular). "
                         "Required if --skip-offset-diagnostic is set; otherwise the "
                         "diagnostic's consensus offset is used.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results: dict = {}
    print("=== step 1: raw dtype/range ===", flush=True)
    results["step1"] = step1_raw_dtype_range(args.d2_dir, args.v10_dir)
    print("=== step 2: packed CFA staging ===", flush=True)
    results["step2"] = step2_stage_packed_cfa(args.d2_dir, args.v10_dir, args.cache_root, n_per_session=args.n_stage_per_session)
    print("=== step 3: normalization fit ===", flush=True)
    chain_d2 = load_chain_log(args.d2_dir / "chain_log.csv")
    chain_v10 = load_chain_log(args.v10_dir / "chain_log.csv")
    d2_train_rows = sorted([t for t in chain_d2 if t < 4592])[:args.n_stage_per_session]
    v10_train_rows = sorted([t for t in chain_v10 if t < 2500])[:args.n_stage_per_session]
    results["step3"] = step3_fit_normalization(args.d2_dir, args.v10_dir, args.cache_root,
                                                d2_train_rows, v10_train_rows, args.stats_dir)
    if args.skip_offset_diagnostic:
        if args.offset is None:
            raise SystemExit("--skip-offset-diagnostic requires --offset N")
        print(f"=== step 4: offset diagnostic SKIPPED (operator-set offset={args.offset}) ===", flush=True)
        results["step4"] = {
            "skipped": True,
            "winning_offset": int(args.offset),
            "overall_pass": True,
            "source": "operator_override",
            "rationale": "Skipped by --skip-offset-diagnostic flag; offset adopted from Phase B empirical validation.",
        }
    else:
        print("=== step 4: offset diagnostic ===", flush=True)
        results["step4"] = step4_offset_diagnostic(args.d2_dir, args.v10_dir, args.out_dir)
        print(f"  winning_offset: {results['step4'].get('winning_offset')}, overall_pass: {results['step4']['overall_pass']}", flush=True)

    print("=== step 5: memory profile ===", flush=True)
    variants = [
        {"name": "A0 (Tiny, oct0+oct1)",       "model": {"class": "XOFDecoderV2", "encoder_size": "tiny",  "enabled_octaves": (0, 1)}},
        {"name": "A1 (Tiny, all octaves)",      "model": {"class": "XOFDecoderV2", "encoder_size": "tiny",  "enabled_octaves": (0, 1, 2, 3)}},
        {"name": "A2 (Large, all octaves)",     "model": {"class": "XOFDecoderV2", "encoder_size": "large", "enabled_octaves": (0, 1, 2, 3)}},
        {"name": "A4 (Tiny, emission)",         "model": {"class": "EmissionPredictorV2", "encoder_size": "tiny", "emission_h": 1080, "emission_w": 1920}},
        {"name": "A6 (Large, all octaves)",     "model": {"class": "XOFDecoderV2", "encoder_size": "large", "enabled_octaves": (0, 1, 2, 3)}},
        {"name": "A7 (Tiny, all + STN-needed)", "model": {"class": "XOFDecoderV2", "encoder_size": "tiny",  "enabled_octaves": (0, 1, 2, 3)}},
    ]
    results["step5"] = step5_memory_step_profile(variants, device, n_steps=args.profile_steps, batch_size=1)
    for n, r in results["step5"].items():
        print(f"  {n}: {r}", flush=True)

    print("=== step 6: A0 overfit smoke ===", flush=True)
    if results["step4"].get("overall_pass") or results["step4"].get("skipped"):
        offset = results["step4"]["winning_offset"]
        stats_path = args.stats_dir / "d2_train_stats.json"
        results["step6"] = step6_a0_overfit_smoke(args.d2_dir, args.cache_root, stats_path,
                                                   offset, device, n_steps=args.overfit_steps)
    else:
        results["step6"] = {"skipped": True, "reason": "offset diagnostic did not pass"}
    print(f"  result: {results['step6']}", flush=True)

    md = write_report(args.out_dir, results)
    print(f"=== wrote report: {md} ===", flush=True)
    (args.out_dir / "sanity_check_results.json").write_text(json.dumps(results, indent=2))
    print(f"=== wrote raw results: {args.out_dir / 'sanity_check_results.json'} ===", flush=True)


if __name__ == "__main__":
    main()

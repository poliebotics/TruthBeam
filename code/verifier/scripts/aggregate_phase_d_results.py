"""Aggregate Phase D matrix results into a single PHASE_C_RESULTS.md + comparison.png.

Pulls per-experiment outputs from experiments/exp001h_*/ on the cloud filesystem
(via local symlink or rsync'd copy):
  - manifest.json
  - val_history.jsonl (per-step Window-FMR@95 stats during training)
  - final_report.json (held-out FMR over D2-val report half + V10 bins)
  - run.log
  - checkpoints/{best_by_loss,best_by_calibration_half_window_fmr,final_step}.pt

Run: `.venv/bin/python scripts/aggregate_phase_d_results.py --root <experiments-dir>`

Writes:
  experiments/PHASE_C_RESULTS.md  (markdown summary)
  experiments/comparison.png       (loss + Window-FMR + per-octave bit recovery)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPERIMENTS = ("exp001h_a0", "exp001h_a1", "exp001h_a2", "exp001h_a4", "exp001h_a6", "exp001h_a7")


def parse_run_log(path: Path) -> dict:
    """Extract per-step loss + per-octave loss (XOF runs) or charb/ms_l1 (emission)."""
    if not path.exists():
        return {"steps": [], "total_loss": [], "per_octave": {}, "charb": [], "ms_l1": []}
    steps, total, charb, ms_l1 = [], [], [], []
    per_oct = {f"oct{i}": [] for i in range(4)}
    pat_step = re.compile(r"\[step (\d+)\] loss=([\d.eE+-]+)")
    pat_oct = re.compile(r"oct(\d)_loss': ([\d.eE+-]+)")
    pat_charb = re.compile(r"'charb': ([\d.eE+-]+)")
    pat_ms = re.compile(r"'ms_l1': ([\d.eE+-]+)")
    with open(path) as f:
        for line in f:
            m = pat_step.search(line)
            if not m:
                continue
            steps.append(int(m.group(1)))
            total.append(float(m.group(2)))
            for om in pat_oct.finditer(line):
                per_oct[f"oct{om.group(1)}"].append(float(om.group(2)))
            cm = pat_charb.search(line)
            if cm:
                charb.append(float(cm.group(1)))
            mm = pat_ms.search(line)
            if mm:
                ms_l1.append(float(mm.group(1)))
    return {
        "steps": steps,
        "total_loss": total,
        "per_octave": {k: v for k, v in per_oct.items() if v},
        "charb": charb,
        "ms_l1": ms_l1,
    }


def parse_val_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def write_findings_for(exp_dir: Path) -> dict:
    """Write findings.md inside exp_dir and return a summary dict."""
    name = exp_dir.name
    log = parse_run_log(exp_dir / "run.log")
    val_history = parse_val_history(exp_dir / "val_history.jsonl")
    manifest_path = exp_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    final_report_path = exp_dir / "final_report.json"
    final_report = json.loads(final_report_path.read_text()) if final_report_path.exists() else {}

    is_emission = "a4" in name or manifest.get("config", {}).get("model", {}).get("class") == "EmissionPredictorV2"

    md = exp_dir / "findings.md"
    with open(md, "w") as f:
        f.write(f"# {name} findings\n\n")
        f.write(f"## Manifest summary\n\n")
        f.write(f"- model: `{manifest.get('config', {}).get('model', {}).get('class', '?')}` "
                f"({manifest.get('config', {}).get('model', {}).get('encoder_size', '?')})\n")
        f.write(f"- offset: `{manifest.get('winning_offset')}` "
                f"(source: `{manifest.get('config', {}).get('offset', {}).get('source', '?')}`)\n")
        f.write(f"- d_search_max: `{manifest.get('d_search_max')}`\n")
        f.write(f"- world size: `{manifest.get('ddp_config', {}).get('world_size', '?')}`\n")
        f.write(f"- code_hash: `{manifest.get('code_hash', '?')[:16]}...`\n\n")

        f.write(f"## Training trajectory\n\n")
        if log["steps"]:
            n_steps = len(log["steps"])
            first_loss = log["total_loss"][0] if log["total_loss"] else None
            last_loss = log["total_loss"][-1] if log["total_loss"] else None
            min_loss = min(log["total_loss"]) if log["total_loss"] else None
            f.write(f"- steps logged: {n_steps}, max step: {max(log['steps']) if log['steps'] else 0}\n")
            f.write(f"- loss: first={first_loss:.4f}, last={last_loss:.4f}, min={min_loss:.4f}\n")
        else:
            f.write(f"- (no steps logged)\n")
        f.write("\n")

        if not is_emission and val_history:
            f.write(f"## Validation Window-FMR@95 history\n\n")
            f.write(f"- val passes: {len(val_history)}\n")
            last = val_history[-1]
            for variant, stats in last.get("window_fmr", {}).items():
                f.write(f"- {variant}: tau95={last['tau95'].get(variant, '?'):.4f}, "
                        f"worst_family={stats.get('worst_family_value', '?'):.4f} "
                        f"({stats.get('worst_family_name', '?')}), "
                        f"macro_avg={stats.get('macro_average', '?'):.4f}, "
                        f"pooled={stats.get('pooled', '?'):.4f}\n")
            f.write("\n")

        if final_report:
            f.write(f"## Final report (held-out)\n\n```\n")
            f.write(json.dumps(final_report, indent=2)[:4000])
            f.write("\n```\n\n")

    return {
        "name": name,
        "is_emission": is_emission,
        "log": log,
        "val_history": val_history,
        "manifest": manifest,
        "final_report": final_report,
    }


def make_comparison_plot(summaries: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax_loss, ax_oct, ax_fmr, ax_psnr = axes.flatten()

    for s in summaries:
        log = s["log"]
        if not log["steps"]:
            continue
        ax_loss.plot(log["steps"], log["total_loss"], label=s["name"])

    ax_loss.set_xlabel("step"); ax_loss.set_ylabel("total loss"); ax_loss.set_yscale("log")
    ax_loss.set_title("Training loss"); ax_loss.legend(fontsize=7); ax_loss.grid(alpha=0.3)

    # Per-octave loss for XOF runs (oct0 only — least demanding)
    for s in summaries:
        if s["is_emission"]:
            continue
        log = s["log"]
        oct0 = log["per_octave"].get("oct0", [])
        if oct0:
            ax_oct.plot(log["steps"][:len(oct0)], oct0, label=s["name"])
    ax_oct.set_xlabel("step"); ax_oct.set_ylabel("oct0 loss"); ax_oct.set_yscale("log")
    ax_oct.set_title("Per-octave (oct0) training loss"); ax_oct.legend(fontsize=7); ax_oct.grid(alpha=0.3)

    # Window-FMR@95 trajectory (worst_family on all_octaves variant)
    for s in summaries:
        if s["is_emission"]:
            continue
        vh = s["val_history"]
        if not vh:
            continue
        steps = [e["step"] for e in vh]
        worst = []
        for e in vh:
            wfmr = e.get("window_fmr", {})
            v = wfmr.get("all_octaves") or wfmr.get("oct0_plus_oct1") or wfmr.get("oct0_only") or {}
            worst.append(v.get("worst_family_value", float("nan")))
        ax_fmr.plot(steps, worst, label=s["name"])
    ax_fmr.set_xlabel("step"); ax_fmr.set_ylabel("worst-family Window-FMR@95")
    ax_fmr.set_title("Validation worst-family FMR (lower = better)"); ax_fmr.legend(fontsize=7); ax_fmr.grid(alpha=0.3)

    # Emission PSNR (final_report charb is in [0,1] units; convert to dB for legibility)
    has_psnr = False
    for s in summaries:
        if not s["is_emission"]:
            continue
        log = s["log"]
        if log["charb"]:
            # PSNR ≈ -20 log10(charb) for small charb (rough proxy on training-time L1)
            charb = np.array(log["charb"])
            psnr_proxy = -20 * np.log10(np.clip(charb, 1e-6, 1.0))
            ax_psnr.plot(log["steps"][:len(charb)], psnr_proxy, label=f"{s['name']} (proxy)")
            has_psnr = True
    if has_psnr:
        ax_psnr.set_xlabel("step"); ax_psnr.set_ylabel("PSNR proxy (dB)")
        ax_psnr.set_title("Emission training PSNR proxy from charb")
        ax_psnr.legend(fontsize=7); ax_psnr.grid(alpha=0.3)
    else:
        ax_psnr.text(0.5, 0.5, "no emission run completed yet", ha="center", va="center", transform=ax_psnr.transAxes)

    fig.suptitle("Phase D Matrix — exp001h_a0 / a1 / a2 / a4 / a6 / a7", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def write_phase_c_results(summaries: list[dict], out_path: Path) -> None:
    with open(out_path, "w") as f:
        f.write("# Phase D matrix results\n\n")
        f.write("## Per-experiment summary\n\n")
        f.write("| exp | encoder | task | last loss | min loss | val passes | final worst-family FMR |\n")
        f.write("|---|---|---|---:|---:|---:|---:|\n")
        for s in summaries:
            log = s["log"]
            last = log["total_loss"][-1] if log["total_loss"] else float("nan")
            mn = min(log["total_loss"]) if log["total_loss"] else float("nan")
            cfg_model = s["manifest"].get("config", {}).get("model", {})
            enc = cfg_model.get("encoder_size", "?")
            task = "emission" if s["is_emission"] else "XOF"
            n_val = len(s["val_history"])
            final_fmr = float("nan")
            if s["val_history"]:
                wf = s["val_history"][-1].get("window_fmr", {})
                v = wf.get("all_octaves") or wf.get("oct0_plus_oct1") or wf.get("oct0_only") or {}
                final_fmr = v.get("worst_family_value", float("nan"))
            f.write(f"| {s['name']} | {enc} | {task} | {last:.4f} | {mn:.4f} | {n_val} | {final_fmr:.4f} |\n")
        f.write("\n")

        f.write("## Notes\n\n")
        f.write("- Window-FMR@95 lower is better (tau95 = 95th percentile of positive scores; FMR = fraction of frames where any wrong candidate scores ≤ tau95).\n")
        f.write("- Emission task does not have a Window-FMR metric; the final_report.json contains PSNR + per-channel L1 instead.\n")
        f.write("- A6's normalization stats source is D2-train ONLY (transfer test); other variants use per-session stats.\n")
        f.write("- A0 only constructs heads for oct0+oct1; its FMR variants are limited to oct0_only and oct0_plus_oct1.\n")
        f.write("\n## See also\n\n")
        f.write("- `comparison.png` — training curves, validation FMR trajectory, emission PSNR proxy\n")
        f.write("- `experiments/<exp>/findings.md` — per-experiment writeup\n")
        f.write("- `experiments/<exp>/manifest.json` — full provenance\n")
        f.write("- `experiments/<exp>/final_report.json` — held-out evaluation (REPORT half + V10 bins)\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True,
                    help="Path to experiments/ directory (containing exp001h_a*/)")
    ap.add_argument("--out-md", type=Path, default=None)
    ap.add_argument("--out-png", type=Path, default=None)
    args = ap.parse_args()

    out_md = args.out_md or args.root / "PHASE_C_RESULTS.md"
    out_png = args.out_png or args.root / "comparison.png"

    summaries = []
    for name in EXPERIMENTS:
        d = args.root / name
        if not d.exists():
            print(f"[skip] {d} (not present)")
            continue
        s = write_findings_for(d)
        summaries.append(s)
        print(f"[done] {name}: {len(s['log']['steps'])} steps logged, "
              f"{len(s['val_history'])} val passes, final={'yes' if s['final_report'] else 'no'}")

    if not summaries:
        print("no experiment dirs found")
        return

    make_comparison_plot(summaries, out_png)
    write_phase_c_results(summaries, out_md)
    print(f"wrote {out_md}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()

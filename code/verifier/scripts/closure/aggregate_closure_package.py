"""Aggregates closure-package outputs into the CLOSURE_PACKAGE_RESULTS.md tables.

Reads:
  experiments/closure_package/exp001h_<v>/{final_report,zero_predictor,oracle_*}.json
  experiments/closure_package/a4_emission_ranking.json
  experiments/closure_package/exact_rgb_xof/final_report.json
  experiments/closure_package/exact_rgb_xof/val_history.jsonl
  experiments/closure_package/blur_noise_sweep.json

Writes:
  experiments/closure_package/CLOSURE_PACKAGE_RESULTS.md (overwrites with filled tables)

Run:
  python scripts/closure/aggregate_closure_package.py --root experiments/closure_package
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def load_json(path):
    if path.exists():
        return json.loads(path.read_text())
    return None


def fmt(v, decimals=3):
    if v is None or (isinstance(v, float) and (v != v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def primary_variant(score_variants_summary):
    """Pick the most-comprehensive variant (all_octaves > oct0_plus_oct1 > oct0_only)."""
    if not score_variants_summary:
        return None
    for v in ("all_octaves", "oct0_plus_oct1_plus_oct2", "oct0_plus_oct1", "oct0_only"):
        if v in score_variants_summary:
            return v
    return next(iter(score_variants_summary))


def task1_table(root: Path) -> str:
    rows = ["| variant | tau95 (calib) | worst-family FMR (report) | top-1 (report) |",
            "|---|---:|---:|---:|"]
    for v in ("a0", "a1", "a2", "a6", "a7"):
        d = load_json(root / f"exp001h_{v}" / "final_report.json")
        if not d:
            rows.append(f"| {v.upper()} | n/a | n/a | n/a |")
            continue
        # Calib tau95 from d2_val_calib split's tau95 (computed per-variant)
        calib_split = d["splits"].get("d2_val_calib") or {}
        report_split = d["splits"].get("d2_val_report") or {}
        pv_calib = primary_variant(calib_split.get("tau95", {}))
        pv_report = primary_variant(report_split.get("window_fmr", {}))
        tau95 = calib_split.get("tau95", {}).get(pv_calib) if pv_calib else None
        worst = report_split.get("window_fmr", {}).get(pv_report, {}).get("worst_family_value") if pv_report else None
        top1 = report_split.get("top1", {}).get(pv_report) if pv_report else None
        rows.append(f"| {v.upper()} | {fmt(tau95, 3)} | {fmt(worst, 3)} | {fmt(top1, 3)} |")
    return "\n".join(rows)


def task2_zero_table(root: Path) -> str:
    rows = ["| variant | zero-pred tau95 | trained tau95 | Δ | worst-family FMR |",
            "|---|---:|---:|---:|---:|"]
    for v in ("a0", "a1", "a2", "a6", "a7"):
        z = load_json(root / f"exp001h_{v}" / "zero_predictor.json")
        t = load_json(root / f"exp001h_{v}" / "final_report.json")
        if not z:
            rows.append(f"| {v.upper()} | n/a | n/a | n/a | n/a |")
            continue
        z_split = z["splits"].get("d2_val_calib") or z["splits"].get("d2_val_report") or {}
        pv = primary_variant(z_split.get("tau95", {}))
        z_tau = z_split.get("tau95", {}).get(pv) if pv else None
        z_worst = z_split.get("window_fmr", {}).get(pv, {}).get("worst_family_value") if pv else None
        # trained tau95 from d2_val_calib (same split)
        t_tau = None
        if t:
            t_split = t["splits"].get("d2_val_calib") or {}
            t_tau = t_split.get("tau95", {}).get(pv) if pv else None
        delta = (z_tau - t_tau) if (z_tau is not None and t_tau is not None) else None
        rows.append(f"| {v.upper()} | {fmt(z_tau, 4)} | {fmt(t_tau, 4)} | {fmt(delta, 4)} | {fmt(z_worst, 3)} |")
    return "\n".join(rows)


def task2_oracle_table(root: Path) -> str:
    rows = ["| noise std | top-1 (A0) | top-1 (A1) | top-1 (A2) | top-1 (A6) | top-1 (A7) |",
            "|---|---:|---:|---:|---:|---:|"]
    for noise_label, fname in [("0.00", "oracle_noiseless.json"), ("0.05", "oracle_noise_005.json")]:
        cells = [f"| {noise_label} "]
        for v in ("a0", "a1", "a2", "a6", "a7"):
            d = load_json(root / f"exp001h_{v}" / fname)
            if not d:
                cells.append("| n/a ")
                continue
            split = d["splits"].get("d2_val_report") or next(iter(d["splits"].values()))
            pv = primary_variant(split.get("top1", {}))
            top1 = split.get("top1", {}).get(pv) if pv else None
            cells.append(f"| {fmt(top1, 3)} ")
        cells.append("|")
        rows.append("".join(cells))
    return "\n".join(rows)


def task3_table(root: Path) -> str:
    rows = ["| split | matched PSNR (mean) | mismatched PSNR (mean) | gap (dB) | top-1 (1-of-33) |",
            "|---|---:|---:|---:|---:|"]
    d = load_json(root / "a4_emission_ranking.json")
    if not d:
        for s in ("D2_val_full", "V10_val_full", "V10_early", "V10_mid", "V10_late"):
            rows.append(f"| {s} | n/a | n/a | n/a | n/a |")
        return "\n".join(rows)
    for s in ("D2_val_full", "V10_val_full", "V10_early", "V10_mid", "V10_late"):
        ss = d["splits"].get(s)
        if not ss or ss.get("n", 0) == 0:
            rows.append(f"| {s} | n/a | n/a | n/a | n/a |")
            continue
        matched = ss.get("matched_psnr_mean")
        cross = ss.get("cross_pair_psnr_mean")
        gap = ss.get("matched_minus_cross_pair_db")
        top1 = ss.get("top1_retrieval_rate")
        rows.append(f"| {s} | {fmt(matched, 2)} | {fmt(cross, 2)} | {fmt(gap, 2)} | {fmt(top1, 3)} |")
    return "\n".join(rows)


def task4_table(root: Path) -> str:
    rows = ["| octave | clean RGB byte recovery | clean RGB matched score |",
            "|---|---:|---:|"]
    d = load_json(root / "exact_rgb_xof" / "final_report.json")
    if not d:
        for o in range(4):
            rows.append(f"| oct{o} | n/a | n/a |")
        return "\n".join(rows)
    rep = d.get("report", {})
    by_oct = rep.get("byte_match_per_octave", {})
    matched = rep.get("matched_score", {})
    for o in range(4):
        bm = by_oct.get(f"oct{o}")
        # Per-octave matched score isn't directly returned (we have score variants);
        # report all_octaves on row 0 and per-octave bit recovery in the others
        rows.append(f"| oct{o} | {fmt(bm, 4)} | (see all_octaves total: {fmt(matched.get('all_octaves'), 4)}) |")
    return "\n".join(rows)


def task5_section(root: Path) -> str:
    d = load_json(root / "blur_noise_sweep.json")
    if not d:
        return "(Task 5 results not yet available.)"
    lines = ["| corruption | oct0 | oct1 | oct2 | oct3 | matched(all) |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in d.get("results", []):
        # Format corruption description
        parts = []
        if r.get("blur_sigma", 0) > 0:
            parts.append(f"blur σ={r['blur_sigma']}")
        if r.get("noise_std", 0) > 0:
            parts.append(f"noise σ={r['noise_std']}")
        if r.get("downsample", 1) > 1:
            parts.append(f"down ×{r['downsample']}")
        if r.get("gamma", 1.0) != 1.0:
            parts.append(f"γ={r['gamma']}")
        label = ", ".join(parts) if parts else "clean"
        bm = r.get("byte_match_per_octave", {})
        matched_all = r.get("matched_score_mean", {}).get("all_octaves")
        lines.append(f"| {label} | {fmt(bm.get('oct0'), 3)} | {fmt(bm.get('oct1'), 3)} | "
                     f"{fmt(bm.get('oct2'), 3)} | {fmt(bm.get('oct3'), 3)} | {fmt(matched_all, 3)} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path,
                    help="experiments/closure_package directory")
    ap.add_argument("--template", type=Path,
                    default=None,
                    help="Path to existing CLOSURE_PACKAGE_RESULTS.md (template). "
                         "If omitted, looks at <root>/CLOSURE_PACKAGE_RESULTS.md")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output path (default: <root>/CLOSURE_PACKAGE_RESULTS.md)")
    args = ap.parse_args()

    template_path = args.template or (args.root / "CLOSURE_PACKAGE_RESULTS.md")
    out_path = args.out or (args.root / "CLOSURE_PACKAGE_RESULTS.md")

    if not template_path.exists():
        raise SystemExit(f"template not found: {template_path}")
    text = template_path.read_text()

    # Each task table replaces the corresponding placeholder block.
    # We delimit blocks with sentinels — first parse the existing tables and
    # replace just the contents using table headers.
    replacements = {
        "TASK1_TABLE": task1_table(args.root),
        "TASK2_ZERO_TABLE": task2_zero_table(args.root),
        "TASK2_ORACLE_TABLE": task2_oracle_table(args.root),
        "TASK3_TABLE": task3_table(args.root),
        "TASK4_TABLE": task4_table(args.root),
        "TASK5_SECTION": task5_section(args.root),
    }

    # Replace tables by header pattern matching
    import re

    def replace_table(text, header_line, new_table):
        # Find the existing table header and replace through to the next blank line
        pat = re.compile(rf"^{re.escape(header_line)}\n\|.*?\n((?:\|.*?\n)*?)(?=\n|\Z)", re.MULTILINE)
        return pat.sub(new_table + "\n", text, count=1)

    text = replace_table(text, "| variant | tau95 (calib) | worst-family FMR (report) | top-1 (report) |",
                          replacements["TASK1_TABLE"])
    text = replace_table(text, "| variant | zero-pred tau95 | trained tau95 | Δ | worst-family FMR |",
                          replacements["TASK2_ZERO_TABLE"])
    text = replace_table(text, "| noise std | top-1 (A0) | top-1 (A1) | top-1 (A2) | top-1 (A6) | top-1 (A7) |",
                          replacements["TASK2_ORACLE_TABLE"])
    text = replace_table(text, "| split | matched PSNR (mean) | mismatched PSNR (mean) | gap (dB) | top-1 (1-of-33) |",
                          replacements["TASK3_TABLE"])
    text = replace_table(text, "| octave | clean RGB byte recovery | clean RGB matched score |",
                          replacements["TASK4_TABLE"])

    # Task 5: append the new section before "## Implications" if present
    if "## Implications for the next direction" in text:
        # Insert sweep table before that heading
        sweep_block = "\n" + replacements["TASK5_SECTION"] + "\n"
        # find existing "(Plots saved..." line and replace following with sweep_block
        text = re.sub(r"(\(Plots saved.*?\.\))\n",
                      r"\1\n" + sweep_block + "\n",
                      text, count=1, flags=re.DOTALL)

    out_path.write_text(text)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

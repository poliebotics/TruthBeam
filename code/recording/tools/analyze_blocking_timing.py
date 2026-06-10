#!/usr/bin/env python3
"""Timing-only diagnostic for blocking-mode sessions.

Independent of optical content. Validates that:
  1. Every chain step corresponds to exactly one delivered capture.
  2. Inter-capture intervals match the blocking cadence predicted from the
     pipeline-wait configuration.
  3. No row violated the pipeline-wait discipline
     (capture_wall_ns[t] >= tile_queued_wall_ns[t-1] + pipeline_wait_ns).
  4. Handler durations never exceeded the pipeline wait.
  5. Camera VSYNC deltas (device timestamps) are regular.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Must match tb_loop.py:READOUT_NS. The analyzer estimates trigger time
# (exposure start in the host clock) as capture_wall_ns - READOUT_NS and
# validates that trigger time landed on or after the previous
# queue_draw_wall_ns + pipeline_wait_ms. Keep in sync.
READOUT_NS = 70_000_000


def _read_csv_with_note(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        first = f.readline()
        if not first.startswith("#"):
            f.seek(0)
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _coerce(rows: list[dict], fields: dict[str, type]) -> dict[str, np.ndarray]:
    out = {}
    for name, t in fields.items():
        vals = []
        for r in rows:
            v = r.get(name, "")
            if v == "" or v is None:
                vals.append(None)
            else:
                try:
                    vals.append(t(v))
                except ValueError:
                    vals.append(None)
        if t is int:
            arr = np.array([-1 if v is None else v for v in vals], dtype=np.int64)
        elif t is str:
            arr = np.array(["" if v is None else v for v in vals], dtype=object)
        else:
            arr = np.array([np.nan if v is None else v for v in vals], dtype=np.float64)
        out[name] = arr
    return out


def load_pipeline_wait_ms(session: Path, cli_override: float | None) -> tuple[float, str]:
    if cli_override is not None:
        return cli_override, "cli"
    meta = session / "session_metadata.json"
    if meta.exists():
        try:
            m = json.loads(meta.read_text())
            if "pipeline_wait_ms" in m:
                return float(m["pipeline_wait_ms"]), f"session_metadata.json: {meta}"
        except Exception:
            pass
    # fallback: read rig calibration in ~/.tb
    tb_dir = Path.home() / ".tb"
    cals = sorted(tb_dir.glob("pipeline_calibration_*.json"))
    if cals:
        try:
            c = json.loads(cals[-1].read_text())
            if "recommended_wait_ms" in c:
                return float(c["recommended_wait_ms"]), f"rig calibration: {cals[-1]}"
        except Exception:
            pass
    return 120.0, "default (120 ms)"


def quantiles(x: np.ndarray, qs: list[float]) -> dict[str, float]:
    if len(x) == 0:
        return {f"p{int(q*100)}": float("nan") for q in qs}
    return {f"p{int(q*100)}": float(np.quantile(x, q)) for q in qs}


def analyze(session: Path, pipeline_wait_ms: float, duration_s_hint: float | None) -> dict:
    chain = _read_csv_with_note(session / "chain_log.csv")
    capture = _read_csv_with_note(session / "capture_log.csv")

    chain_fields = {
        "t": int,
        "capture_frame_id": int,
        "aravis_device_timestamp_ns": int,
        "capture_wall_ns": int,
        "tile_queued_wall_ns": int,
    }
    cap_fields = {
        "capture_idx": int,
        "capture_frame_id": int,
        "aravis_device_timestamp_ns": int,
        "capture_wall_ns": int,
        "consumed_into_chain": str,
        "consumed_as_t": int,
    }
    c = _coerce(chain, chain_fields)
    cap = _coerce(capture, cap_fields)

    n_chain = len(chain)
    n_capture = len(capture)
    n_consumed = int((np.char.lower(cap["consumed_into_chain"].astype(str)) == "true").sum())

    cap_wall = c["capture_wall_ns"].astype(np.float64)
    queued = c["tile_queued_wall_ns"].astype(np.float64)
    dev_ts = c["aravis_device_timestamp_ns"].astype(np.float64)
    t_col = c["t"].astype(np.int64)
    frame_ids = c["capture_frame_id"].astype(np.int64)

    order = np.argsort(t_col)
    t_col = t_col[order]; cap_wall = cap_wall[order]; queued = queued[order]
    dev_ts = dev_ts[order]; frame_ids = frame_ids[order]

    # handler duration: per-row time from capture -> queue_draw
    handler_ns = queued - cap_wall
    # inter-capture wall gap (reference: capture arrivals on wall clock)
    inter_cap_ns = np.diff(cap_wall)
    # pipeline wait ACTUAL: gap between queued[t-1] and capture_wall[t]
    wait_actual_ns = cap_wall[1:] - queued[:-1]
    # device-timestamp deltas (hardware VSYNC)
    dev_dt_ns = np.diff(dev_ts)
    # camera frame-id deltas (should be 1 per consumed capture; gaps indicate
    # trigger drops or camera-side frames not consumed)
    fid_dt = np.diff(frame_ids)

    wait_budget_ns = pipeline_wait_ms * 1e6
    handler_violations = int((handler_ns > wait_budget_ns).sum())
    # Wait discipline on trigger time, not receive time: a frame whose
    # exposure began before queue_draw + pipeline_wait_ms is stale, even
    # if its delivery landed after. trigger_host_ns ≈ capture_wall_ns
    # - READOUT_NS; the -1 ns slack tolerates int rounding around the
    # threshold boundary.
    trigger_wait_ns = wait_actual_ns - READOUT_NS
    wait_violations = int((trigger_wait_ns + 1 < wait_budget_ns).sum())
    wait_violation_idx = np.where(trigger_wait_ns + 1 < wait_budget_ns)[0].tolist()[:10]

    duration_measured_s = (cap_wall[-1] - cap_wall[0]) / 1e9 if n_chain >= 2 else 0.0
    median_inter_ms = float(np.median(inter_cap_ns)) / 1e6 if inter_cap_ns.size else float("nan")
    measured_rate_hz = 1000.0 / median_inter_ms if median_inter_ms > 0 else float("nan")
    expected_chain_rate = 1000.0 / (pipeline_wait_ms + 80.0)

    qs = [0.05, 0.5, 0.95, 0.99]
    inter_q = quantiles(inter_cap_ns, qs)
    handler_q = quantiles(handler_ns, qs)
    wait_q = quantiles(wait_actual_ns, qs)
    dev_q = quantiles(dev_dt_ns, qs)

    # vsync gaps: a dropped trigger would look like dev_dt > ~2 * nominal.
    # 60 Hz vsync = 16.67 ms. Between two CONSUMED captures we expect some
    # integer multiple of 16.67 ms; single-step normal at 5-6 vsyncs.
    nominal_vsync_ms = 1000.0 / 60.0
    dev_dt_vsyncs = dev_dt_ns / 1e6 / nominal_vsync_ms
    # a "gap" = difference between successive consumed captures that is
    # significantly larger than the median (>150%).
    if dev_dt_ns.size:
        median_dev_ms = float(np.median(dev_dt_ns)) / 1e6
        gap_threshold = 1.5 * median_dev_ms
        gaps_idx = np.where(dev_dt_ns / 1e6 > gap_threshold)[0].tolist()
    else:
        median_dev_ms = float("nan"); gaps_idx = []

    # inter-capture "stalls": wall > 2x median or > 500 ms
    stall_threshold_ns = max(2.0 * float(np.median(inter_cap_ns)) if inter_cap_ns.size else 0, 500e6)
    stall_idx = np.where(inter_cap_ns > stall_threshold_ns)[0].tolist()

    # expected chain steps given measured duration and measured rate — we
    # reason relative to what actually happened, not a nominal 5 Hz claim.
    expected_chain_from_duration = int(duration_measured_s * measured_rate_hz) if measured_rate_hz > 0 else 0

    # slip-rate prediction for async mode: fraction of handlers that would
    # have overrun the 16.67 ms vsync budget if the same code ran once per
    # camera frame.
    vsync_budget_ns = int(nominal_vsync_ms * 1e6)
    async_overrun_fraction = float((handler_ns > vsync_budget_ns).mean()) if handler_ns.size else 0.0

    # pass/fail
    checks = {
        "every_chain_step_has_one_capture": n_chain == n_consumed,
        "no_wait_discipline_violations": wait_violations == 0,
        "no_handler_overruns_vs_pipeline_wait": handler_violations == 0,
        "median_inter_capture_in_tolerance": (
            inter_cap_ns.size > 0 and
            abs(median_inter_ms - (pipeline_wait_ms + 80.0)) <= 0.2 * (pipeline_wait_ms + 80.0)
        ),
        "p99_inter_capture_below_400ms": inter_cap_ns.size > 0 and inter_q["p99"] < 400e6,
        "no_stalls_over_500ms": len(stall_idx) == 0,
        "no_vsync_gaps": len(gaps_idx) == 0,
    }
    passed = all(checks.values())

    return {
        "session": str(session),
        "pipeline_wait_ms": pipeline_wait_ms,
        "n_chain": n_chain,
        "n_capture": n_capture,
        "n_consumed": n_consumed,
        "duration_measured_s": duration_measured_s,
        "measured_rate_hz": measured_rate_hz,
        "predicted_rate_hz": expected_chain_rate,
        "inter_capture_ms": {
            "min": float(inter_cap_ns.min()) / 1e6 if inter_cap_ns.size else float("nan"),
            "median": median_inter_ms,
            "max": float(inter_cap_ns.max()) / 1e6 if inter_cap_ns.size else float("nan"),
            **{k: v / 1e6 for k, v in inter_q.items()},
        },
        "handler_duration_ms": {
            "min": float(handler_ns.min()) / 1e6 if handler_ns.size else float("nan"),
            "median": float(np.median(handler_ns)) / 1e6 if handler_ns.size else float("nan"),
            "max": float(handler_ns.max()) / 1e6 if handler_ns.size else float("nan"),
            **{k: v / 1e6 for k, v in handler_q.items()},
        },
        "pipeline_wait_actual_ms": {
            "min": float(wait_actual_ns.min()) / 1e6 if wait_actual_ns.size else float("nan"),
            "median": float(np.median(wait_actual_ns)) / 1e6 if wait_actual_ns.size else float("nan"),
            "max": float(wait_actual_ns.max()) / 1e6 if wait_actual_ns.size else float("nan"),
            **{k: v / 1e6 for k, v in wait_q.items()},
        },
        "device_timestamp_delta_ms": {
            "median": median_dev_ms,
            **{k: v / 1e6 for k, v in dev_q.items()},
        },
        "vsync_nominal_ms": nominal_vsync_ms,
        "handler_overruns_over_pipeline_wait": handler_violations,
        "wait_discipline_violations": wait_violations,
        "wait_violation_row_idx": wait_violation_idx,
        "stall_idx": stall_idx,
        "vsync_gap_idx": gaps_idx,
        "async_mode_overrun_rate_estimate": async_overrun_fraction,
        "checks": checks,
        "passed": passed,
        # per-row arrays for CSV dump
        "_per_row": {
            "t": t_col,
            "capture_wall_ns": cap_wall,
            "tile_queued_wall_ns": queued,
            "handler_ns": handler_ns,
            "inter_capture_ns": np.concatenate([[0], inter_cap_ns]),
            "pipeline_wait_actual_ns": np.concatenate([[0], wait_actual_ns]),
            "device_timestamp_ns": dev_ts,
            "device_dt_ns": np.concatenate([[0], dev_dt_ns]),
            "capture_frame_id": frame_ids,
            "frame_id_dt": np.concatenate([[0], fid_dt]),
        },
    }


def write_csv(report: dict, path: Path) -> None:
    pr = report["_per_row"]
    keys = list(pr.keys())
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        n = len(pr["t"])
        for i in range(n):
            w.writerow([pr[k][i] for k in keys])


def write_plots(report: dict, out_dir: Path) -> None:
    pr = report["_per_row"]
    inter_ms = pr["inter_capture_ns"][1:] / 1e6
    handler_ms = pr["handler_ns"] / 1e6
    wait_ms = pr["pipeline_wait_actual_ns"][1:] / 1e6

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(inter_ms, bins=60, color="steelblue", edgecolor="black")
    axes[0].axvline(report["inter_capture_ms"]["median"], color="red", linestyle="--",
                    label=f'median {report["inter_capture_ms"]["median"]:.1f} ms')
    axes[0].axvline(report["inter_capture_ms"]["p99"], color="orange", linestyle=":",
                    label=f'p99 {report["inter_capture_ms"]["p99"]:.1f} ms')
    axes[0].set_xlabel("inter-capture wall gap (ms)")
    axes[0].set_ylabel("count")
    axes[0].set_title("inter-capture distribution")
    axes[0].legend()

    axes[1].hist(handler_ms, bins=60, color="olivedrab", edgecolor="black")
    axes[1].axvline(report["pipeline_wait_ms"], color="red", linestyle="--",
                    label=f'pipeline_wait {report["pipeline_wait_ms"]:.0f} ms')
    axes[1].axvline(report["vsync_nominal_ms"], color="purple", linestyle=":",
                    label=f'vsync 16.67 ms')
    axes[1].set_xlabel("handler duration (ms)")
    axes[1].set_title(f"handler duration  (over pipeline_wait: "
                      f"{report['handler_overruns_over_pipeline_wait']})")
    axes[1].legend()

    axes[2].hist(wait_ms, bins=60, color="darkorange", edgecolor="black")
    axes[2].axvline(report["pipeline_wait_ms"], color="red", linestyle="--",
                    label=f'config {report["pipeline_wait_ms"]:.0f} ms')
    axes[2].set_xlabel("pipeline_wait actual (ms)")
    axes[2].set_title(f"post-queue wait before next capture  "
                      f"(violations: {report['wait_discipline_violations']})")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "timing_distribution.png", dpi=120)
    plt.close(fig)

    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    t = pr["t"]
    axes2[0].plot(t[1:], inter_ms, ".", markersize=3, color="steelblue")
    axes2[0].axhline(report["inter_capture_ms"]["median"], color="red", linestyle="--")
    axes2[0].set_ylabel("inter-capture (ms)")
    axes2[0].set_title("timing trace (lower = higher chain rate)")
    axes2[1].plot(t, handler_ms, ".", markersize=3, color="olivedrab")
    axes2[1].axhline(report["pipeline_wait_ms"], color="red", linestyle="--",
                     label=f'pipeline_wait {report["pipeline_wait_ms"]:.0f} ms')
    axes2[1].axhline(report["vsync_nominal_ms"], color="purple", linestyle=":",
                     label=f'vsync 16.67 ms')
    axes2[1].set_ylabel("handler (ms)")
    axes2[1].set_xlabel("chain step t")
    axes2[1].legend()
    fig2.tight_layout()
    fig2.savefig(out_dir / "inter_capture_intervals.png", dpi=120)
    plt.close(fig2)


def write_markdown(report: dict, path: Path) -> None:
    c = report["checks"]
    lines = [
        f"# Blocking-mode timing diagnostic",
        "",
        f"- session: `{report['session']}`",
        f"- pipeline_wait_ms (configured): {report['pipeline_wait_ms']:.0f}",
        f"- n_chain: {report['n_chain']}  n_capture: {report['n_capture']}  "
        f"n_consumed: {report['n_consumed']}",
        f"- measured duration: {report['duration_measured_s']:.1f} s",
        f"- measured chain rate: {report['measured_rate_hz']:.3f} Hz  "
        f"(naive predicted: {report['predicted_rate_hz']:.3f} Hz)",
        "",
        "## Pass/fail",
        "",
        f"- **overall: {'PASS' if report['passed'] else 'FAIL'}**",
        "",
    ]
    for k, v in c.items():
        lines.append(f"- {'OK' if v else 'FAIL'} — {k}")
    lines += [
        "",
        "## Inter-capture wall-clock gap (ms)",
        "",
        "| min | p05 | median | p95 | p99 | max |",
        "|----:|----:|-------:|----:|----:|----:|",
        (f"| {report['inter_capture_ms']['min']:.2f} "
         f"| {report['inter_capture_ms']['p5']:.2f} "
         f"| {report['inter_capture_ms']['median']:.2f} "
         f"| {report['inter_capture_ms']['p95']:.2f} "
         f"| {report['inter_capture_ms']['p99']:.2f} "
         f"| {report['inter_capture_ms']['max']:.2f} |"),
        "",
        "## Handler duration (ms) — capture arrival to tile-queued",
        "",
        "| min | p05 | median | p95 | p99 | max |",
        "|----:|----:|-------:|----:|----:|----:|",
        (f"| {report['handler_duration_ms']['min']:.2f} "
         f"| {report['handler_duration_ms']['p5']:.2f} "
         f"| {report['handler_duration_ms']['median']:.2f} "
         f"| {report['handler_duration_ms']['p95']:.2f} "
         f"| {report['handler_duration_ms']['p99']:.2f} "
         f"| {report['handler_duration_ms']['max']:.2f} |"),
        "",
        "## Pipeline-wait actual (ms) — tile-queued to next capture",
        "",
        "| min | p05 | median | p95 | p99 | max |",
        "|----:|----:|-------:|----:|----:|----:|",
        (f"| {report['pipeline_wait_actual_ms']['min']:.2f} "
         f"| {report['pipeline_wait_actual_ms']['p5']:.2f} "
         f"| {report['pipeline_wait_actual_ms']['median']:.2f} "
         f"| {report['pipeline_wait_actual_ms']['p95']:.2f} "
         f"| {report['pipeline_wait_actual_ms']['p99']:.2f} "
         f"| {report['pipeline_wait_actual_ms']['max']:.2f} |"),
        "",
        "## Camera device-timestamp delta (ms) between consumed captures",
        "",
        (f"- median: {report['device_timestamp_delta_ms']['median']:.2f} ms"
         f"  (nominal vsync {report['vsync_nominal_ms']:.2f} ms)"),
        (f"- p95/p99: {report['device_timestamp_delta_ms']['p95']:.2f} / "
         f"{report['device_timestamp_delta_ms']['p99']:.2f}"),
        f"- vsync gaps detected: {len(report['vsync_gap_idx'])} "
        f"(indices: {report['vsync_gap_idx'][:10]}{'…' if len(report['vsync_gap_idx'])>10 else ''})",
        "",
        "## Anomalies",
        "",
        f"- handler overruns (> pipeline_wait): {report['handler_overruns_over_pipeline_wait']}",
        f"- wait-discipline violations: {report['wait_discipline_violations']}"
        + (f"  (first indices: {report['wait_violation_row_idx']})"
           if report['wait_violation_row_idx'] else ""),
        f"- stalls > max(2×median, 500 ms): {len(report['stall_idx'])} events",
        "",
        "## Async-mode slip-rate prediction",
        "",
        (f"Fraction of handlers whose wall duration exceeded the 60 Hz vsync "
         f"budget ({report['vsync_nominal_ms']:.2f} ms): "
         f"**{report['async_mode_overrun_rate_estimate']*100:.1f}%**. "
         f"This is a lower-bound estimate for async-mode slip: a handler that "
         f"overruns the vsync budget cannot consume the next camera frame, "
         f"so the chain falls behind by at least one vsync interval."),
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path,
                    help="session directory with chain_log.csv + capture_log.csv")
    ap.add_argument("--pipeline-wait-ms", type=float, default=None,
                    help="override; else read from session_metadata.json or "
                         "rig calibration, else 120 ms default.")
    ap.add_argument("--duration-s", type=float, default=None,
                    help="expected duration for sanity comparison (optional)")
    args = ap.parse_args(argv)

    session = args.session.resolve()
    pwait_ms, src = load_pipeline_wait_ms(session, args.pipeline_wait_ms)
    print(f"pipeline_wait_ms = {pwait_ms:.0f} (source: {src})")

    report = analyze(session, pwait_ms, args.duration_s)

    out = session / "timing_analysis.md"
    out_csv = session / "timing_rows.csv"
    write_markdown(report, out)
    write_csv(report, out_csv)
    write_plots(report, session)
    report.pop("_per_row", None)
    (session / "timing_analysis.json").write_text(json.dumps(report, indent=2))

    print(f"n_chain={report['n_chain']} n_consumed={report['n_consumed']}")
    print(f"measured {report['measured_rate_hz']:.3f} Hz over "
          f"{report['duration_measured_s']:.1f} s")
    print(f"pass: {report['passed']}")
    for k, v in report["checks"].items():
        print(f"  {'OK' if v else 'FAIL'}  {k}")
    print(f"report: {out}")
    print(f"csv:    {out_csv}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

#!/usr/bin/env python3
"""Empirical pipeline-delay calibration for v8 blocking-mode sessions.

Projects alternating white/black tiles at varying wait durations and
measures how long it takes for the DMD (plus HDMI buffering + any other
pipeline stages) to reach the new state. Output is a JSON file at
~/.tb/pipeline_calibration_<rig_hash>.json that tb_loop.py loads at
session start when --mode blocking is in effect.

Method:
  1. Display solid WHITE tile, wait STABLE_MS to let the rig settle.
  2. For each wait_ms in the sweep list:
     For each trial in range(N_TRIALS):
       - Emit solid BLACK tile via queue_draw
       - time.sleep(wait_ms / 1000)
       - Drain any captures queued before the wait window closed
       - Block for the next capture (its exposure started >= wait_ms
         after the black emission)
       - Record mean(raw_bayer_bytes) of that capture
       - Re-emit solid WHITE, wait STABLE_MS to restore baseline
  3. Repeat the above in the reverse direction (BLACK→WHITE), starting
     from a stable BLACK baseline.

Classification:
  - stable_white_mean, stable_black_mean: mean intensity in each
    stable phase (measured from calibration captures).
  - threshold = (stable_white_mean + stable_black_mean) / 2
  - A capture whose mean crosses the threshold (below, for W→B; above,
    for B→W) is classified as "transitioned". Borderline captures
    (mean on the wrong side of the threshold) are conservatively
    classified as "still old tile" — that is what the spec calls for.
  - per_wait_correct_fraction is the fraction of trials at each
    wait_ms that were classified as "transitioned".
  - p99_wait_ms: smallest wait_ms where correct_fraction >= 0.99.
  - p100_wait_ms: smallest wait_ms where correct_fraction == 1.00.
  - recommended_wait_ms = max(W_to_B.p100, B_to_W.p100) + SAFETY_MS.

CLI:
  python3 calibrate_pipeline_delay.py --connector HDMI-1
                                       --n-trials 100
                                       --output ~/.tb/
"""

import os
os.environ.setdefault("WAYLAND_DISPLAY", "wayland-0")
os.environ.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
os.environ.setdefault("GDK_BACKEND", "wayland")

import argparse
import datetime as dt
import statistics
import sys
import threading
import time
from pathlib import Path

# tools/ scripts reach protocol/ via sys.path insertion — keeps the layout
# flat-executable while keeping the protocol/ dir as the canonical home of
# shared code like session_schema.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "protocol"))

import numpy as np
from blake3 import blake3

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
gi.require_version("Aravis", "0.8")
from gi.repository import Aravis, Gdk, GLib, Gtk  # noqa: E402

from session_schema import canonical_json_bytes


TILE_W, TILE_H = 1920, 1080
STABLE_MS = 500         # wait after restoring baseline between trials
SAFETY_MS = 20          # added to the worst p100 when recommending
DEFAULT_WAIT_SWEEP_MS = [0, 50, 100, 150, 200, 250, 300,
                          350, 400, 450, 500, 600, 700, 800]


def _probe_edid_fingerprint(connector):
    candidates = [connector]
    if "-A-" not in connector:
        candidates.append(connector.replace("HDMI-", "HDMI-A-"))
    if "-A-" in connector:
        candidates.append(connector.replace("HDMI-A-", "HDMI-"))
    drm_root = Path("/sys/class/drm")
    if not drm_root.exists():
        return None
    for entry in drm_root.iterdir():
        for cand in candidates:
            if entry.name.endswith(f"-{cand}"):
                edid_path = entry / "edid"
                if edid_path.exists():
                    try:
                        data = edid_path.read_bytes()
                    except OSError:
                        continue
                    if data:
                        return blake3(data).hexdigest()
    return None


def _probe_camera_identity(device, dev_id):
    serial = None
    for feat in ("DeviceSerialNumber", "DeviceID"):
        try:
            v = device.get_string_feature_value(feat)
            if v:
                serial = str(v).strip()
                break
        except Exception:
            pass
    if serial is None and dev_id:
        tail = ""
        for ch in reversed(dev_id):
            if ch.isdigit():
                tail = ch + tail
            else:
                break
        if tail:
            serial = tail
    firmware = None
    try:
        firmware = device.get_string_feature_value("DeviceFirmwareVersion")
        if firmware:
            firmware = str(firmware).strip()
    except Exception:
        pass
    return serial, firmware


def compute_rig_hash(rig_config: dict) -> str:
    """Canonical-JSON blake3 of the rig_config dict. Same algorithm as
    bundle_hash / manifest_hash_* so verifiers can reproduce it."""
    return blake3(canonical_json_bytes(rig_config)).hexdigest()


class CalibrationApp(Gtk.Application):
    def __init__(self, monitor, exposure_us, gain_db, n_trials,
                 wait_sweep_ms, format_str="BayerRG8",
                 output_dir=None, report_dir=None):
        super().__init__(application_id="local.tb.calibrate")
        self.monitor = monitor
        self.exposure_us = exposure_us
        self.gain_db = gain_db
        self.n_trials = n_trials
        self.wait_sweep_ms = list(wait_sweep_ms)
        self.format_str = format_str
        self.output_dir = Path(output_dir or (Path.home() / ".tb")).expanduser()
        self.report_dir = Path(report_dir or "reports").expanduser()

        self.window = None
        self.area = None
        self._camera = None
        self._stream = None
        self._device = None
        self.real_roi = None
        self.payload = None
        self.camera_serial = None
        self.camera_firmware = None
        self.edid_fp = None

        # Tile state shown by _draw(). Updated from worker thread via
        # GLib.idle_add(_set_state, color). Binary: "white" or "black".
        self.current_color = "white"
        self._draw_lock = threading.Lock()

        # Results.
        self.stable_white_samples = []
        self.stable_black_samples = []
        self.w_to_b = {wait_ms: [] for wait_ms in self.wait_sweep_ms}
        self.b_to_w = {wait_ms: [] for wait_ms in self.wait_sweep_ms}
        self.error = None
        self.calibration = None
        self.calibration_path = None
        self.report_path = None

    def do_activate(self):
        self.window = Gtk.ApplicationWindow(application=self)
        self.window.set_decorated(False)
        self.area = Gtk.DrawingArea()
        self.area.set_draw_func(self._draw)
        self.window.set_child(self.area)
        self.window.fullscreen_on_monitor(self.monitor["monitor"])
        try:
            cur = Gdk.Cursor.new_from_name("none", None)
            self.window.set_cursor(cur)
        except Exception:
            pass
        self.window.present()
        GLib.timeout_add(300, self._start)

    def _draw(self, area, cr, w, h):
        with self._draw_lock:
            color = self.current_color
        if color == "white":
            cr.set_source_rgb(1, 1, 1)
        else:
            cr.set_source_rgb(0, 0, 0)
        cr.rectangle(0, 0, w, h)
        cr.fill()

    def _set_color(self, color):
        """Blocking tile swap: schedule on GTK main thread, wait for it
        to actually redraw. Returns the wall_ns at the moment the
        idle_add callback fired on the GTK thread."""
        evt = threading.Event()
        wall = [None]

        def _post():
            with self._draw_lock:
                self.current_color = color
            wall[0] = time.monotonic_ns()
            self.area.queue_draw()
            evt.set()
            return False

        GLib.idle_add(_post)
        evt.wait(timeout=2.0)
        return wall[0] if wall[0] is not None else time.monotonic_ns()

    def _start(self):
        try:
            self._setup_camera()
            t = threading.Thread(target=self._run_calibration, daemon=True,
                                 name="tb-calib")
            t.start()
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            import traceback
            traceback.print_exc()
            GLib.idle_add(self.quit)
        return False

    def _setup_camera(self):
        Aravis.update_device_list()
        if Aravis.get_n_devices() == 0:
            raise RuntimeError("no Aravis cameras")
        dev_id = Aravis.get_device_id(0)
        print(f"[calib] opening camera: {dev_id}", flush=True)
        self._camera = Aravis.Camera.new(dev_id)
        self._camera.set_pixel_format_from_string(self.format_str)
        self._device = self._camera.get_device()

        self.camera_serial, self.camera_firmware = _probe_camera_identity(
            self._device, dev_id
        )

        sw, sh = self._camera.get_sensor_size()
        for name, value in (("OffsetX", 0), ("OffsetY", 0),
                            ("Width", sw), ("Height", sh)):
            try:
                self._device.set_integer_feature_value(name, int(value))
            except Exception:
                pass
        rx = self._device.get_integer_feature_value("OffsetX")
        ry = self._device.get_integer_feature_value("OffsetY")
        rw = self._device.get_integer_feature_value("Width")
        rh = self._device.get_integer_feature_value("Height")
        self.real_roi = (rx, ry, rw, rh)
        self.payload = self._camera.get_payload()

        for n, v in (("ExposureAuto", "Off"),
                     ("GainAuto", "Off"),
                     ("BalanceWhiteAuto", "Off")):
            try:
                self._device.set_string_feature_value(n, v)
            except Exception:
                pass
        self._device.set_float_feature_value("ExposureTime", float(self.exposure_us))
        self._device.set_float_feature_value("Gain", float(self.gain_db))
        for n, v in (("TriggerSelector", "FrameStart"),
                     ("TriggerMode", "On"),
                     ("TriggerSource", "Line1"),
                     ("TriggerActivation", "RisingEdge")):
            self._device.set_string_feature_value(n, v)

        self._stream = self._camera.create_stream(None, None)
        for _ in range(32):
            self._stream.push_buffer(Aravis.Buffer.new_allocate(self.payload))
        self._camera.start_acquisition()
        self.edid_fp = _probe_edid_fingerprint(self.monitor["connector"])
        print(f"[calib] ROI={rw}x{rh}  exposure={self.exposure_us}us  "
              f"gain={self.gain_db}dB", flush=True)
        print(f"[calib] camera_serial={self.camera_serial!r}  "
              f"firmware={self.camera_firmware!r}", flush=True)
        print(f"[calib] edid_fingerprint="
              f"{(self.edid_fp or 'null')[:16]}…", flush=True)

    def _drain_and_pop(self, after_wall_ns, timeout_us=1_000_000):
        """Unconditionally drain every currently-queued buffer, then
        block for the next freshly-delivered one.

        BUG FIX (2026-04-19): the previous implementation kept pop-time
        wall_ns comparisons (`wall = time.monotonic_ns()` at pop time
        vs `after_wall_ns`) and short-circuited on the first buffer.
        Since callers `time.sleep` until `after_wall_ns` BEFORE this
        function runs, every pop already satisfied `pop_wall >=
        after_wall_ns` by construction, so the `wall < after_wall_ns`
        discard path never fired — stale buffers queued during the
        sleep (exposures taken during the pre-transition state) were
        returned as "fresh." Confirmed by calibrate_debug_trace.py:
        every after_B capture showed the same mean as pre_W.

        The `after_wall_ns` argument is kept in the signature for
        backward-compatibility with _capture_one's call site but is
        no longer used — the unconditional drain renders it
        unnecessary.
        """
        # 1. Unconditional drain: pop everything currently queued (they
        # were delivered during the sleep window, while the DMD may
        # still have been in its previous state or transitioning).
        while True:
            buf = self._stream.timeout_pop_buffer(1_000)   # 1 ms probe
            if buf is None:
                break
            self._stream.push_buffer(buf)

        # 2. Block for the next fresh delivery. Its exposure started
        # at or after the drain instant, i.e. strictly after
        # after_wall_ns.
        buf = self._stream.timeout_pop_buffer(timeout_us)
        if buf is None:
            return None, None
        wall = time.monotonic_ns()
        if buf.get_status() != Aravis.BufferStatus.SUCCESS:
            self._stream.push_buffer(buf)
            return None, None
        data = bytes(buf.get_data())
        self._stream.push_buffer(buf)
        return data, wall

    def _capture_one(self, min_wait_ms=0, after_wall_ns=None):
        """Drain + wait + pop. Returns (mean_intensity, wall_ns) or
        (None, None) on timeout."""
        if after_wall_ns is None:
            after_wall_ns = time.monotonic_ns() + int(min_wait_ms * 1e6)
        elif min_wait_ms > 0:
            after_wall_ns += int(min_wait_ms * 1e6)
        # Sleep until the wait window closes before draining.
        now = time.monotonic_ns()
        if after_wall_ns > now:
            time.sleep((after_wall_ns - now) / 1e9)
        raw, wall = self._drain_and_pop(after_wall_ns)
        if raw is None:
            return None, None
        arr = np.frombuffer(raw, dtype=np.uint8)
        return float(arr.mean()), wall

    def _measure_stable_baseline(self, color, n_samples=20):
        """Set color, wait STABLE_MS, then grab n_samples captures
        (each ~camera_period apart) to characterise the stable-state
        intensity distribution."""
        self._set_color(color)
        time.sleep(STABLE_MS / 1000.0)
        means = []
        for _ in range(n_samples):
            m, _ = self._capture_one()
            if m is not None:
                means.append(m)
        return means

    def _one_trial(self, src_color, dst_color, wait_ms):
        """Do one (src→dst) transition trial at the given wait_ms.
        Assumes rig is already in src_color stable state. Returns the
        mean intensity of the captured frame, or None on timeout.
        Leaves the rig back at src_color stable state, ready for the
        next trial."""
        emission_wall = self._set_color(dst_color)
        m, _ = self._capture_one(min_wait_ms=wait_ms, after_wall_ns=emission_wall)
        # Restore baseline for next trial.
        self._set_color(src_color)
        time.sleep(STABLE_MS / 1000.0)
        return m

    def _run_calibration(self):
        try:
            print(f"[calib] sampling stable-white baseline "
                  f"({STABLE_MS} ms settle)…", flush=True)
            self.stable_white_samples = self._measure_stable_baseline("white")
            print(f"[calib]   white_mean={statistics.mean(self.stable_white_samples):.1f} "
                  f"std={statistics.pstdev(self.stable_white_samples):.2f} "
                  f"(n={len(self.stable_white_samples)})", flush=True)

            print(f"[calib] sampling stable-black baseline "
                  f"({STABLE_MS} ms settle)…", flush=True)
            self.stable_black_samples = self._measure_stable_baseline("black")
            print(f"[calib]   black_mean={statistics.mean(self.stable_black_samples):.1f} "
                  f"std={statistics.pstdev(self.stable_black_samples):.2f} "
                  f"(n={len(self.stable_black_samples)})", flush=True)

            w_mean = statistics.mean(self.stable_white_samples)
            b_mean = statistics.mean(self.stable_black_samples)
            threshold = (w_mean + b_mean) / 2.0
            print(f"[calib] threshold = {threshold:.1f}  "
                  f"(midpoint of w_mean={w_mean:.1f}, b_mean={b_mean:.1f})",
                  flush=True)

            if abs(w_mean - b_mean) < 30:
                print(f"[calib] WARN: white/black stable means differ by "
                      f"{abs(w_mean - b_mean):.1f} (< 30); calibration "
                      f"will be unreliable. Dim the rig and retry.",
                      flush=True)

            # W → B sweep. Restore white baseline between trials.
            self._set_color("white")
            time.sleep(STABLE_MS / 1000.0)
            for wait_ms in self.wait_sweep_ms:
                samples = []
                for _ in range(self.n_trials):
                    m = self._one_trial("white", "black", wait_ms)
                    if m is not None:
                        samples.append(m)
                self.w_to_b[wait_ms] = samples
                # "transitioned" means the capture crossed below threshold.
                transitioned = sum(1 for m in samples if m < threshold)
                total = len(samples)
                frac = transitioned / total if total else 0.0
                print(f"[calib] W→B wait={wait_ms:3d}ms  "
                      f"correct={transitioned}/{total} = {frac:.2f}",
                      flush=True)

            # B → W sweep. Restore black baseline between trials.
            self._set_color("black")
            time.sleep(STABLE_MS / 1000.0)
            for wait_ms in self.wait_sweep_ms:
                samples = []
                for _ in range(self.n_trials):
                    m = self._one_trial("black", "white", wait_ms)
                    if m is not None:
                        samples.append(m)
                self.b_to_w[wait_ms] = samples
                # For B→W, "transitioned" means the capture crossed ABOVE threshold.
                transitioned = sum(1 for m in samples if m > threshold)
                total = len(samples)
                frac = transitioned / total if total else 0.0
                print(f"[calib] B→W wait={wait_ms:3d}ms  "
                      f"correct={transitioned}/{total} = {frac:.2f}",
                      flush=True)

            self.calibration = self._build_calibration(w_mean, b_mean, threshold)
            self._persist()
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            import traceback
            traceback.print_exc()
        finally:
            try:
                self._camera.stop_acquisition()
            except Exception:
                pass
            GLib.idle_add(self.quit)

    def _build_calibration(self, w_mean, b_mean, threshold):
        def _per_wait_fractions(samples_by_wait, direction):
            out = {}
            for wait_ms, samples in samples_by_wait.items():
                if not samples:
                    out[str(wait_ms)] = 0.0
                    continue
                if direction == "W_to_B":
                    correct = sum(1 for m in samples if m < threshold)
                else:
                    correct = sum(1 for m in samples if m > threshold)
                out[str(wait_ms)] = round(correct / len(samples), 4)
            return out

        def _find_p(frac_by_wait, q):
            # smallest wait_ms where fraction >= q.
            ordered = sorted(int(k) for k in frac_by_wait.keys())
            for w in ordered:
                if frac_by_wait[str(w)] >= q:
                    return w
            return None

        w2b_frac = _per_wait_fractions(self.w_to_b, "W_to_B")
        b2w_frac = _per_wait_fractions(self.b_to_w, "B_to_W")

        w2b_p99 = _find_p(w2b_frac, 0.99)
        w2b_p100 = _find_p(w2b_frac, 1.00)
        b2w_p99 = _find_p(b2w_frac, 0.99)
        b2w_p100 = _find_p(b2w_frac, 1.00)

        p100_candidates = [x for x in (w2b_p100, b2w_p100) if x is not None]
        recommended = None
        confidence = ""
        if p100_candidates:
            recommended = max(p100_candidates) + SAFETY_MS
            n = self.n_trials
            confidence = (f"{n}/{n} trials transitioned cleanly at the "
                          f"p100 wait for both directions; recommended = "
                          f"max(W→B.p100={w2b_p100}, B→W.p100={b2w_p100}) "
                          f"+ {SAFETY_MS}ms safety margin = {recommended} ms")
        else:
            # No wait reached 100%. Fall back to the highest p99; if that
            # is None too, use the largest swept wait with a warning.
            p99_candidates = [x for x in (w2b_p99, b2w_p99) if x is not None]
            if p99_candidates:
                recommended = max(p99_candidates) + SAFETY_MS
                confidence = (f"WARNING: no wait reached 100% in one or "
                              f"both directions; falling back to max p99 "
                              f"(W→B.p99={w2b_p99}, B→W.p99={b2w_p99}) + "
                              f"{SAFETY_MS}ms = {recommended} ms")
            else:
                recommended = max(self.wait_sweep_ms) + SAFETY_MS
                confidence = (f"WARNING: calibration failed to find a "
                              f"reliable wait in either direction; "
                              f"falling back to largest swept + safety = "
                              f"{recommended} ms. Calibration is NOT "
                              f"trustworthy; investigate the rig "
                              f"(ambient light? camera exposure? "
                              f"projector signal?).")

        rig_config = {
            "exposure_us": int(self.exposure_us),
            "gain_db": float(self.gain_db),
            "projector_connector": self.monitor["connector"],
            "hdmi_edid_fingerprint": self.edid_fp,
            "camera_serial": self.camera_serial,
            "camera_firmware": self.camera_firmware,
        }
        rig_hash_hex = compute_rig_hash(rig_config)

        def _dist(samples):
            return {
                "mean": round(statistics.mean(samples), 4),
                "std": round(statistics.pstdev(samples), 4),
                "n": len(samples),
            }

        return {
            "method": "white_black_empirical",
            "n_trials_per_wait": self.n_trials,
            "wait_ms_tested": self.wait_sweep_ms,
            "W_to_B": {
                "intensity_distribution_stable_white": _dist(self.stable_white_samples),
                "intensity_distribution_stable_black": _dist(self.stable_black_samples),
                "classification_threshold": round(threshold, 4),
                "per_wait_correct_fraction": w2b_frac,
                "p99_wait_ms": w2b_p99,
                "p100_wait_ms": w2b_p100,
            },
            "B_to_W": {
                "intensity_distribution_stable_white": _dist(self.stable_white_samples),
                "intensity_distribution_stable_black": _dist(self.stable_black_samples),
                "classification_threshold": round(threshold, 4),
                "per_wait_correct_fraction": b2w_frac,
                "p99_wait_ms": b2w_p99,
                "p100_wait_ms": b2w_p100,
            },
            "recommended_wait_ms": recommended,
            "confidence": confidence,
            "measurement_timestamp": dt.datetime.now(dt.timezone.utc).isoformat(
                timespec="seconds"),
            "rig_config": rig_config,
            "rig_hash": rig_hash_hex,
        }

    def _persist(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        rig_hash = self.calibration["rig_hash"]
        out_path = self.output_dir / f"pipeline_calibration_{rig_hash[:16]}.json"
        # Use canonical bytes so reloaders can recompute the hash.
        out_path.write_bytes(canonical_json_bytes(self.calibration))
        self.calibration_path = out_path
        print(f"[calib] wrote {out_path}", flush=True)

        # Markdown report.
        self.report_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        rp = self.report_dir / f"pipeline_calibration_{stamp}.md"
        self.report_path = rp
        c = self.calibration
        rc = c["rig_config"]
        lines = [
            f"# Pipeline-delay calibration — {stamp}",
            "",
            f"- method: `{c['method']}`",
            f"- rig_hash: `{c['rig_hash']}`",
            f"- rig_config:",
            f"  - exposure_us: {rc['exposure_us']}",
            f"  - gain_db: {rc['gain_db']}",
            f"  - projector_connector: {rc['projector_connector']}",
            f"  - hdmi_edid_fingerprint: {rc['hdmi_edid_fingerprint']}",
            f"  - camera_serial: {rc['camera_serial']}",
            f"  - camera_firmware: {rc['camera_firmware']}",
            f"- n_trials_per_wait: {c['n_trials_per_wait']}",
            f"- wait_ms_tested: {c['wait_ms_tested']}",
            "",
            f"## Stable-state distributions",
            "",
            f"- stable_white_mean: {c['W_to_B']['intensity_distribution_stable_white']['mean']}  (std "
            f"{c['W_to_B']['intensity_distribution_stable_white']['std']}, "
            f"n {c['W_to_B']['intensity_distribution_stable_white']['n']})",
            f"- stable_black_mean: {c['W_to_B']['intensity_distribution_stable_black']['mean']}  (std "
            f"{c['W_to_B']['intensity_distribution_stable_black']['std']}, "
            f"n {c['W_to_B']['intensity_distribution_stable_black']['n']})",
            f"- classification_threshold: {c['W_to_B']['classification_threshold']}",
            "",
            f"## W → B correct fraction by wait",
            "",
            f"| wait_ms | correct_fraction |",
            f"| ---: | ---: |",
        ]
        for wait_ms in c["wait_ms_tested"]:
            f = c["W_to_B"]["per_wait_correct_fraction"][str(wait_ms)]
            lines.append(f"| {wait_ms} | {f:.2f} |")
        lines += [
            "",
            f"- p99_wait_ms: {c['W_to_B']['p99_wait_ms']}",
            f"- p100_wait_ms: {c['W_to_B']['p100_wait_ms']}",
            "",
            f"## B → W correct fraction by wait",
            "",
            f"| wait_ms | correct_fraction |",
            f"| ---: | ---: |",
        ]
        for wait_ms in c["wait_ms_tested"]:
            f = c["B_to_W"]["per_wait_correct_fraction"][str(wait_ms)]
            lines.append(f"| {wait_ms} | {f:.2f} |")
        lines += [
            "",
            f"- p99_wait_ms: {c['B_to_W']['p99_wait_ms']}",
            f"- p100_wait_ms: {c['B_to_W']['p100_wait_ms']}",
            "",
            f"## Recommendation",
            "",
            f"- **recommended_wait_ms**: {c['recommended_wait_ms']}",
            f"- confidence: {c['confidence']}",
            "",
            f"Calibration JSON written to `{self.calibration_path}`.",
        ]
        rp.write_text("\n".join(lines))
        print(f"[calib] wrote {rp}", flush=True)


def pick_monitor(args):
    display = Gdk.Display.get_default()
    if display is None:
        raise RuntimeError("no Gdk.Display")
    mons = display.get_monitors()
    items = []
    for i in range(mons.get_n_items()):
        m = mons.get_item(i)
        geo = m.get_geometry()
        items.append({"index": i, "connector": m.get_connector() or "?",
                      "geometry": (geo.x, geo.y, geo.width, geo.height),
                      "monitor": m})
    if args.monitor is not None:
        return items[args.monitor]
    for it in items:
        if it["connector"] == args.connector:
            return it
    raise RuntimeError(f"monitor {args.monitor or args.connector} not found")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--monitor", type=int)
    g.add_argument("--connector", type=str)
    p.add_argument("--exposure-us", type=float, default=16000.0)
    p.add_argument("--gain-db", type=float, default=24.0)
    p.add_argument("--n-trials", type=int, default=50,
                   help="Trials per wait value in each direction. "
                        "Default 50: enough to detect p99 (50/50 → "
                        "pipeline delay is reliably below this wait). "
                        "Bump to 100 for tighter statistics.")
    p.add_argument("--waits", type=str, default=None,
                   help="Optional comma-separated list of wait_ms values "
                        f"to sweep. Default: {DEFAULT_WAIT_SWEEP_MS}")
    p.add_argument("--format", type=str, default="BayerRG8",
                   choices=["BayerRG8"])
    p.add_argument("--output", type=str, default=None,
                   help="Directory to write the calibration JSON into "
                        "(default ~/.tb/).")
    p.add_argument("--reports-dir", type=str, default="reports")
    args = p.parse_args()

    if args.n_trials < 1:
        print("ERROR: --n-trials must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.waits is None:
        waits = DEFAULT_WAIT_SWEEP_MS
    else:
        try:
            waits = sorted({int(x) for x in args.waits.split(",") if x.strip()})
        except ValueError:
            print(f"ERROR: --waits must be comma-separated integers",
                  file=sys.stderr)
            sys.exit(1)
        if not waits:
            print("ERROR: --waits must be non-empty", file=sys.stderr)
            sys.exit(1)

    mon = pick_monitor(args)
    print(f"target monitor: idx={mon['index']} "
          f"connector={mon['connector']} geo={mon['geometry']}")
    app = CalibrationApp(
        monitor=mon, exposure_us=args.exposure_us, gain_db=args.gain_db,
        n_trials=args.n_trials, wait_sweep_ms=waits,
        format_str=args.format, output_dir=args.output,
        report_dir=args.reports_dir,
    )
    app.run([])
    if app.error is not None:
        print(f"[calib] FAILED: {app.error}", file=sys.stderr)
        sys.exit(3)
    if app.calibration is None:
        print(f"[calib] FAILED: calibration did not complete",
              file=sys.stderr)
        sys.exit(3)
    c = app.calibration
    print()
    print(f"=== recommended_wait_ms: {c['recommended_wait_ms']} ===")
    print(f"confidence: {c['confidence']}")
    sys.exit(0)


if __name__ == "__main__":
    main()

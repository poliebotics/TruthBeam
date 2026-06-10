#!/usr/bin/env python3
"""Quick feasibility capture: 5 single-shot captures at staircase exposures
with subject in projection cone. No chain, no manifest, no session schema.

  1. ambient   48 ms   projector OFF (noise floor / ambient)
  2. illum    16 ms    projector ON  (random emission)
  3. illum    32 ms    projector ON
  4. illum    48 ms    projector ON  (commonly useful target)
  5. illum    64 ms    projector ON  (generous upper bound)

For each capture: save raw BayerRG8 and a debayered preview PNG, print
per-capture stats (mean, max, p99, saturated fraction, dim fraction),
and save a CSV summary.

Usage:
    python3 feasibility_capture.py \\
        --connector HDMI-1 \\
        --out-dir /tmp/feasibility_test_$(date +%Y%m%d_%H%M%S)
"""
import os
os.environ.setdefault("WAYLAND_DISPLAY", "wayland-0")
os.environ.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
os.environ.setdefault("GDK_BACKEND", "wayland")

import argparse
import csv
import sys
import threading
import time
from pathlib import Path

# tools/ scripts import directly from protocol/ — keep the layout flat but
# reachable. Prepend the sibling protocol directory to sys.path so the
# existing module-level imports below resolve without a package install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "protocol"))

import cairo
import cv2
import numpy as np
from PIL import Image

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
gi.require_version("Aravis", "0.8")
from gi.repository import Aravis, Gdk, GLib, Gtk  # noqa: E402

from blake3 import blake3
from session_schema import compute_xof_seeds
import tile_gpu

TILE_W, TILE_H = 1920, 1080
GAIN_DB = 24.0
ACTIVATION = "RisingEdge"
FORMAT_STR = "BayerRG8"
SETTLE_MS = 200  # let DMD settle after queue_draw before capture

CAPTURE_SPECS = [
    {"label": "01_ambient_48ms",     "exposure_ms": 48, "projector_on": False},
    {"label": "02_illuminated_16ms", "exposure_ms": 16, "projector_on": True},
    {"label": "03_illuminated_32ms", "exposure_ms": 32, "projector_on": True},
    {"label": "04_illuminated_48ms", "exposure_ms": 48, "projector_on": True},
    {"label": "05_illuminated_64ms", "exposure_ms": 64, "projector_on": True},
]


def pick_monitor(connector):
    disp = Gdk.Display.get_default() or Gdk.Display.open(None)
    if disp is None:
        raise RuntimeError("no Gdk display")
    mons = disp.get_monitors()
    for i in range(mons.get_n_items()):
        m = mons.get_item(i)
        if m.get_connector() == connector:
            return m
    raise RuntimeError(f"monitor {connector} not found")


def stats(data_u8, label, exposure_ms, projector_on):
    mean = float(data_u8.mean())
    mx = int(data_u8.max())
    p99 = float(np.percentile(data_u8, 99))
    sat = float((data_u8 == 255).mean())
    dim = float((data_u8 < 20).mean())
    print(f"  {label}  exp={exposure_ms:>3} ms  proj={'ON ' if projector_on else 'OFF'}"
          f"  mean={mean:6.2f}  p99={p99:5.1f}  max={mx:3d}"
          f"  sat={sat*100:6.2f}%  dim<20={dim*100:6.2f}%", flush=True)
    return {"label": label, "exposure_ms": exposure_ms,
            "projector_on": projector_on, "mean": mean, "max": mx,
            "p99": p99, "sat_frac": sat, "dim_frac": dim}


class FeasibilityApp(Gtk.Application):
    def __init__(self, connector, out_dir, seed):
        super().__init__(application_id="local.tb.feasibility")
        self.connector = connector
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.tile_lock = threading.Lock()
        self.current_tile = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
        self.scratch = np.empty((TILE_H, TILE_W, 4), dtype=np.uint8)
        self._camera = None
        self._device = None
        self._stream = None
        self.payload = 0
        self.sensor_w = 0
        self.sensor_h = 0

    def do_activate(self):
        monitor = pick_monitor(self.connector)
        geo = monitor.get_geometry()
        print(f"target monitor: {self.connector} geo={geo.x},{geo.y},"
              f"{geo.width}x{geo.height}", flush=True)
        self.window = Gtk.ApplicationWindow(application=self)
        self.area = Gtk.DrawingArea()
        self.area.set_draw_func(self._draw)
        self.window.set_child(self.area)
        self.window.fullscreen_on_monitor(monitor)
        try:
            cur = Gdk.Cursor.new_from_name("none", None)
            if cur:
                self.window.set_cursor(cur)
        except Exception:
            pass
        self.window.present()
        GLib.timeout_add(500, self._start_worker)

    def _start_worker(self):
        threading.Thread(target=self._worker, daemon=True).start()
        return False

    def _draw(self, area, cr, w, h):
        with self.tile_lock:
            tile = self.current_tile
        # BGRA packing for cairo FORMAT_RGB24 (little-endian XRGB)
        self.scratch[..., 0] = tile[..., 2]
        self.scratch[..., 1] = tile[..., 1]
        self.scratch[..., 2] = tile[..., 0]
        surf = cairo.ImageSurface.create_for_data(
            self.scratch, cairo.FORMAT_RGB24, TILE_W, TILE_H, TILE_W * 4)
        cr.scale(w / TILE_W, h / TILE_H)
        cr.set_source_surface(surf, 0, 0)
        cr.paint()

    def _queue_tile(self, tile):
        evt = threading.Event()
        def _post():
            with self.tile_lock:
                self.current_tile = tile
            self.area.queue_draw()
            evt.set()
            return False
        GLib.idle_add(_post)
        evt.wait(timeout=2.0)

    def _setup_camera(self):
        Aravis.update_device_list()
        if Aravis.get_n_devices() == 0:
            raise RuntimeError("no Aravis cameras detected")
        dev_id = Aravis.get_device_id(0)
        print(f"camera: {dev_id}", flush=True)
        self._camera = Aravis.Camera.new(dev_id)
        self._camera.set_pixel_format_from_string(FORMAT_STR)
        self._device = self._camera.get_device()

        sw, sh = self._camera.get_sensor_size()
        self.sensor_w, self.sensor_h = sw, sh
        for n, v in (("OffsetX", 0), ("OffsetY", 0),
                     ("Width", sw), ("Height", sh)):
            try:
                self._device.set_integer_feature_value(n, int(v))
            except Exception:
                pass
        self.payload = self._camera.get_payload()
        print(f"  ROI={sw}x{sh}  payload={self.payload}", flush=True)

        for n, v in (("ExposureAuto", "Off"),
                     ("GainAuto", "Off"),
                     ("BalanceWhiteAuto", "Off")):
            try:
                self._device.set_string_feature_value(n, v)
            except Exception:
                pass
        self._device.set_float_feature_value("Gain", GAIN_DB)
        for n, v in (("TriggerSelector", "FrameStart"),
                     ("TriggerMode", "On"),
                     ("TriggerSource", "Line1"),
                     ("TriggerActivation", ACTIVATION)):
            try:
                self._device.set_string_feature_value(n, v)
            except Exception as e:
                print(f"  WARN set {n}={v}: {e}", flush=True)

        self._stream = self._camera.create_stream(None, None)
        for _ in range(8):
            self._stream.push_buffer(Aravis.Buffer.new_allocate(self.payload))
        # Start acquisition once and keep it running across all captures.
        # Exposure is changed on the fly; a per-capture drain flushes any
        # frames that were exposed at the previous setting.
        self._device.set_float_feature_value(
            "ExposureTime", float(CAPTURE_SPECS[0]["exposure_ms"] * 1000))
        self._camera.start_acquisition()
        print(f"  gain={GAIN_DB}dB  trigger=VSYNC/Line1  buffers=8  "
              f"acquisition started", flush=True)

    def _drain_all(self):
        while True:
            buf = self._stream.timeout_pop_buffer(1_000)
            if buf is None:
                return
            self._stream.push_buffer(buf)

    def _acquire_one_at_exposure(self, exposure_ms, timeout_us=3_000_000,
                                  transition_frames=4, max_bad=15):
        """Set exposure on the fly, drain pipeline, pop first SUCCESS frame.
        Tolerates transitional bad-status frames (MISSING_PACKETS,
        PARTIAL_DATA, etc.) up to max_bad before giving up."""
        self._device.set_float_feature_value("ExposureTime", float(exposure_ms * 1000))
        # Drain a few frames to flush the capture pipeline at the new
        # exposure setting. Non-SUCCESS buffers are simply recycled.
        for _ in range(transition_frames):
            buf = self._stream.timeout_pop_buffer(timeout_us)
            if buf is None:
                continue
            self._stream.push_buffer(buf)

        bad_statuses = []
        for _ in range(max_bad):
            buf = self._stream.timeout_pop_buffer(timeout_us)
            if buf is None:
                raise RuntimeError(
                    f"timeout at exposure {exposure_ms} ms  (bad-so-far="
                    f"{bad_statuses})"
                )
            status = buf.get_status()
            if status == Aravis.BufferStatus.SUCCESS:
                data = bytes(buf.get_data())
                self._stream.push_buffer(buf)
                return data
            bad_statuses.append(int(status))
            self._stream.push_buffer(buf)
        raise RuntimeError(
            f"no SUCCESS frame after {max_bad} tries at exposure "
            f"{exposure_ms} ms  statuses={bad_statuses}"
        )

    def _worker(self):
        rng = np.random.default_rng(self.seed)
        summary = []
        try:
            self._setup_camera()
            # Warm up the GPU tile generator so the first real emission
            # doesn't pay the CUDA init cost on the critical path.
            warm_seed = blake3(
                b"tb-feasibility-prewarm").digest()
            wx_r, wx_g, wx_b = compute_xof_seeds(warm_seed)
            for _ in range(3):
                tile_gpu.gen_rgb_tile_cuda(wx_r, wx_g, wx_b)
            print(f"  tile generator: tile_gpu (XOF → multi-octave "
                  f"upsampled grids, matches protocol)", flush=True)
            H, W = self.sensor_h, self.sensor_w
            black = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)

            for spec in CAPTURE_SPECS:
                label = spec["label"]
                exposure_ms = spec["exposure_ms"]
                projector_on = spec["projector_on"]
                print(f"[{label}] starting", flush=True)

                if projector_on:
                    # Generate a fresh pseudo-random 32-byte S_t seed and
                    # derive the real protocol emission from it (XOF →
                    # NUM_OCTAVES multi-octave grids → upsampled RGB).
                    # Visually this has coherent low-frequency blobs,
                    # unlike per-pixel uniform random noise; it's what
                    # the main session will actually project.
                    s_bytes = rng.bytes(32)
                    xof_r, xof_g, xof_b = compute_xof_seeds(s_bytes)
                    emission = tile_gpu.gen_rgb_tile_cuda(xof_r, xof_g, xof_b)
                    self._queue_tile(emission)
                    Image.fromarray(
                        np.ascontiguousarray(emission), mode="RGB"
                    ).save(self.out_dir / f"emission_{label}.png",
                           compress_level=1)
                else:
                    self._queue_tile(black)
                time.sleep(SETTLE_MS / 1000.0)

                raw = self._acquire_one_at_exposure(exposure_ms)
                arr = np.frombuffer(raw, dtype=np.uint8)
                if arr.size != H * W:
                    raise RuntimeError(
                        f"capture size {arr.size} != sensor {H}x{W}={H*W}")

                (self.out_dir / f"{label}.raw").write_bytes(raw)
                bayer = arr.reshape(H, W)
                rgb = cv2.cvtColor(bayer, cv2.COLOR_BayerRG2RGB)
                preview = cv2.resize(rgb, (1330, 1150), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(self.out_dir / f"{label}.png"),
                            cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
                summary.append(stats(arr, label, exposure_ms, projector_on))

            self._queue_tile(black)
            time.sleep(0.2)

            fields = ["label", "exposure_ms", "projector_on",
                      "mean", "max", "p99", "sat_frac", "dim_frac"]
            with open(self.out_dir / "summary.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for row in summary:
                    w.writerow(row)
            print(f"\nartifacts: {self.out_dir}", flush=True)
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
        finally:
            try:
                self._camera.stop_acquisition()
            except Exception:
                pass
            GLib.idle_add(self.quit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--connector", default="HDMI-1")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    app = FeasibilityApp(args.connector, args.out_dir, args.seed)
    return app.run([])


if __name__ == "__main__":
    sys.exit(main())

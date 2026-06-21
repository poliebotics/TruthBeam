"""camera.py — Aravis camera setup, configuration, and capture primitives.

Scope: every Aravis/GenICam call in the project. A future agent swapping
to a non-Aravis camera library replaces THIS FILE and nothing else
outside projector.py should need editing.

Adapt this module for:
- Different camera library (Spinnaker, Vimba, libgphoto2) — replace the
  body of setup() and acquire_*() while keeping the same function
  signatures; callers see only a CameraHandle
- Different pixel format / bit depth
- Different trigger wiring (Line1 vs. Line2, RisingEdge vs. FallingEdge)
- Different sensor-native resolution or ROI

Do NOT put here:
- Chain / hashing math — that's chain.py
- CSV row construction — that's session_logs.py
- Clock-offset calibration above the raw timestamp level — that logic
  currently lives on TBLoopV8 in tb_loop.py; a later refactor moves the
  accept/reject policy to a dedicated capture_blocking module.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import gi
gi.require_version("Aravis", "0.8")
from gi.repository import Aravis  # noqa: E402

from blake3 import blake3


@dataclass
class CameraHandle:
    """All Aravis state that the rest of the system accesses only through
    module-level functions. Populated by setup(); passed to acquire_*()."""
    camera: object = None           # Aravis.Camera
    device: object = None           # Aravis.Device
    stream: object = None           # Aravis.Stream
    payload: int = 0
    sensor_w: int = 0
    sensor_h: int = 0
    real_roi: tuple = ()            # (x, y, w, h) actually applied
    camera_serial: Optional[str] = None
    camera_firmware: Optional[str] = None
    aravis_version: Optional[str] = None


# ----- probes (pure functions over an already-open device) -----

def probe_aravis_version() -> str:
    try:
        maj = Aravis.get_major_version()
        mnr = Aravis.get_minor_version()
        mic = Aravis.get_micro_version()
        return f"{maj}.{mnr}.{mic}"
    except Exception:
        pass
    try:
        import subprocess
        v = subprocess.run(
            ["pkg-config", "--modversion", "aravis-0.8"],
            capture_output=True, text=True, timeout=2,
        )
        if v.returncode == 0 and v.stdout.strip():
            return v.stdout.strip()
    except Exception:
        pass
    return f"gir-{getattr(Aravis, '_version', 'unknown')}"


def probe_camera_serial(device, dev_id) -> Optional[str]:
    for feature in ("DeviceSerialNumber", "DeviceID"):
        try:
            v = device.get_string_feature_value(feature)
            if v:
                return str(v).strip()
        except Exception:
            pass
    if dev_id:
        tail = ""
        for ch in reversed(dev_id):
            if ch.isdigit():
                tail = ch + tail
            else:
                break
        if tail:
            return tail
    return None


def probe_camera_firmware(device) -> Optional[str]:
    try:
        v = device.get_string_feature_value("DeviceFirmwareVersion")
        return str(v).strip() if v else None
    except Exception:
        return None


# ----- setup / acquire / cleanup -----

def setup(format_str: str, exposure_us: float, gain_db: float,
          activation: str, iso_now: callable) -> CameraHandle:
    """Open the first available Aravis camera, configure ROI to full
    sensor, set exposure/gain/activation, and pre-allocate 32 buffers
    on a fresh stream. Returns a CameraHandle. Does NOT start acquisition
    (call start_acquisition() when ready).

    `iso_now` is a callable returning a human-readable timestamp for
    log lines, supplied by the orchestrator so all lines share a clock.
    """
    Aravis.update_device_list()
    if Aravis.get_n_devices() == 0:
        raise RuntimeError("no Aravis cameras")
    dev_id = Aravis.get_device_id(0)
    print(f"[{iso_now()}] opening camera: {dev_id}", flush=True)
    camera = Aravis.Camera.new(dev_id)
    camera.set_pixel_format_from_string(format_str)
    device = camera.get_device()

    camera_serial = probe_camera_serial(device, dev_id)
    camera_firmware = probe_camera_firmware(device)
    aravis_version = probe_aravis_version()

    sw, sh = camera.get_sensor_size()
    for name, value in (("OffsetX", 0), ("OffsetY", 0),
                        ("Width", sw), ("Height", sh)):
        try:
            device.set_integer_feature_value(name, int(value))
        except Exception as e:
            print(f"  WARN set {name}={value}: {e}", flush=True)
    rx = device.get_integer_feature_value("OffsetX")
    ry = device.get_integer_feature_value("OffsetY")
    rw = device.get_integer_feature_value("Width")
    rh = device.get_integer_feature_value("Height")
    real_roi = (rx, ry, rw, rh)
    payload = camera.get_payload()
    print(f"  ROI={rw}x{rh}+{rx}+{ry}  payload={payload} B", flush=True)

    for n, v in (("ExposureAuto", "Off"),
                 ("GainAuto", "Off"),
                 ("BalanceWhiteAuto", "Off")):
        try:
            device.set_string_feature_value(n, v)
        except Exception as e:
            print(f"  WARN set {n}={v}: {e}", flush=True)
    device.set_float_feature_value("ExposureTime", float(exposure_us))
    device.set_float_feature_value("Gain", float(gain_db))
    for n, v in (("TriggerSelector", "FrameStart"),
                 ("TriggerMode", "On"),
                 ("TriggerSource", "Line1"),
                 ("TriggerActivation", activation)):
        device.set_string_feature_value(n, v)

    for k in ("ExposureTime", "Gain"):
        print(f"  {k} = {device.get_float_feature_value(k)!r}", flush=True)

    stream = camera.create_stream(None, None)
    for _ in range(32):
        stream.push_buffer(Aravis.Buffer.new_allocate(payload))
    print(f"[{iso_now()}] camera configured (acquisition deferred)", flush=True)

    return CameraHandle(
        camera=camera, device=device, stream=stream, payload=payload,
        sensor_w=sw, sensor_h=sh, real_roi=real_roi,
        camera_serial=camera_serial, camera_firmware=camera_firmware,
        aravis_version=aravis_version,
    )


def start_acquisition(handle: CameraHandle, iso_now: callable) -> int:
    """Start camera acquisition. Returns the host monotonic_ns at start."""
    handle.camera.start_acquisition()
    start_ns = time.monotonic_ns()
    print(f"[{iso_now()}] acquisition started (wall_ns={start_ns})", flush=True)
    return start_ns


def drain_queue(handle: CameraHandle) -> None:
    """Pop every buffer currently in the stream's output queue and push
    it back to the pool. Used in blocking mode to discard captures whose
    exposure started while the DMD was still transitioning."""
    while True:
        buf = handle.stream.timeout_pop_buffer(1_000)  # 1 ms
        if buf is None:
            return
        handle.stream.push_buffer(buf)


def pop_one_buffer(handle: CameraHandle, timeout_us: int):
    """Pop a single successful buffer. Returns (buf, capture_wall_ns) or
    None on timeout. A buffer with non-SUCCESS status is pushed back and
    None is returned. The caller owns the returned buf and must
    push_buffer it back."""
    buf = handle.stream.timeout_pop_buffer(timeout_us)
    if buf is None:
        return None
    wall = time.monotonic_ns()
    if buf.get_status() != Aravis.BufferStatus.SUCCESS:
        handle.stream.push_buffer(buf)
        return None
    return buf, wall


def drain_and_receive(handle: CameraHandle, timeout_us: int = 2_000_000):
    """Drain stale queued buffers, then block for the next fresh one.
    Returns (raw_bytes, frame_id, ts_ns, wall_ns) or None on timeout /
    bad status. Retained for calibrate/other callers; the blocking chain
    loop uses the timestamp-based accept path in capture_blocking.py."""
    drain_queue(handle)
    buf = handle.stream.timeout_pop_buffer(timeout_us)
    if buf is None:
        return None
    wall = time.monotonic_ns()
    if buf.get_status() != Aravis.BufferStatus.SUCCESS:
        handle.stream.push_buffer(buf)
        return None
    data = bytes(buf.get_data())
    frame_id = buf.get_frame_id()
    ts_ns = buf.get_timestamp()
    handle.stream.push_buffer(buf)
    return data, frame_id, ts_ns, wall


def stop_acquisition(handle: CameraHandle) -> None:
    try:
        if handle.camera is not None:
            handle.camera.stop_acquisition()
    except Exception:
        pass

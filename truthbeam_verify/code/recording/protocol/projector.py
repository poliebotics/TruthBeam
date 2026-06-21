"""projector.py — GTK/Gdk/Cairo projection of RGB tiles on a chosen monitor.

Scope: the fullscreen GTK window, Cairo rendering of a (1080, 1920, 3)
uint8 tile, and GTK-thread-safe tile updates from background threads.

Adapt this module for:
- Different display stack (Wayland-specific vs. X11-specific pathways;
  or a headless/offscreen target like EGL/Vulkan)
- Different projector resolution (change tile_params.TILE_H/TILE_W and
  audit the Cairo surface setup here)
- Sync-locked multi-monitor setups

Do NOT put here:
- Camera / triggering concerns — camera.py
- Chain / hash concerns — chain.py
- Pixel-format negotiation with the projector at the EDID level — the
  operator is expected to have already chosen a compatible mode
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import cairo
from pathlib import Path
import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
from gi.repository import Gdk, GLib, Gtk  # noqa: E402

from blake3 import blake3

from tile_params import TILE_W, TILE_H


def probe_edid_fingerprint(connector: str):
    """Return a BLAKE3 hex of the raw EDID bytes for the named connector,
    or None if the EDID can't be read. Linux-specific (/sys/class/drm)."""
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


def pick_monitor(args):
    """Pick the target monitor by --monitor index or --connector name.
    Returns a dict with keys: index, connector, geometry, monitor.
    Raises RuntimeError if no matching monitor is found."""
    display = Gdk.Display.get_default()
    if display is None:
        raise RuntimeError("no Gdk.Display (Wayland connection failed)")
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


@dataclass
class ProjectorHandle:
    """Per-session GTK/Cairo state. Created by an enclosing Gtk.Application
    (usually the orchestrator in tb_loop.py), not directly constructed."""
    window: object = None           # Gtk.ApplicationWindow
    area: object = None             # Gtk.DrawingArea
    current_tile: Optional[np.ndarray] = None
    scratch: Optional[np.ndarray] = None     # BGRA scratch buffer for Cairo
    tile_lock: Optional[threading.Lock] = None


def make_handle() -> ProjectorHandle:
    """Build a ProjectorHandle with its scratch buffer + lock initialised.
    The window + area are attached later by attach_window()."""
    scratch = np.empty((TILE_H, TILE_W, 4), dtype=np.uint8)
    scratch[..., 3] = 255
    return ProjectorHandle(
        scratch=scratch,
        tile_lock=threading.Lock(),
    )


def attach_window(handle: ProjectorHandle, application: Gtk.Application,
                  monitor: dict) -> None:
    """Attach a fullscreen window on the chosen monitor to the handle and
    wire up the draw callback. Runs on the GTK main thread."""
    window = Gtk.ApplicationWindow(application=application)
    window.set_decorated(False)
    area = Gtk.DrawingArea()
    area.set_draw_func(lambda area, cr, w, h: _draw(handle, area, cr, w, h))
    window.set_child(area)
    window.fullscreen_on_monitor(monitor["monitor"])
    try:
        cursor = Gdk.Cursor.new_from_name("none", None)
        window.set_cursor(cursor)
    except Exception as e:
        print(f"[gtk] WARN: failed to hide cursor: {e!r}", flush=True)
    window.present()
    handle.window = window
    handle.area = area


def _draw(handle: ProjectorHandle, area, cr, w, h):
    with handle.tile_lock:
        tile = handle.current_tile
    if tile is None:
        cr.set_source_rgb(0, 0, 0)
        cr.rectangle(0, 0, w, h)
        cr.fill()
        return
    handle.scratch[..., 0] = tile[..., 2]
    handle.scratch[..., 1] = tile[..., 1]
    handle.scratch[..., 2] = tile[..., 0]
    surf = cairo.ImageSurface.create_for_data(
        memoryview(handle.scratch), cairo.FORMAT_ARGB32,
        TILE_W, TILE_H, TILE_W * 4,
    )
    cr.set_source_surface(surf, 0, 0)
    cr.paint()


def queue_tile(handle: ProjectorHandle, tile: np.ndarray) -> int:
    """GTK-thread-safe tile update. Returns the host monotonic_ns at which
    the queue_draw() call happened (recorded inside the GTK main thread).

    The caller can treat the returned wall_ns as the 'emission queued'
    timestamp. The actual pixel-delivery-to-DMD latency after that point
    is the domain of the pipeline calibration."""
    evt = threading.Event()
    wall_holder = [None]

    def _post():
        with handle.tile_lock:
            handle.current_tile = tile
        wall_holder[0] = time.monotonic_ns()
        handle.area.queue_draw()
        evt.set()
        return False

    GLib.idle_add(_post)
    evt.wait(timeout=2.0)
    return wall_holder[0] if wall_holder[0] is not None else time.monotonic_ns()

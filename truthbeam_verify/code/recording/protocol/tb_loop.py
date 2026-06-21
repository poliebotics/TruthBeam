#!/usr/bin/env python3
"""Truth Beam recording loop, v9: hash-first capture-driven chain.

The chain advances per CAPTURED FRAME (hash-first pipeline). The v9 transition
additionally folds the per-row drand round (number + value) into the state, so
the row domain tag is ``TB:ROW:v9`` (an existing v8 chain is not re-walkable
under v9). Row t's emission columns commit E_t (the emission live during cap_t).
See session_schema.py:PROTOCOL_VERSION for the authoritative protocol note.

    S_{t+1} = blake3(b"TB:ROW:v9" || S_t
                      || len32(bayer_hash)        || bayer_hash
                      || len32(meta_bytes)        || meta_bytes
                      || len32(drand_round_be8)   || drand_round_be8
                      || len32(drand_round_value) || drand_round_value)

Session layout: <session_dir>/
    .gitignore                                   — contains 'derived/'
    verification_bundle.json (+ .pretty.json)    — pinned algorithm contract
    manifest.json (+ .pretty.json)               — session identity, S_0
    chain_log.csv                                — one row per captured frame
                                                    consumed; authoritative chain
    capture_log.csv                              — one row per captured frame
                                                    delivered (consumed or not)
    anchor_txs.csv                               — interior-anchor pulse log
    Recordings/frame_NNNNNN.raw                  — captured Bayer (authoritative)
    derived/                                     — deletable, regenerable
        Recordings_previews/frame_NNNNNN.png     — debayered preview
        Emissions/tile_NNNNNN.png                — rendered tile (best-effort)
"""

import os
os.environ.setdefault("WAYLAND_DISPLAY", "wayland-0")
os.environ.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
os.environ.setdefault("GDK_BACKEND", "wayland")

import csv
import datetime as dt
import gc
import queue
import socket
import struct
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from blake3 import blake3

import cairo
import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
gi.require_version("Aravis", "0.8")
from gi.repository import Aravis, Gdk, GLib, Gtk  # noqa: E402

from tile_params import (
    TILE_W, TILE_H, NUM_OCTAVES, GRID_H_TABLE, GRID_W_TABLE,
)
from session_schema import (
    PROTOCOL_VERSION, SessionManifest, VerificationBundle,
    compute_chain_row, compute_xof_seeds,
    derive_s0, validate_bundle_manifest_pair,
    canonical_json_bytes,
    SESSION_MODE_BLOCKING,
    SESSION_MODE_ASYNC_TIGHTENED,
    UNANCHORED_BLOCK_HASH_HEX, UNANCHORED_BLOCK_HEIGHT,
)
import async_mitigations
from rsk_anchor import (
    fetch_current_tip, wait_for_next_block, fetch_chain_id,
)
from drand_client import (
    DrandClient, DrandError,
    QUICKNET_CHAIN_HASH, QUICKNET_PUBKEY_HEX, DEFAULT_MAX_STALENESS_S,
)

# Extracted helpers (relocated to dedicated modules during the academic
# refactor). Aliased here to their old underscore-prefixed names so the
# TBLoopV8 class methods below can keep calling them unchanged.
from camera import (
    probe_aravis_version as _probe_aravis_version,
    probe_camera_serial as _probe_camera_serial,
    probe_camera_firmware as _probe_camera_firmware,
)
from projector import probe_edid_fingerprint as _probe_edid_fingerprint

INTERIOR_ANCHOR_RPC_DEFAULT = "https://public-node.testnet.rsk.co"
NETWORK_CHAIN_IDS = {"testnet": 31, "mainnet": 30}

# v0.8 CAPTURE meta struct. Authoritative definition lives in chain.py
# (single source of truth for the wire format hashed into S_{t+1}).
# Re-imported here so the class-internal call sites below keep their
# existing names unchanged.
from chain import META_STRUCT, META_LEN  # noqa: E402

PREROLL_BLACK_MS = 200
END_BLACK_MS = 150

# Camera fourcc map. Keys are GenICam pixel_format strings; values are the
# 4-char ASCII packed into meta_bytes.fourcc.
FOURCC_MAP = {"BayerRG8": "RG08"}

# Default exposure/gain tuned for 60Hz HDMI DMD switching: 16ms exposure =
# one HDMI frame, prevents mid-exposure tile transitions. 24 dB compensates
# for the shorter exposure vs v7.1's 32ms/18 dB.
DEFAULT_EXPOSURE_US = 16000.0
DEFAULT_GAIN_DB = 24.0

# IMX540 BayerRG8 full-sensor readout (~66 ms) plus host driver receipt
# margin. Used by blocking mode to map a frame's host receive time back to
# the trigger (exposure-start) host time:
#   trigger_host_ns ≈ capture_wall_ns - READOUT_NS
# Kept in sync with analyze_blocking_timing.py's READOUT_NS constant so
# the wait-discipline check there matches the loop's acceptance rule.
READOUT_NS = 70_000_000

# Hash pool and disk pool sizes (v8 pipeline architecture).
HASH_POOL_WORKERS = 4
DISK_POOL_WORKERS = 2

# Number of GPU pre-warm tiles run before the first capture is consumed.
GPU_PREWARM_TILES = 10


# ---- v0.5.1: identity probes (relocated) ----
# _probe_aravis_version, _probe_camera_serial, _probe_camera_firmware →
#   imported from camera.py above.
# _probe_edid_fingerprint → imported from projector.py above.
# _probe_host_config → imported from host_info.py (see below).
# debayer_bayerrg_nn → session_finalize.py.
# _load_tile_generator → tile_backend.py (see below).
# _save_tile_png_safe → session_finalize.py (see below).
# The import aliases below preserve the old names so the TBLoopV8 class
# body below continues to compile unchanged.
from host_info import probe_host_config as _probe_host_config  # noqa: E402
from tile_backend import load_tile_generator as _load_tile_generator  # noqa: E402
from session_finalize import save_tile_png_safe as _save_tile_png_safe  # noqa: E402


# ---- Durability helper ----

class DurabilityThread(threading.Thread):
    """Single thread that consumes fsync requests in FIFO order. Callers
    submit a request via submit() which returns a threading.Event the
    caller can wait on for completion.
    """

    def __init__(self, stop_event):
        super().__init__(daemon=True, name="tb-durability")
        self._stop = stop_event
        self._q: "queue.Queue" = queue.Queue()

    def submit(self, fd_or_path, event, dir_sync=False):
        """Submit a fsync request. fd_or_path is either an OS file
        descriptor (int) or a path (str/Path) to open O_RDONLY for
        directory fsync. event is signalled when the fsync has
        completed (successfully or not).
        """
        self._q.put((fd_or_path, event, dir_sync))

    def run(self):
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.25)
            except queue.Empty:
                continue
            if item is None:
                break
            fd_or_path, event, dir_sync = item
            try:
                if isinstance(fd_or_path, int):
                    os.fsync(fd_or_path)
                else:
                    dfd = os.open(str(fd_or_path), os.O_RDONLY)
                    try:
                        os.fsync(dfd)
                    finally:
                        os.close(dfd)
            except OSError:
                pass
            finally:
                event.set()

    def shutdown(self):
        self._q.put(None)


# ---- Chain thread ordered queue ----

class OrderedChainQueue:
    """Buffer of completed (capture_idx, bayer_hash, ...) tuples pushed
    from hash-pool workers in completion order. The chain thread consumes
    strictly in capture_idx order, waiting for the next_expected entry.
    """

    def __init__(self):
        self._buf: dict[int, tuple] = {}
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._closed = False

    def put(self, capture_idx, item):
        with self._cond:
            self._buf[capture_idx] = item
            self._cond.notify_all()

    def get_at(self, capture_idx, timeout=None):
        """Wait for the entry at the specified capture_idx and pop it.
        Returns None if closed before the entry arrives."""
        with self._cond:
            while capture_idx not in self._buf and not self._closed:
                if not self._cond.wait(timeout=timeout):
                    return None
            if capture_idx in self._buf:
                return self._buf.pop(capture_idx)
            return None

    def close(self):
        with self._cond:
            self._closed = True
            self._cond.notify_all()


# ---- App ----

class TBLoopV8(Gtk.Application):
    def __init__(self, monitor, duration, exposure_us, gain_db,
                 activation, format_str, session_dir,
                 cpu_only=False, session_mode="blocking",
                 calibration_dir=None,
                 rsk_rpc_url=None, interior_publisher=None,
                 network="testnet", wallet_address=None):
        super().__init__(application_id="local.tb.loop.v8")
        self.monitor = monitor
        self.duration = duration
        self.exposure_us = exposure_us
        self.gain_db = gain_db
        self.activation = activation
        self.format_str = format_str
        self.cpu_only = cpu_only
        self.session_mode = session_mode
        self.calibration_dir = Path(calibration_dir or
                                     (Path.home() / ".tb")).expanduser()
        # Populated by _load_calibration_if_blocking when mode=blocking;
        # stays None for async.
        self.rig_pipeline_calibration = None
        self.rig_pipeline_calibration_hash = ""
        self.pipeline_wait_ms = 0
        # Device-to-host clock offset calibrated at blocking-mode chain
        # start. Used to map Aravis device timestamps to host monotonic_ns
        # for the threshold-based accept rule. Fallback (capture_wall_ns
        # minus READOUT_NS) is used if calibration fails or device clock
        # lands out of sane range.
        self._device_ts_offset_ns = 0
        self._device_ts_offset_valid = False
        self.rsk_rpc_url = rsk_rpc_url
        self.interior_publisher = interior_publisher
        self.network = network
        self.wallet_address = wallet_address
        self.session_dir = Path(session_dir)
        self.recordings_dir = self.session_dir / "Recordings"
        self.derived_dir = self.session_dir / "derived"
        self.recordings_previews_dir = self.derived_dir / "Recordings_previews"
        self.emissions_dir = self.derived_dir / "Emissions"
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self.recordings_dir.mkdir(exist_ok=False)
        self.derived_dir.mkdir(exist_ok=False)
        self.recordings_previews_dir.mkdir(exist_ok=False)
        self.emissions_dir.mkdir(exist_ok=False)
        (self.session_dir / ".gitignore").write_text("derived/\n")

        self.chain_log_path = self.session_dir / "chain_log.csv"
        self.capture_log_path = self.session_dir / "capture_log.csv"

        self.bundle = None
        self.manifest = None
        self.session_iso_utc = None
        self.S0 = None

        # Finalization tracking.
        self.signal_caught = False
        self.chain_error = None
        self.setup_error = None
        self.last_S_next = None

        self.stop_event = threading.Event()
        # Signals the chain loop it should finish the current duration.
        # The capture thread sets this after duration expires; chain thread
        # drains its remaining ordered queue, then closes.
        self.capture_stop_event = threading.Event()

        # v8 pipeline state.
        self.capture_count = 0  # captures delivered
        self.N_chain = 0        # chain rows consumed
        self.N_captures = 0     # captures delivered (final count)

        # Tile buffers for drawing.
        self.scratch = np.empty((TILE_H, TILE_W, 4), dtype=np.uint8)
        self.scratch[..., 3] = 255
        self.current_tile = None
        self.tile_lock = threading.Lock()

        # Threads / pools.
        self.hash_pool = None
        self.disk_pool = None
        self.durability_thread = None
        self.png_pool = None
        self._capture_thread = None
        self._chain_thread = None
        self._cpu_pool = None

        # Tile generator.
        self.backend_name = None
        self.gen_tile_fn = None
        self.generator_code_hash = None

        # Camera identity.
        self._camera = None
        self._stream = None
        self._device = None
        self.real_roi = None
        self.payload = None
        self.camera_serial = None
        self.camera_firmware = None
        self.aravis_version = None
        self.edid_fingerprint = None
        self.host_config = None

        # Acquisition bracket.
        self.camera_acquisition_start_wall_ns = None
        self.camera_acquisition_end_wall_ns = None

        # Ordered chain queue (hash pool → chain thread).
        self.chain_queue = OrderedChainQueue()

        # Disk-log lock (disk workers serialise capture_log.csv appends).
        self._capture_log_lock = threading.Lock()
        # Header flags for exclusive-create CSVs.
        self._chain_log_fields = None
        self._capture_log_fields = None

        # GTK window.
        self.window = None
        self.area = None

        # Hard-quit handle.
        self._hard_quit_source_id = None

        # Timing stats collected on the chain thread.
        self.chain_step_ms = []
        self.cuda_gen_ms = []
        self.hash_latency_ms = []

        # v8.3 async-tightened bookkeeping.
        # `mitigations_applied` collects short labels for every
        # mitigation that actually landed — these end up in the
        # finalized manifest. `mitigation_notes` keeps the human-
        # readable detail for the session report.
        self.mitigations_applied: list[str] = []
        self.mitigation_notes: list[str] = []
        # Held-open resources that need to be released at session end.
        self._mlockall_libc = None
        self._cpu_dma_latency_fd = None
        # Pre-allocated capture ring buffer (populated in async-tightened).
        self.capture_ring = None
        self.capture_ring_slot = 0
        # Real-time slip counter for async-tightened.
        self.tightened_slow_iter_count = 0

    # ---- GTK activation / wiring ----

    def do_activate(self):
        self.window = Gtk.ApplicationWindow(application=self)
        self.window.set_decorated(False)
        self.area = Gtk.DrawingArea()
        self.area.set_draw_func(self._draw)
        self.window.set_child(self.area)
        self.window.fullscreen_on_monitor(self.monitor["monitor"])
        try:
            cursor = Gdk.Cursor.new_from_name("none", None)
            self.window.set_cursor(cursor)
        except Exception as e:
            print(f"[gtk] WARN: failed to hide cursor: {e!r}", flush=True)
        self.window.present()
        GLib.timeout_add(500, self._start_session)

    def _hard_quit(self):
        print(f"[{self._iso_now()}] hard_quit fired (duration+15s cap)", flush=True)
        self.stop_event.set()
        self.quit()
        return False

    def _iso_now(self):
        return dt.datetime.now().isoformat(timespec="seconds")

    def _draw(self, area, cr, w, h):
        with self.tile_lock:
            tile = self.current_tile
        if tile is None:
            cr.set_source_rgb(0, 0, 0)
            cr.rectangle(0, 0, w, h)
            cr.fill()
            return
        self.scratch[..., 0] = tile[..., 2]
        self.scratch[..., 1] = tile[..., 1]
        self.scratch[..., 2] = tile[..., 0]
        surf = cairo.ImageSurface.create_for_data(
            memoryview(self.scratch), cairo.FORMAT_ARGB32,
            TILE_W, TILE_H, TILE_W * 4,
        )
        cr.set_source_surface(surf, 0, 0)
        cr.paint()

    def _queue_tile(self, tile):
        """GTK-thread-safe tile update. Returns the wall_ns at which the
        queue_draw call happened (recorded on the GTK main thread).

        Raises RuntimeError on GTK idle-add timeout (2.0 s). v8.2.3
        (2026-04-24): a fake wall_ns is never synthesized when the tile
        was not actually queued. Callers (blocking bootstrap, blocking
        per-row, async per-row) use the returned wall time as the
        reference for the NEXT iteration's pipeline_wait threshold, so
        a "now"-stamp for a tile that never reached the DMD would
        silently poison downstream timing in exactly the same class of
        failure as bug 1. End-of-session black-out callers already wrap
        in try/except Exception and will log + continue; bootstrap /
        per-row callers propagate to the chain loop's outer handler and
        abort the session cleanly.
        """
        evt = threading.Event()
        wall_holder = [None]

        def _post():
            with self.tile_lock:
                self.current_tile = tile
            wall_holder[0] = time.monotonic_ns()
            self.area.queue_draw()
            evt.set()
            return False

        GLib.idle_add(_post)
        if not evt.wait(timeout=2.0):
            raise RuntimeError(
                "_queue_tile: GTK idle_add did not fire within 2.0 s; "
                "tile was NOT projected. Refusing to synthesize a "
                "wall_ns — downstream timing (pipeline_wait threshold "
                "for next accept) would be silently poisoned."
            )
        # evt.wait returned True, so _post ran to completion before
        # setting evt; wall_holder[0] is guaranteed non-None.
        return wall_holder[0]

    # ---- Session setup ----

    def _start_session(self):
        try:
            # v9 deprecation gate: async capture paths (_run_chain /
            # _chain_loop / _capture_loop / hash-pool / disk-pool) have not
            # been updated for v9 B++ live-attribution semantics. Under v9
            # blocking mode is the only supported capture path. Async
            # remains in the source as structural scaffolding for a future
            # symmetric refactor; running it now would produce chain_log
            # rows whose emission-adjacent fields carry v8 (next-emission)
            # attribution even though the bundle's protocol_version reads
            # "TB-v0.9" — a silent correctness failure. Fail fast here,
            # before any artefact is written, rather than later in the
            # async chain thread.
            if self.session_mode != SESSION_MODE_BLOCKING:
                raise NotImplementedError(
                    f"v9 protocol: session_mode={self.session_mode!r} is "
                    f"deprecated. Async capture paths have not been "
                    f"updated for v9 live-attribution semantics and will "
                    f"produce incorrect chain_log rows. Use "
                    f"--mode blocking (the only supported capture path; see "
                    f"METHODS.md and session_schema.py:PROTOCOL_VERSION)."
                )

            # v9 corroborating-evidence mode (2026-04-24): drand is
            # soft. If the client can't reach drand or fails /info
            # validation, log and proceed — RSK is the primary timing
            # anchor. Rows will record drand_round_number=0 and
            # drand_round_value=zeros as sentinels. The verifier will
            # report drand_availability=MISSING/PARTIAL but will not
            # gate PASS/FAIL on it.
            self.drand_client = DrandClient(
                chain_hash=QUICKNET_CHAIN_HASH,
                pubkey_hex=QUICKNET_PUBKEY_HEX,
                max_staleness_s=DEFAULT_MAX_STALENESS_S,
                log_fn=lambda s: print(f"[{self._iso_now()}] {s}",
                                       flush=True),
            )
            try:
                self.drand_client.start()
            except DrandError as e:
                print(f"[{self._iso_now()}] WARN drand start failed: "
                      f"{e}. Session will proceed with drand=MISSING; "
                      f"verifier will flag drand_availability accordingly.",
                      flush=True)

            # Prewarm + load the tile backend BEFORE any artefact is written;
            # generator_code_hash needs the active backend's source hash.
            self._load_and_prewarm_backend()
            self._setup_camera()
            # v8.2: load pipeline calibration if --mode blocking (must come
            # after camera setup so serial/firmware are known for rig_hash,
            # but before bundle write so the calibration can be committed).
            self._load_calibration_if_blocking()
            self._build_and_write_session_artefacts()
            self._open_logs()
            self._begin_acquisition()
            # Arm hard-quit AFTER setup. duration + 15s covers the chain
            # tail + finalize bracket. Blocking mode runs slower so we
            # give it a more generous budget.
            hard_quit_budget = self.duration + (
                60 if self.session_mode == "blocking" else 15
            )
            self._hard_quit_source_id = GLib.timeout_add(
                int(hard_quit_budget * 1000), self._hard_quit
            )
            # Disable GC for the duration of the session; re-enabled in finalize.
            gc.disable()
            if self.session_mode == SESSION_MODE_BLOCKING:
                # Blocking mode: one synchronous chain thread. No pools,
                # no capture thread, no durability thread. All fsyncs
                # on the critical path.
                self._chain_thread = threading.Thread(
                    target=self._chain_loop_blocking, daemon=True,
                    name="tb-chain-blocking"
                )
                self._chain_thread.start()
            else:
                tightened = (self.session_mode
                             == SESSION_MODE_ASYNC_TIGHTENED)
                if tightened:
                    self._apply_async_tightened_mitigations()

                # Pool initializer: pin each worker to CPUs 2-3 and
                # (attempt to) raise SCHED_FIFO. Disk at prio 20
                # (spec), hash at 60 (below chain 70, above disk 20).
                hash_init = None
                disk_init = None
                png_init = None
                if tightened:
                    def _hash_init():
                        res = async_mitigations.thread_tighten({2, 3}, 60)
                        for (_, label, detail) in res:
                            # Thread-local; captured once the first
                            # worker runs initializer. Append to shared
                            # list — threadsafe enough for our purposes,
                            # and the labels de-dupe downstream.
                            self.mitigations_applied.append(
                                f"hash/{label}")
                            self.mitigation_notes.append(
                                f"hash pool thread: {detail}")

                    def _disk_init():
                        res = async_mitigations.thread_tighten({2, 3}, 20)
                        for (_, label, detail) in res:
                            self.mitigations_applied.append(
                                f"disk/{label}")
                            self.mitigation_notes.append(
                                f"disk pool thread: {detail}")

                    def _png_init():
                        # PNG pool: low priority, same disk CPUs.
                        res = async_mitigations.thread_tighten({2, 3}, 10)
                        for (_, label, detail) in res:
                            self.mitigations_applied.append(
                                f"png/{label}")

                    hash_init = _hash_init
                    disk_init = _disk_init
                    png_init = _png_init

                self.durability_thread = DurabilityThread(self.stop_event)
                self.durability_thread.start()
                self.hash_pool = ThreadPoolExecutor(
                    max_workers=HASH_POOL_WORKERS,
                    thread_name_prefix="tb-hash",
                    initializer=hash_init,
                )
                self.disk_pool = ThreadPoolExecutor(
                    max_workers=DISK_POOL_WORKERS,
                    thread_name_prefix="tb-disk",
                    initializer=disk_init,
                )
                self.png_pool = ThreadPoolExecutor(
                    max_workers=2, thread_name_prefix="tb-png",
                    initializer=png_init,
                )
                self._capture_thread = threading.Thread(
                    target=self._capture_loop, daemon=True,
                    name="tb-capture"
                )
                self._chain_thread = threading.Thread(
                    target=self._chain_loop, daemon=True, name="tb-chain"
                )
                self._capture_thread.start()
                self._chain_thread.start()
        except Exception as e:
            self.setup_error = f"{type(e).__name__}: {e}"
            import traceback
            traceback.print_exc()
            print(f"[setup] FATAL — {self.setup_error}", flush=True)
            self._safe_cleanup()
        return False

    def _load_calibration_if_blocking(self):
        """Blocking mode requires a committed calibration for this rig.
        Locates ~/.tb/pipeline_calibration_<rig_hash[:16]>.json, verifies
        the rig_hash matches, warns on age, and populates
        self.rig_pipeline_calibration + self.pipeline_wait_ms. Raises
        RuntimeError with a clear message if missing or stale."""
        if self.session_mode != "blocking":
            print(f"[{self._iso_now()}] mode: async (no calibration required)",
                  flush=True)
            return

        # Build the rig_config (must match calibrate_pipeline_delay.py).
        import json
        rig_config = {
            "exposure_us": int(self.exposure_us),
            "gain_db": float(self.gain_db),
            "projector_connector": self.monitor["connector"],
            "hdmi_edid_fingerprint": _probe_edid_fingerprint(
                self.monitor["connector"]),
            "camera_serial": self.camera_serial,
            "camera_firmware": self.camera_firmware,
        }
        rig_hash_hex = blake3(canonical_json_bytes(rig_config)).hexdigest()

        cal_path = self.calibration_dir / (
            f"pipeline_calibration_{rig_hash_hex[:16]}.json"
        )
        if not cal_path.exists():
            raise RuntimeError(
                f"No pipeline delay calibration found for this rig. "
                f"Expected at {cal_path}. Run "
                f"calibrate_pipeline_delay.py --connector "
                f"{self.monitor['connector']} --exposure-us "
                f"{int(self.exposure_us)} --gain-db {self.gain_db} first."
            )
        raw = cal_path.read_bytes()
        cal = json.loads(raw.decode("utf-8"))
        # Rig-hash consistency check.
        if cal.get("rig_hash") != rig_hash_hex:
            raise RuntimeError(
                f"Calibration file {cal_path} has rig_hash="
                f"{cal.get('rig_hash')!r}; this session's rig hashes to "
                f"{rig_hash_hex!r}. Re-run calibrate_pipeline_delay.py."
            )
        rw = cal.get("recommended_wait_ms")
        if not isinstance(rw, int) or rw <= 0:
            raise RuntimeError(
                f"Calibration file {cal_path} has bad "
                f"recommended_wait_ms={rw!r}"
            )

        # Age check — warn (don't abort) if > 24h old.
        ts = cal.get("measurement_timestamp")
        try:
            meas = dt.datetime.fromisoformat(ts)
            age_s = (dt.datetime.now(dt.timezone.utc) - meas).total_seconds()
            if age_s > 24 * 3600:
                print(f"[{self._iso_now()}] WARN: calibration is "
                      f"{age_s/3600:.1f} h old (> 24h). Consider re-running "
                      f"calibrate_pipeline_delay.py.", flush=True)
            else:
                print(f"[{self._iso_now()}] calibration age "
                      f"{age_s/3600:.1f} h (< 24h)", flush=True)
        except Exception:
            print(f"[{self._iso_now()}] WARN: calibration "
                  f"measurement_timestamp unreadable: {ts!r}", flush=True)

        # Canonical-JSON hash of the calibration dict (to commit into bundle).
        self.rig_pipeline_calibration = cal
        self.rig_pipeline_calibration_hash = blake3(
            canonical_json_bytes(cal)
        ).hexdigest()
        self.pipeline_wait_ms = int(rw)
        print(f"[{self._iso_now()}] calibration loaded: "
              f"recommended_wait_ms={rw}  rig_hash={rig_hash_hex[:16]}…",
              flush=True)
        print(f"[{self._iso_now()}] calibration confidence: "
              f"{cal.get('confidence', '(none)')}", flush=True)

    def _safe_cleanup(self):
        try:
            if self._camera is not None:
                try:
                    self._camera.stop_acquisition()
                except Exception:
                    pass
        except Exception:
            pass
        self.stop_event.set()
        try:
            GLib.idle_add(self.quit)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Async-tightened mitigations (v8.3)
    # ------------------------------------------------------------------

    def _apply_async_tightened_mitigations(self):
        """Probe the environment and apply every mitigation the host
        permits. Populates self.mitigations_applied (short labels) and
        self.mitigation_notes (human-readable detail). Never aborts —
        any mitigation that fails is logged and the session continues."""
        print(f"[{self._iso_now()}] tightened: probing capabilities…",
              flush=True)
        cap = async_mitigations.probe_capabilities()
        summary = async_mitigations.summarize_capabilities(cap)
        for line in summary.splitlines():
            print(f"[tightened] cap  {line}", flush=True)
        for warn in async_mitigations.service_warnings(cap):
            print(f"[tightened] WARN {warn}", flush=True)
            self.mitigation_notes.append(f"WARN: {warn}")

        # Governor is report-only (we don't auto-set).
        ok, label, detail = async_mitigations.governor_summary(cap)
        self.mitigations_applied.append(label)
        self.mitigation_notes.append(detail)

        # Session-wide: mlockall.
        ok, label, detail, libc = async_mitigations.try_mlockall()
        self.mitigations_applied.append(label)
        self.mitigation_notes.append(detail)
        self._mlockall_libc = libc

        # /dev/cpu_dma_latency = 0 µs.
        ok, label, detail, fd = async_mitigations.try_cpu_dma_latency_zero()
        self.mitigations_applied.append(label)
        self.mitigation_notes.append(detail)
        self._cpu_dma_latency_fd = fd

        # GC lockdown (keep the prior gc.disable() after this; our
        # lockdown does collect+freeze+disable on top of it).
        ok, label, detail = async_mitigations.gc_lockdown()
        self.mitigations_applied.append(label)
        self.mitigation_notes.append(detail)

        # Pre-allocate capture ring buffer for the tightened capture
        # loop. 16 slots × ROI bytes = ~400 MB at full-sensor
        # 5320×4600 BayerRG8. With ring size 16 and ~15 Hz throughput,
        # the oldest slot is re-used ~1 s after its previous use —
        # far longer than the hash (8 ms) + disk (few ms) worker's
        # read window.
        rw, rh = self.real_roi[2], self.real_roi[3]
        self.capture_ring = [
            np.empty(rw * rh, dtype=np.uint8) for _ in range(16)
        ]
        self.capture_ring_slot = 0
        self.mitigations_applied.append("capture_ring:16×roi")
        self.mitigation_notes.append(
            f"Pre-allocated capture ring: 16 × {rw * rh} bytes "
            f"= {16 * rw * rh / (1024*1024):.0f} MB")

        print(f"[{self._iso_now()}] tightened: applied "
              f"{len(self.mitigations_applied)} mitigations: "
              f"{self.mitigations_applied}", flush=True)

    def _release_async_tightened_mitigations(self):
        """Reverse side-effects of _apply_…. Safe to call multiple times
        and safe to call when nothing was applied."""
        try:
            async_mitigations.munlockall(self._mlockall_libc)
        except Exception:
            pass
        self._mlockall_libc = None
        try:
            async_mitigations.close_cpu_dma_latency(self._cpu_dma_latency_fd)
        except Exception:
            pass
        self._cpu_dma_latency_fd = None
        # gc will be re-enabled by the main finalize path; we also
        # unfreeze here to release any frozen objects.
        try:
            async_mitigations.gc_release()
        except Exception:
            pass

    def _load_and_prewarm_backend(self):
        """Load the active tile generator, pre-warm it with GPU_PREWARM_TILES,
        then run a deterministic byte-match check against the CPU reference.
        """
        print(f"[{self._iso_now()}] backend: loading "
              f"{'CPU' if self.cpu_only else 'GPU'} tile generator…", flush=True)
        self.backend_name, self.gen_tile_fn, self.generator_code_hash, self._cpu_pool = \
            _load_tile_generator(self.cpu_only)
        print(f"[{self._iso_now()}] backend: {self.backend_name}  "
              f"generator_code_hash={self.generator_code_hash[:16]}…", flush=True)

        # Pre-warm.
        prewarm_seed = blake3(b"tb-v8-prewarm").digest()
        for i in range(GPU_PREWARM_TILES):
            seed_i = blake3(prewarm_seed + i.to_bytes(4, "big")).digest()
            xr, xg, xb = compute_xof_seeds(seed_i)
            self.gen_tile_fn(xr, xg, xb)
        if not self.cpu_only:
            import torch
            torch.cuda.synchronize()
        print(f"[{self._iso_now()}] backend: pre-warmed "
              f"{GPU_PREWARM_TILES} tiles", flush=True)

        # Deterministic check: one CPU tile + one backend tile, same seed.
        # Skipped when cpu_only — the active backend IS the CPU path.
        if not self.cpu_only:
            import tile_cpu
            check_pool = ThreadPoolExecutor(max_workers=3,
                                             thread_name_prefix="tb-det-cpu")
            try:
                det_seed = blake3(b"tb-v8-determinism-check").digest()
                xr, xg, xb = compute_xof_seeds(det_seed)
                gpu_tile = self.gen_tile_fn(xr, xg, xb)
                cpu_tile = tile_cpu.gen_rgb_v2(xr, xg, xb, check_pool)
                if not np.array_equal(gpu_tile, cpu_tile):
                    gpu_sha = blake3(gpu_tile.tobytes()).hexdigest()[:16]
                    cpu_sha = blake3(cpu_tile.tobytes()).hexdigest()[:16]
                    raise RuntimeError(
                        f"GPU/CPU determinism check FAILED: "
                        f"gpu={gpu_sha}… vs cpu={cpu_sha}…"
                    )
                print(f"[{self._iso_now()}] backend: determinism check PASSED "
                      f"(GPU tile bytes == CPU tile bytes)", flush=True)
            finally:
                check_pool.shutdown(wait=True)
        else:
            print(f"[{self._iso_now()}] backend: determinism check skipped "
                  f"(--cpu-only; active path is the CPU reference)", flush=True)

    def _build_and_write_session_artefacts(self):
        rx, ry, rw, rh = self.real_roi
        self.edid_fingerprint = _probe_edid_fingerprint(self.monitor["connector"])
        if self.edid_fingerprint is None:
            print(f"[{self._iso_now()}] WARN: could not read EDID for "
                  f"{self.monitor['connector']!r}", flush=True)
        else:
            print(f"[{self._iso_now()}] projector EDID fingerprint = "
                  f"{self.edid_fingerprint[:16]}…", flush=True)
        self.host_config = _probe_host_config()
        print(f"[{self._iso_now()}] host_config = {self.host_config['hostname']}"
              f" / {self.host_config['os_release']}", flush=True)

        bundle = VerificationBundle(
            protocol_version=PROTOCOL_VERSION,
            generator_code_hash=self.generator_code_hash,
            wallet_address=(self.wallet_address or ""),
            chain_id=NETWORK_CHAIN_IDS[self.network],
            generator_config={
                "NUM_OCTAVES": NUM_OCTAVES,
                "GRID_H_TABLE": list(GRID_H_TABLE),
                "GRID_W_TABLE": list(GRID_W_TABLE),
                "bit_depth": 8,
                "upsample_method": "integer_bilinear_16_16_fixed",
                "persistence_shift_per_octave": ">> octave",
            },
            camera_config={
                "model": "DFK 38UX540",
                "sensor_size": [5320, 4600],
                "pixel_format": self.format_str,
                "roi": [int(rx), int(ry), int(rw), int(rh)],
                "exposure_us": float(self.exposure_us),
                "gain_db": float(self.gain_db),
                "white_balance_off": True,
                "trigger_config": {
                    "source": "Line1",
                    "mode": "On",
                    "selector": "FrameStart",
                    "activation": self.activation,
                },
                "serial_number": self.camera_serial,
                "firmware_version": self.camera_firmware,
                "aravis_version": self.aravis_version,
            },
            projector_config={
                "model": "EKB 4750LC",
                "connector": self.monitor["connector"],
                "resolution": [TILE_W, TILE_H],
                "refresh_rate_hz": 60,
                "color_space": "sRGB",
                "serial_number": None,
                "edid_fingerprint": self.edid_fingerprint,
            },
            tile_config={
                # tile_config pins the tile output buffer for the
                # reconstructor. v8 chain no longer commits tile bytes,
                # but reconstructor replays tile generation from the
                # chain's xof_seed columns to check optical binding.
                "pixel_format": "RGB888",
                "bit_depth": 8,
                "dimensions": [TILE_W, TILE_H],
            },
            chain_config={
                "hash": "blake3",
                "state_bytes": 32,
            },
            metadata_schema={
                "struct_format": META_STRUCT,
                "endianness": "big",
                "total_bytes": META_LEN,
                "fields": [
                    {"name": "t",                          "offset":  0, "length_bytes": 4, "type": "uint32"},
                    {"name": "aravis_device_timestamp_ns", "offset":  4, "length_bytes": 8, "type": "uint64"},
                    {"name": "capture_wall_ns",            "offset": 12, "length_bytes": 8, "type": "uint64"},
                    {"name": "exposure_us",                "offset": 20, "length_bytes": 4, "type": "uint32"},
                    {"name": "fourcc",                     "offset": 24, "length_bytes": 4, "type": "ascii4"},
                ],
                "fourcc_map": FOURCC_MAP,
            },
            host_config=self.host_config,
            # v8.2: mode + calibration. For async, calibration is null
            # and hash is "". For blocking, both are populated by
            # _load_calibration_if_blocking.
            session_mode=self.session_mode,
            rig_pipeline_calibration=self.rig_pipeline_calibration,
            rig_pipeline_calibration_hash=self.rig_pipeline_calibration_hash,
        )
        bundle.finalize()
        bundle.write(self.session_dir / "verification_bundle.json")
        self.bundle = bundle

        if self.interior_publisher is not None:
            anchor_policy = {
                "enabled": True,
                "interval_s": float(self.interior_publisher.anchor_interval_s),
                "chain_id": int(self.interior_publisher.chain_id),
                "network": str(self.interior_publisher.network),
            }
        else:
            anchor_policy = {"enabled": False}

        self.session_iso_utc = dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="microseconds")
        manifest_path = self.session_dir / "manifest.json"

        # v9: generate a 32-byte local CSPRNG nonce ONCE at session open.
        # Published in manifest.local_nonce_hex (covered by
        # manifest_hash_open) and explicitly re-committed inside derive_s0
        # below. Raises S_0 unpredictability above what the RSK fresh-block
        # hash alone provides: anyone watching RSK can predict the
        # next-block hash under some assumptions, but they cannot predict
        # our local urandom output.
        local_nonce = os.urandom(32)

        session_id = str(uuid.uuid4())
        # v9: pin the drand anchor in the manifest BEFORE manifest_hash_open
        # is finalised, so the session-level drand commitment is part of
        # the opening snapshot (parallel to anchor_start for RSK). Also
        # pin the RSK RPC endpoints list (primary + fallbacks) so the
        # operator cannot retroactively claim a different set was used.
        rsk_rpc_endpoints = []
        if self.interior_publisher is not None:
            rsk_rpc_endpoints = list(self.interior_publisher.rpc_urls)

        manifest = SessionManifest(
            session_id=session_id,
            session_iso_utc_start=self.session_iso_utc,
            bundle_hash=bundle.bundle_hash,
            S_0_hex="",
            device_id=socket.gethostname(),
            local_nonce_hex=local_nonce.hex(),
            drand_chain_hash=self.drand_client.chain_hash,
            drand_public_key=self.drand_client.pubkey_hex,
            drand_round_at_session_open=self.drand_client.initial_round_number,
            rsk_rpc_endpoints=rsk_rpc_endpoints,
            anchor_policy=anchor_policy,
            anchor_start=None,
        )
        manifest.finalize_open()
        # v9 layered derive_s0: rsk_block_hash/height are zero sentinels at
        # this first call (anchor_start still None); re-derived below after
        # the anchor wait if --rsk-rpc is on.
        self.S0 = derive_s0(
            rsk_block_hash_hex=UNANCHORED_BLOCK_HASH_HEX,
            rsk_block_height=UNANCHORED_BLOCK_HEIGHT,
            local_nonce_hex=manifest.local_nonce_hex,
            session_id=manifest.session_id,
            manifest_hash_open_hex=manifest.manifest_hash_open,
        )
        manifest.S_0_hex = self.S0.hex()

        # v9: bind the interior-anchor publisher to this session_id so
        # pulse commitments can be computed. The publisher's thread is
        # already running by this point but skips cleanly until bound.
        if self.interior_publisher is not None:
            self.interior_publisher.set_session_id(manifest.session_id)
        validate_bundle_manifest_pair(bundle.to_dict(), manifest.to_dict(),
                                       path="session-build-pre-anchor")
        manifest.write(manifest_path)
        self.manifest = manifest

        print(f"[{self._iso_now()}] verification_bundle: "
              f"bundle_hash={bundle.bundle_hash[:32]}…", flush=True)
        print(f"[{self._iso_now()}] manifest (pre-anchor): "
              f"manifest_hash_open={manifest.manifest_hash_open[:32]}…",
              flush=True)

        if self.rsk_rpc_url:
            expected_chain_id = NETWORK_CHAIN_IDS[self.network]
            print(f"[{self._iso_now()}] rsk: verifying chain_id at "
                  f"{self.rsk_rpc_url}…", flush=True)
            actual_chain_id = fetch_chain_id(self.rsk_rpc_url)
            if actual_chain_id != expected_chain_id:
                raise RuntimeError(
                    f"start-anchor RPC chain_id mismatch: --network "
                    f"{self.network} expects chain_id {expected_chain_id}, "
                    f"but {self.rsk_rpc_url} returned {actual_chain_id}."
                )
            print(f"[{self._iso_now()}] rsk: chain_id verified ({actual_chain_id})",
                  flush=True)
            print(f"[{self._iso_now()}] rsk: fetching current tip…", flush=True)
            cur = fetch_current_tip(self.rsk_rpc_url)
            print(f"[{self._iso_now()}] rsk: current_tip "
                  f"#{cur['block_number']}  hash={cur['block_hash'][:16]}…",
                  flush=True)
            print(f"[{self._iso_now()}] rsk: waiting for next block…",
                  flush=True)
            t_wait = time.monotonic()
            new = wait_for_next_block(self.rsk_rpc_url, cur)
            wait_s = time.monotonic() - t_wait
            observed_at_utc = dt.datetime.now(dt.timezone.utc).isoformat(
                timespec="microseconds")
            anchor_start = {
                "chain": f"rsk-{self.network}",
                "chain_id": NETWORK_CHAIN_IDS[self.network],
                "block_number": new["block_number"],
                "block_hash": new["block_hash"],
                "block_timestamp_utc": new["timestamp_utc"],
                "parent_hash": new["parent_hash"],
                "observed_at_utc": observed_at_utc,
                # v9 audit trail: explicit record of the freshness wait so
                # the verifier can prove the post-wait block (block_hash)
                # was published AFTER the pre-wait tip was observed.
                # parent_hash should equal pre_wait_tip_block_hash for a
                # clean single-block wait (not a reorg or skip).
                "pre_wait_tip_block_number": cur["block_number"],
                "pre_wait_tip_block_hash": cur["block_hash"],
                "wait_duration_s": float(wait_s),
            }
            print(f"[{self._iso_now()}] rsk: anchor_start "
                  f"#{new['block_number']}  hash={new['block_hash'][:16]}…  "
                  f"({wait_s:.2f}s wait)", flush=True)
            if new["parent_hash"] != cur["block_hash"]:
                print(f"[{self._iso_now()}] rsk: WARNING "
                      f"new block's parent_hash {new['parent_hash'][:16]}… "
                      f"!= prior tip's hash {cur['block_hash'][:16]}… — "
                      f"possible reorg between fetches; continuing", flush=True)

            manifest.update_open_state_anchor_start(anchor_start)
            # v9 layered derive_s0: now with real anchor fields. This is
            # the authoritative S_0 the chain commits to (the pre-anchor
            # derivation above is overwritten).
            self.S0 = derive_s0(
                rsk_block_hash_hex=anchor_start["block_hash"],
                rsk_block_height=anchor_start["block_number"],
                local_nonce_hex=manifest.local_nonce_hex,
                session_id=manifest.session_id,
                manifest_hash_open_hex=manifest.manifest_hash_open,
            )
            manifest.S_0_hex = self.S0.hex()
            validate_bundle_manifest_pair(bundle.to_dict(), manifest.to_dict(),
                                           path="session-build-post-anchor")
            manifest.write(manifest_path)
            print(f"[{self._iso_now()}] manifest (post-anchor): "
                  f"manifest_hash_open={manifest.manifest_hash_open[:32]}…",
                  flush=True)
        else:
            print(f"[{self._iso_now()}] rsk: no --rsk-rpc — UNANCHORED "
                  f"(offline-demo mode)", flush=True)

        print(f"[{self._iso_now()}] S_0 = {self.S0.hex()}", flush=True)

    def _setup_camera(self):
        Aravis.update_device_list()
        if Aravis.get_n_devices() == 0:
            raise RuntimeError("no Aravis cameras")
        dev_id = Aravis.get_device_id(0)
        print(f"[{self._iso_now()}] opening camera: {dev_id}", flush=True)
        self._camera = Aravis.Camera.new(dev_id)
        self._camera.set_pixel_format_from_string(self.format_str)
        self._device = self._camera.get_device()

        self.camera_serial = _probe_camera_serial(self._device, dev_id)
        self.camera_firmware = _probe_camera_firmware(self._device)
        self.aravis_version = _probe_aravis_version()

        sw, sh = self._camera.get_sensor_size()
        for name, value in (("OffsetX", 0), ("OffsetY", 0),
                            ("Width", sw), ("Height", sh)):
            try:
                self._device.set_integer_feature_value(name, int(value))
            except Exception as e:
                print(f"  WARN set {name}={value}: {e}", flush=True)
        rx = self._device.get_integer_feature_value("OffsetX")
        ry = self._device.get_integer_feature_value("OffsetY")
        rw = self._device.get_integer_feature_value("Width")
        rh = self._device.get_integer_feature_value("Height")
        self.real_roi = (rx, ry, rw, rh)
        self.payload = self._camera.get_payload()
        print(f"  ROI={rw}x{rh}+{rx}+{ry}  payload={self.payload} B", flush=True)

        for n, v in (("ExposureAuto", "Off"),
                     ("GainAuto", "Off"),
                     ("BalanceWhiteAuto", "Off")):
            try:
                self._device.set_string_feature_value(n, v)
            except Exception as e:
                print(f"  WARN set {n}={v}: {e}", flush=True)
        self._device.set_float_feature_value("ExposureTime", float(self.exposure_us))
        self._device.set_float_feature_value("Gain", float(self.gain_db))
        for n, v in (("TriggerSelector", "FrameStart"),
                     ("TriggerMode", "On"),
                     ("TriggerSource", "Line1"),
                     ("TriggerActivation", self.activation)):
            self._device.set_string_feature_value(n, v)

        for k in ("ExposureTime", "Gain"):
            print(f"  {k} = {self._device.get_float_feature_value(k)!r}",
                  flush=True)

        self._stream = self._camera.create_stream(None, None)
        for _ in range(32):
            self._stream.push_buffer(Aravis.Buffer.new_allocate(self.payload))
        print(f"[{self._iso_now()}] camera configured (acquisition deferred)",
              flush=True)

    def _begin_acquisition(self):
        self._camera.start_acquisition()
        self.camera_acquisition_start_wall_ns = time.monotonic_ns()
        print(f"[{self._iso_now()}] acquisition started "
              f"(wall_ns={self.camera_acquisition_start_wall_ns})", flush=True)

    def _open_logs(self):
        # v9 B++ schema. Canonical source is session_logs.CHAIN_LOG_FIELDS;
        # duplicated here because this module does not import session_logs
        # for the write path. Keep the two in sync.
        # v9 drand columns (drand_round_number, drand_round_value_hex)
        # append at the end. See session_logs.CHAIN_LOG_FIELDS docstring
        # for the semantic description of each column.
        self._chain_log_fields = [
            "t", "S_t_hex",
            "bayer_blake3_hex",
            "capture_frame_id",
            "aravis_device_timestamp_ns",
            "capture_wall_ns",
            "emission_live_pixel_blake3_hex",
            "emission_live_queued_wall_ns",
            "meta_hex",
            "emission_png_path",
            "drand_round_number",
            "drand_round_value_hex",
            "drand_staleness_ms",
        ]
        with open(self.chain_log_path, "x", newline="") as f:
            f.write(f"# session_iso_utc={self.session_iso_utc}\n")
            w = csv.DictWriter(f, fieldnames=self._chain_log_fields)
            w.writeheader()
            f.flush()
            os.fsync(f.fileno())

        self._capture_log_fields = [
            "capture_idx", "capture_frame_id",
            "aravis_device_timestamp_ns", "capture_wall_ns",
            "bayer_blake3_hex", "raw_path",
            "consumed_into_chain", "consumed_as_t",
        ]
        with open(self.capture_log_path, "x", newline="") as f:
            f.write(f"# session_iso_utc={self.session_iso_utc}\n")
            w = csv.DictWriter(f, fieldnames=self._capture_log_fields)
            w.writeheader()
            f.flush()
            os.fsync(f.fileno())

        try:
            dfd = os.open(self.session_dir, os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except OSError:
            pass

    # ---- Thread loops ----

    def _capture_loop(self):
        """THREAD 1. Aravis-driven. One iteration = one delivered buffer.
        Submits to hash pool and disk pool; returns the Aravis buffer
        ASAP so the next trigger doesn't starve."""
        tightened = (self.session_mode == SESSION_MODE_ASYNC_TIGHTENED)
        if tightened:
            # Capture thread is the most latency-sensitive: CPU 0,
            # SCHED_FIFO priority 80.
            for (_, label, detail) in async_mitigations.thread_tighten(
                    {0}, 80):
                self.mitigations_applied.append(f"capture/{label}")
                self.mitigation_notes.append(f"capture thread: {detail}")
        capture_idx = 0
        session_start_wall = time.monotonic_ns()
        duration_ns = int(self.duration * 1e9)
        slow_threshold_ns = 50_000_000  # 50 ms
        try:
            while not self.stop_event.is_set():
                if time.monotonic_ns() - session_start_wall >= duration_ns:
                    # Duration reached. Signal chain loop to drain and exit.
                    break
                buf = self._stream.timeout_pop_buffer(250_000)
                if buf is None:
                    continue
                if buf.get_status() != Aravis.BufferStatus.SUCCESS:
                    self._stream.push_buffer(buf)
                    continue
                capture_wall_ns = time.monotonic_ns()
                frame_id = buf.get_frame_id()
                ts_ns = buf.get_timestamp()

                rx, ry, rw, rh = self.real_roi
                expected = rw * rh if self.format_str == "BayerRG8" else None

                if tightened and self.capture_ring is not None:
                    # Zero-allocation capture: memcpy into pre-allocated
                    # ring slot. Aravis buffer can go back immediately.
                    slot_i = self.capture_ring_slot % len(self.capture_ring)
                    slot = self.capture_ring[slot_i]
                    src_view = np.frombuffer(buf.get_data(), dtype=np.uint8)
                    if expected is not None and src_view.size != expected:
                        self._stream.push_buffer(buf)
                        print(f"[capture] WARN: capture_idx={capture_idx} "
                              f"size {src_view.size} != {expected}",
                              flush=True)
                        continue
                    np.copyto(slot, src_view)
                    self._stream.push_buffer(buf)
                    payload = slot
                    self.capture_ring_slot += 1
                else:
                    data = bytes(buf.get_data())
                    self._stream.push_buffer(buf)
                    if expected is not None and len(data) != expected:
                        print(f"[capture] WARN: capture_idx={capture_idx} "
                              f"payload size {len(data)} != {expected} — "
                              f"dropping", flush=True)
                        continue
                    payload = data

                # Submit hash job. `payload` is buffer-protocol-compatible
                # (bytes or np.uint8 array); workers use it transparently.
                self.hash_pool.submit(
                    self._hash_worker,
                    capture_idx, payload, frame_id, ts_ns, capture_wall_ns
                )
                # Submit disk job.
                self.disk_pool.submit(
                    self._disk_worker,
                    capture_idx, payload, frame_id, ts_ns, capture_wall_ns
                )
                handler_done_ns = time.monotonic_ns()
                if tightened and (handler_done_ns - capture_wall_ns
                                  > slow_threshold_ns):
                    self.tightened_slow_iter_count += 1
                capture_idx += 1

            self.capture_count = capture_idx
        except Exception as e:
            print(f"[capture] EXCEPTION: {e!r}", flush=True)
            import traceback
            traceback.print_exc()
        finally:
            # Order matters: set capture_count before the event so the chain
            # thread observes the final count the instant it sees the event.
            self.capture_count = capture_idx
            self.N_captures = capture_idx
            self.capture_stop_event.set()
            try:
                self._camera.stop_acquisition()
            except Exception:
                pass
            self.camera_acquisition_end_wall_ns = time.monotonic_ns()
            print(f"[{self._iso_now()}] capture: stopped after {capture_idx} "
                  f"frames delivered", flush=True)

    def _hash_worker(self, capture_idx, raw_bytes, frame_id, ts_ns, capture_wall_ns):
        """THREAD 2 (pool worker). Compute blake3, put into ordered queue."""
        t_start = time.monotonic_ns()
        try:
            bayer_hash = blake3(raw_bytes).digest()
            hash_latency_ms = (time.monotonic_ns() - t_start) / 1e6
            self.chain_queue.put(
                capture_idx,
                (bayer_hash, frame_id, ts_ns, capture_wall_ns, hash_latency_ms)
            )
        except Exception as e:
            print(f"[hash] WARN capture_idx={capture_idx}: {e!r}", flush=True)

    def _disk_worker(self, capture_idx, raw_bytes, frame_id, ts_ns, capture_wall_ns):
        """THREAD 4 (pool worker). Atomic raw write, capture_log append,
        submit fsync to durability thread."""
        raw_final = self.recordings_dir / f"frame_{capture_idx:06d}.raw"
        raw_tmp = self.recordings_dir / f".frame_{capture_idx:06d}.raw.tmp"
        try:
            with open(raw_tmp, "wb") as rf:
                rf.write(raw_bytes)
                rf.flush()
                os.fsync(rf.fileno())
            os.replace(raw_tmp, raw_final)
            # Ask durability thread to fsync the recordings dir.
            dir_evt = threading.Event()
            self.durability_thread.submit(self.recordings_dir, dir_evt,
                                           dir_sync=True)
            dir_evt.wait(timeout=5.0)
            bayer_blake3_hex = blake3(raw_bytes).hexdigest()
            row = {
                "capture_idx": capture_idx,
                "capture_frame_id": frame_id,
                "aravis_device_timestamp_ns": ts_ns,
                "capture_wall_ns": capture_wall_ns,
                "bayer_blake3_hex": bayer_blake3_hex,
                "raw_path": f"Recordings/{raw_final.name}",
                # These columns finalized offline after session end (see
                # _post_session_fill_capture_log). Written now with
                # conservative placeholder values so the row is well-formed
                # if the session aborts mid-run.
                "consumed_into_chain": "false",
                "consumed_as_t": -1,
            }
            with self._capture_log_lock:
                with open(self.capture_log_path, "a", newline="") as cf:
                    cw = csv.DictWriter(cf, fieldnames=self._capture_log_fields)
                    cw.writerow(row)
                    cf.flush()
                    file_evt = threading.Event()
                    self.durability_thread.submit(
                        cf.fileno(), file_evt, dir_sync=False
                    )
                    # Important: fileno is valid only while file is open;
                    # wait before leaving the `with` block.
                    file_evt.wait(timeout=5.0)
        except Exception as e:
            print(f"[disk] WARN capture_idx={capture_idx}: {e!r}", flush=True)

    def _chain_loop(self):
        """THREAD 3. Strictly in-order consumer of the ordered queue.
        Per capture_idx: pack meta, advance chain, derive xof seeds,
        generate tile, queue_draw, append chain_log, fsync via
        durability thread, update publisher."""
        try:
            self._run_chain()
        except Exception as e:
            self.chain_error = repr(e)
            print(f"[chain] EXCEPTION: {e}", flush=True)
            import traceback
            traceback.print_exc()
        finally:
            self.stop_event.set()
            self.chain_queue.close()
            # End-of-session black-out — not a chain step.
            try:
                black = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
                self._queue_tile(black)
                t_black = time.monotonic()
                while time.monotonic() - t_black < END_BLACK_MS / 1000.0:
                    while GLib.MainContext.default().iteration(may_block=False):
                        pass
                    time.sleep(0.005)
                print(f"[{self._iso_now()}] projector: black-out emitted "
                      f"({END_BLACK_MS} ms GLib spin)", flush=True)
            except Exception as e:
                print(f"[{self._iso_now()}] WARN: end black-out failed: "
                      f"{e!r}", flush=True)
            GLib.idle_add(self.quit)

    def _run_chain(self):
        print(f"[{self._iso_now()}] chain: starting (hash-first, per-capture)",
              flush=True)
        if self.session_mode == SESSION_MODE_ASYNC_TIGHTENED:
            # Chain thread: CPU 1, SCHED_FIFO priority 70.
            for (_, label, detail) in async_mitigations.thread_tighten(
                    {1}, 70):
                self.mitigations_applied.append(f"chain/{label}")
                self.mitigation_notes.append(f"chain thread: {detail}")
        # Pre-roll black so the projector isn't stuck on whatever the
        # window was showing before.
        black = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
        self._queue_tile(black)
        time.sleep(PREROLL_BLACK_MS / 1000.0)

        S_t = self.S0
        next_expected = 0
        fourcc_bytes = FOURCC_MAP[self.format_str].encode("ascii")
        exposure_us_int = int(round(self.exposure_us))

        drain_deadline = None
        while not self.stop_event.is_set():
            item = self.chain_queue.get_at(next_expected, timeout=0.5)
            if item is None:
                if self.capture_stop_event.is_set():
                    # Capture has stopped. Wait for in-flight hashes up to
                    # capture_count, then exit. Hash pool is ~8 ms / capture
                    # so a 10-second deadline covers any reasonable tail.
                    if next_expected >= self.capture_count:
                        break
                    if drain_deadline is None:
                        drain_deadline = time.monotonic() + 10.0
                    if time.monotonic() > drain_deadline:
                        print(f"[chain] WARN drain deadline: "
                              f"next_expected={next_expected} "
                              f"capture_count={self.capture_count}; "
                              f"exiting with missing tail captures",
                              flush=True)
                        break
                continue
            step_t_start = time.monotonic_ns()
            (bayer_hash, frame_id, ts_ns, capture_wall_ns,
             hash_latency_ms) = item
            t = next_expected
            meta_bytes = struct.pack(
                META_STRUCT,
                t, int(ts_ns), int(capture_wall_ns),
                exposure_us_int, fourcc_bytes,
            )
            S_next = compute_chain_row(S_t, bayer_hash, meta_bytes)
            self.last_S_next = S_next

            xof_r, xof_g, xof_b = compute_xof_seeds(S_next)

            # Tile generation.
            t_gen0 = time.monotonic_ns()
            tile = self.gen_tile_fn(xof_r, xof_g, xof_b)
            cuda_ms = (time.monotonic_ns() - t_gen0) / 1e6

            tile_bytes = bytes(np.ascontiguousarray(tile).tobytes())
            tile_pixel_sha = blake3(tile_bytes).hexdigest()

            # Present — non-blocking from chain thread's POV (GTK idle_add).
            tile_queued_wall_ns = self._queue_tile(tile)

            # Append chain_log row (no inline fsync; durability handles it).
            emission_png_rel = ""
            try:
                emission_png_path = self.emissions_dir / f"tile_{t:06d}.png"
                tile_copy = np.ascontiguousarray(tile).copy()
                self.png_pool.submit(
                    _save_tile_png_safe, tile_copy, emission_png_path, t
                )
                emission_png_rel = f"derived/Emissions/{emission_png_path.name}"
            except Exception as e:
                print(f"[chain] WARN tile PNG submit at t={t}: {e!r}",
                      flush=True)

            row = {
                "t": t,
                "S_t_hex": S_t.hex(),
                "bayer_blake3_hex": bayer_hash.hex(),
                "capture_frame_id": frame_id,
                "aravis_device_timestamp_ns": ts_ns,
                "capture_wall_ns": capture_wall_ns,
                "xof_seed_R_hex": xof_r.hex(),
                "xof_seed_G_hex": xof_g.hex(),
                "xof_seed_B_hex": xof_b.hex(),
                "tile_pixel_sha_hex": tile_pixel_sha,
                "tile_queued_wall_ns": tile_queued_wall_ns,
                "meta_hex": meta_bytes.hex(),
                "emission_png_path": emission_png_rel,
            }
            with open(self.chain_log_path, "a", newline="") as cf:
                cw = csv.DictWriter(cf, fieldnames=self._chain_log_fields)
                cw.writerow(row)
                cf.flush()
                evt = threading.Event()
                self.durability_thread.submit(cf.fileno(), evt, dir_sync=False)
                # Wait for durability BEFORE advancing the chain; spec invariant.
                evt.wait(timeout=5.0)

            # Publisher gets the new chain state.
            if self.interior_publisher is not None:
                self.interior_publisher.update_chain_state(t + 1, S_next.hex())

            step_ms = (time.monotonic_ns() - step_t_start) / 1e6
            self.chain_step_ms.append(step_ms)
            self.cuda_gen_ms.append(cuda_ms)
            self.hash_latency_ms.append(hash_latency_ms)

            S_t = S_next
            next_expected += 1

        self.N_chain = next_expected
        print(f"[{self._iso_now()}] chain: completed {self.N_chain} rows",
              flush=True)

    # ------------------------------------------------------------------
    # Blocking-mode chain loop (v8.2)
    # ------------------------------------------------------------------

    def _drain_aravis_queue(self):
        """Pop everything currently queued on the Aravis stream and push
        the buffers back. Used in blocking mode to discard captures
        taken while the DMD was transitioning to the new emission."""
        while True:
            buf = self._stream.timeout_pop_buffer(1_000)  # 1 ms
            if buf is None:
                return
            self._stream.push_buffer(buf)

    def _drain_and_receive(self, timeout_us=2_000_000):
        """Drain stale queued buffers, then block for the next fresh
        one. Returns (raw_bytes, frame_id, ts_ns, wall_ns) or None on
        timeout / bad status. Retained for calibrate/other callers; the
        blocking chain loop uses the timestamp-based accept path below."""
        self._drain_aravis_queue()
        buf = self._stream.timeout_pop_buffer(timeout_us)
        if buf is None:
            return None
        wall = time.monotonic_ns()
        if buf.get_status() != Aravis.BufferStatus.SUCCESS:
            self._stream.push_buffer(buf)
            return None
        data = bytes(buf.get_data())
        frame_id = buf.get_frame_id()
        ts_ns = buf.get_timestamp()
        self._stream.push_buffer(buf)
        return data, frame_id, ts_ns, wall

    # ------------------------------------------------------------------
    # Timestamp-based accept path (blocking mode, v8.2.1)
    # ------------------------------------------------------------------

    def _pop_one_buffer(self, timeout_us):
        """Pop a single successful buffer. Returns (buf, capture_wall_ns)
        or None on timeout. A buffer with non-SUCCESS status is pushed
        back and None is returned. The caller owns the returned buf and
        must push_buffer it back."""
        buf = self._stream.timeout_pop_buffer(timeout_us)
        if buf is None:
            return None
        wall = time.monotonic_ns()
        if buf.get_status() != Aravis.BufferStatus.SUCCESS:
            self._stream.push_buffer(buf)
            return None
        return buf, wall

    def _calibrate_device_clock_offset(self):
        """Measure the device-to-host clock offset used to map a frame's
        camera timestamp (buf.get_timestamp, arbitrary epoch) into the
        host monotonic_ns domain.

        Strategy: discard one warm-up frame, then on the next frame record
        (device_ts_ns, capture_wall_ns). The trigger fired roughly
        READOUT_NS before the buffer was delivered to userland, so:

            host_ns_at_trigger ≈ capture_wall_ns - READOUT_NS
            OFFSET = host_ns_at_trigger - device_ts_ns
                   = capture_wall_ns - device_ts_ns - READOUT_NS

        Main-loop use: trigger_host_ns = device_ts_ns + OFFSET, which is
        algebraically identical to `capture_wall_ns - READOUT_NS`, but the
        device-timestamp form carries the hardware's sub-millisecond jitter
        instead of the kernel/USB delivery jitter of the host timestamp.

        If the computed offset lands outside a sane range (±1 hour), the
        device clock is considered untrusted and the host fallback
        (`capture_wall_ns - READOUT_NS`) is used instead."""
        # Drain any pre-existing buffers (e.g. from pre-roll BLACK) so the
        # calibration frame is a fresh arrival. Without this, capture_wall_ns
        # on the popped buffer reflects user-land pop time of a stale frame
        # whose delivery lagged its true exposure by however long it sat
        # queued, biasing offset_ns by that backlog age.
        self._drain_aravis_queue()
        res = self._pop_one_buffer(timeout_us=2_000_000)
        if res is None:
            raise RuntimeError(
                "blocking clock-offset calibration: warm-up frame timed out"
            )
        warm_buf, _ = res
        self._stream.push_buffer(warm_buf)

        res = self._pop_one_buffer(timeout_us=2_000_000)
        if res is None:
            raise RuntimeError(
                "blocking clock-offset calibration: calibration frame timed out"
            )
        calib_buf, capture_wall_ns = res
        device_ts_ns = calib_buf.get_timestamp()
        self._stream.push_buffer(calib_buf)

        offset_ns = capture_wall_ns - device_ts_ns - READOUT_NS
        # The camera's internal clock epoch is arbitrary (typically a
        # power-on uptime counter), so the offset magnitude scales with
        # camera uptime. Reject only wildly nonsensical values — anything
        # inside ±30 days covers plausible uptimes and catches the cases
        # we actually want to guard against (uninitialised 0, overflow,
        # driver bug returning garbage).
        thirty_days_ns = 30 * 24 * 3600 * 1_000_000_000
        if device_ts_ns > 0 and abs(offset_ns) < thirty_days_ns:
            self._device_ts_offset_ns = int(offset_ns)
            self._device_ts_offset_valid = True
            print(f"[{self._iso_now()}] blocking: device clock offset "
                  f"calibrated  offset={offset_ns} ns  "
                  f"(device_ts={device_ts_ns}, capture_wall={capture_wall_ns}, "
                  f"readout={READOUT_NS})", flush=True)
        else:
            self._device_ts_offset_ns = 0
            self._device_ts_offset_valid = False
            print(f"[{self._iso_now()}] blocking: device-ts offset "
                  f"{offset_ns} ns outside sane range; falling back to "
                  f"host-clock path (capture_wall_ns - READOUT_NS)",
                  flush=True)

    def _trigger_host_ns(self, device_ts_ns, capture_wall_ns):
        """Return the host-clock estimate of when this frame's exposure
        began. Device-timestamp path if calibration succeeded; host
        fallback otherwise. Both paths are conservative lower bounds on
        the true trigger time (we'd rather reject a marginal frame than
        accept one whose exposure started before the DMD settled)."""
        if self._device_ts_offset_valid:
            return device_ts_ns + self._device_ts_offset_ns
        return capture_wall_ns - READOUT_NS

    def _accept_frame_after(self, threshold_ns, timeout_us=2_000_000):
        """Pop buffers until one whose trigger_host_ns >= threshold_ns is
        found. Stale buffers (triggered before the DMD finished settling
        on the previous emission) are pushed back to the pool immediately.

        Returns (raw_bytes, frame_id, device_ts_ns, capture_wall_ns).
        Raises RuntimeError on timeout (camera has stopped delivering).
        Each iteration of the inner pop loop respects timeout_us; outer
        rejections do not accumulate delay because rejected buffers are
        returned to the pool for the driver to refill.

        Hard invariant (v8.2.2, 2026-04-24): capture_wall_ns is the
        user-land pop time of the buffer and cannot predate the photons
        hitting the sensor. So capture_wall_ns < threshold_ns means the
        buffer was physically acquired before the pipeline-wait window
        elapsed — reject regardless of what _trigger_host_ns reports,
        because the device-clock offset can be poisoned by backlog."""
        while True:
            if self.stop_event.is_set():
                raise RuntimeError("blocking: stop_event set while waiting "
                                   "for capture")
            res = self._pop_one_buffer(timeout_us=timeout_us)
            if res is None:
                raise RuntimeError(
                    f"blocking: no capture delivered within "
                    f"{timeout_us} us while waiting for trigger_host_ns >= "
                    f"{threshold_ns}"
                )
            buf, capture_wall_ns = res
            device_ts_ns = buf.get_timestamp()
            # Hard physical invariant: pop time must itself be after the
            # threshold. Cheap, correct, independent of clock-offset math.
            if capture_wall_ns < threshold_ns:
                self._stream.push_buffer(buf)
                continue
            trigger_host_ns = self._trigger_host_ns(device_ts_ns, capture_wall_ns)
            if trigger_host_ns >= threshold_ns:
                raw_bytes = bytes(buf.get_data())
                frame_id = buf.get_frame_id()
                self._stream.push_buffer(buf)
                return raw_bytes, frame_id, device_ts_ns, capture_wall_ns
            # Frame triggered before the settle threshold. Return to pool
            # and try again — the driver will refill it with a newer one.
            self._stream.push_buffer(buf)

    def _chain_loop_blocking(self):
        """Single-thread blocking chain loop for --mode blocking. All
        fsyncs on the critical path. Chain advances exactly once per
        consumed capture, and every consumed capture is taken after the
        DMD has been holding the previous row's emission for at least
        pipeline_wait_ms."""
        try:
            self._run_chain_blocking()
        except Exception as e:
            self.chain_error = repr(e)
            print(f"[chain] EXCEPTION: {e}", flush=True)
            import traceback
            traceback.print_exc()
        finally:
            self.stop_event.set()
            # Stop acquisition synchronously — blocking mode owns the
            # Aravis stream on this thread.
            try:
                self._camera.stop_acquisition()
            except Exception:
                pass
            self.camera_acquisition_end_wall_ns = time.monotonic_ns()
            # v9: stop the drand background poller.
            try:
                if getattr(self, "drand_client", None) is not None:
                    self.drand_client.stop()
            except Exception as e:
                print(f"[{self._iso_now()}] WARN: drand_client.stop() "
                      f"failed: {e!r}", flush=True)
            # End-of-session black-out.
            try:
                black = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
                self._queue_tile(black)
                t_black = time.monotonic()
                while time.monotonic() - t_black < END_BLACK_MS / 1000.0:
                    while GLib.MainContext.default().iteration(may_block=False):
                        pass
                    time.sleep(0.005)
                print(f"[{self._iso_now()}] projector: black-out emitted "
                      f"({END_BLACK_MS} ms GLib spin)", flush=True)
            except Exception as e:
                print(f"[{self._iso_now()}] WARN: end black-out failed: "
                      f"{e!r}", flush=True)
            GLib.idle_add(self.quit)

    def _run_chain_blocking(self):
        """v9 B++ blocking-mode chain loop.

        Invariant: row t in chain_log.csv commits the emission that was
        LIVE on the DMD during cap_t (= E_t = gen(xof(S_t))), not the
        emission queued after cap_t for cap_{t+1}. All emission-adjacent
        chain_log fields (emission_live_pixel_blake3_hex,
        emission_live_queued_wall_ns, emission_png_path) refer to E_t.
        The on-disk PNG `derived/Emissions/tile_{t:06d}.png` contains
        E_t's pixel bytes.

        Structure: carry current_emission state (pixel buffer + hash +
        queued_wall_ns + relative png path) across iterations. Inside
        iter t: accept cap_t, hash bayer + write raw, advance chain,
        generate + queue + hash E_{t+1}, save tile_{t:06d}.png (= E_t)
        during the DMD transition, write row t from CURRENT state, then
        rotate CURRENT ← NEXT.

        No dangling PNGs: iter N-1 saves tile_{N-1:06d}.png as its
        last act; E_N is generated and queued but never saved, because
        no row was ever written to reference it.

        Bootstrap failure policy: tile_000000.png = E_0 MUST save
        successfully before the main loop starts. If the save raises,
        abort the whole session — row 0 would reference a missing PNG,
        which is a correctness violation that cannot be logged-past.
        """
        print(f"[{self._iso_now()}] chain: BLOCKING mode starting "
              f"(pipeline_wait_ms={self.pipeline_wait_ms}, "
              f"readout_ns={READOUT_NS})", flush=True)
        if self.pipeline_wait_ms <= 0:
            raise RuntimeError("blocking mode: pipeline_wait_ms not set")
        pipeline_wait_ns = int(self.pipeline_wait_ms * 1_000_000)
        fourcc_bytes = FOURCC_MAP[self.format_str].encode("ascii")
        exposure_us_int = int(round(self.exposure_us))

        # Pre-roll black so the DMD starts from a known state.
        black = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
        self._queue_tile(black)
        time.sleep(PREROLL_BLACK_MS / 1000.0)

        # Bootstrap (v9 B++): derive E_0 = gen(xof(S_0)), queue it to the
        # DMD, hash its pixel bytes, save tile_000000.png. Row 0 (written
        # inside iter 0 below) will commit these as its live-emission
        # fields.
        S_t = self.S0
        xof_r, xof_g, xof_b = compute_xof_seeds(S_t)
        current_emission = self.gen_tile_fn(xof_r, xof_g, xof_b)
        current_emission_bytes = bytes(
            np.ascontiguousarray(current_emission).tobytes())
        current_emission_hash = blake3(current_emission_bytes).hexdigest()
        current_emission_queued_wall_ns = self._queue_tile(current_emission)
        current_emission_png_rel = "derived/Emissions/tile_000000.png"

        # Save tile_000000.png synchronously BEFORE the chain loop starts.
        # v9 B++ correctness requires row 0's emission_png_path to exist
        # on disk. If the save fails we cannot write a valid row 0, so
        # abort the session here with a clear exception — the chain loop
        # wrapper (_chain_loop_blocking) catches and records it.
        try:
            png_0 = self.emissions_dir / "tile_000000.png"
            Image.fromarray(
                np.ascontiguousarray(current_emission), mode="RGB"
            ).save(png_0, compress_level=1)
        except Exception as e:
            raise RuntimeError(
                f"bootstrap tile_000000.png save failed: {e!r}; "
                f"refusing to start chain — row 0 would reference a "
                f"missing PNG and violate v9 B++ live-attribution"
            ) from e

        # Establish the device-to-host clock offset. Consumes two frames
        # (one warm-up, one calibration); their timing is not chained.
        # Kept here so the bootstrap DMD transition happens in parallel
        # with the calibration reads rather than a blind sleep.
        self._calibrate_device_clock_offset()

        # Drain any pre-roll / bootstrap-era buffers that survived
        # calibration. At long exposures (96 ms) and 2 s pre-roll, up to
        # ~20 buffers can queue up; calibration consumes 2; the rest must
        # be dropped so iter 0's _accept_frame_after does not pop a stale
        # pre-roll-era frame whose device_ts maps (via the now-correct
        # offset) into the pre-bootstrap past.
        self._drain_aravis_queue()

        t = 0
        session_start_wall = time.monotonic_ns()
        duration_ns = int(self.duration * 1e9)
        prev_queue_draw_wall_ns = current_emission_queued_wall_ns

        while not self.stop_event.is_set():
            if time.monotonic_ns() - session_start_wall >= duration_ns:
                break
            step_t_start = time.monotonic_ns()

            # Accept cap_t: the first buffered frame whose trigger
            # (exposure start, in host clock) landed at or after the
            # previous queue_draw + pipeline_wait_ms threshold. Photons
            # integrated during this exposure are under current_emission
            # (= E_t), which was queued at current_emission_queued_wall_ns.
            threshold_ns = prev_queue_draw_wall_ns + pipeline_wait_ns
            raw_bytes, frame_id, ts_ns, capture_wall_ns = (
                self._accept_frame_after(threshold_ns, timeout_us=2_000_000)
            )

            # Hash synchronously on this thread.
            hash_t0 = time.monotonic_ns()
            bayer_hash = blake3(raw_bytes).digest()
            hash_ms = (time.monotonic_ns() - hash_t0) / 1e6

            # Raw file write + fsync + dirsync.
            raw_final = self.recordings_dir / f"frame_{t:06d}.raw"
            raw_tmp = self.recordings_dir / f".frame_{t:06d}.raw.tmp"
            with open(raw_tmp, "wb") as rf:
                rf.write(raw_bytes)
                rf.flush()
                os.fsync(rf.fileno())
            os.replace(raw_tmp, raw_final)
            try:
                dfd = os.open(self.recordings_dir, os.O_RDONLY)
                try:
                    os.fsync(dfd)
                finally:
                    os.close(dfd)
            except OSError:
                pass

            # v9 corroborating-evidence: drand is soft. get_current()
            # returns (round_number, value, staleness_ms). round_number=0
            # and value=zeros are valid sentinels when drand is
            # unavailable; staleness_ms=-1 signals "never fetched."
            drand_round_number, drand_round_value, drand_staleness_ms = (
                self.drand_client.get_current(require_fresh=False))

            # Chain transition.
            meta_bytes = struct.pack(
                META_STRUCT, t, int(ts_ns), int(capture_wall_ns),
                exposure_us_int, fourcc_bytes,
            )
            S_next = compute_chain_row(
                S_t, bayer_hash, meta_bytes,
                drand_round_number, drand_round_value,
            )
            self.last_S_next = S_next

            # Generate NEXT emission E_{t+1} from S_next; this will be
            # queued below and consumed during cap_{t+1}. Row t+1 (future
            # iter) will commit its hash/wall_ns/png as that iter's live-
            # emission fields.
            next_xof_r, next_xof_g, next_xof_b = compute_xof_seeds(S_next)
            gen_t0 = time.monotonic_ns()
            next_emission = self.gen_tile_fn(
                next_xof_r, next_xof_g, next_xof_b)
            cuda_ms = (time.monotonic_ns() - gen_t0) / 1e6
            next_emission_bytes = bytes(
                np.ascontiguousarray(next_emission).tobytes())
            next_emission_hash = blake3(next_emission_bytes).hexdigest()

            # Queue E_{t+1} to the DMD. _queue_tile blocks on GTK idle-add
            # and returns the wall_ns at which queue_draw fired; this
            # becomes the reference for iter t+1's pipeline_wait threshold
            # AND row t+1's emission_live_queued_wall_ns.
            next_queued_wall_ns = self._queue_tile(next_emission)
            prev_queue_draw_wall_ns = next_queued_wall_ns

            # Save tile_{t:06d}.png = current_emission (E_t) here — i.e.
            # we save the emission this iter's row references, NOT the
            # one we just queued. This placement gives the DMD transition
            # of E_{t+1} (physical few-ms settle) overlapping with the
            # PNG encode of E_t, same latency as the original loop. It
            # also eliminates the dangling-PNG trap: an N-row session
            # produces exactly tile_0 … tile_{N-1}.png (tile_0 from
            # bootstrap, tile_{t>0} from inside iter t); E_N is generated
            # and queued but never saved, because no row was ever written
            # to reference it.
            #
            # Bootstrap already saved tile_000000.png before the loop, so
            # iter 0 skips its own save to avoid a redundant rewrite of
            # identical bytes. For t >= 1 the save is best-effort; a
            # transient PNG failure warns and continues rather than
            # aborting the whole session. (Bootstrap's save is strict
            # because row 0's correctness anchor depends on it.)
            if t >= 1:
                try:
                    png_here = self.emissions_dir / f"tile_{t:06d}.png"
                    Image.fromarray(
                        np.ascontiguousarray(current_emission), mode="RGB"
                    ).save(png_here, compress_level=1)
                except Exception as e:
                    print(f"[chain] WARN tile_{t:06d}.png save failed: "
                          f"{e!r}", flush=True)

            # Append chain row t. v9 B++ live attribution: emission-
            # adjacent fields refer to current_emission (= E_t, the
            # emission LIVE on the DMD during cap_t), not E_{t+1}.
            chain_row = {
                "t": t,
                "S_t_hex": S_t.hex(),
                "bayer_blake3_hex": bayer_hash.hex(),
                "capture_frame_id": frame_id,
                "aravis_device_timestamp_ns": ts_ns,
                "capture_wall_ns": capture_wall_ns,
                "emission_live_pixel_blake3_hex": current_emission_hash,
                "emission_live_queued_wall_ns":
                    current_emission_queued_wall_ns,
                "meta_hex": meta_bytes.hex(),
                "emission_png_path": current_emission_png_rel,
                # v9 drand columns. Same round may appear across multiple
                # rows when blocking-mode step-time < drand period of 3s.
                # Under corroborating-evidence mode, round=0 + value=zeros
                # + staleness=-1 means "drand was unavailable at this
                # row." staleness_ms grows without bound if drand stays
                # unreachable mid-session.
                "drand_round_number": drand_round_number,
                "drand_round_value_hex": drand_round_value.hex(),
                "drand_staleness_ms": drand_staleness_ms,
            }
            with open(self.chain_log_path, "a", newline="") as cf:
                cw = csv.DictWriter(cf, fieldnames=self._chain_log_fields)
                cw.writerow(chain_row)
                cf.flush()
                os.fsync(cf.fileno())

            # Append capture row, fsync. In blocking mode every capture
            # is consumed immediately into the chain, so we can write
            # the reconciled columns directly.
            capture_row = {
                "capture_idx": t,
                "capture_frame_id": frame_id,
                "aravis_device_timestamp_ns": ts_ns,
                "capture_wall_ns": capture_wall_ns,
                "bayer_blake3_hex": bayer_hash.hex(),
                "raw_path": f"Recordings/{raw_final.name}",
                "consumed_into_chain": "true",
                "consumed_as_t": t,
            }
            with open(self.capture_log_path, "a", newline="") as cf:
                cw = csv.DictWriter(cf, fieldnames=self._capture_log_fields)
                cw.writerow(capture_row)
                cf.flush()
                os.fsync(cf.fileno())

            # Publisher gets the new chain state.
            if self.interior_publisher is not None:
                self.interior_publisher.update_chain_state(
                    t + 1, S_next.hex())

            step_ms = (time.monotonic_ns() - step_t_start) / 1e6
            self.chain_step_ms.append(step_ms)
            self.cuda_gen_ms.append(cuda_ms)
            self.hash_latency_ms.append(hash_ms)

            # No explicit pipeline-wait sleep here (v8.2.1): the next
            # iteration's _accept_frame_after enforces the pipeline_wait
            # discipline by rejecting frames whose trigger arrived before
            # prev_queue_draw_wall_ns + pipeline_wait_ms.

            # Rotate CURRENT ← NEXT. After this, iter t+1 sees the
            # just-queued E_{t+1} as its current emission. The png path
            # is a pure function of t (f"tile_{t:06d}.png") so we don't
            # need to carry it across iterations.
            S_t = S_next
            current_emission = next_emission
            current_emission_bytes = next_emission_bytes
            current_emission_hash = next_emission_hash
            current_emission_queued_wall_ns = next_queued_wall_ns
            t += 1
            current_emission_png_rel = (
                f"derived/Emissions/tile_{t:06d}.png")

        self.N_chain = t
        self.N_captures = t  # every consumed capture is also a chain row
        print(f"[{self._iso_now()}] chain: blocking completed {t} rows",
              flush=True)



# ----------------------------------------------------------------------------
# Entry-point shim.
#
# CLI parsing + main() moved to tb_main.py as part of the 2026-04-21 academic
# refactor. The import is deferred to inside the `__main__` guard to avoid a
# circular import at module-load time (tb_main.py imports TBLoopV8 from here).
# Backward compat: `python3 protocol/tb_loop.py ...` still works.
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    from tb_main import main
    main()

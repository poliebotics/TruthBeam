"""Unit tests for v9 B++ blocking-mode writer invariants.

Exercises `TBLoopV8._run_chain_blocking` against synthetic inputs, verifying
the live-attribution invariants of the v9 writer (see METHODS.md and
session_schema.py:PROTOCOL_VERSION):

    I1  row 0's S_t_hex == manifest.S_0_hex
    I2  for t≥1: row t's S_t_hex == advance(S_{t-1}, bayer_hash_{t-1}, meta_{t-1})
    I3  bayer_blake3_hex == blake3(frame_{t:06d}.raw bytes)
    I4  emission_live_pixel_blake3_hex[t] == blake3(gen(xof(S_t)))
    I5  tile_{t:06d}.png decoded bytes hash == emission_live_pixel_blake3_hex[t]
    I6  capture_wall_ns[t] >= emission_live_queued_wall_ns[t] + pipeline_wait_ms
        (NOT tested here — governed by `_accept_frame_after`, stubbed out)
    I7  emission_live_queued_wall_ns[t] == successor queue time of t-1
    I8  N PNG files on disk for N chain rows (no dangling tile_N.png)
    I9  bootstrap: E_0 queued before first accept; tile_000000.png exists

Approach: construct a `FakeLoop` stand-in with just the attributes and method
stubs `_run_chain_blocking` reads, and invoke `TBLoopV8._run_chain_blocking`
as an unbound function with `fake_loop` as self. This avoids booting GTK /
Aravis / the real camera, but still exercises every line of the production
chain-loop body verbatim.
"""
from __future__ import annotations

import csv
import os
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from blake3 import blake3
from PIL import Image

HERE = Path(__file__).resolve().parent
PROTOCOL_DIR = HERE.parent / "protocol"
sys.path.insert(0, str(PROTOCOL_DIR))

import tb_loop                                           # noqa: E402
from chain import META_STRUCT                            # noqa: E402
from session_schema import (                             # noqa: E402
    compute_chain_row, compute_xof_seeds,
)
from session_logs import CHAIN_LOG_FIELDS                # noqa: E402
from tile_params import TILE_W, TILE_H                   # noqa: E402


# -- Helpers ----------------------------------------------------------------


class FakeDrandClient:
    """Deterministic stand-in for DrandClient. Returns a monotonic
    pseudo-round with a pseudo-random value derived from (round, session).
    No HTTP, no BLS. Just enough for unit tests to exercise the v9
    chain-advance formula's drand fold-in."""

    def __init__(self, session_seed: bytes = b"test-drand-session",
                 start_round: int = 10_000_000):
        self._session_seed = session_seed
        self._round = int(start_round)
        self.chain_hash = "ab" * 32
        self.pubkey_hex = "cd" * 96

    @property
    def initial_round_number(self) -> int:
        return self._round

    def get_current(self, require_fresh: bool = False) -> tuple[int, bytes, int]:
        # Value = blake3(session_seed || round_be8), deterministic.
        # Returns (round, value, staleness_ms). v9 corroborating-evidence
        # signature: stand-in reports a fixed 42 ms staleness so tests
        # can distinguish "drand was available" from the zero-sentinel
        # path.
        val = blake3(self._session_seed +
                     self._round.to_bytes(8, "big")).digest()
        return self._round, val, 42

    def stop(self):
        pass


class FakeLoop:
    """Minimal stand-in with the attributes `_run_chain_blocking` reads.

    Not a subclass of TBLoopV8 — we invoke the unbound method
    `TBLoopV8._run_chain_blocking(fake_loop)` and Python's duck typing does
    the rest.
    """

    def __init__(self, tmp_dir: Path, n_synthetic_captures: int,
                 S0_seed: bytes = b"test-S0"):
        self.pipeline_wait_ms = 340
        self.format_str = "BayerRG8"
        self.exposure_us = 96000.0
        self.S0 = blake3(S0_seed).digest()
        self.duration = 3600.0  # capped by stop_event instead
        self.last_S_next = None
        self.interior_publisher = None
        self.chain_step_ms = []
        self.cuda_gen_ms = []
        self.hash_latency_ms = []
        # v9: drand client stand-in. Real DrandClient makes HTTPS calls
        # to drand.sh and BLS-verifies; the fake is deterministic.
        self.drand_client = FakeDrandClient(
            session_seed=b"fake-drand-" + S0_seed)

        self._iso_fixed = "2026-04-24T00:00:00+00:00"
        self._capture_counter = 0
        self._n_synthetic = n_synthetic_captures
        self._queue_counter = 0
        self.queued_wall_ns_log = []
        self.stop_event = threading.Event()

        self.recordings_dir = tmp_dir / "Recordings"
        self.emissions_dir = tmp_dir / "derived" / "Emissions"
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.emissions_dir.mkdir(parents=True, exist_ok=True)
        self.chain_log_path = tmp_dir / "chain_log.csv"
        self.capture_log_path = tmp_dir / "capture_log.csv"

        # v9 B++ field list (must match session_logs.CHAIN_LOG_FIELDS).
        self._chain_log_fields = list(CHAIN_LOG_FIELDS)
        self._capture_log_fields = [
            "capture_idx", "capture_frame_id",
            "aravis_device_timestamp_ns", "capture_wall_ns",
            "bayer_blake3_hex", "raw_path",
            "consumed_into_chain", "consumed_as_t",
        ]
        # Headers (simulate what _open_logs would do).
        with open(self.chain_log_path, "w", newline="") as f:
            f.write(f"# session_iso_utc={self._iso_fixed}\n")
            csv.DictWriter(f, fieldnames=self._chain_log_fields).writeheader()
        with open(self.capture_log_path, "w", newline="") as f:
            f.write(f"# session_iso_utc={self._iso_fixed}\n")
            csv.DictWriter(f, fieldnames=self._capture_log_fields).writeheader()

    def _iso_now(self) -> str:
        return self._iso_fixed

    def gen_tile_fn(self, xof_r: bytes, xof_g: bytes, xof_b: bytes) -> np.ndarray:
        """Deterministic test stand-in for the real fBm generator.

        Real production path is tile_gpu.gen_fbm_tile; that implementation is
        a 3-channel integer-arithmetic multi-octave fBm. For these invariant
        tests we only need the output to be a deterministic function of the
        XOF seeds; the byte pattern does not need to match the production
        algorithm.
        """
        seed = blake3(xof_r + xof_g + xof_b).digest()
        nbytes = TILE_H * TILE_W * 3
        # Tile per-byte pattern via blake3 XOF — deterministic, well-distributed.
        pattern = blake3(seed).digest(length=nbytes)
        return np.frombuffer(pattern, dtype=np.uint8).reshape(TILE_H, TILE_W, 3).copy()

    def _queue_tile(self, tile: np.ndarray) -> int:
        """Simulate DMD queue. Returns a monotonic wall_ns and records it."""
        self._queue_counter += 1
        wall_ns = time.monotonic_ns()
        self.queued_wall_ns_log.append(wall_ns)
        return wall_ns

    def _calibrate_device_clock_offset(self):
        pass  # no-op in test

    def _drain_aravis_queue(self):
        pass  # no-op in test

    def _accept_frame_after(self, threshold_ns, timeout_us=None):
        """Return synthetic capture. Set stop_event when exhausted so the
        NEXT while-check exits the loop cleanly."""
        assert self._capture_counter < self._n_synthetic, (
            "test harness exhausted; stop_event should have exited the loop")
        # Tiny synthetic raw (bayer bytes) — size doesn't matter for the
        # writer's logic, only blake3(raw) does.
        raw_bytes = blake3(
            f"synth-cap-{self._capture_counter}".encode()
        ).digest(length=256)
        frame_id = 1000 + self._capture_counter
        capture_wall_ns = threshold_ns + 500_000_000  # 500 ms past threshold
        ts_ns = capture_wall_ns - 70_000_000  # ROI readout lag
        self._capture_counter += 1
        if self._capture_counter >= self._n_synthetic:
            # Set stop_event AFTER returning this frame. The loop processes
            # this row to completion then exits on next while-check.
            self.stop_event.set()
        return raw_bytes, frame_id, ts_ns, capture_wall_ns


def _read_chain_rows(path: Path) -> list:
    with open(path) as f:
        _hdr = f.readline()  # skip comment
        return list(csv.DictReader(f))


def _read_capture_rows(path: Path) -> list:
    with open(path) as f:
        _hdr = f.readline()
        return list(csv.DictReader(f))


def _run(fake_loop: FakeLoop):
    """Invoke the unbound method on the fake."""
    tb_loop.TBLoopV8._run_chain_blocking(fake_loop)


# -- Schema alignment -------------------------------------------------------


def test_open_logs_chain_fields_match_session_logs(tmp_path):
    """FakeLoop uses session_logs.CHAIN_LOG_FIELDS directly; this test
    sanity-checks that tb_loop.TBLoopV8._open_logs' inline list is also
    in sync. Parses the source text (no instantiation needed)."""
    src = (PROTOCOL_DIR / "tb_loop.py").read_text()
    # Find the _open_logs inline field list. Brittle but sufficient.
    import re
    m = re.search(
        r"def _open_logs\(self\):.*?self\._chain_log_fields\s*=\s*\[(.*?)\]",
        src, re.DOTALL)
    assert m is not None, "could not find _open_logs chain fields"
    inline = tuple(
        s.strip().strip('"').strip("'")
        for s in m.group(1).split(",")
        if s.strip() and not s.strip().startswith("#")
    )
    # Remove any empty entries from trailing commas
    inline = tuple(x for x in inline if x)
    assert inline == tuple(CHAIN_LOG_FIELDS), (
        f"tb_loop._open_logs inline list diverged from "
        f"session_logs.CHAIN_LOG_FIELDS:\n"
        f"  inline = {inline}\n"
        f"  canon  = {CHAIN_LOG_FIELDS}"
    )


# -- Bootstrap / I9 ---------------------------------------------------------


def test_bootstrap_saves_tile_000000(tmp_path):
    fl = FakeLoop(tmp_path, n_synthetic_captures=3)
    _run(fl)
    assert (tmp_path / "derived" / "Emissions" / "tile_000000.png").exists()


def test_bootstrap_png_failure_aborts(tmp_path):
    """I9: if tile_000000.png save fails, _run_chain_blocking raises."""
    fl = FakeLoop(tmp_path, n_synthetic_captures=3)
    # Point emissions_dir at a file (not a dir) so PIL.save raises.
    sabotage = tmp_path / "not_a_dir"
    sabotage.write_text("I am a file, not a dir")
    fl.emissions_dir = sabotage
    with pytest.raises(RuntimeError, match="bootstrap tile_000000.png"):
        _run(fl)


def test_bootstrap_queues_e0_before_first_accept(tmp_path):
    """I9: E_0 is queued before _accept_frame_after is ever called."""
    fl = FakeLoop(tmp_path, n_synthetic_captures=2)

    # Instrument: record the capture counter at the moment each _queue_tile
    # fires. Bootstrap's E_0 queue must happen while counter is still 0.
    original_queue = fl._queue_tile
    queue_call_capture_counters = []

    def tracking_queue(tile):
        queue_call_capture_counters.append(fl._capture_counter)
        return original_queue(tile)

    fl._queue_tile = tracking_queue
    _run(fl)

    # Sequence: _queue_tile(black) (pre-roll), _queue_tile(E_0) (bootstrap)
    # → then _accept_frame_after is called (capture_counter increments to 1)
    # → _queue_tile(E_1) inside iter 0 (counter is now 1), etc.
    assert len(queue_call_capture_counters) >= 2
    assert queue_call_capture_counters[0] == 0  # pre-roll black
    assert queue_call_capture_counters[1] == 0  # bootstrap E_0


# -- Row count + no dangler / I8 --------------------------------------------


def test_n_rows_and_no_dangling_png(tmp_path):
    n = 5
    fl = FakeLoop(tmp_path, n_synthetic_captures=n)
    _run(fl)

    chain_rows = _read_chain_rows(tmp_path / "chain_log.csv")
    cap_rows = _read_capture_rows(tmp_path / "capture_log.csv")
    assert len(chain_rows) == n
    assert len(cap_rows) == n

    pngs = sorted((tmp_path / "derived" / "Emissions").glob("tile_*.png"))
    # N rows → exactly N PNGs (tile_0 from bootstrap, tile_1..tile_{N-1}
    # from iters 1..N-1). E_N is generated and queued but never saved.
    assert len(pngs) == n, (
        f"expected {n} PNGs for {n} rows; got {len(pngs)}: "
        f"{[p.name for p in pngs]}"
    )
    assert pngs[0].name == "tile_000000.png"
    assert pngs[-1].name == f"tile_{n-1:06d}.png"


def test_single_row_session(tmp_path):
    """Edge: N=1. Must still produce exactly 1 row, 1 PNG, 1 raw, no dangler."""
    fl = FakeLoop(tmp_path, n_synthetic_captures=1)
    _run(fl)
    chain_rows = _read_chain_rows(tmp_path / "chain_log.csv")
    assert len(chain_rows) == 1
    pngs = sorted((tmp_path / "derived" / "Emissions").glob("tile_*.png"))
    assert [p.name for p in pngs] == ["tile_000000.png"]
    raws = sorted((tmp_path / "Recordings").glob("frame_*.raw"))
    assert [p.name for p in raws] == ["frame_000000.raw"]


# -- Chain transition / I1, I2 ---------------------------------------------


def test_row_zero_commits_to_S0(tmp_path):
    """I1: row 0's S_t_hex == manifest.S_0_hex (here fl.S0)."""
    fl = FakeLoop(tmp_path, n_synthetic_captures=3)
    _run(fl)
    rows = _read_chain_rows(tmp_path / "chain_log.csv")
    assert rows[0]["S_t_hex"] == fl.S0.hex()


def test_chain_advance_rewalkable(tmp_path):
    """I2 (v9): row t's S_t_hex ==
       advance(S_{t-1}, bayer_{t-1}, meta_{t-1},
               drand_round_{t-1}, drand_value_{t-1}) for t≥1.
    v9 folds drand round number + value into each chain step, so the
    verifier recomputation also takes them."""
    n = 6
    fl = FakeLoop(tmp_path, n_synthetic_captures=n)
    _run(fl)
    rows = _read_chain_rows(tmp_path / "chain_log.csv")
    for t in range(1, n):
        prev = rows[t - 1]
        S_next_expected = compute_chain_row(
            bytes.fromhex(prev["S_t_hex"]),
            bytes.fromhex(prev["bayer_blake3_hex"]),
            bytes.fromhex(prev["meta_hex"]),
            int(prev["drand_round_number"]),
            bytes.fromhex(prev["drand_round_value_hex"]),
        )
        assert rows[t]["S_t_hex"] == S_next_expected.hex(), \
            f"chain diverges at row t={t}"


# -- Bayer hash / I3 -------------------------------------------------------


def test_bayer_blake3_matches_raw_file(tmp_path):
    """I3: row t's bayer_blake3_hex == blake3(frame_{t:06d}.raw bytes)."""
    n = 4
    fl = FakeLoop(tmp_path, n_synthetic_captures=n)
    _run(fl)
    rows = _read_chain_rows(tmp_path / "chain_log.csv")
    for t, row in enumerate(rows):
        raw_path = tmp_path / "Recordings" / f"frame_{t:06d}.raw"
        assert raw_path.exists()
        assert row["bayer_blake3_hex"] == blake3(
            raw_path.read_bytes()).hexdigest(), f"bayer hash mismatch at t={t}"


# -- Live-emission hash derivation / I4 -------------------------------------


def test_live_emission_hash_equals_derived_from_S_t(tmp_path):
    """I4: emission_live_pixel_blake3_hex[t] == blake3(gen(xof(S_t))).

    This is the core v9 B++ invariant. Under v8 this would have failed
    because tile_pixel_sha_hex was hash(gen(xof(S_{t+1}))).
    """
    n = 5
    fl = FakeLoop(tmp_path, n_synthetic_captures=n)
    _run(fl)
    rows = _read_chain_rows(tmp_path / "chain_log.csv")
    for t, row in enumerate(rows):
        S_t = bytes.fromhex(row["S_t_hex"])
        xof_r, xof_g, xof_b = compute_xof_seeds(S_t)
        expected = fl.gen_tile_fn(xof_r, xof_g, xof_b)
        expected_hash = blake3(
            bytes(np.ascontiguousarray(expected).tobytes())
        ).hexdigest()
        assert row["emission_live_pixel_blake3_hex"] == expected_hash, \
            f"live hash ≠ derived-from-S_t at row t={t}"


# -- PNG ↔ hash consistency / I5 -------------------------------------------


def test_png_pixel_bytes_match_live_hash(tmp_path):
    """I5: tile_{t:06d}.png's decoded RGB bytes, hashed, == emission_live_pixel_blake3_hex."""
    n = 3
    fl = FakeLoop(tmp_path, n_synthetic_captures=n)
    _run(fl)
    rows = _read_chain_rows(tmp_path / "chain_log.csv")
    for row in rows:
        png_path = tmp_path / row["emission_png_path"]
        assert png_path.exists(), f"missing PNG: {png_path}"
        img = Image.open(png_path).convert("RGB")
        arr = np.asarray(img)
        png_hash = blake3(bytes(np.ascontiguousarray(arr).tobytes())).hexdigest()
        assert row["emission_live_pixel_blake3_hex"] == png_hash, \
            f"PNG↔hash mismatch at t={row['t']}"


# -- Queued wall-ns chaining / I7 -------------------------------------------


def test_emission_live_queued_wall_ns_chains_across_rows(tmp_path):
    """I7: emission_live_queued_wall_ns[t] for t≥1 == the NEXT queue time
    recorded during iter t-1 (i.e. the queue time of E_t itself).

    FakeLoop logs every _queue_tile call's returned wall_ns. Sequence is:
        [pre-roll-black, bootstrap-E_0, iter-0-E_1, iter-1-E_2, ...]
    So `queued_wall_ns_log[1]` is E_0's queue → row 0's
    emission_live_queued_wall_ns. `queued_wall_ns_log[2]` is E_1's queue
    → row 1's field. Etc.
    """
    n = 4
    fl = FakeLoop(tmp_path, n_synthetic_captures=n)
    _run(fl)
    rows = _read_chain_rows(tmp_path / "chain_log.csv")
    # Skip log[0] (pre-roll black); log[1] is E_0 bootstrap queue; log[k+1]
    # is E_k's queue for k=0..n-1 (we queue E_0..E_n, but E_n is queued in
    # iter n-1 as "next" and never consumed).
    for t, row in enumerate(rows):
        expected_wall_ns = fl.queued_wall_ns_log[1 + t]
        assert int(row["emission_live_queued_wall_ns"]) == expected_wall_ns, \
            f"queued_wall_ns chain broken at t={t}"


# -- Capture log reconciliation ---------------------------------------------


def test_capture_log_consistent_with_chain_log(tmp_path):
    """Every chain row t has a capture row with capture_idx=t and
    matching bayer_blake3_hex."""
    n = 4
    fl = FakeLoop(tmp_path, n_synthetic_captures=n)
    _run(fl)
    chain_rows = _read_chain_rows(tmp_path / "chain_log.csv")
    cap_rows = _read_capture_rows(tmp_path / "capture_log.csv")
    assert len(chain_rows) == len(cap_rows)
    for ch, cp in zip(chain_rows, cap_rows):
        assert ch["t"] == cp["capture_idx"] == cp["consumed_as_t"]
        assert ch["bayer_blake3_hex"] == cp["bayer_blake3_hex"]
        assert cp["consumed_into_chain"] == "true"


# -- State non-leakage across two independent invocations ------------------


def test_two_independent_loops_do_not_share_state(tmp_path):
    """Running _run_chain_blocking twice (two separate FakeLoop instances,
    two separate tmp dirs, two separate S_0 seeds) must not produce rows
    that share chain state. Guards against any accidental module-level
    state leaking between sessions."""
    loop_a = FakeLoop(tmp_path / "a", n_synthetic_captures=3,
                      S0_seed=b"session-A")
    loop_b = FakeLoop(tmp_path / "b", n_synthetic_captures=3,
                      S0_seed=b"session-B")
    _run(loop_a)
    _run(loop_b)
    rows_a = _read_chain_rows((tmp_path / "a") / "chain_log.csv")
    rows_b = _read_chain_rows((tmp_path / "b") / "chain_log.csv")
    assert rows_a[0]["S_t_hex"] != rows_b[0]["S_t_hex"]
    for r_a, r_b in zip(rows_a, rows_b):
        assert r_a["emission_live_pixel_blake3_hex"] != \
               r_b["emission_live_pixel_blake3_hex"], \
            "state appears shared across sessions"


# -- Stop-event-before-any-accept edge --------------------------------------


def test_drand_columns_present_and_consistent(tmp_path):
    """v9 drand invariant: every row records drand_round_number +
    drand_round_value_hex, and the value matches what the drand client
    (or stand-in) returned at that row's capture."""
    n = 4
    fl = FakeLoop(tmp_path, n_synthetic_captures=n)
    _run(fl)
    rows = _read_chain_rows(tmp_path / "chain_log.csv")
    for row in rows:
        assert "drand_round_number" in row
        assert "drand_round_value_hex" in row
        # FakeDrandClient returns a fixed round; value derivable from seed.
        rn = int(row["drand_round_number"])
        assert rn == fl.drand_client.initial_round_number
        expected_val = blake3(
            fl.drand_client._session_seed + rn.to_bytes(8, "big")
        ).digest()
        assert row["drand_round_value_hex"] == expected_val.hex()


def test_stop_event_before_iter_0(tmp_path):
    """If stop_event is set before the first accept, no rows should be
    written. tile_000000.png exists from bootstrap; that's acceptable
    (one PNG, zero rows). Verifier sees an empty chain_log."""
    fl = FakeLoop(tmp_path, n_synthetic_captures=3)
    fl.stop_event.set()  # fire pre-loop
    _run(fl)
    rows = _read_chain_rows(tmp_path / "chain_log.csv")
    assert rows == []
    # Bootstrap PNG was still saved (strict correctness).
    assert (tmp_path / "derived" / "Emissions" / "tile_000000.png").exists()

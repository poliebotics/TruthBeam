"""End-to-end writer↔verifier round-trip test.

Builds a synthetic session via the same FakeLoop machinery as
test_v9_writer_invariants, then runs the verifier against it offline
and asserts the report matches expectations.

This is the high-value integration test: if the writer and verifier
ever disagree on what `compute_chain_row` does (say: one bumps the
ROW_DOMAIN_TAG and the other doesn't), this test fails loudly.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import threading
from pathlib import Path

import pytest
from blake3 import blake3

HERE = Path(__file__).resolve().parent
PROTOCOL_DIR = HERE.parent / "protocol"
VERIFY_DIR = HERE.parent / "verify"
sys.path.insert(0, str(PROTOCOL_DIR))
sys.path.insert(0, str(VERIFY_DIR))

import tb_loop                                               # noqa: E402
from session_schema import (                                 # noqa: E402
    derive_s0, UNANCHORED_BLOCK_HASH_HEX, UNANCHORED_BLOCK_HEIGHT,
    SessionManifest, VerificationBundle, PROTOCOL_VERSION,
    canonical_json_bytes,
)
from verify_v9 import verify_session                         # noqa: E402

# Reuse the stand-ins from the writer invariants test
from test_v9_writer_invariants import FakeLoop, FakeDrandClient  # noqa: E402


def _write_minimal_bundle_and_manifest(tmp_path: Path,
                                       fake_loop: FakeLoop) -> None:
    """Write minimum-viable manifest.json + verification_bundle.json so
    the verifier can find them. Bundle values are placeholders; only
    `protocol_version` is semantically checked by the verifier."""
    bundle = {
        "protocol_version": PROTOCOL_VERSION,
        "generator_code_hash": "00" * 32,
        "generator_config": {
            "NUM_OCTAVES": 4, "GRID_H_TABLE": [17, 34, 68, 135],
            "GRID_W_TABLE": [30, 60, 120, 240], "bit_depth": 8,
            "upsample_method": "nearest", "persistence_shift_per_octave": 1,
        },
        "camera_config": {
            "model": "test", "sensor_size": [5320, 4600],
            "pixel_format": "BayerRG8", "roi": [0, 0, 5320, 4600],
            "exposure_us": 96000, "gain_db": 24, "white_balance_off": True,
            "trigger_config": {"source": "Line1", "mode": "On",
                               "selector": "FrameStart",
                               "activation": "RisingEdge"},
            "serial_number": "TEST-SERIAL", "firmware_version": "TEST-FW",
            "aravis_version": "TEST-ARAVIS",
        },
        "projector_config": {
            "model": "test", "connector": "HDMI-1",
            "resolution": [1920, 1080], "refresh_rate_hz": 60,
            "color_space": "RGB",
            "serial_number": None, "edid_fingerprint": None,
        },
        "tile_config": {
            "pixel_format": "RGB888", "bit_depth": 8,
            "dimensions": [1920, 1080],
        },
        "chain_config": {"hash": "blake3", "state_bytes": 32},
        "metadata_schema": {
            "struct_format": ">IQQI4s", "endianness": "big",
            "total_bytes": 28, "fourcc_map": {"BayerRG8": "RG08"},
            "fields": [
                {"name": "t", "offset": 0, "length_bytes": 4, "type": "uint32"},
                {"name": "aravis_device_timestamp_ns", "offset": 4,
                 "length_bytes": 8, "type": "uint64"},
                {"name": "capture_wall_ns", "offset": 12,
                 "length_bytes": 8, "type": "uint64"},
                {"name": "exposure_us", "offset": 20,
                 "length_bytes": 4, "type": "uint32"},
                {"name": "fourcc", "offset": 24,
                 "length_bytes": 4, "type": "ascii4"},
            ],
        },
        "host_config": {
            "hostname": "test-host", "os_release": "test-os",
            "kernel_version": "test-kernel", "python_version": "3.14",
            "cpu_model": "test-cpu",
        },
        "wallet_address": "",
        "chain_id": 31,
        "session_mode": "blocking",
        "rig_pipeline_calibration": None,
        "rig_pipeline_calibration_hash": "",
        "bundle_hash": "",
    }
    # Compute bundle_hash
    import hashlib
    bh_input = {k: v for k, v in bundle.items() if k != "bundle_hash"}
    bundle["bundle_hash"] = blake3(canonical_json_bytes(bh_input)).hexdigest()
    (tmp_path / "verification_bundle.json").write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n")

    # Manifest
    session_id = "test-session-00000000-0000-4000-8000-000000000000"
    # local_nonce — invent a deterministic one based on S0_seed
    local_nonce = blake3(b"test-nonce").digest()
    manifest = {
        "session_id": session_id,
        "session_iso_utc_start": "2026-04-24T00:00:00+00:00",
        "bundle_hash": bundle["bundle_hash"],
        "S_0_hex": "",  # fill in below
        "device_id": "test-device",
        "local_nonce_hex": local_nonce.hex(),
        "drand_chain_hash": fake_loop.drand_client.chain_hash,
        "drand_public_key": fake_loop.drand_client.pubkey_hex,
        "drand_round_at_session_open": fake_loop.drand_client._round,
        "rsk_rpc_endpoints": [],
        "anchor_policy": {"enabled": False},
        "session_status": "open",
        "manifest_hash_open": "",
    }
    # Compute manifest_hash_open from _OPEN_SNAPSHOT_FIELDS via SessionManifest.
    sm = SessionManifest(
        session_id=session_id,
        session_iso_utc_start=manifest["session_iso_utc_start"],
        bundle_hash=manifest["bundle_hash"],
        S_0_hex="",
        device_id="test-device",
        local_nonce_hex=manifest["local_nonce_hex"],
        drand_chain_hash=manifest["drand_chain_hash"],
        drand_public_key=manifest["drand_public_key"],
        drand_round_at_session_open=manifest["drand_round_at_session_open"],
        rsk_rpc_endpoints=[],
        anchor_policy={"enabled": False},
        anchor_start=None,
    )
    sm.finalize_open()
    manifest["manifest_hash_open"] = sm.manifest_hash_open

    # Derive S_0 using the same function the writer uses. But the writer
    # FakeLoop used `blake3(S0_seed).digest()` directly — we need the
    # real S_0 here to match what the writer actually wrote into row 0.
    # The test FakeLoop's S0 is set directly; it does NOT go through
    # derive_s0. So the test setup either:
    #   (a) set FakeLoop.S0 to the real derive_s0(...) result, OR
    #   (b) accept that row 0's S_t_hex won't match our re-derived S_0.
    # (a) is correct. We'll compute it here and pass it into FakeLoop
    # via the caller's construction of FakeLoop — see the test below.
    S_0 = derive_s0(
        rsk_block_hash_hex=UNANCHORED_BLOCK_HASH_HEX,
        rsk_block_height=UNANCHORED_BLOCK_HEIGHT,
        local_nonce_hex=manifest["local_nonce_hex"],
        session_id=session_id,
        manifest_hash_open_hex=manifest["manifest_hash_open"],
    )
    manifest["S_0_hex"] = S_0.hex()
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    # Mutate fake_loop.S0 to match what the verifier will recompute.
    fake_loop.S0 = S_0


def test_verifier_reports_chain_integrity_pass(tmp_path):
    """End-to-end: build a synthetic 5-row session, verify it, expect PASS."""
    n = 5
    fl = FakeLoop(tmp_path, n_synthetic_captures=n)
    # Write manifest + bundle FIRST so we have the real S_0 to seed FakeLoop.
    _write_minimal_bundle_and_manifest(tmp_path, fl)
    # Run the writer against the synthesized S_0 — producing chain_log +
    # capture_log + tile PNGs + raw files.
    tb_loop.TBLoopV8._run_chain_blocking(fl)

    report = verify_session(tmp_path, online=False)

    assert report.schema_ok, report.errors
    assert report.n_rows == n, (report.n_rows, n)
    assert report.n_rows_chain_advance_ok == n, \
        (report.n_rows_chain_advance_ok, report.row_errors[:5])
    assert report.n_rows_bayer_hash_ok == n
    assert report.n_rows_emission_png_hash_ok == n
    assert report.chain_integrity == "PASS", \
        (report.errors[:5], report.row_errors[:5])
    assert report.cycle_closure_verdict == "PASS"
    # FakeDrandClient always returns round != 0, so FULL availability.
    assert report.drand_availability == "FULL"


def test_verifier_detects_tampered_row(tmp_path):
    """Flip a bit in chain_log.csv row 2 — verifier should FAIL."""
    n = 4
    fl = FakeLoop(tmp_path, n_synthetic_captures=n)
    _write_minimal_bundle_and_manifest(tmp_path, fl)
    tb_loop.TBLoopV8._run_chain_blocking(fl)

    # Tamper: flip the first hex char of bayer_blake3_hex on row 2.
    chain_log = tmp_path / "chain_log.csv"
    with open(chain_log) as f:
        lines = f.readlines()
    # Line 0 = comment, line 1 = header, line 2 = row 0, line 4 = row 2.
    row2_line = lines[4]
    cells = row2_line.split(",")
    # bayer_blake3_hex is the 3rd field (after t, S_t_hex).
    cells[2] = ("0" if cells[2][0] != "0" else "f") + cells[2][1:]
    lines[4] = ",".join(cells)
    chain_log.write_text("".join(lines))

    report = verify_session(tmp_path, online=False)
    # Chain advance for row 3 uses row 2's bayer_hash → mismatch.
    assert report.chain_integrity == "FAIL"
    assert report.n_rows_bayer_hash_ok < n or \
           report.n_rows_chain_advance_ok < n


def test_verifier_runs_on_empty_session(tmp_path):
    """Session dir with no chain_log — verifier should FAIL gracefully."""
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "verification_bundle.json").write_text("{}")
    report = verify_session(tmp_path, online=False)
    assert report.chain_integrity == "FAIL"
    assert any("manifest" in e.lower() or "bundle" in e.lower()
               for e in report.errors)

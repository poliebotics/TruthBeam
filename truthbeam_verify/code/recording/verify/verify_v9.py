#!/usr/bin/env python3
"""Truth Beam v9 session verifier.

Reads a session directory and reports, at multiple granularities:

  per row (chain_log.csv):
    - chain_advance_ok:        bool  — S_t == advance(S_{t-1}, bayer_{t-1}, ...)
    - bayer_hash_ok:           bool  — bayer_blake3_hex == blake3(frame_t.raw)
    - emission_png_hash_ok:    bool  — emission_live_pixel_blake3_hex ==
                                       blake3(decode(tile_t.png))

  per pulse (anchor_txs.csv, is_final=0):
    - payload_commitment_verified: bool — recomputed from session state
                                          matches anchor_txs.csv field
    - rsk_inclusion_status:   NOT_FOUND | PENDING | INCLUDED_1_CONF |
                              INCLUDED_6_CONF | STABLE_100_CONF  (online only)
    - block_depth:            int  (online only)
    - block_explorer_url:     derived string

  session-level:
    - chain_integrity:        PASS | FAIL | LOGS_ONLY
    - cycle_closure_verdict:  PASS | FAIL | INCONCLUSIVE
    - drand_availability:     FULL | PARTIAL | MISSING
    - drand_rounds_verified:  int   (online only)
    - drand_rounds_stale:     int
    - entropy_source_vector:  list  (sources that were actually verified)

chain_integrity gates on the three per-row invariants only (chain
advance + raw-Bayer hash + emission-PNG hash); in --logs-only mode the
media hashes are skipped and it reports LOGS_ONLY. The pulse
commitments AND the manifest/bundle/log-hash commitment checks gate
cycle_closure_verdict, not chain_integrity. RSK inclusion status and
drand verification status are REPORTED but DO NOT gate the verdict
under the v9 corroborating-evidence policy.

Honest reporting:
  - Fields the verifier couldn't check are reported as null /
    status-tier value, never as silent PASS.
  - Offline mode (default) skips RSK inclusion + drand refetch;
    they become PENDING / NOT_VERIFIED in the output.
  - Online mode (`--online`) performs network checks; still
    soft-fails on individual RPC errors rather than aborting.

Usage:
    python3 verify_v9.py --session-dir <path> [--online]
    python3 -m truth_beam_recording.verify.verify_v9 --session-dir <path>

Dependencies:
    blake3                 — chain / hash math (always required)
    Pillow (PIL)           — emission PNG decode (always required)
    py_ecc                 — drand BLS re-verify (optional, --online)
    web3                   — RSK tx inclusion check (optional, --online)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from blake3 import blake3
from PIL import Image

# Allow running as a script OR as a package module.
HERE = Path(__file__).resolve().parent
PROTOCOL_DIR = HERE.parent / "protocol"
if str(PROTOCOL_DIR) not in sys.path:
    sys.path.insert(0, str(PROTOCOL_DIR))

from session_schema import (  # noqa: E402
    PROTOCOL_VERSION, ROW_DOMAIN_TAG,
    compute_chain_row, compute_xof_seeds,
    derive_s0, UNANCHORED_BLOCK_HASH_HEX, UNANCHORED_BLOCK_HEIGHT,
    compute_pulse_commitment, UNANCHORED_PREV_PULSE_COMMITMENT,
    compute_chain_log_hash, compute_capture_log_hash,
)
from chain import META_STRUCT                                # noqa: E402
from session_logs import CHAIN_LOG_FIELDS                    # noqa: E402


# ----------------------------------------------------------------------
# Derived-URL helpers (documented in README_BUNDLE.md)
# ----------------------------------------------------------------------

RSK_EXPLORER_BASE = {
    30: "https://explorer.rsk.co",        # mainnet
    31: "https://explorer.testnet.rsk.co",  # testnet
}


def block_explorer_url(chain_id: int, tx_hash: str) -> str:
    """One-line derivation. (chain_id, tx_hash) → explorer URL.

    chain_id: RSK mainnet=30, testnet=31.
    tx_hash: "0x..." (66 hex chars).
    """
    base = RSK_EXPLORER_BASE.get(int(chain_id))
    if base is None:
        return ""
    return f"{base}/transactions/{tx_hash}"


# ----------------------------------------------------------------------
# Report dataclasses (JSON-serializable)
# ----------------------------------------------------------------------


@dataclass
class RowReport:
    t: int
    chain_advance_ok: bool
    bayer_hash_ok: bool
    emission_png_hash_ok: bool
    errors: list = field(default_factory=list)


@dataclass
class PulseReport:
    pulse_index: int
    tx_hash: str
    frame_start: int
    frame_end: int
    payload_commitment_verified: bool
    rsk_inclusion_status: str            # tier value
    block_depth: Optional[int]
    block_explorer_url: str
    errors: list = field(default_factory=list)


@dataclass
class VerifyReport:
    session_dir: str
    protocol_version_expected: str
    protocol_version_found: str
    schema_ok: bool

    # Session-level verdicts
    chain_integrity: str = "FAIL"        # PASS | FAIL
    cycle_closure_verdict: str = "INCONCLUSIVE"  # PASS | FAIL | INCONCLUSIVE
    drand_availability: str = "MISSING"  # FULL | PARTIAL | MISSING

    # Counts
    n_rows: int = 0
    n_rows_chain_advance_ok: int = 0
    n_rows_bayer_hash_ok: int = 0
    n_rows_emission_png_hash_ok: int = 0
    n_pulses: int = 0
    n_pulses_commitment_ok: int = 0

    # Drand
    drand_rounds_present: int = 0        # distinct non-zero round numbers
    drand_rounds_verified: int = 0
    drand_rounds_stale: int = 0
    drand_rounds_failed_verify: int = 0
    drand_staleness_max_ms: int = 0

    # Pulses
    pulse_reports: list = field(default_factory=list)
    row_errors: list = field(default_factory=list)

    # Entropy source vector
    entropy_source_vector: list = field(default_factory=list)

    # Top-level session errors (missing files, etc.)
    errors: list = field(default_factory=list)

    # Terminal state, chain log/capture log hashes.
    # last_logged_S_t_hex is the *pre*-state of the final row (S_{N-1});
    # terminal_S_N_hex is the computed post-final-transition state S_N,
    # compared against manifest.S_N_hex (the value anchored in the final
    # RSK pulse). Verifier versions before 2026-06-10 mislabelled the
    # final row's pre-state as terminal_S_N_hex.
    last_logged_S_t_hex: str = ""
    terminal_S_N_hex: str = ""
    terminal_state_matches_manifest: str = "UNCHECKED"  # PASS | FAIL | UNCHECKED
    chain_log_hash_recomputed: str = ""
    capture_log_hash_recomputed: str = ""
    # Commitment checks (2026-06-10 hardening): each is PASS | FAIL |
    # UNCHECKED and gates cycle_closure_verdict.
    manifest_hash_final_matches: str = "UNCHECKED"
    bundle_hash_matches: str = "UNCHECKED"
    chain_log_hash_matches_manifest: str = "UNCHECKED"
    capture_log_hash_matches_manifest: str = "UNCHECKED"
    # anchor_log_hash is computed at manifest-finalize time, BEFORE the
    # trailing final_root tx row is appended; PASS means the file matches
    # either as-is or after stripping trailing final_root rows.
    anchor_log_hash_matches_manifest: str = "UNCHECKED"
    # The single trailing final_root row is appended AFTER the anchor_log_hash
    # snapshot, so it is not covered by that hash; this check authenticates it
    # against manifest.anchor_end (which manifest_hash_final does cover).
    anchor_final_row_matches: str = "UNCHECKED"
    pulse_chain_continuity: str = "UNCHECKED"
    # True when the media (raw frame / emission PNG) hash checks were
    # deliberately skipped; chain_integrity reports LOGS_ONLY in that mode.
    logs_only: bool = False

    def to_dict(self) -> dict:
        # Convert dataclass → dict (dataclass.asdict would recurse into pulses too)
        import dataclasses
        d = dataclasses.asdict(self)
        return d


# ----------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------


def _load_json(path: Path) -> Optional[dict]:
    """Strict JSON load: reject duplicate keys and NaN/Infinity, exactly as
    the protocol's writer/schema does. Using plain json.loads here would let
    an attacker hide contradictory values behind duplicate keys (Python keeps
    the last) so the verifier and a strict-loading party disagree on what the
    document is. Falls back to plain json.loads only if the strict loader is
    unavailable."""
    if not path.exists():
        return None
    try:
        from session_schema import _strict_json_load
        return _strict_json_load(path.read_bytes())
    except ImportError:
        return json.loads(path.read_text())


def _load_chain_log(path: Path) -> list:
    """Return list of dict rows (header comment skipped, DictReader)."""
    if not path.exists():
        return []
    with open(path) as f:
        header = f.readline()
        if not header.startswith("# session_iso_utc="):
            return []
        return list(csv.DictReader(f))


def _load_capture_log(path: Path) -> list:
    if not path.exists():
        return []
    with open(path) as f:
        f.readline()  # skip comment
        return list(csv.DictReader(f))


def _load_anchor_txs(path: Path) -> list:
    """anchor_txs.csv is canonical-form (no DictWriter); parse manually."""
    if not path.exists():
        return []
    with open(path, "rb") as f:
        raw = f.read().decode("ascii")
    lines = raw.strip("\n").split("\n")
    if not lines:
        return []
    header = lines[0].split(",")
    rows = []
    for line in lines[1:]:
        if not line:
            continue
        vals = line.split(",")
        if len(vals) != len(header):
            # Do not silently drop a malformed row — surface it so the
            # caller can fail, rather than letting an attacker hide a
            # dropped pulse behind a deliberately mangled line. The
            # sentinel row is recognisable by its single \"__malformed__\"
            # marker and is filtered before commitment recomputation.
            rows.append({"__malformed__": "1"})
            continue
        rows.append(dict(zip(header, vals)))
    return rows


# ----------------------------------------------------------------------
# Row-level invariants
# ----------------------------------------------------------------------


def _check_manifest_bundle_hashes(report, manifest: dict, bundle: dict) -> None:
    """Recompute manifest_hash_final and bundle_hash from the loaded dicts
    (canonical JSON minus the self-hash field) and check the manifest↔bundle
    linkage. Catches any post-hoc edit to either JSON document.

    For a finalized session these commitments are MANDATORY: a missing
    manifest_hash_final, bundle.bundle_hash, or manifest.bundle_hash is a
    FAIL, not an UNCHECKED. Otherwise an attacker could neutralise the
    entire commitment layer simply by deleting the pinned fields (they are
    not folded into S_0 / manifest_hash_open, so removing them does not
    break the chain walk)."""
    finalized = manifest.get("session_status") == "finalized"
    try:
        from session_schema import canonical_json_hash_hex
        mhf = manifest.get("manifest_hash_final")
        if mhf:
            d = dict(manifest)
            d.pop("manifest_hash_final", None)
            report.manifest_hash_final_matches = (
                "PASS" if canonical_json_hash_hex(d) == mhf else "FAIL")
            if report.manifest_hash_final_matches == "FAIL":
                report.errors.append(
                    "manifest_hash_final does not match recomputed hash of "
                    "manifest contents")
        elif finalized:
            report.manifest_hash_final_matches = "FAIL"
            report.errors.append(
                "session_status is 'finalized' but manifest_hash_final is "
                "absent (the field that transitively pins the log and "
                "bundle hashes)")

        # Drive the bundle check from the MANIFEST side: the manifest's
        # bundle_hash is covered by manifest_hash_final, so it cannot be
        # stripped without detection above. Require the bundle to carry a
        # matching self-hash. (Checking only `if bundle.bundle_hash` would
        # let an attacker swap in a whole bundle that omits the field.)
        m_bh = manifest.get("bundle_hash")
        b_bh = bundle.get("bundle_hash")
        if m_bh or b_bh or finalized:
            db = dict(bundle)
            db.pop("bundle_hash", None)
            recomputed = canonical_json_hash_hex(db)
            self_ok = bool(b_bh) and recomputed == b_bh
            link_ok = bool(m_bh) and bool(b_bh) and m_bh == b_bh
            report.bundle_hash_matches = (
                "PASS" if (self_ok and link_ok) else "FAIL")
            if not b_bh:
                report.errors.append(
                    "verification_bundle.json is missing its bundle_hash "
                    "self-commitment")
            elif not self_ok:
                report.errors.append(
                    "bundle_hash does not match recomputed hash of "
                    "verification_bundle contents")
            if not m_bh:
                report.errors.append(
                    "manifest.json is missing its bundle_hash linkage field")
            elif not link_ok:
                report.errors.append(
                    "manifest.bundle_hash != verification_bundle.bundle_hash")
    except Exception as e:
        report.errors.append(f"manifest/bundle hash check raised {e!r}")


def _final_root_field_index(anchor_header: bytes):
    """Index of the payload_kind column in the anchor_txs.csv header, or
    None if the header cannot be parsed."""
    try:
        cols = anchor_header.decode("utf-8").strip().split(",")
        return cols.index("payload_kind")
    except Exception:
        return None


def _authenticate_final_root_row(report, manifest, header, cells) -> None:
    """Authenticate the stripped trailing final_root row against
    manifest.anchor_end (covered by manifest_hash_final). The final_root row
    is appended after the anchor_log_hash snapshot, so without this it would
    be an unauthenticated region a downstream consumer might trust as the
    on-chain anchor pointer. Skipped (UNCHECKED) when the manifest carries no
    anchor_end (e.g. the final tx did not confirm)."""
    anchor_end = manifest.get("anchor_end")
    if not anchor_end:
        return
    try:
        col = {name: i for i, name in enumerate(header)}

        def cell(name):
            i = col.get(name)
            return cells[i] if i is not None and i < len(cells) else None

        row_tx = cell("tx_hash")
        row_payload = cell("payload_S_hex")
        ok = True
        if row_tx is not None and row_tx != anchor_end.get("tx_hash"):
            ok = False
            report.errors.append(
                "final_root row tx_hash does not match manifest.anchor_end."
                "tx_hash")
        # For a final_root row the payload column carries the final-root
        # commitment; accept a match against either anchor_end payload.
        if row_payload is not None and row_payload not in (
                anchor_end.get("payload_final_root_hex"),
                anchor_end.get("payload_S_N_hex")):
            ok = False
            report.errors.append(
                "final_root row payload does not match manifest.anchor_end "
                "payload_final_root_hex / payload_S_N_hex")
        report.anchor_final_row_matches = "PASS" if ok else "FAIL"
    except Exception as e:
        report.anchor_final_row_matches = "FAIL"
        report.errors.append(f"final_root row authentication raised {e!r}")


def _check_log_hashes(report, manifest: dict, anchor_txs_path: Path) -> None:
    """Compare the recomputed chain/capture log hashes against the values
    pinned in the manifest (which manifest_hash_final covers), and check
    anchor_txs.csv against anchor_log_hash. anchor_log_hash is snapshotted
    at finalize time, BEFORE the single trailing final_root row is appended,
    so a match after stripping exactly one well-formed final_root row also
    counts as PASS.

    For a finalized session a pinned-but-unverifiable hash is a FAIL: if the
    manifest pins a log hash but the recompute produced nothing (a swallowed
    exception leaving the recomputed string empty), that is FAIL, not an
    UNCHECKED that the verdict would tolerate."""
    finalized = manifest.get("session_status") == "finalized"
    try:
        want = manifest.get("chain_log_hash")
        if want:
            if report.chain_log_hash_recomputed:
                report.chain_log_hash_matches_manifest = (
                    "PASS" if report.chain_log_hash_recomputed == want
                    else "FAIL")
                if report.chain_log_hash_matches_manifest == "FAIL":
                    report.errors.append(
                        "chain_log.csv hash does not match manifest."
                        "chain_log_hash")
            else:
                report.chain_log_hash_matches_manifest = "FAIL"
                report.errors.append(
                    "manifest pins chain_log_hash but it could not be "
                    "recomputed")
        elif finalized:
            report.chain_log_hash_matches_manifest = "FAIL"
            report.errors.append(
                "finalized session manifest is missing chain_log_hash")

        want = manifest.get("capture_log_hash")
        if want:
            if report.capture_log_hash_recomputed:
                report.capture_log_hash_matches_manifest = (
                    "PASS" if report.capture_log_hash_recomputed == want
                    else "FAIL")
                if report.capture_log_hash_matches_manifest == "FAIL":
                    report.errors.append(
                        "capture_log.csv hash does not match manifest."
                        "capture_log_hash")
            else:
                report.capture_log_hash_matches_manifest = "FAIL"
                report.errors.append(
                    "manifest pins capture_log_hash but it could not be "
                    "recomputed")
        elif finalized:
            report.capture_log_hash_matches_manifest = "FAIL"
            report.errors.append(
                "finalized session manifest is missing capture_log_hash")

        want = manifest.get("anchor_log_hash")
        if want:
            if not anchor_txs_path.exists():
                report.anchor_log_hash_matches_manifest = "FAIL"
                report.errors.append(
                    "manifest.anchor_log_hash is set but anchor_txs.csv is "
                    "missing")
            else:
                raw = anchor_txs_path.read_bytes()
                if blake3(raw).hexdigest() == want:
                    report.anchor_log_hash_matches_manifest = "PASS"
                else:
                    # At-finalize snapshot: strip AT MOST ONE trailing row,
                    # and only if it is a structurally well-formed
                    # final_root row (payload_kind column == "final_root"),
                    # not merely any line containing that substring.
                    lines = raw.split(b"\n")
                    while lines and lines[-1] == b"":
                        lines.pop()
                    header_cells = lines[0].decode("utf-8", "replace").split(
                        ",") if lines else []
                    kind_idx = _final_root_field_index(lines[0]) if lines \
                        else None
                    stripped = False
                    final_row_cells = None
                    if kind_idx is not None and len(lines) > 1:
                        cells = lines[-1].decode(
                            "utf-8", "replace").split(",")
                        if len(cells) > kind_idx and \
                                cells[kind_idx] == "final_root":
                            final_row_cells = cells
                            lines.pop()
                            stripped = True
                    cand = b"\n".join(lines) + b"\n"
                    if stripped and blake3(cand).hexdigest() == want:
                        report.anchor_log_hash_matches_manifest = "PASS"
                        _authenticate_final_root_row(
                            report, manifest, header_cells, final_row_cells)
                    else:
                        report.anchor_log_hash_matches_manifest = "FAIL"
                        report.errors.append(
                            "anchor_txs.csv hash does not match manifest."
                            "anchor_log_hash (with or without the single "
                            "trailing final_root row)")
        elif finalized and anchor_txs_path.exists():
            # An anchored, finalized session pins anchor_log_hash; its
            # absence while a pulse log is present is suspicious.
            report.anchor_log_hash_matches_manifest = "FAIL"
            report.errors.append(
                "anchor_txs.csv is present but the finalized manifest does "
                "not pin anchor_log_hash")
    except Exception as e:
        report.errors.append(f"log hash comparison raised {e!r}")


def _check_pulse_chain(report, pulses: list, malformed_anchor_rows: int = 0,
                       final_chain_row: Optional[int] = None) -> None:
    """Enforce pulse-chain continuity: pulse_index strictly sequential from
    0, each prev_pulse_commitment equal to the previous pulse's commitment
    (32 zero bytes for the first), and frame ranges contiguous; and FAIL on
    any malformed anchor row (an attacker must not be able to hide a dropped
    pulse behind a mangled line). Catches deletion or reordering of interior
    pulse rows, which per-pulse commitment recomputation alone does not.

    Note: the protocol does NOT require the last state pulse to reach the
    final chain row — interior pulses are periodic and the trailing segment
    is committed by the manifest's S_N (terminal_state_matches_manifest), so
    final_chain_row is accepted for API symmetry but not used as an equality
    constraint. A dropped terminal pulse instead surfaces via the mandatory
    anchor_log_hash commitment (a clean drop changes anchor_txs.csv) or the
    malformed-row check (a mangled drop)."""
    if not pulses and not malformed_anchor_rows:
        return
    try:
        ok = True
        if malformed_anchor_rows:
            ok = False
            report.errors.append(
                f"pulse chain: {malformed_anchor_rows} malformed anchor row(s) "
                f"(wrong column count) — refusing to verify a mangled log")
        zero = "00" * 32
        for i, pr in enumerate(pulses):
            if int(pr["pulse_index"]) != i:
                ok = False
                report.errors.append(
                    f"pulse chain: index {pr['pulse_index']} at position "
                    f"{i} (expected {i})")
            want_prev = zero if i == 0 else \
                pulses[i - 1]["payload_commitment_hex"]
            if pr["prev_pulse_commitment_hex"] != want_prev:
                ok = False
                report.errors.append(
                    f"pulse chain: pulse {i} prev_pulse_commitment does "
                    f"not match previous pulse's commitment")
            if i > 0 and int(pr["frame_start"]) != \
                    int(pulses[i - 1]["frame_end"]) + 1:
                ok = False
                report.errors.append(
                    f"pulse chain: pulse {i} frame_start "
                    f"{pr['frame_start']} not contiguous with previous "
                    f"frame_end {pulses[i - 1]['frame_end']}")
        report.pulse_chain_continuity = "PASS" if ok else "FAIL"
    except Exception as e:
        report.pulse_chain_continuity = "FAIL"
        report.errors.append(f"pulse chain continuity check raised {e!r}")


def _commitment_checks_ok(report) -> bool:
    """No commitment check may be FAIL. A check stays UNCHECKED only when the
    manifest legitimately does not pin the corresponding value; for a
    finalized session the pin fields are mandatory and their absence is
    recorded as FAIL upstream (see _check_manifest_bundle_hashes /
    _check_log_hashes), so UNCHECKED here cannot be attacker-induced on a
    session that claims to be finalized."""
    return all(v != "FAIL" for v in (
        report.manifest_hash_final_matches,
        report.bundle_hash_matches,
        report.chain_log_hash_matches_manifest,
        report.capture_log_hash_matches_manifest,
        report.anchor_log_hash_matches_manifest,
        report.anchor_final_row_matches,
        report.pulse_chain_continuity,
    ))


def _verify_row(t: int, row: dict, prev_row: Optional[dict],
                session_dir: Path, S_0_recomputed: bytes,
                recordings_dir: Path, logs_only: bool = False) -> RowReport:
    """Verify one chain_log row against its derivable invariants."""
    errors = []

    # --- Chain advance ---
    chain_advance_ok = False
    try:
        if t == 0:
            # Row 0's S_t_hex must equal S_0 recomputed from manifest fields.
            chain_advance_ok = row["S_t_hex"] == S_0_recomputed.hex()
            if not chain_advance_ok:
                errors.append(
                    f"row 0: S_t_hex != S_0 recomputed from manifest")
        else:
            # Row t's S_t_hex == advance(S_{t-1}, bayer_{t-1}, meta_{t-1},
            # drand_round_{t-1}, drand_value_{t-1}).
            expected = compute_chain_row(
                bytes.fromhex(prev_row["S_t_hex"]),
                bytes.fromhex(prev_row["bayer_blake3_hex"]),
                bytes.fromhex(prev_row["meta_hex"]),
                int(prev_row["drand_round_number"]),
                bytes.fromhex(prev_row["drand_round_value_hex"]),
            )
            chain_advance_ok = row["S_t_hex"] == expected.hex()
            if not chain_advance_ok:
                errors.append(
                    f"t={t}: S_t_hex != advance(prev state + bayer + meta + drand)"
                )
    except Exception as e:
        errors.append(f"t={t}: chain advance check raised {e!r}")

    # --- Bayer hash vs raw file (skipped in --logs-only mode) ---
    bayer_hash_ok = False
    if not logs_only:
        try:
            raw_path = recordings_dir / f"frame_{t:06d}.raw"
            if raw_path.exists():
                actual = blake3(raw_path.read_bytes()).hexdigest()
                bayer_hash_ok = actual == row["bayer_blake3_hex"]
                if not bayer_hash_ok:
                    errors.append(
                        f"t={t}: bayer_blake3_hex mismatch with "
                        f"frame_{t:06d}.raw")
            else:
                errors.append(f"t={t}: missing raw file {raw_path}")
        except Exception as e:
            errors.append(f"t={t}: bayer hash check raised {e!r}")

    # --- Emission PNG pixel hash (skipped in --logs-only mode) ---
    emission_png_hash_ok = False
    if logs_only:
        return RowReport(
            t=t,
            chain_advance_ok=chain_advance_ok,
            bayer_hash_ok=False,
            emission_png_hash_ok=False,
            errors=errors,
        )
    try:
        png_rel = row["emission_png_path"]
        if png_rel:
            png_path = session_dir / png_rel
            if png_path.exists():
                img = Image.open(png_path).convert("RGB")
                arr = np.asarray(img)
                actual = blake3(
                    bytes(np.ascontiguousarray(arr).tobytes())
                ).hexdigest()
                emission_png_hash_ok = (
                    actual == row["emission_live_pixel_blake3_hex"])
                if not emission_png_hash_ok:
                    errors.append(
                        f"t={t}: emission_png pixel hash mismatch "
                        f"(png bytes {actual[:16]}… vs "
                        f"live {row['emission_live_pixel_blake3_hex'][:16]}…)"
                    )
            else:
                errors.append(f"t={t}: missing PNG {png_path}")
        else:
            errors.append(f"t={t}: emission_png_path is empty")
    except Exception as e:
        errors.append(f"t={t}: PNG hash check raised {e!r}")

    return RowReport(
        t=t,
        chain_advance_ok=chain_advance_ok,
        bayer_hash_ok=bayer_hash_ok,
        emission_png_hash_ok=emission_png_hash_ok,
        errors=errors,
    )


# ----------------------------------------------------------------------
# Pulse-commitment verification
# ----------------------------------------------------------------------


def _v9_row_advance(row: dict) -> str:
    """Recompute the v9 post-row state from one chain_log row's fields."""
    return compute_chain_row(
        bytes.fromhex(row["S_t_hex"]),
        bytes.fromhex(row["bayer_blake3_hex"]),
        bytes.fromhex(row["meta_hex"]),
        int(row["drand_round_number"]),
        bytes.fromhex(row["drand_round_value_hex"]),
    ).hex()


def _verify_pulse_commitment(pulse_row: dict, session_id: str,
                             chain_rows_by_t: dict,
                             advance_fn=None) -> PulseReport:
    """Reconstruct the pulse commitment from session_id + chain state
    and compare to payload_commitment_hex in anchor_txs.csv. A malformed
    numeric field in one pulse row fails THAT pulse rather than aborting the
    whole verification with an uncaught traceback."""
    errors = []
    try:
        pulse_index = int(pulse_row["pulse_index"])
        frame_start = int(pulse_row["frame_start"])
        frame_end = int(pulse_row["frame_end"])
        prev_commit_hex = pulse_row["prev_pulse_commitment_hex"]
        recorded_commit_hex = pulse_row["payload_commitment_hex"]
        S_hex = pulse_row["payload_S_hex"]
        tx_hash = pulse_row["tx_hash"]
        chain_id = int(pulse_row["chain_id"])
    except (KeyError, ValueError) as e:
        return PulseReport(
            pulse_index=-1, tx_hash=str(pulse_row.get("tx_hash", "")),
            frame_start=-1, frame_end=-1,
            payload_commitment_verified=False,
            rsk_inclusion_status="NOT_CHECKED", block_depth=None,
            block_explorer_url="",
            errors=[f"malformed pulse row (could not parse fields): {e!r}"])

    try:
        expected = compute_pulse_commitment(
            session_id=session_id,
            pulse_index=pulse_index,
            frame_start=frame_start,
            frame_end=frame_end,
            prev_pulse_commitment=bytes.fromhex(prev_commit_hex),
            S_t=bytes.fromhex(S_hex),
        )
        ok = expected.hex() == recorded_commit_hex
        if not ok:
            errors.append(
                f"pulse {pulse_index}: recomputed commitment "
                f"{expected.hex()[:16]}… != recorded "
                f"{recorded_commit_hex[:16]}…"
            )
    except Exception as e:
        ok = False
        errors.append(f"pulse {pulse_index}: recompute raised {e!r}")

    # Optional cross-check: S_hex should match chain_log row at frame_end's
    # S_next (= chain_row[frame_end+1].S_t_hex if present, or terminal S_N).
    # Pulse covers frames [frame_start..frame_end]; state_index = frame_end+1.
    expected_S_from_chain = None
    if frame_end + 1 in chain_rows_by_t:
        expected_S_from_chain = chain_rows_by_t[frame_end + 1]["S_t_hex"]
    elif chain_rows_by_t and frame_end == max(chain_rows_by_t):
        # Last-pulse fallback: frame_end is the final row, so the expected
        # payload state is the post-final-transition terminal S_N =
        # advance(last row's S_t, bayer, meta, drand[, ai_root]).
        try:
            fn = advance_fn or _v9_row_advance
            expected_S_from_chain = fn(chain_rows_by_t[frame_end])
        except Exception as e:
            errors.append(
                f"pulse {pulse_index}: terminal-state fallback raised {e!r}")
    if expected_S_from_chain is not None and S_hex != expected_S_from_chain:
        errors.append(
            f"pulse {pulse_index}: payload_S_hex {S_hex[:16]}… != "
            f"chain_log[{frame_end+1}].S_t_hex "
            f"{expected_S_from_chain[:16]}…"
        )
        ok = False

    return PulseReport(
        pulse_index=pulse_index,
        tx_hash=tx_hash,
        frame_start=frame_start,
        frame_end=frame_end,
        payload_commitment_verified=ok,
        rsk_inclusion_status="NOT_CHECKED",    # filled by online pass
        block_depth=None,
        block_explorer_url=block_explorer_url(chain_id, tx_hash),
        errors=errors,
    )


# ----------------------------------------------------------------------
# Online checks (optional)
# ----------------------------------------------------------------------


def _check_rsk_inclusion(pulse: PulseReport, rpc_urls: list) -> None:
    """Mutate pulse.rsk_inclusion_status and pulse.block_depth via RPC
    lookups. Errors are logged into pulse.errors; pulse.rsk_inclusion_status
    remains NOT_FOUND or PENDING on error."""
    try:
        from web3 import Web3
    except ImportError:
        pulse.errors.append("web3 not installed — skipping RSK inclusion")
        pulse.rsk_inclusion_status = "NOT_CHECKED"
        return

    last_err = None
    for url in rpc_urls:
        try:
            w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 10}))
            tx = w3.eth.get_transaction(pulse.tx_hash)
            if tx is None:
                pulse.rsk_inclusion_status = "NOT_FOUND"
                return
            block_number = tx.get("blockNumber")
            if block_number is None:
                pulse.rsk_inclusion_status = "PENDING"
                return
            latest = int(w3.eth.block_number)
            depth = latest - int(block_number) + 1
            pulse.block_depth = depth
            if depth >= 100:
                pulse.rsk_inclusion_status = "STABLE_100_CONF"
            elif depth >= 6:
                pulse.rsk_inclusion_status = "INCLUDED_6_CONF"
            else:
                pulse.rsk_inclusion_status = "INCLUDED_1_CONF"
            return
        except Exception as e:
            last_err = e
            continue

    pulse.errors.append(
        f"RSK inclusion: all RPCs failed: {last_err!r}")
    pulse.rsk_inclusion_status = "NOT_CHECKED"


def _check_drand_rounds(unique_rounds: list, chain_hash: str,
                        pubkey_hex: str, report: VerifyReport) -> None:
    """Refetch and BLS-verify each unique drand round seen in chain_log.
    Results update report.drand_rounds_verified /
    drand_rounds_failed_verify."""
    try:
        from drand_client import fetch_round, verify_round_full
    except ImportError:
        report.errors.append(
            "drand_client import failed — skipping BLS re-verify")
        return

    for rn in unique_rounds:
        if rn == 0:
            # Zero sentinel = drand unavailable at that row; skip.
            continue
        try:
            rd = fetch_round(rn, chain_hash=chain_hash)
        except Exception as e:
            report.drand_rounds_failed_verify += 1
            report.errors.append(
                f"drand: refetch round {rn} failed: {e!r}")
            continue
        try:
            ok = verify_round_full(
                rn, rd["randomness"], rd["signature"],
                pubkey_hex=pubkey_hex,
            )
        except Exception as e:
            report.drand_rounds_failed_verify += 1
            report.errors.append(
                f"drand: BLS verify round {rn} raised: {e!r}")
            continue
        if ok:
            report.drand_rounds_verified += 1
        else:
            report.drand_rounds_failed_verify += 1


# ----------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------


def verify_session(session_dir: Path, online: bool = False,
                   logs_only: bool = False) -> VerifyReport:
    session_dir = Path(session_dir)
    # Locate key files. chain_log.csv + capture_log.csv may live at the
    # session root OR under a "chain/" subdir depending on session
    # convention; check both.
    chain_log_path = session_dir / "chain_log.csv"
    capture_log_path = session_dir / "capture_log.csv"
    if not chain_log_path.exists():
        chain_log_path = session_dir / "chain" / "chain_log.csv"
    if not capture_log_path.exists():
        capture_log_path = session_dir / "chain" / "capture_log.csv"

    manifest_path = session_dir / "manifest.json"
    if not manifest_path.exists():
        manifest_path = session_dir / "chain" / "session_manifest.json"
    bundle_path = session_dir / "verification_bundle.json"
    if not bundle_path.exists():
        bundle_path = session_dir / "chain" / "verification_bundle.json"
    anchor_txs_path = session_dir / "anchor_txs.csv"
    recordings_dir = session_dir / "Recordings"

    manifest = _load_json(manifest_path)
    bundle = _load_json(bundle_path)

    report = VerifyReport(
        session_dir=str(session_dir),
        protocol_version_expected=PROTOCOL_VERSION,
        protocol_version_found=(bundle or {}).get(
            "protocol_version", "MISSING"),
        schema_ok=False,
    )
    if manifest is None:
        report.errors.append(f"manifest.json not found at {manifest_path}")
        return report
    if bundle is None:
        report.errors.append(
            f"verification_bundle.json not found at {bundle_path}")
        return report
    if report.protocol_version_found != PROTOCOL_VERSION:
        report.errors.append(
            f"protocol_version mismatch: bundle says "
            f"{report.protocol_version_found!r}, verifier expects "
            f"{PROTOCOL_VERSION!r}"
        )
        return report
    report.schema_ok = True

    # --- Recompute S_0 ---
    try:
        anchor_start = manifest.get("anchor_start")
        if anchor_start:
            rsk_bh = anchor_start["block_hash"]
            rsk_bn = int(anchor_start["block_number"])
        else:
            rsk_bh = UNANCHORED_BLOCK_HASH_HEX
            rsk_bn = UNANCHORED_BLOCK_HEIGHT
        S_0 = derive_s0(
            rsk_block_hash_hex=rsk_bh,
            rsk_block_height=rsk_bn,
            local_nonce_hex=manifest["local_nonce_hex"],
            session_id=manifest["session_id"],
            manifest_hash_open_hex=manifest["manifest_hash_open"],
        )
        if S_0.hex() != manifest["S_0_hex"]:
            report.errors.append(
                f"S_0 mismatch: recomputed {S_0.hex()[:16]}… != "
                f"manifest.S_0_hex {manifest['S_0_hex'][:16]}…"
            )
            return report
    except Exception as e:
        report.errors.append(f"S_0 recompute raised: {e!r}")
        return report

    # --- Chain log walk ---
    rows = _load_chain_log(chain_log_path)
    report.n_rows = len(rows)
    if not rows:
        report.errors.append(f"chain_log.csv empty or missing: {chain_log_path}")
        report.chain_integrity = "FAIL"
        return report
    prev_row = None
    row_reports = []
    unique_drand_rounds = set()
    for i, row in enumerate(rows):
        t = int(row["t"])
        rr = _verify_row(
            t, row, prev_row, session_dir, S_0, recordings_dir,
            logs_only=logs_only)
        row_reports.append(rr)
        if rr.chain_advance_ok:
            report.n_rows_chain_advance_ok += 1
        if rr.bayer_hash_ok:
            report.n_rows_bayer_hash_ok += 1
        if rr.emission_png_hash_ok:
            report.n_rows_emission_png_hash_ok += 1
        if rr.errors:
            report.row_errors.extend(rr.errors[:3])  # cap per-row noise
        # Collect drand info
        rn = int(row["drand_round_number"])
        if rn != 0:
            unique_drand_rounds.add(rn)
        try:
            sm = int(row["drand_staleness_ms"])
            if sm > report.drand_staleness_max_ms:
                report.drand_staleness_max_ms = sm
        except (KeyError, ValueError):
            pass
        prev_row = row

    report.last_logged_S_t_hex = rows[-1]["S_t_hex"]
    # The true terminal state S_N is the post-final-transition state: one
    # more chain advance from the final row's logged fields. Compare it to
    # manifest.S_N_hex, which is the value carried in the final RSK anchor.
    try:
        report.terminal_S_N_hex = _v9_row_advance(rows[-1])
        manifest_S_N = (manifest.get("S_N_hex") or "").lower()
        if manifest_S_N:
            report.terminal_state_matches_manifest = (
                "PASS" if manifest_S_N == report.terminal_S_N_hex.lower()
                else "FAIL")
            if report.terminal_state_matches_manifest == "FAIL":
                report.errors.append(
                    "terminal S_N mismatch: computed "
                    f"{report.terminal_S_N_hex[:16]}… != manifest.S_N_hex "
                    f"{manifest_S_N[:16]}…")
    except Exception as e:
        report.errors.append(f"terminal S_N recompute raised: {e!r}")
    report.drand_rounds_present = len(unique_drand_rounds)

    # Chain integrity: all three per-row invariants must hold for ALL rows.
    # In --logs-only mode the media hashes are deliberately not exercised,
    # so the verdict is LOGS_ONLY (chain math only), never PASS.
    report.logs_only = logs_only
    if logs_only:
        report.chain_integrity = (
            "LOGS_ONLY" if report.n_rows_chain_advance_ok == report.n_rows
            else "FAIL")
    elif (report.n_rows_chain_advance_ok == report.n_rows and
            report.n_rows_bayer_hash_ok == report.n_rows and
            report.n_rows_emission_png_hash_ok == report.n_rows):
        report.chain_integrity = "PASS"
    else:
        report.chain_integrity = "FAIL"

    # --- Chain-log / capture-log file hashes ---
    try:
        report.chain_log_hash_recomputed = compute_chain_log_hash(
            chain_log_path)
    except Exception as e:
        report.errors.append(f"chain_log_hash recompute failed: {e!r}")
    try:
        report.capture_log_hash_recomputed = compute_capture_log_hash(
            capture_log_path)
    except Exception as e:
        report.errors.append(f"capture_log_hash recompute failed: {e!r}")

    # --- Commitment checks (manifest/bundle/log hashes) ---
    _check_manifest_bundle_hashes(report, manifest, bundle)
    _check_log_hashes(report, manifest, anchor_txs_path)

    # --- Drand availability tier ---
    if report.drand_rounds_present == 0:
        report.drand_availability = "MISSING"
    elif any(int(r["drand_round_number"]) == 0 for r in rows):
        report.drand_availability = "PARTIAL"
    else:
        report.drand_availability = "FULL"

    # --- Pulse commitments ---
    if manifest.get("anchor_log_hash") and not anchor_txs_path.exists():
        report.errors.append(
            "anchor_txs.csv missing but manifest.anchor_log_hash is set "
            "(anchored session): pulse log required")
    anchor_rows = _load_anchor_txs(anchor_txs_path)
    malformed_anchor = sum(1 for r in anchor_rows if "__malformed__" in r)
    pulses = [r for r in anchor_rows if r.get("payload_kind") == "state"]
    report.n_pulses = len(pulses)
    chain_rows_by_t = {int(r["t"]): r for r in rows}
    session_id = manifest["session_id"]
    for pr in pulses:
        rep = _verify_pulse_commitment(pr, session_id, chain_rows_by_t)
        if rep.payload_commitment_verified:
            report.n_pulses_commitment_ok += 1
        report.pulse_reports.append(rep)
    _check_pulse_chain(report, pulses, malformed_anchor_rows=malformed_anchor,
                       final_chain_row=int(rows[-1]["t"]) if rows else None)

    # Cycle closure: requires every commitment surface to hold — all chain
    # integrity, at least one pulse with every commitment verified, the
    # pulse chain continuous, the computed terminal state matching the
    # manifest/anchored S_N, and the manifest/bundle/log hash commitments
    # intact. A vacuous pulse set (n_pulses == 0) can never produce PASS.
    if (report.chain_integrity in ("PASS", "LOGS_ONLY") and
            report.n_pulses > 0 and
            report.n_pulses_commitment_ok == report.n_pulses and
            report.pulse_chain_continuity == "PASS" and
            report.terminal_state_matches_manifest == "PASS" and
            _commitment_checks_ok(report)):
        report.cycle_closure_verdict = "PASS"
    elif (report.chain_integrity == "FAIL" or
            report.terminal_state_matches_manifest == "FAIL" or
            report.pulse_chain_continuity == "FAIL" or
            not _commitment_checks_ok(report) or
            (report.n_pulses > 0 and
             report.n_pulses_commitment_ok != report.n_pulses)):
        report.cycle_closure_verdict = "FAIL"
    else:
        report.cycle_closure_verdict = "INCONCLUSIVE"

    # --- Entropy source vector (always-verified sources) ---
    # Every session contributes local_nonce and the S_0 derivation.
    # Drand/RSK depend on availability + online mode.
    report.entropy_source_vector.append("local_nonce")
    anchor_start = manifest.get("anchor_start")
    if anchor_start:
        report.entropy_source_vector.append("rsk_block_hash")
        # v9 audit-trail check: if pre-wait fields are recorded, verify
        # parent_hash linkage (proves the recorded fresh-block was
        # published AFTER the pre-wait tip was observed) and single-
        # block wait (block_number == pre_wait_tip + 1).
        pre_h = anchor_start.get("pre_wait_tip_block_hash")
        pre_n = anchor_start.get("pre_wait_tip_block_number")
        if pre_h is not None and pre_n is not None:
            if anchor_start.get("parent_hash") != pre_h:
                report.errors.append(
                    f"anchor_start: parent_hash "
                    f"{anchor_start.get('parent_hash', '?')[:16]}… does "
                    f"not match recorded pre_wait_tip_block_hash "
                    f"{pre_h[:16]}…  — fresh-block linkage broken "
                    f"(possible reorg or audit-trail tampering)"
                )
            if anchor_start.get("block_number") != int(pre_n) + 1:
                report.errors.append(
                    f"anchor_start: block_number "
                    f"{anchor_start.get('block_number')} != "
                    f"pre_wait_tip_block_number + 1 "
                    f"({int(pre_n) + 1}) — fresh-block wait did not "
                    f"capture exactly one new block"
                )
            else:
                report.entropy_source_vector.append(
                    "rsk_fresh_block_wait_audited")
    if report.drand_rounds_present > 0:
        report.entropy_source_vector.append("drand_quicknet")

    # --- Online checks ---
    if online:
        rpc_urls = list(manifest.get("rsk_rpc_endpoints", []))
        if rpc_urls and pulses:
            for pulse in report.pulse_reports:
                _check_rsk_inclusion(pulse, rpc_urls)
        drand_chain_hash = manifest.get("drand_chain_hash", "")
        drand_public_key = manifest.get("drand_public_key", "")
        if unique_drand_rounds and drand_chain_hash and drand_public_key:
            _check_drand_rounds(
                sorted(unique_drand_rounds), drand_chain_hash,
                drand_public_key, report)
        else:
            report.drand_rounds_stale = 0  # nothing to check

    return report


# ----------------------------------------------------------------------
# Pretty-printed summary
# ----------------------------------------------------------------------


def format_summary(r: VerifyReport) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append(f"Truth Beam v9 verifier — session {r.session_dir}")
    lines.append("=" * 72)
    lines.append(f"protocol_version: expected {r.protocol_version_expected},  "
                 f"found {r.protocol_version_found}")
    lines.append(f"schema_ok:        {r.schema_ok}")
    if r.errors:
        lines.append("")
        lines.append("top-level errors:")
        for e in r.errors[:10]:
            lines.append(f"  - {e}")
        if len(r.errors) > 10:
            lines.append(f"  ... ({len(r.errors) - 10} more)")
    lines.append("")
    lines.append("SESSION-LEVEL VERDICTS")
    lines.append(f"  chain_integrity:        {r.chain_integrity}")
    lines.append(f"  cycle_closure_verdict:  {r.cycle_closure_verdict}")
    lines.append(f"  drand_availability:     {r.drand_availability}")
    lines.append("")
    lines.append("PER-ROW INVARIANTS")
    lines.append(f"  rows:                        {r.n_rows}")
    lines.append(f"  chain_advance_ok:            "
                 f"{r.n_rows_chain_advance_ok}/{r.n_rows}")
    lines.append(f"  bayer_hash_ok:               "
                 f"{r.n_rows_bayer_hash_ok}/{r.n_rows}")
    lines.append(f"  emission_png_hash_ok:        "
                 f"{r.n_rows_emission_png_hash_ok}/{r.n_rows}")
    if r.terminal_S_N_hex:
        lines.append(f"  terminal S_N (computed):     "
                     f"{r.terminal_S_N_hex[:32]}…")
        lines.append(f"  terminal matches manifest:   "
                     f"{r.terminal_state_matches_manifest}")
    lines.append("")
    lines.append("COMMITMENT CHECKS")
    lines.append(f"  manifest_hash_final:          {r.manifest_hash_final_matches}")
    lines.append(f"  bundle_hash (+linkage):       {r.bundle_hash_matches}")
    lines.append(f"  chain_log_hash vs manifest:   {r.chain_log_hash_matches_manifest}")
    lines.append(f"  capture_log_hash vs manifest: {r.capture_log_hash_matches_manifest}")
    lines.append(f"  anchor_log_hash vs manifest:  {r.anchor_log_hash_matches_manifest}")
    lines.append(f"  final_root row vs anchor_end: {r.anchor_final_row_matches}")
    lines.append(f"  pulse chain continuity:       {r.pulse_chain_continuity}")
    if r.row_errors:
        lines.append("")
        lines.append("row errors (capped at 10):")
        for e in r.row_errors[:10]:
            lines.append(f"  - {e}")
        if len(r.row_errors) > 10:
            lines.append(f"  ... ({len(r.row_errors) - 10} more)")
    lines.append("")
    lines.append("PULSES")
    lines.append(f"  n_pulses:                    {r.n_pulses}")
    lines.append(f"  commitment_verified:         "
                 f"{r.n_pulses_commitment_ok}/{r.n_pulses}")
    if r.pulse_reports:
        inclusion_counts = {}
        for p in r.pulse_reports:
            inclusion_counts[p.rsk_inclusion_status] = \
                inclusion_counts.get(p.rsk_inclusion_status, 0) + 1
        lines.append("  rsk_inclusion_status tiers:")
        for status, count in sorted(inclusion_counts.items()):
            lines.append(f"    {status:<20} {count}")
    lines.append("")
    lines.append("DRAND")
    lines.append(f"  rounds_present:              {r.drand_rounds_present}")
    lines.append(f"  rounds_verified (online):    {r.drand_rounds_verified}")
    lines.append(f"  rounds_failed (online):      {r.drand_rounds_failed_verify}")
    lines.append(f"  staleness_max_ms:            {r.drand_staleness_max_ms}")
    lines.append("")
    lines.append("ENTROPY SOURCES (verified)")
    for src in r.entropy_source_vector:
        lines.append(f"  - {src}")
    lines.append("")
    lines.append("=" * 72)
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session-dir", required=True, type=Path)
    ap.add_argument("--online", action="store_true",
                    help="Perform RSK inclusion + drand BLS re-verification "
                         "via public endpoints. Default: offline.")
    ap.add_argument("--logs-only", action="store_true",
                    help="Verify the chain math from chain_log.csv + "
                         "manifest alone (no raw frames / PNGs needed). "
                         "chain_integrity reports LOGS_ONLY, not PASS.")
    ap.add_argument("--json", action="store_true",
                    help="Emit machine-readable JSON report to stdout "
                         "(in addition to the human-readable summary on "
                         "stderr).")
    args = ap.parse_args(argv)

    try:
        report = verify_session(args.session_dir, online=args.online,
                                logs_only=args.logs_only)
    except Exception as e:
        # Fail closed and informatively: a malformed CSV / JSON (e.g. a
        # non-integer chain row index, or duplicate JSON keys rejected by the
        # strict loader) must yield a clean FAIL, never an uncaught traceback.
        print(f"verification aborted: {e!r}", file=sys.stderr)
        err = {"session_dir": str(args.session_dir),
               "chain_integrity": "FAIL",
               "cycle_closure_verdict": "FAIL",
               "fatal_error": repr(e)}
        if args.json:
            print(json.dumps(err, indent=2))
        return 1
    print(format_summary(report), file=sys.stderr)
    if args.json:
        # JSON report goes to stdout so piping works cleanly.
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        # Without --json, the JSON is written to a file alongside the
        # session so a later reader can consume it.
        out_path = Path(args.session_dir) / "verify_report.json"
        try:
            out_path.write_text(
                json.dumps(report.to_dict(), indent=2, default=str) + "\n")
            print(f"\nwrote {out_path}", file=sys.stderr)
        except Exception as e:
            print(f"\nWARN: could not write {out_path}: {e!r}",
                  file=sys.stderr)

    # Exit code: 0 only when the chain walk passed AND no commitment
    # surface failed (cycle closure carries terminal-state, pulse-chain,
    # and manifest/bundle/log hash failures). INCONCLUSIVE (e.g. a
    # genuinely unanchored session) does not fail the run.
    ok = (report.chain_integrity in ("PASS", "LOGS_ONLY") and
          report.cycle_closure_verdict != "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

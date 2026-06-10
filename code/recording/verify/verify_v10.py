#!/usr/bin/env python3
"""Truth Beam v10 chain verifier — third-party chain walk for TB-v0.10 sessions.

v10 differs from v9 (see verify_v9.py) in exactly two ways:

  * the per-row transition length-prefixes the prior state S_t and folds an
    additional 32-byte `ai_payload_root` field, under domain tag TB:ROW:v10:

        S_{t+1} = blake3(
            b"TB:ROW:v10"
            || len32(S_t)              || S_t                 (32 B)
            || len32(bayer_hash)       || bayer_hash          (32 B)
            || len32(meta_bytes)       || meta_bytes          (28 B)
            || len32(drand_round_be8)  || struct(">Q", round) (8  B)
            || len32(drand_round_value)|| drand_round_value   (32 B)
            || len32(ai_payload_root)  || ai_payload_root     (32 B)
        )

  * `chain_log.csv` gains two columns: `ai_payload_count`,
    `ai_payload_root_hex` (15 columns total). An empty payload list is
    committed as the sentinel root of 32 zero bytes.

Everything else — the S_0 derivation (TB:S0:v1), pulse commitments
(TB:PULSE:v1), drand sentinel semantics, the 28-byte capture-meta struct,
and the on-disk layout — is shared with v9 and reused from verify_v9.py /
session_schema.py.

Checked here: S_0 recompute; per-row v10 chain advance; bayer hash vs raw
frames; emission PNG pixel hash; computed terminal S_N vs manifest.S_N_hex
(the value carried in the final RSK anchor); pulse commitments including
the final-pulse terminal fallback; drand availability; and ai-payload
column sanity (count == 0  ⇒  root == 32 zero bytes).

NOT checked here: recomputing `ai_payload_root` from the payload *files*
(`ai_payloads/included/...`). The writer-side serialization of payload
digests into the root is not part of this code snapshot. The root value
itself is still cryptographically committed — it is folded into every
S_{t+1}, so the chain advance above verifies it row by row.

Usage:
    python3 code/recording/verify/verify_v10.py --session-dir /path/to/session
    python3 code/recording/verify/verify_v10.py --session-dir ... --logs-only

`--logs-only` skips the raw-frame and emission-PNG hash checks so the chain
math (S_0, every transition, terminal S_N, pulse commitments) can be
verified from the ~2 MB chain_log.csv + manifest alone, without the bulk
session data. In that mode `chain_integrity` reports `LOGS_ONLY` rather
than PASS/FAIL, because the media hashes were not exercised.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from blake3 import blake3

HERE = Path(__file__).resolve().parent
PROTOCOL_DIR = HERE.parent / "protocol"
if str(PROTOCOL_DIR) not in sys.path:
    sys.path.insert(0, str(PROTOCOL_DIR))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from session_schema import (  # noqa: E402
    derive_s0, UNANCHORED_BLOCK_HASH_HEX, UNANCHORED_BLOCK_HEIGHT,
    compute_chain_log_hash, compute_capture_log_hash,
)
import verify_v9 as v9  # noqa: E402

PROTOCOL_VERSION_V10 = "TB-v0.10"
ROW_DOMAIN_TAG_V10 = b"TB:ROW:v10"
AI_PAYLOAD_EMPTY_ROOT_HEX = "00" * 32


def compute_chain_row_v10(S_t, bayer_hash, meta_bytes,
                          drand_round_number, drand_round_value,
                          ai_payload_root):
    """v10 chain transition (length-prefixed prior state + ai_payload_root)."""
    for name, val, n in (("S_t", S_t, 32), ("bayer_hash", bayer_hash, 32),
                         ("drand_round_value", drand_round_value, 32),
                         ("ai_payload_root", ai_payload_root, 32)):
        if len(val) != n:
            raise ValueError(f"{name} must be {n} bytes, got {len(val)}")
    drand_round_be8 = int(drand_round_number).to_bytes(8, "big", signed=False)
    h = blake3()
    h.update(ROW_DOMAIN_TAG_V10)
    for fld in (S_t, bayer_hash, meta_bytes, drand_round_be8,
                drand_round_value, ai_payload_root):
        h.update(len(fld).to_bytes(4, "big"))
        h.update(fld)
    return h.digest()


def _v10_row_advance(row: dict) -> str:
    """Recompute the v10 post-row state from one chain_log row's fields."""
    return compute_chain_row_v10(
        bytes.fromhex(row["S_t_hex"]),
        bytes.fromhex(row["bayer_blake3_hex"]),
        bytes.fromhex(row["meta_hex"]),
        int(row["drand_round_number"]),
        bytes.fromhex(row["drand_round_value_hex"]),
        bytes.fromhex(row["ai_payload_root_hex"]),
    ).hex()


@dataclass
class VerifyReportV10(v9.VerifyReport):
    # v10 extras
    n_ai_payload_rows: int = 0          # rows with ai_payload_count > 0
    n_ai_payloads_total: int = 0        # sum of ai_payload_count
    n_rows_ai_sentinel_ok: int = 0      # count==0 rows whose root == sentinel
    n_rows_ai_inconsistent: int = 0     # rows violating the count⇔root rule
    ai_sentinel_check: str = "UNCHECKED"  # PASS | FAIL | UNCHECKED


def verify_session(session_dir: Path, online: bool = False,
                   logs_only: bool = False) -> VerifyReportV10:
    session_dir = Path(session_dir)
    chain_log_path = session_dir / "chain_log.csv"
    capture_log_path = session_dir / "capture_log.csv"
    manifest_path = session_dir / "manifest.json"
    bundle_path = session_dir / "verification_bundle.json"
    anchor_txs_path = session_dir / "anchor_txs.csv"
    recordings_dir = session_dir / "Recordings"

    manifest = v9._load_json(manifest_path)
    bundle = v9._load_json(bundle_path)

    report = VerifyReportV10(
        session_dir=str(session_dir),
        protocol_version_expected=PROTOCOL_VERSION_V10,
        protocol_version_found=(bundle or {}).get(
            "protocol_version", "MISSING"),
        schema_ok=False,
        logs_only=logs_only,
    )
    if manifest is None:
        report.errors.append(f"manifest.json not found at {manifest_path}")
        return report
    if bundle is None:
        report.errors.append(
            f"verification_bundle.json not found at {bundle_path}")
        return report
    if report.protocol_version_found != PROTOCOL_VERSION_V10:
        report.errors.append(
            f"protocol_version mismatch: bundle says "
            f"{report.protocol_version_found!r}, verifier expects "
            f"{PROTOCOL_VERSION_V10!r} (for v9 sessions use verify_v9.py)")
        return report
    report.schema_ok = True

    # --- Recompute S_0 (same TB:S0:v1 derivation as v9) ---
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
                f"manifest.S_0_hex {manifest['S_0_hex'][:16]}…")
            return report
    except Exception as e:
        report.errors.append(f"S_0 recompute raised: {e!r}")
        return report

    # --- Chain log walk ---
    rows = v9._load_chain_log(chain_log_path)
    report.n_rows = len(rows)
    if not rows:
        report.errors.append(
            f"chain_log.csv empty or missing: {chain_log_path}")
        report.chain_integrity = "FAIL"
        return report

    unique_drand_rounds = set()
    prev_row = None
    for row in rows:
        try:
            t = int(row["t"])
        except (KeyError, ValueError) as e:
            # A malformed row index is not advanced, so n_rows_chain_advance_ok
            # falls short of n_rows and chain_integrity is FAIL — no traceback.
            report.row_errors.append(
                f"malformed chain row index 't' ({row.get('t')!r}): {e!r}")
            prev_row = row
            continue
        errors = []

        # Chain advance (v10).
        chain_advance_ok = False
        try:
            if t == 0:
                chain_advance_ok = row["S_t_hex"] == S_0.hex()
                if not chain_advance_ok:
                    errors.append("row 0: S_t_hex != S_0 recomputed")
            else:
                chain_advance_ok = (
                    row["S_t_hex"] == _v10_row_advance(prev_row))
                if not chain_advance_ok:
                    errors.append(
                        f"t={t}: S_t_hex != v10 advance(prev row)")
        except Exception as e:
            errors.append(f"t={t}: chain advance check raised {e!r}")
        if chain_advance_ok:
            report.n_rows_chain_advance_ok += 1

        # AI-payload sanity: BOTH directions of the biconditional
        # (count==0  ⇔  root == 32-zero-byte sentinel) must hold, and the
        # root must be a well-formed 64-hex value. Any violation increments
        # n_rows_ai_inconsistent, which forces ai_sentinel_check = FAIL —
        # row_errors alone gate nothing.
        try:
            count = int(row["ai_payload_count"])
            root_hex = row["ai_payload_root_hex"].lower()
            bad = False
            if len(root_hex) != 64 or any(
                    c not in "0123456789abcdef" for c in root_hex):
                bad = True
                errors.append(
                    f"t={t}: ai_payload_root_hex is not 64 hex chars")
            if count < 0:
                bad = True
                errors.append(
                    f"t={t}: ai_payload_count is negative ({count})")
            elif count > 0:
                report.n_ai_payload_rows += 1
                report.n_ai_payloads_total += count
                if root_hex == AI_PAYLOAD_EMPTY_ROOT_HEX:
                    bad = True
                    errors.append(
                        f"t={t}: ai_payload_count={count} but root is the "
                        f"empty sentinel (empty-commit forgery)")
            else:  # count == 0
                if root_hex == AI_PAYLOAD_EMPTY_ROOT_HEX:
                    report.n_rows_ai_sentinel_ok += 1
                else:
                    bad = True
                    errors.append(
                        f"t={t}: ai_payload_count=0 but root != sentinel")
            if bad:
                report.n_rows_ai_inconsistent += 1
        except Exception as e:
            report.n_rows_ai_inconsistent += 1
            errors.append(f"t={t}: ai-payload sanity check raised {e!r}")

        # Media hashes (skipped in --logs-only mode).
        bayer_hash_ok = False
        emission_png_hash_ok = False
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
            try:
                png_rel = row["emission_png_path"]
                png_path = session_dir / png_rel if png_rel else None
                if png_path and png_path.exists():
                    import numpy as np
                    from PIL import Image
                    img = Image.open(png_path).convert("RGB")
                    arr = np.asarray(img)
                    actual = blake3(
                        bytes(np.ascontiguousarray(arr).tobytes())
                    ).hexdigest()
                    emission_png_hash_ok = (
                        actual == row["emission_live_pixel_blake3_hex"])
                    if not emission_png_hash_ok:
                        errors.append(
                            f"t={t}: emission_png pixel hash mismatch")
                else:
                    errors.append(f"t={t}: missing PNG {png_path}")
            except Exception as e:
                errors.append(f"t={t}: PNG hash check raised {e!r}")
        if bayer_hash_ok:
            report.n_rows_bayer_hash_ok += 1
        if emission_png_hash_ok:
            report.n_rows_emission_png_hash_ok += 1
        if errors:
            report.row_errors.extend(errors[:3])

        try:
            rn = int(row["drand_round_number"])
        except (KeyError, ValueError):
            rn = 0
            report.row_errors.append(
                f"t={t}: malformed drand_round_number "
                f"({row.get('drand_round_number')!r})")
            report.n_rows_ai_inconsistent += 1  # force non-PASS verdict
        if rn != 0:
            unique_drand_rounds.add(rn)
        try:
            sm = int(row["drand_staleness_ms"])
            if sm > report.drand_staleness_max_ms:
                report.drand_staleness_max_ms = sm
        except (KeyError, ValueError):
            pass
        prev_row = row

    report.ai_sentinel_check = (
        "PASS" if (report.n_rows_ai_inconsistent == 0 and
                   report.n_rows_ai_sentinel_ok ==
                   report.n_rows - report.n_ai_payload_rows) else "FAIL")

    # Terminal state: one more v10 advance from the final row, compared to
    # the manifest's S_N (the value carried in the final RSK anchor).
    report.last_logged_S_t_hex = rows[-1]["S_t_hex"]
    try:
        report.terminal_S_N_hex = _v10_row_advance(rows[-1])
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

    # Chain integrity verdict.
    if logs_only:
        report.chain_integrity = (
            "LOGS_ONLY" if report.n_rows_chain_advance_ok == report.n_rows
            and report.ai_sentinel_check == "PASS" else "FAIL")
    elif (report.n_rows_chain_advance_ok == report.n_rows and
            report.n_rows_bayer_hash_ok == report.n_rows and
            report.n_rows_emission_png_hash_ok == report.n_rows and
            report.ai_sentinel_check == "PASS"):
        report.chain_integrity = "PASS"
    else:
        report.chain_integrity = "FAIL"

    # Chain-log / capture-log file hashes.
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

    # Drand availability tier.
    if report.drand_rounds_present == 0:
        report.drand_availability = "MISSING"
    elif any(int(r["drand_round_number"]) == 0 for r in rows):
        report.drand_availability = "PARTIAL"
    else:
        report.drand_availability = "FULL"

    # Pulse commitments (TB:PULSE:v1, shared with v9), with the v10
    # advance supplying the final-pulse terminal fallback.
    if manifest.get("anchor_log_hash") and not anchor_txs_path.exists():
        report.errors.append(
            "anchor_txs.csv missing but manifest.anchor_log_hash is set "
            "(anchored session): pulse log required")
    anchor_rows = v9._load_anchor_txs(anchor_txs_path)
    malformed_anchor = sum(1 for r in anchor_rows if "__malformed__" in r)
    pulses = [r for r in anchor_rows if r.get("payload_kind") == "state"]
    report.n_pulses = len(pulses)
    chain_rows_by_t = {int(r["t"]): r for r in rows}
    for pr in pulses:
        rep = v9._verify_pulse_commitment(
            pr, manifest["session_id"], chain_rows_by_t,
            advance_fn=_v10_row_advance)
        if rep.payload_commitment_verified:
            report.n_pulses_commitment_ok += 1
        report.pulse_reports.append(rep)
    v9._check_pulse_chain(report, pulses,
                          malformed_anchor_rows=malformed_anchor,
                          final_chain_row=int(rows[-1]["t"]) if rows else None)

    # Commitment checks (shared helpers; same semantics as verify_v9).
    v9._check_manifest_bundle_hashes(report, manifest, bundle)
    v9._check_log_hashes(report, manifest, anchor_txs_path)

    chain_ok = report.chain_integrity in ("PASS", "LOGS_ONLY")
    if (chain_ok and report.n_pulses > 0 and
            report.n_pulses_commitment_ok == report.n_pulses and
            report.pulse_chain_continuity == "PASS" and
            report.terminal_state_matches_manifest == "PASS" and
            v9._commitment_checks_ok(report)):
        report.cycle_closure_verdict = "PASS"
    elif (report.chain_integrity == "FAIL" or
            report.terminal_state_matches_manifest == "FAIL" or
            report.pulse_chain_continuity == "FAIL" or
            not v9._commitment_checks_ok(report) or
            (report.n_pulses > 0 and
             report.n_pulses_commitment_ok != report.n_pulses)):
        report.cycle_closure_verdict = "FAIL"
    else:
        report.cycle_closure_verdict = "INCONCLUSIVE"

    # Entropy source vector (same semantics as v9).
    report.entropy_source_vector.append("local_nonce")
    if manifest.get("anchor_start"):
        report.entropy_source_vector.append("rsk_block_hash")
    if report.drand_availability in ("FULL", "PARTIAL"):
        report.entropy_source_vector.append("drand_quicknet")

    if online:
        rpc_urls = manifest.get("rsk_rpc_endpoints") or []
        for pulse in report.pulse_reports:
            v9._check_rsk_inclusion(pulse, rpc_urls)

    return report


def format_summary(r: VerifyReportV10) -> str:
    base = v9.format_summary(r).replace(
        "Truth Beam v9 verifier", "Truth Beam v10 verifier")
    extra = [
        "V10 EXTRAS",
        f"  logs_only mode:              {r.logs_only}",
        f"  ai-payload rows:             {r.n_ai_payload_rows} "
        f"({r.n_ai_payloads_total} payloads)",
        f"  empty-root sentinel check:   {r.ai_sentinel_check}",
        "=" * 72,
    ]
    return base + "\n" + "\n".join(extra)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session-dir", required=True, type=Path)
    ap.add_argument("--online", action="store_true",
                    help="Perform RSK inclusion checks via public "
                         "endpoints. Default: offline.")
    ap.add_argument("--logs-only", action="store_true",
                    help="Verify the chain math from chain_log.csv + "
                         "manifest alone (no raw frames / PNGs needed).")
    ap.add_argument("--json", action="store_true",
                    help="Emit machine-readable JSON report to stdout.")
    args = ap.parse_args(argv)

    try:
        report = verify_session(args.session_dir, online=args.online,
                                logs_only=args.logs_only)
    except Exception as e:
        # Fail closed: a malformed CSV/JSON yields a clean FAIL, not a
        # traceback (and no verify_report.json with partial state).
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
        print(json.dumps(dataclasses.asdict(report), indent=2, default=str))
    else:
        out_path = Path(args.session_dir) / "verify_report.json"
        try:
            out_path.write_text(
                json.dumps(dataclasses.asdict(report), indent=2,
                           default=str) + "\n")
            print(f"\nwrote {out_path}", file=sys.stderr)
        except Exception as e:
            print(f"\nWARN: could not write {out_path}: {e!r}",
                  file=sys.stderr)

    ok = (report.chain_integrity in ("PASS", "LOGS_ONLY") and
          report.cycle_closure_verdict != "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

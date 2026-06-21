#!/usr/bin/env python3
"""golden_diff.py — session-pair differ for refactor validation.

Compares two session directories (one "golden", one "candidate") and
reports every per-row / per-field difference, classified against a
policy that marks each field as MUST_MATCH, WILL_DIFFER, or CONDITIONAL.

Used for Phase-0 vs Phase-A refactor validation (run twice on same
config, same lighting; diff). See GOLDEN_DIFF.md for the field policy
and the reasoning behind each classification.

Usage:
    python3 tools/golden_diff.py <golden_session_dir> <candidate_session_dir>

Exit codes:
    0 — no MUST_MATCH violations
    1 — at least one MUST_MATCH field differs
    2 — one of the sessions is missing a required file
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


# ---- Field policy -----------------------------------------------------

# MUST_MATCH: two recordings made with the same code + same config must
# produce equal values here. A difference indicates a real regression.
CHAIN_LOG_MUST_MATCH = {
    # Row index and column names (schema-level): checked via fieldnames.
    "t",
    # emission_png_path is deterministic from t.
    "emission_png_path",
}

# WILL_DIFFER: values are wall-clock / device-clock / session-id
# dependent and MUST differ across two real recordings. A *match* on
# these would actually be suspicious.
CHAIN_LOG_WILL_DIFFER = {
    "capture_wall_ns",
    "aravis_device_timestamp_ns",
    "tile_queued_wall_ns",
}

# CONDITIONAL: these depend on the chain state, which in turn depends
# on capture_wall_ns + aravis_device_timestamp_ns feeding into meta_bytes
# that gets hashed into S_{t+1}. Therefore from row t >= 1 onward, the
# chain state will *always* differ between two recordings even on
# identical code — NOT because of a regression but because wall-clock
# values differ and the chain is wall-clock-coupled.
#
# For t == 0 specifically, S_t_hex == S_0, which is derived from the
# manifest hash, which itself includes session_iso_utc_start and
# session_id (random UUID) — so S_0 also differs across two recordings.
#
# Upshot: the chain_log golden diff provides *schema-level* evidence
# (column names, row counts, data types) but cannot provide bit-level
# evidence without deterministic timestamp injection. The synthetic-
# input test in ast_equivalence_check.py's companion (not yet scaffolded)
# is the bit-level check.
CHAIN_LOG_CONDITIONAL_ON_TIMESTAMPS = {
    "S_t_hex",
    "bayer_blake3_hex",   # differs with even nanosecond-scale lighting / sensor-noise drift
    "capture_frame_id",   # Aravis counter may or may not reset per session
    "xof_seed_R_hex", "xof_seed_G_hex", "xof_seed_B_hex",
    "tile_pixel_sha_hex",
    "meta_hex",
}

CAPTURE_LOG_MUST_MATCH = {"capture_idx"}
CAPTURE_LOG_WILL_DIFFER = {
    "capture_wall_ns", "aravis_device_timestamp_ns",
}
CAPTURE_LOG_CONDITIONAL_ON_TIMESTAMPS = {
    "capture_frame_id", "bayer_blake3_hex", "raw_path",
    "consumed_into_chain", "consumed_as_t",
}

# manifest.json — nested JSON. Top-level keys the diff classifies.
MANIFEST_MUST_MATCH = {
    # bundle_hash commits to rig config, which should be identical across
    # two runs on the same rig. A mismatch means a code-path difference
    # in how bundle fields are populated.
    "bundle_hash",
}
MANIFEST_WILL_DIFFER = {
    "session_id", "session_iso_utc_start",
    # S_0 depends on manifest_hash_open, which depends on
    # session_iso_utc_start — differs per run.
    "S_0_hex",
    "manifest_hash_open",
    "manifest_hash_final",  # includes anchor + final_root + per-run timing
    "S_N_hex",              # chain-dependent
    "final_root_hex",
    "anchor_start", "anchor_end",  # tx hashes, block hashes differ
    "session_status",              # may legitimately be different (e.g.
                                   # one aborted, one didn't)
}

# verification_bundle.json — top-level keys.
# Everything here SHOULD match bit-for-bit across two runs with same config.
BUNDLE_MUST_MATCH = {
    "protocol_version", "generator_code_hash",
    "wallet_address", "chain_id",
    "generator_config", "camera_config", "projector_config",
    "tile_config", "chain_config", "metadata_schema",
    "host_config",
    "session_mode",
    "rig_pipeline_calibration", "rig_pipeline_calibration_hash",
    "bundle_hash",
}


# ---- Diff helpers ------------------------------------------------------

class Result:
    def __init__(self):
        self.must_match_violations = []
        self.will_differ_satisfied = []
        self.conditional_differences = []
        self.schema_differences = []
        self.missing_files = []

    def report(self) -> int:
        """Print findings. Returns 0 if no MUST_MATCH violation, 1 otherwise."""
        def _h(title): print(f"\n=== {title} ===")

        if self.missing_files:
            _h("MISSING FILES")
            for f in self.missing_files:
                print(f"  {f}")

        if self.schema_differences:
            _h("SCHEMA DIFFERENCES (blocker)")
            for s in self.schema_differences:
                print(f"  {s}")

        if self.must_match_violations:
            _h("MUST_MATCH VIOLATIONS (blocker)")
            for v in self.must_match_violations:
                print(f"  {v}")

        if self.will_differ_satisfied:
            _h("WILL_DIFFER — expected and observed")
            print(f"  {len(self.will_differ_satisfied)} field(s) "
                  f"differed as expected.")

        if self.conditional_differences:
            _h("CONDITIONAL differences (expected unless timestamps were pinned)")
            for c in self.conditional_differences[:10]:
                print(f"  {c}")
            if len(self.conditional_differences) > 10:
                print(f"  … ({len(self.conditional_differences) - 10} more)")

        print()
        if self.must_match_violations or self.schema_differences or self.missing_files:
            print("RESULT: FAIL")
            return 1
        print("RESULT: PASS (MUST_MATCH fields match; WILL_DIFFER differ as expected)")
        return 0


def _load_csv(path: Path):
    with open(path) as f:
        # Accept an optional leading `#` comment line.
        start = f.tell()
        first = f.readline()
        if not first.startswith("#"):
            f.seek(start)
        reader = csv.DictReader(f)
        rows = list(reader)
        return reader.fieldnames or [], rows


def _diff_csv(
    golden: Path, candidate: Path, label: str,
    must_match: set[str], will_differ: set[str],
    conditional: set[str], result: Result,
) -> None:
    if not golden.exists():
        result.missing_files.append(f"golden {label}: {golden}")
        return
    if not candidate.exists():
        result.missing_files.append(f"candidate {label}: {candidate}")
        return

    g_cols, g_rows = _load_csv(golden)
    c_cols, c_rows = _load_csv(candidate)
    if g_cols != c_cols:
        result.schema_differences.append(
            f"{label}: column layout differs\n"
            f"    golden:    {g_cols}\n"
            f"    candidate: {c_cols}"
        )
        return
    if len(g_rows) != len(c_rows):
        result.must_match_violations.append(
            f"{label}: row count differs — golden {len(g_rows)} vs "
            f"candidate {len(c_rows)}"
        )

    n = min(len(g_rows), len(c_rows))
    will_differ_hits = 0
    for i in range(n):
        g, c = g_rows[i], c_rows[i]
        for col in g_cols:
            gv, cv = g.get(col), c.get(col)
            if gv == cv:
                continue
            if col in must_match:
                result.must_match_violations.append(
                    f"{label} row {i} col={col}: "
                    f"golden={gv!r} vs candidate={cv!r}")
            elif col in will_differ:
                will_differ_hits += 1
            elif col in conditional:
                result.conditional_differences.append(
                    f"{label} row {i} col={col}: differs (conditional)")
            else:
                # Unclassified — strict about schema; any new column
                # not in the policy is an audit gap.
                result.schema_differences.append(
                    f"{label} col={col}: differs at row {i} and is not in "
                    f"the diff policy. Update CHAIN_/CAPTURE_/... sets.")
    result.will_differ_satisfied.extend(
        [f"{label}"] * will_differ_hits)


def _diff_json(
    golden: Path, candidate: Path, label: str,
    must_match: set[str], will_differ: set[str],
    result: Result,
) -> None:
    if not golden.exists():
        result.missing_files.append(f"golden {label}: {golden}")
        return
    if not candidate.exists():
        result.missing_files.append(f"candidate {label}: {candidate}")
        return
    g = json.loads(golden.read_text())
    c = json.loads(candidate.read_text())
    for key in sorted(set(g) | set(c)):
        gv, cv = g.get(key, "<<absent>>"), c.get(key, "<<absent>>")
        if gv == cv:
            continue
        if key in must_match:
            result.must_match_violations.append(
                f"{label} key={key}: differs (MUST_MATCH)\n"
                f"    golden:    {gv!r}\n"
                f"    candidate: {cv!r}")
        elif key in will_differ:
            result.will_differ_satisfied.append(f"{label}:{key}")
        else:
            result.schema_differences.append(
                f"{label} key={key}: differs and is not in the diff policy.")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    golden_dir = Path(argv[1])
    cand_dir = Path(argv[2])
    result = Result()

    _diff_csv(
        golden_dir / "chain_log.csv",
        cand_dir / "chain_log.csv",
        "chain_log.csv",
        CHAIN_LOG_MUST_MATCH,
        CHAIN_LOG_WILL_DIFFER,
        CHAIN_LOG_CONDITIONAL_ON_TIMESTAMPS,
        result,
    )
    _diff_csv(
        golden_dir / "capture_log.csv",
        cand_dir / "capture_log.csv",
        "capture_log.csv",
        CAPTURE_LOG_MUST_MATCH,
        CAPTURE_LOG_WILL_DIFFER,
        CAPTURE_LOG_CONDITIONAL_ON_TIMESTAMPS,
        result,
    )
    _diff_json(
        golden_dir / "manifest.json",
        cand_dir / "manifest.json",
        "manifest.json",
        MANIFEST_MUST_MATCH,
        MANIFEST_WILL_DIFFER,
        result,
    )
    _diff_json(
        golden_dir / "verification_bundle.json",
        cand_dir / "verification_bundle.json",
        "verification_bundle.json",
        BUNDLE_MUST_MATCH,
        set(),   # every bundle key is MUST_MATCH
        result,
    )
    return result.report()


if __name__ == "__main__":
    sys.exit(main(sys.argv))

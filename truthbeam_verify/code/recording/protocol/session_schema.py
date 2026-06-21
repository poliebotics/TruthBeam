"""Session artefact schemas for Truth Beam.

Two JSON schemas:

- VerificationBundle: pinned algorithm contract (protocol version + generator
  source hash + generator/camera/projector/chain config + metadata layout).
  bundle_hash binds all of these together. Same algorithm + same hardware ⇒
  identical bundle_hash across sessions.

- SessionManifest: per-session identity (uuid, iso, S_0, session_status).
  Manifest binds to a specific bundle via bundle_hash. S_0 is DERIVED from
  manifest_hash_open in v2 — see derive_s0() — so any of the fields covered
  by manifest_hash_open (session_id, device_id, bundle_hash, iso_start,
  anchor_start) feeds transitively into S_0, and the chain walk catches
  any post-hoc tamper.

  Two hashes on a SessionManifest:
    manifest_hash_open   — computed ONCE at session start over the open-state
                           snapshot. Hardcodes session_status="open" in the
                           hash input, so it never changes as session_status
                           transitions to "finalized" or "aborted". Acts as
                           a permanent commitment to what the session looked
                           like at the moment it opened.
    manifest_hash_final  — computed at finalization over ALL current fields
                           (status="finalized", end fields populated)
                           INCLUDING manifest_hash_open as a pinned reference
                           to the opening state. Cryptographically binds the
                           finalized state to the open state. Absent on
                           open/aborted sessions.

  session_status transitions "open" → "finalized" (by finalizer after a
  successful chain) OR "open" → "aborted" (by abort handler on signal/
  exception). manifest_hash_open is NOT recomputed across transitions —
  it's a snapshot of the opening state. Only manifest_hash_final reflects
  the terminal state, and only if the session was actually finalized.

Canonical JSON form: sort_keys=True, separators=(",", ":"), ensure_ascii=True,
UTF-8 encode. This is the form used for any hash-over-contents field. Pretty-
printed copies are written alongside (.pretty.json) for humans, but the
canonical single-line form is the only thing that's hashed.

manifest_hash_final is EXCLUDED from its own hash input (self-hash is a
paradox); it's computed last and inserted. manifest_hash_open is INCLUDED
in manifest_hash_final's input as an ordinary field — the pinning reference.

NO SIGNATURES. Identity claim ("who produced this") is deliberately not made
for the academic demonstration. Tamper-evidence of post-hoc modification is
handled by the hash chain; bundle↔session binding is handled by deriving S_0
from bundle_hash. That is the entire crypto story at this stage.
"""

import datetime as _dt
import inspect
import json
import os
import re
import struct
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from blake3 import blake3


# v5: hex-string and address validation regexes. All-lowercase or all-mixed
# is permitted on load (canonical writers emit lowercase via .hex()), but
# the format check is case-insensitive.
_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")
_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")


def _atomic_write_bytes(path, data):
    """Durably write bytes to path: tmp file → fsync → rename → fsync dir.

    Used by VerificationBundle.write and SessionManifest.write. Crash-safety
    invariant: at any point, on-disk path is either the previous version or
    the new version, never partial.
    """
    path = Path(path)
    parent = path.parent
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                     dir=str(parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
        # Fsync the parent dir so the rename is durable.
        try:
            dir_fd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            # Some filesystems don't support directory fsync; tolerate.
            pass
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _validate_hex(value, expected_len, field_path):
    """Validate value is a hex string of expected length (chars, not bytes)."""
    if not isinstance(value, str):
        raise ValueError(f"{field_path}: must be a string, got "
                         f"{type(value).__name__}")
    if expected_len is not None and len(value) != expected_len:
        raise ValueError(f"{field_path}: must be {expected_len} hex chars, "
                         f"got {len(value)}")
    if not _HEX_RE.match(value):
        raise ValueError(f"{field_path}: must contain only hex characters")


def _validate_iso8601_utc(value, field_path):
    """Validate value is a UTC ISO-8601 timestamp."""
    if not isinstance(value, str):
        raise ValueError(f"{field_path}: must be a string, got "
                         f"{type(value).__name__}")
    try:
        # Python 3.11+ accepts both '+00:00' and 'Z' suffix.
        parsed = _dt.datetime.fromisoformat(value)
    except (ValueError, TypeError) as e:
        raise ValueError(f"{field_path}: not a valid ISO-8601 timestamp: "
                         f"{value!r}") from e
    if parsed.tzinfo is None:
        raise ValueError(f"{field_path}: missing timezone (must be UTC)")
    offset = parsed.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise ValueError(f"{field_path}: must be UTC (offset = {offset})")


def _validate_chain_id(value, field_path):
    """Validate value is a known RSK chain_id."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_path}: must be an int, got "
                         f"{type(value).__name__}")
    if value not in (30, 31):
        raise ValueError(f"{field_path}: must be 30 (mainnet) or 31 "
                         f"(testnet), got {value}")


def _validate_address(value, field_path, allow_empty=False):
    """Validate value is '0x' + 40 hex chars, or '' if allow_empty=True."""
    if not isinstance(value, str):
        raise ValueError(f"{field_path}: must be a string, got "
                         f"{type(value).__name__}")
    if value == "":
        if allow_empty:
            return
        raise ValueError(f"{field_path}: empty wallet address not allowed here")
    if not _ADDRESS_RE.match(value):
        raise ValueError(f"{field_path}: must be '0x' followed by 40 hex "
                         f"characters")


PROTOCOL_VERSION = "TB-v0.9"

# v9 NOTE ON CRYPTOGRAPHIC CHAIN. The v8 -> v9 change has two parts:
#   (1) an attribution fix -- row t's emission columns commit E_t (the
#       emission live during cap_t), not E_{t+1} (see the block below); and
#   (2) a real cryptographic change -- the per-row chain-advance input now
#       folds in the drand round (number + value), so ROW_DOMAIN_TAG is
#       bumped to b"TB:ROW:v9". An existing v8 chain_log.csv is therefore
#       NOT re-walkable under v9 code, and there are no v8 sessions to
#       preserve (simplified-scope directive).
# The XOF-seed derivation, S_0 derivation, and final_root composition are
# unchanged; their domain tags retain the v8 label deliberately, because
# the tag is part of the hashed input and must not change while the math
# is identical (changing the label would change the hash).

# v8 BREAKING CHANGE: chain advances per CAPTURED FRAME (hash-first pipeline).
# v7.x advanced per projected tile; v8 inverts the dependency.
#
#     v7.x:  S_{t+1} = blake3(S_t || tile_bytes_sha || tile_meta)
#     v8:    S_{t+1} = blake3(b"TB:ROW:v8" || S_t
#                              || len32(bayer_hash) || bayer_hash
#                              || len32(meta_bytes) || meta_bytes)
#
# v9 ATTRIBUTION FIX (2026-04-24): row t's emission-adjacent chain_log
# columns now commit the emission that was LIVE during cap_t (= E_t),
# not the next emission queued for cap_{t+1} (= E_{t+1}). This fixes an
# off-by-one in the column semantics that caused any naïve same-index
# reader (external verifier, training loader) to silently mispair
# every row. The on-disk PNG filename `tile_{t:06d}.png` now contains
# E_t; under v8 it contained E_{t+1}. See bug 2 design space in the
# session notes. The cryptographic chain (S_t, bayer_hash, meta_bytes,
# XOF seeds, final_root) is byte-identical between v8 and v9 for the
# same physical inputs.
#
# Domain tags.
# v9 CRYPTOGRAPHIC NOTE: ROW_DOMAIN_TAG bumps from v8 to v9 because the
# per-row chain-advance input structure is extended to include drand
# round number + value (the unpredictable-at-capture-time entropy the
# rig cannot produce itself). This is a real cryptographic change, not
# just an attribution rename. An existing v8 chain_log.csv is NOT
# rewalkable under v9 code. (There are no v8 sessions to preserve per
# the simplified-scope directive.)
FINAL_ROOT_DOMAIN_TAG = b"TB:FINAL:v8"
ROW_DOMAIN_TAG = b"TB:ROW:v9"  # v9 bump: drand folded into input
# XOF seed derivation and final_root composition are unchanged;
# retaining v8 tags for those.
SEED_DOMAIN_TAG_R = b"TB:SEED:R:v8"
SEED_DOMAIN_TAG_G = b"TB:SEED:G:v8"
SEED_DOMAIN_TAG_B = b"TB:SEED:B:v8"

# v9 LAYERED S_0 derivation. `TB:S0:v1` is a fresh domain tag for the
# new layered-commitment scheme (distinct from the legacy `TB:S0:v8`
# that used manifest_hash_open as the sole input). "v1" = first version
# of the layered S_0 primitive, not a reset of the overall protocol
# version. See derive_s0() below for the full formula.
S0_DOMAIN_TAG = b"TB:S0:v1"


def canonical_json_bytes(obj):
    """Canonical JSON encoding: sort_keys, no whitespace, ASCII-only, UTF-8.

    v3: allow_nan=False. NaN/Infinity are rejected at serialize time; a
    verifier following the schema will never need to reason about them.
    """
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("utf-8")


def canonical_json_hash_hex(obj):
    """blake3 hex digest of canonical JSON encoding."""
    return blake3(canonical_json_bytes(obj)).hexdigest()


def pretty_json_str(obj):
    """Indented JSON for human inspection. NOT used for hashing."""
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False,
                      allow_nan=False)


def _reject_nonfinite(value):
    """parse_constant hook for json.loads: rejects NaN/Infinity at parse time."""
    raise ValueError(f"non-finite JSON value not permitted: {value!r}")


def _strict_json_load(raw_bytes_or_path):
    """Parse JSON, rejecting duplicate keys AND NaN/Infinity. Accepts bytes,
    str, or Path.

    v3: duplicate keys at any nesting level raise ValueError. Python's
    default json.loads silently keeps the last duplicate; we reject to
    prevent an attacker from hiding contradictory values in the same file.

    v6: NaN/Infinity are also rejected at parse time (Python's default
    accepts them). canonical_json_bytes already rejects on serialize via
    allow_nan=False; v6 makes the load path symmetric.
    """
    if hasattr(raw_bytes_or_path, "read_bytes"):
        raw = raw_bytes_or_path.read_bytes()
    elif isinstance(raw_bytes_or_path, (bytes, bytearray)):
        raw = bytes(raw_bytes_or_path)
    else:
        raw = raw_bytes_or_path.encode("utf-8") if isinstance(raw_bytes_or_path, str) else raw_bytes_or_path

    def reject_dupes(pairs):
        seen = set()
        for k, _ in pairs:
            if k in seen:
                raise ValueError(f"duplicate key in JSON object: {k!r}")
            seen.add(k)
        return dict(pairs)

    return json.loads(raw.decode("utf-8"),
                       object_pairs_hook=reject_dupes,
                       parse_constant=_reject_nonfinite)


def hash_generator_source(funcs):
    """blake3 of inspect.getsource() of each callable, concatenated in order.

    Order matters — callers must pass the same list in the same order to get
    a reproducible hash.
    """
    h = blake3()
    for fn in funcs:
        h.update(inspect.getsource(fn).encode("utf-8"))
    return h.hexdigest()


def compute_chain_row(S_t, bayer_hash, meta_bytes,
                      drand_round_number, drand_round_value):
    """v9 chain transition (drand-folded).

        S_{t+1} = blake3(
            b"TB:ROW:v9" || S_t
            || len32(bayer_hash)       || bayer_hash              (32 B)
            || len32(meta_bytes)       || meta_bytes              (28 B)
            || len32(drand_round_be8)  || struct(">Q", round_n)   (8  B)
            || len32(drand_round_value) || drand_round_value      (32 B)
        )

    Inputs:
      - `S_t`:               32 bytes (prior chain state).
      - `bayer_hash`:        32 bytes (blake3 over raw bayer).
      - `meta_bytes`:        28 bytes (big-endian capture meta struct;
                             see chain.META_STRUCT).
      - `drand_round_number`: int, drand quicknet round live at this
                             row. Serialised as 8-byte big-endian u64.
      - `drand_round_value`: 32 bytes, drand round's `randomness`
                             (sha256 of the round's BLS signature).

    The writer MUST have verified the drand signature locally
    (see drand_client.verify_round) before passing the round to
    this function. The chain transition itself does not re-verify —
    verification is the caller's responsibility, because the verifier
    re-runs it on its own copy of the chain_log.

    Returns the 32-byte digest.
    """
    if len(S_t) != 32:
        raise ValueError(f"S_t must be 32 bytes, got {len(S_t)}")
    if len(bayer_hash) != 32:
        raise ValueError(f"bayer_hash must be 32 bytes, got {len(bayer_hash)}")
    if not isinstance(drand_round_value, (bytes, bytearray)) or \
            len(drand_round_value) != 32:
        raise ValueError(
            f"drand_round_value must be 32 bytes, got "
            f"{len(drand_round_value) if isinstance(drand_round_value, (bytes, bytearray)) else type(drand_round_value).__name__}"
        )
    # v9 corroborating-evidence mode: drand_round_number=0 and
    # drand_round_value=zeros are valid sentinels meaning "drand was
    # unavailable at this row." The chain advance still folds them in
    # verbatim (no special casing) — their BLAKE3 input is literally 8
    # zero bytes + 32 zero bytes, which is fine; the verifier notes
    # drand_availability=PARTIAL/MISSING but does not gate PASS/FAIL on it.
    drand_round_be8 = int(drand_round_number).to_bytes(8, "big", signed=False)

    h = blake3()
    h.update(ROW_DOMAIN_TAG)
    h.update(S_t)
    h.update(len(bayer_hash).to_bytes(4, "big"))
    h.update(bayer_hash)
    h.update(len(meta_bytes).to_bytes(4, "big"))
    h.update(meta_bytes)
    h.update(len(drand_round_be8).to_bytes(4, "big"))
    h.update(drand_round_be8)
    h.update(len(drand_round_value).to_bytes(4, "big"))
    h.update(bytes(drand_round_value))
    return h.digest()


def compute_xof_seeds(S_next):
    """v8 XOF seed derivation, domain-separated per channel:

        xof_seed_{R,G,B} = blake3(b"TB:SEED:{R,G,B}:v8" || S_{t+1})

    Returns (seed_r, seed_g, seed_b) as 32-byte bytes.
    """
    if len(S_next) != 32:
        raise ValueError(f"S_next must be 32 bytes, got {len(S_next)}")
    return (
        blake3(SEED_DOMAIN_TAG_R + S_next).digest(),
        blake3(SEED_DOMAIN_TAG_G + S_next).digest(),
        blake3(SEED_DOMAIN_TAG_B + S_next).digest(),
    )


def compute_final_root(S_N_hex, bundle_hash_hex, manifest_hash_open_hex,
                        anchor_log_hash_hex=None):
    """Compute the v8 final_root that the final anchor tx commits to.

    Protocol v8:
        final_root = blake3(
            b"TB:FINAL:v8"
            || bytes.fromhex(S_N_hex)                 # 32
            || bytes.fromhex(bundle_hash_hex)         # 32
            || bytes.fromhex(manifest_hash_open_hex)  # 32
            || bytes.fromhex(anchor_log_hash_hex)     # 32, or bytes(32) if None
        )

    The 32-byte zero fallback is used when no interior-anchor log exists
    (e.g. --interior-anchor was not passed). In practice v4 only computes
    final_root when interior-anchor IS used (the publisher fires the
    final tx), so anchor_log_hash_hex is typically non-None.

    Returns the 32-byte digest (bytes). Caller calls .hex() for the hex str.
    """
    if anchor_log_hash_hex is None:
        anchor_log_bytes = b"\x00" * 32
    else:
        anchor_log_bytes = bytes.fromhex(anchor_log_hash_hex)
        if len(anchor_log_bytes) != 32:
            raise ValueError(
                f"anchor_log_hash_hex must be 32 bytes (64 hex chars); "
                f"got {len(anchor_log_bytes)} bytes"
            )

    h = blake3()
    h.update(FINAL_ROOT_DOMAIN_TAG)
    h.update(bytes.fromhex(S_N_hex))
    h.update(bytes.fromhex(bundle_hash_hex))
    h.update(bytes.fromhex(manifest_hash_open_hex))
    h.update(anchor_log_bytes)
    return h.digest()


def compute_capture_log_hash(csv_path):
    """v0.7: blake3 over the canonical line-serialisation of capture_log.csv.

    Computed at finalize over the file's exact bytes (same approach as
    anchor_log_hash). Pinned by manifest_hash_final so a verifier rejects
    sessions where capture rows were stripped or rewritten without
    touching the chain.

    If the capture_log.csv file does not exist (zero captures arrived,
    e.g. camera fully starved), the digest is over the empty byte string
    so the hash is well-defined and verifier-recomputable.
    """
    csv_path = Path(csv_path)
    if csv_path.exists():
        return blake3(csv_path.read_bytes()).hexdigest()
    return blake3(b"").hexdigest()


def compute_chain_log_hash(csv_path):
    """v0.8: blake3 over the raw bytes of chain_log.csv.

    Unlike the chain walk (which verifies every S_t transition), chain_log_hash
    commits the CSV file as a whole — protecting auxiliary columns and row
    order from tampering that would not alter the per-row chain transitions.
    Pinned by manifest_hash_final.
    """
    csv_path = Path(csv_path)
    if csv_path.exists():
        return blake3(csv_path.read_bytes()).hexdigest()
    return blake3(b"").hexdigest()


def derive_s0(rsk_block_hash_hex, rsk_block_height,
              local_nonce_hex, session_id,
              manifest_hash_open_hex):
    """v9 layered S_0 derivation.

        S_0 = blake3(DS("TB:S0:v1")
                     || len32(rsk_block_hash)   || rsk_block_hash    (32 B)
                     || len32(rsk_block_height) || rsk_block_height  (8 B, u64 BE)
                     || len32(local_nonce)      || local_nonce       (32 B)
                     || len32(session_id_utf8)  || session_id_utf8   (variable)
                     || len32(manifest_hash_open) || manifest_hash_open (32 B))

    Inputs:
      - `rsk_block_hash_hex`: 64-char hex of the RSK fresh-block hash
        committed in `manifest.anchor_start.block_hash`. Pass 32 zero
        bytes hex (`"00" * 32`) for unanchored sessions.
      - `rsk_block_height`: integer block number from
        `manifest.anchor_start.block_number`. Serialized as 8-byte
        big-endian unsigned. Pass 0 for unanchored.
      - `local_nonce_hex`: 64-char hex of 32 random bytes generated
        locally by the rig at session open (`os.urandom(32)`) and
        published in `manifest.local_nonce_hex`. Unpredictable to
        anyone without rig access at session-open time.
      - `session_id`: uuid-style string (e.g. the UUID in
        `manifest.session_id`); hashed as its UTF-8 bytes.
      - `manifest_hash_open_hex`: 64-char hex of 32-byte
        `manifest.manifest_hash_open`. Redundant with the explicit
        fields above (manifest_hash_open already covers session_id,
        anchor_start, local_nonce via _OPEN_SNAPSHOT_FIELDS) — kept as
        a belt-and-braces commitment that ties S_0 to the full open
        state of the manifest, not just the individually enumerated
        fields.

    Each input is length-prefixed (4-byte big-endian) to prevent any
    concatenation ambiguity between adjacent fields.

    Returns the 32-byte S_0 digest (bytes).
    """
    rsk_block_hash = bytes.fromhex(rsk_block_hash_hex)
    if len(rsk_block_hash) != 32:
        raise ValueError(
            f"rsk_block_hash must be 32 bytes (64 hex chars); got "
            f"{len(rsk_block_hash)} bytes"
        )
    rsk_block_height_bytes = int(rsk_block_height).to_bytes(
        8, "big", signed=False)
    local_nonce = bytes.fromhex(local_nonce_hex)
    if len(local_nonce) != 32:
        raise ValueError(
            f"local_nonce must be 32 bytes (64 hex chars); got "
            f"{len(local_nonce)} bytes"
        )
    session_id_bytes = str(session_id).encode("utf-8")
    manifest_hash_open_bytes = bytes.fromhex(manifest_hash_open_hex)
    if len(manifest_hash_open_bytes) != 32:
        raise ValueError(
            f"manifest_hash_open must be 32 bytes (64 hex chars); got "
            f"{len(manifest_hash_open_bytes)} bytes"
        )

    h = blake3()
    h.update(S0_DOMAIN_TAG)
    for segment in (rsk_block_hash,
                    rsk_block_height_bytes,
                    local_nonce,
                    session_id_bytes,
                    manifest_hash_open_bytes):
        h.update(len(segment).to_bytes(4, "big"))
        h.update(segment)
    return h.digest()


# Sentinel for unanchored sessions: rsk_block_hash when anchor_start is None.
UNANCHORED_BLOCK_HASH_HEX = "00" * 32
UNANCHORED_BLOCK_HEIGHT = 0


# v9 pulse commitment-digest (interior-anchor periodic pulses).
#
# `TB:PULSE:v1` is a fresh domain tag for the periodic-pulse commitment
# primitive. Previously (v8) the pulse tx's `data` field was the RAW
# 32-byte S_t; an on-chain observer reading the mempool could see the
# exact chain state, creating an emission-lookahead vulnerability.
# v9 replaces raw-S_t-in-data with a domain-separated commitment that
# binds session_id + pulse_index + frame range + prior-pulse commitment
# + S_t, hiding the raw state behind a BLAKE3 wall while preserving
# downstream verifiability (the verifier recomputes the commitment from
# anchor_txs.csv's recorded fields and compares to on-chain tx data).
PULSE_DOMAIN_TAG = b"TB:PULSE:v1"

# First pulse in a session has no predecessor. Sentinel value.
UNANCHORED_PREV_PULSE_COMMITMENT = b"\x00" * 32


def compute_pulse_commitment(session_id: str, pulse_index: int,
                             frame_start: int, frame_end: int,
                             prev_pulse_commitment: bytes,
                             S_t: bytes) -> bytes:
    """v9 interior-anchor pulse commitment.

        pulse_commitment = blake3(
            DS("TB:PULSE:v1")
            || len32 || session_id_utf8
            || len32 || struct(">I", pulse_index)   (4 B u32 BE)
            || len32 || struct(">I", frame_start)   (4 B u32 BE)
            || len32 || struct(">I", frame_end)     (4 B u32 BE)
            || len32 || prev_pulse_commitment       (32 B; zeros for first pulse)
            || len32 || S_t                         (32 B)
        )

    Returns the 32-byte digest. This is what goes in the pulse tx's
    `data` field and is recorded as `payload_commitment_hex` in
    anchor_txs.csv. The raw `S_t` is ALSO recorded in
    `payload_S_hex` for local verification convenience — the verifier
    recomputes the commitment from S_t and the other fields and
    checks it matches both the CSV record AND the on-chain tx data.

    Inputs:
      - `session_id`: the uuid-style string from manifest.session_id.
        Hashed as UTF-8 bytes.
      - `pulse_index`: 0-based counter; first pulse in a session is 0.
      - `frame_start`, `frame_end`: inclusive range of chain row
        indices this pulse covers. For the first pulse,
        frame_start = 0; frame_end = the last row written before the
        pulse fired. Subsequent pulses: frame_start = (previous
        pulse's frame_end) + 1.
      - `prev_pulse_commitment`: 32 bytes. First pulse:
        UNANCHORED_PREV_PULSE_COMMITMENT (32 zero bytes). Subsequent
        pulses: the previous pulse's 32-byte commitment output.
        Forms a hash chain of pulses across the session.
      - `S_t`: 32 bytes. The chain state committed by this pulse —
        typically S_{last_row+1} = S_{frame_end+1}, i.e. the chain
        state AFTER the last row in the pulse's range was written.
    """
    if not isinstance(prev_pulse_commitment, (bytes, bytearray)):
        raise TypeError(
            f"prev_pulse_commitment must be bytes, got "
            f"{type(prev_pulse_commitment).__name__}"
        )
    if len(prev_pulse_commitment) != 32:
        raise ValueError(
            f"prev_pulse_commitment must be 32 bytes, got "
            f"{len(prev_pulse_commitment)}"
        )
    if not isinstance(S_t, (bytes, bytearray)) or len(S_t) != 32:
        raise ValueError(
            f"S_t must be 32 bytes, got "
            f"{len(S_t) if isinstance(S_t, (bytes, bytearray)) else type(S_t).__name__}"
        )
    if pulse_index < 0 or frame_start < 0 or frame_end < frame_start:
        raise ValueError(
            f"invalid pulse range: pulse_index={pulse_index}, "
            f"frame_start={frame_start}, frame_end={frame_end}"
        )

    sid_bytes = str(session_id).encode("utf-8")
    pi_bytes = struct.pack(">I", int(pulse_index))
    fs_bytes = struct.pack(">I", int(frame_start))
    fe_bytes = struct.pack(">I", int(frame_end))

    h = blake3()
    h.update(PULSE_DOMAIN_TAG)
    for segment in (sid_bytes, pi_bytes, fs_bytes, fe_bytes,
                    bytes(prev_pulse_commitment), bytes(S_t)):
        h.update(len(segment).to_bytes(4, "big"))
        h.update(segment)
    return h.digest()


# ----------------------------------------------------------------------------
# VerificationBundle
# ----------------------------------------------------------------------------

# v3: bundle_hash covers EVERY non-bundle_hash field (matching the README
# wording and fixing the v2 divergence). Closed-schema rejection on load
# (below) keeps unknown fields from sneaking in.
#
# v3 adds chain_id — the EVM chainId (31 for RSK testnet, 30 for RSK mainnet).
# Bundle load refuses to proceed if the live RPC disagrees.
_BUNDLE_KNOWN_FIELDS = frozenset({
    "protocol_version",
    "generator_code_hash",
    "generator_config",
    "camera_config",       # v0.7 role: describes the capture stream, NOT chain input
    "projector_config",
    "tile_config",         # v0.7 NEW: tile output buffer pixel_format/bit_depth/dimensions (reconstructor target in v8)
    "chain_config",
    "metadata_schema",     # v0.8: describes the 28-byte CAPTURE meta struct
    "host_config",
    "wallet_address",
    "chain_id",
    # v8.2 additions — operational polish, not a protocol bump:
    # session_mode selects between synchronous blocking mode (with
    # empirical pipeline-delay wait) and the original v8.0/v8.1 async
    # five-thread pipeline. rig_pipeline_calibration holds the full
    # calibration JSON committed at session start (null for async
    # mode). rig_pipeline_calibration_hash is blake3 over the
    # canonical-JSON bytes of rig_pipeline_calibration — redundant
    # with bundle_hash but explicit so verifiers can cross-check the
    # hash against a standalone calibration file on disk.
    "session_mode",
    "rig_pipeline_calibration",
    "rig_pipeline_calibration_hash",
    "bundle_hash",
})

SESSION_MODE_BLOCKING = "blocking"
SESSION_MODE_ASYNC = "async"
# v8.3: "async-tightened" is operationally async (no calibration required,
# decoupled five-thread pipeline) but with SCHED_FIFO, CPU affinity,
# mlockall, /dev/cpu_dma_latency, GC lockdown, and pre-allocated capture
# ring buffer applied to reduce slip rate. Treated as async-equivalent
# by the calibration-presence validator.
SESSION_MODE_ASYNC_TIGHTENED = "async-tightened"
SESSION_MODE_VALUES = frozenset({
    SESSION_MODE_BLOCKING,
    SESSION_MODE_ASYNC,
    SESSION_MODE_ASYNC_TIGHTENED,
})

# v6: nested schemas declare BOTH allowed AND required keys. Allowed
# rejects unknowns (closed schema); required rejects missing keys
# (well-formed schema). For TB-v0.5 every nested key is both required
# and allowed — there are no optional nested keys. Future protocol
# bumps could split these.
_BUNDLE_NESTED_SCHEMA = {
    "generator_config": frozenset({
        "NUM_OCTAVES", "GRID_H_TABLE", "GRID_W_TABLE",
        "bit_depth", "upsample_method", "persistence_shift_per_octave",
    }),
    # v0.7 ROLE SHIFT: camera_config still pins the capture-stream contract
    # (model, ROI, exposure, gain, trigger, identity) but the captures it
    # describes are NOT chain-committed. Captures land in capture_log.csv
    # in parallel and are committed to manifest_hash_final via
    # capture_log_hash, NOT into the per-tile chain transition.
    "camera_config": frozenset({
        "model", "sensor_size", "pixel_format", "roi",
        "exposure_us", "gain_db", "white_balance_off", "trigger_config",
        "serial_number", "firmware_version", "aravis_version",
    }),
    "projector_config": frozenset({
        "model", "connector", "resolution", "refresh_rate_hz", "color_space",
        "serial_number", "edid_fingerprint",
    }),
    # v0.7 NEW: tile_config pins the chain-committed tile buffer contract.
    # pixel_format identifies the byte layout (e.g. RGB888 — 3 bytes per
    # pixel, R then G then B); bit_depth is per-channel; dimensions is
    # [width, height] and MUST equal projector_config.resolution. The
    # bytes blake3'd into the chain are exactly width*height*bytes_per_pixel
    # bytes long.
    "tile_config": frozenset({
        "pixel_format", "bit_depth", "dimensions",
    }),
    "chain_config": frozenset({"hash", "state_bytes"}),
    # v0.8: metadata_schema describes the 28-byte CAPTURE meta struct
    # (format ">IQQI4s", 4+8+8+4+4 bytes = 28 total; v7.1 had a TILE meta
    # of the same layout and size). fourcc_map enumerates the camera
    # pixel formats this writer can emit.
    "metadata_schema": frozenset({
        "struct_format", "endianness", "total_bytes", "fields", "fourcc_map",
    }),
    "host_config": frozenset({
        "hostname", "os_release", "kernel_version",
        "python_version", "cpu_model",
    }),
}
# Required == allowed for v0.7 (no optional nested keys exist).
_BUNDLE_NESTED_REQUIRED = _BUNDLE_NESTED_SCHEMA

# Doubly-nested: camera_config.trigger_config
_TRIGGER_CONFIG_FIELDS = frozenset({
    "source", "mode", "selector", "activation",
})
_TRIGGER_CONFIG_REQUIRED = _TRIGGER_CONFIG_FIELDS

# metadata_schema.fields is a list of {name, offset, length_bytes, type} dicts.
_METADATA_FIELD_FIELDS = frozenset({
    "name", "offset", "length_bytes", "type",
})
_METADATA_FIELD_REQUIRED = _METADATA_FIELD_FIELDS


def _validate_bundle_nested(d, path):
    """Walk bundle nested dicts against the registry; reject unknowns AND
    require known keys (v6) AND enforce value-level rules."""
    for top_key, allowed in _BUNDLE_NESTED_SCHEMA.items():
        nested = d.get(top_key)
        if not isinstance(nested, dict):
            raise ValueError(
                f"{path}: bundle.{top_key} must be a dict, got "
                f"{type(nested).__name__}"
            )
        unknown = set(nested.keys()) - allowed
        if unknown:
            raise ValueError(
                f"{path}: bundle.{top_key}: unknown nested fields "
                f"{sorted(unknown)} — closed schema for {PROTOCOL_VERSION}"
            )
        # v6: required keys for TB-v0.5.
        required = _BUNDLE_NESTED_REQUIRED[top_key]
        missing = required - set(nested.keys())
        if missing:
            raise ValueError(
                f"{path}: bundle.{top_key}: missing required nested fields "
                f"{sorted(missing)} for {PROTOCOL_VERSION}"
            )
    # camera_config.trigger_config
    tc = d["camera_config"].get("trigger_config")
    if not isinstance(tc, dict):
        raise ValueError(
            f"{path}: camera_config.trigger_config must be a dict"
        )
    unknown = set(tc.keys()) - _TRIGGER_CONFIG_FIELDS
    if unknown:
        raise ValueError(
            f"{path}: camera_config.trigger_config: unknown fields "
            f"{sorted(unknown)}"
        )
    missing = _TRIGGER_CONFIG_REQUIRED - set(tc.keys())
    if missing:
        raise ValueError(
            f"{path}: camera_config.trigger_config: missing fields "
            f"{sorted(missing)}"
        )
    # metadata_schema.fields (list of dicts)
    fields = d["metadata_schema"].get("fields")
    if not isinstance(fields, list):
        raise ValueError(
            f"{path}: metadata_schema.fields must be a list"
        )
    for i, f in enumerate(fields):
        if not isinstance(f, dict):
            raise ValueError(
                f"{path}: metadata_schema.fields[{i}] must be a dict"
            )
        unknown = set(f.keys()) - _METADATA_FIELD_FIELDS
        if unknown:
            raise ValueError(
                f"{path}: metadata_schema.fields[{i}]: unknown fields "
                f"{sorted(unknown)}"
            )
        missing = _METADATA_FIELD_REQUIRED - set(f.keys())
        if missing:
            raise ValueError(
                f"{path}: metadata_schema.fields[{i}]: missing fields "
                f"{sorted(missing)}"
            )

    # v5: value-level checks.
    _validate_chain_id(d.get("chain_id"), f"{path}: bundle.chain_id")
    _validate_hex(d.get("generator_code_hash"), 64,
                  f"{path}: bundle.generator_code_hash")
    _validate_hex(d.get("bundle_hash"), 64, f"{path}: bundle.bundle_hash")
    # wallet_address: structural check only here — it's either '' or a valid
    # 0x-address. Cross-field rule (must be empty when anchor_policy.enabled
    # is false; must be a real address when enabled is true) is enforced in
    # validate_bundle_manifest_pair() at write-time and at verifier-time.
    _validate_address(d.get("wallet_address"),
                       f"{path}: bundle.wallet_address",
                       allow_empty=True)

    # v0.5.1: value-level checks on the new identity fields.
    cam = d["camera_config"]
    for k in ("serial_number", "firmware_version", "aravis_version"):
        v = cam.get(k)
        if v is not None and not isinstance(v, str):
            raise ValueError(
                f"{path}: camera_config.{k} must be a string or null, "
                f"got {type(v).__name__}"
            )
    proj = d["projector_config"]
    if proj.get("serial_number") is not None and not isinstance(proj["serial_number"], str):
        raise ValueError(
            f"{path}: projector_config.serial_number must be a string or null"
        )
    edid = proj.get("edid_fingerprint")
    if edid is not None:
        # blake3 hex (64 chars). Allowed to be null when EDID couldn't be read.
        _validate_hex(edid, 64, f"{path}: projector_config.edid_fingerprint")

    # metadata_schema.fourcc_map: dict[str, str] mapping pixel_format → 4-char fourcc.
    fmap = d["metadata_schema"].get("fourcc_map")
    if not isinstance(fmap, dict):
        raise ValueError(
            f"{path}: metadata_schema.fourcc_map must be a dict, got "
            f"{type(fmap).__name__}"
        )
    for k, v in fmap.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError(
                f"{path}: metadata_schema.fourcc_map keys/values must be "
                f"strings; got {k!r}: {v!r}"
            )
        if len(v) != 4:
            raise ValueError(
                f"{path}: metadata_schema.fourcc_map[{k!r}] must be 4 ASCII "
                f"chars (fourcc); got {v!r} ({len(v)} chars)"
            )
    # v0.8 cross-check: the CAMERA pixel_format MUST be in the fourcc_map.
    # The chain commits bayer_hash over raw camera bytes, and meta_bytes
    # packs the camera fourcc. tile_config remains in the bundle so the
    # reconstructor can re-render tiles from chain_log.xof_seed_* — but the
    # chain transition no longer depends on tile bytes.
    cpf = cam.get("pixel_format")
    if not isinstance(cpf, str) or not cpf:
        raise ValueError(
            f"{path}: camera_config.pixel_format must be a non-empty string, "
            f"got {cpf!r}"
        )
    if cpf not in fmap:
        raise ValueError(
            f"{path}: camera_config.pixel_format {cpf!r} not present in "
            f"metadata_schema.fourcc_map (keys: {sorted(fmap.keys())})"
        )
    tc = d["tile_config"]
    if not isinstance(tc, dict):
        raise ValueError(f"{path}: tile_config must be a dict")
    tpf = tc.get("pixel_format")
    if not isinstance(tpf, str) or not tpf:
        raise ValueError(
            f"{path}: tile_config.pixel_format must be a non-empty string, "
            f"got {tpf!r}"
        )
    tbd = tc.get("bit_depth")
    if not isinstance(tbd, int) or isinstance(tbd, bool) or tbd <= 0:
        raise ValueError(
            f"{path}: tile_config.bit_depth must be positive int, got {tbd!r}"
        )
    tdims = tc.get("dimensions")
    if not (isinstance(tdims, list) and len(tdims) == 2
            and all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in tdims)):
        raise ValueError(
            f"{path}: tile_config.dimensions must be [width, height] of "
            f"positive ints, got {tdims!r}"
        )
    # Cross-check: tile dimensions must match projector resolution. The
    # tile is what gets drawn to the projector surface; if these disagree
    # the bundle's claim is internally inconsistent.
    pres = proj.get("resolution")
    if pres != tdims:
        raise ValueError(
            f"{path}: tile_config.dimensions {tdims!r} != "
            f"projector_config.resolution {pres!r} — tile MUST match the "
            f"surface it's drawn to"
        )

    # host_config: required dict (already covered by NESTED_SCHEMA closed-set
    # check above). All values must be strings (no nulls — host fields are
    # always queryable).
    hc = d["host_config"]
    for k in _BUNDLE_NESTED_SCHEMA["host_config"]:
        v = hc.get(k)
        if not isinstance(v, str) or not v:
            raise ValueError(
                f"{path}: host_config.{k} must be a non-empty string, got "
                f"{v!r}"
            )

    # v8.2: session_mode + rig_pipeline_calibration consistency.
    mode = d.get("session_mode")
    if mode not in SESSION_MODE_VALUES:
        raise ValueError(
            f"{path}: session_mode must be one of "
            f"{sorted(SESSION_MODE_VALUES)}, got {mode!r}"
        )
    cal = d.get("rig_pipeline_calibration")
    cal_hash = d.get("rig_pipeline_calibration_hash")
    if mode == SESSION_MODE_BLOCKING:
        if not isinstance(cal, dict):
            raise ValueError(
                f"{path}: session_mode=blocking requires "
                f"rig_pipeline_calibration to be a dict, got "
                f"{type(cal).__name__}"
            )
        if "recommended_wait_ms" not in cal or "rig_hash" not in cal:
            raise ValueError(
                f"{path}: rig_pipeline_calibration missing required keys "
                f"'recommended_wait_ms' and/or 'rig_hash'"
            )
        rw = cal["recommended_wait_ms"]
        if not isinstance(rw, int) or isinstance(rw, bool) or rw <= 0:
            raise ValueError(
                f"{path}: rig_pipeline_calibration.recommended_wait_ms "
                f"must be positive int, got {rw!r}"
            )
        if not isinstance(cal_hash, str):
            raise ValueError(
                f"{path}: rig_pipeline_calibration_hash must be str"
            )
        _validate_hex(cal_hash, 64,
                      f"{path}: rig_pipeline_calibration_hash")
        recomputed = blake3(canonical_json_bytes(cal)).hexdigest()
        if recomputed != cal_hash:
            raise ValueError(
                f"{path}: rig_pipeline_calibration_hash mismatch "
                f"(claimed {cal_hash[:16]}…, recomputed "
                f"{recomputed[:16]}…)"
            )
    else:  # async or async-tightened
        if cal is not None:
            raise ValueError(
                f"{path}: session_mode={mode} requires "
                f"rig_pipeline_calibration to be null"
            )
        if cal_hash != "":
            raise ValueError(
                f"{path}: session_mode={mode} requires "
                f"rig_pipeline_calibration_hash to be empty string, "
                f"got {cal_hash!r}"
            )


def validate_bundle_manifest_pair(bundle_d, manifest_d, path=""):
    """v5 cross-field rule: bundle.wallet_address ↔ manifest.anchor_policy.enabled.

    Called by:
      - tb_loop AFTER both bundle and manifest are constructed
      - any verifier that has loaded both files
    """
    enabled = manifest_d.get("anchor_policy", {}).get("enabled")
    addr = bundle_d.get("wallet_address")
    if enabled is True:
        if addr == "" or not _ADDRESS_RE.match(addr or ""):
            raise ValueError(
                f"{path}: bundle.wallet_address must be a valid '0x'+40-hex "
                f"address when anchor_policy.enabled=true; got {addr!r}"
            )
    elif enabled is False:
        if addr != "":
            raise ValueError(
                f"{path}: bundle.wallet_address must be the empty string "
                f"when anchor_policy.enabled=false; got {addr!r}"
            )
    # If enabled is something else, _validate_manifest_nested has already
    # rejected it.


@dataclass
class VerificationBundle:
    protocol_version: str
    generator_code_hash: str
    generator_config: dict
    camera_config: dict
    projector_config: dict
    tile_config: dict      # tile output buffer format (reconstructor target in v8)
    chain_config: dict
    metadata_schema: dict  # v0.8: 28-byte CAPTURE meta struct (layout in tb_loop.META_STRUCT)
    host_config: dict
    wallet_address: str  # 0x-prefixed RSK/EVM address; pins the anchor tx sender
    chain_id: int        # EVM chainId; 31=testnet, 30=mainnet
    # v8.2 — blocking mode selector + pipeline delay calibration.
    # session_mode: "blocking" (default) or "async".
    # rig_pipeline_calibration: full calibration JSON dict (blocking)
    # or None (async). rig_pipeline_calibration_hash: 64-hex blake3 of
    # canonical_json_bytes(rig_pipeline_calibration) (blocking) or ""
    # (async). Default keeps the older v8.0/v8.1 behaviour if callers
    # omit these — but the bundle writer in tb_loop.py always sets
    # them explicitly.
    session_mode: str = "blocking"
    rig_pipeline_calibration: dict | None = None
    rig_pipeline_calibration_hash: str = ""
    bundle_hash: str = ""  # populated by finalize()

    def compute_bundle_hash(self):
        # v3: hash every field except bundle_hash itself. Combined with
        # closed-schema rejection on load, this makes the bundle airtight
        # against both extra-field injection and extra-field omission.
        d = asdict(self)
        d.pop("bundle_hash", None)
        return canonical_json_hash_hex(d)

    def finalize(self):
        self.bundle_hash = self.compute_bundle_hash()
        return self.bundle_hash

    def to_dict(self):
        return asdict(self)

    def validate(self):
        """v6: run reader-equivalent validation BEFORE writing. The writer
        and the reader use the SAME validator — no path can produce an
        artefact load_and_verify() would reject."""
        VerificationBundle.validate_dict(self.to_dict(),
                                          path="<unwritten bundle>")

    @staticmethod
    def validate_dict(d, path):
        """v6: validate a bundle dict against the v0.5 schema. Raises
        ValueError with a clear message on any violation. Called by both
        VerificationBundle.write() and load_and_verify()."""
        # Top-level closed schema.
        unknown = set(d.keys()) - _BUNDLE_KNOWN_FIELDS
        if unknown:
            raise ValueError(
                f"{path}: unknown bundle fields {sorted(unknown)} — "
                f"closed schema for protocol_version {PROTOCOL_VERSION}"
            )
        required = _BUNDLE_KNOWN_FIELDS - {"bundle_hash"}
        missing = [f for f in required if f not in d]
        if missing:
            raise ValueError(
                f"{path}: missing required bundle fields: {missing}"
            )
        # protocol_version exact match.
        if d.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError(
                f"{path}: unsupported protocol_version "
                f"{d.get('protocol_version')!r} — this loader only handles "
                f"{PROTOCOL_VERSION}"
            )
        # Nested schema + value-level.
        _validate_bundle_nested(d, path)

    def write(self, path):
        if not self.bundle_hash:
            raise RuntimeError("VerificationBundle.write() before finalize()")
        # v6: validate BEFORE writing. If validation fails, the write is
        # refused and on-disk state is not touched.
        try:
            self.validate()
        except ValueError as e:
            raise ValueError(
                f"bundle fails schema validation before write: {e}"
            ) from e
        path = Path(path)
        d = self.to_dict()
        # v5: atomic write (tmp → fsync → rename → dir fsync).
        _atomic_write_bytes(path, canonical_json_bytes(d))
        pretty_path = path.parent / (path.stem + ".pretty.json")
        _atomic_write_bytes(pretty_path,
                             (pretty_json_str(d) + "\n").encode("utf-8"))

    @staticmethod
    def load_and_verify(path):
        """Load JSON (strict), verify closed schema, recompute bundle_hash.

        v3: rejects unknown fields (closed schema per protocol_version).
        Rejects duplicate JSON keys via object_pairs_hook. Hashes every
        non-bundle_hash field.

        Returns (dict, bundle_hash_hex). Raises ValueError on any failure.
        """
        path = Path(path)
        d = _strict_json_load(path)
        # v6: full schema validation via validate_dict. This is the SAME
        # validator the writer calls before write() — reader/writer symmetry.
        VerificationBundle.validate_dict(d, str(path))
        if "bundle_hash" not in d:
            raise ValueError(f"{path}: missing 'bundle_hash' field")
        claimed = d["bundle_hash"]
        # Hash every non-bundle_hash field.
        hash_input = {k: v for k, v in d.items() if k != "bundle_hash"}
        recomputed = canonical_json_hash_hex(hash_input)
        if claimed != recomputed:
            raise ValueError(
                f"{path}: bundle_hash mismatch — "
                f"claimed {claimed[:16]}…, recomputed {recomputed[:16]}…"
            )
        return d, claimed


# ----------------------------------------------------------------------------
# SessionManifest
# ----------------------------------------------------------------------------

# Allowed values for SessionManifest.session_status.
SESSION_STATUS_OPEN = "open"            # written at session start
SESSION_STATUS_FINALIZED = "finalized"  # set by finalizer after chain completes
SESSION_STATUS_ABORTED = "aborted"      # generic abort (signal during chain, etc)
# v5: more granular abort reasons surface the failure mode in the manifest
# itself rather than only in stdout logs. All six are valid session_status
# values; verifiers MUST accept any.
SESSION_STATUS_ABORTED_PUBLISHER_STUCK = "aborted_publisher_stuck"
SESSION_STATUS_ABORTED_SETUP_FAILED = "aborted_setup_failed"
SESSION_STATUS_ABORTED_SIGNAL_DURING_FINALIZE = "aborted_signal_during_finalize"
SESSION_STATUS_VALUES = frozenset({
    SESSION_STATUS_OPEN,
    SESSION_STATUS_FINALIZED,
    SESSION_STATUS_ABORTED,
    SESSION_STATUS_ABORTED_PUBLISHER_STUCK,
    SESSION_STATUS_ABORTED_SETUP_FAILED,
    SESSION_STATUS_ABORTED_SIGNAL_DURING_FINALIZE,
})
SESSION_STATUS_ABORT_REASONS = frozenset({
    SESSION_STATUS_ABORTED,
    SESSION_STATUS_ABORTED_PUBLISHER_STUCK,
    SESSION_STATUS_ABORTED_SETUP_FAILED,
    SESSION_STATUS_ABORTED_SIGNAL_DURING_FINALIZE,
})

# Required fields (always present in JSON on a fresh-schema session).
# v3 adds anchor_policy to the required set — even disabled, it's recorded.
# v9 adds local_nonce_hex: a 32-byte CSPRNG value generated locally at
# session open, published in manifest, hashed into manifest_hash_open and
# explicitly re-committed inside derive_s0. Raises S_0 unpredictability
# above what a public RSK block hash provides.
_MANIFEST_REQUIRED_FIELDS = (
    "session_id",
    "session_iso_utc_start",
    "bundle_hash",
    "S_0_hex",
    "device_id",
    "session_status",
    "anchor_policy",
    "manifest_hash_open",
    "local_nonce_hex",
    "drand_chain_hash",
    "drand_public_key",
    "drand_round_at_session_open",
)

# v0.8 manifest schema. Renamed from v0.7:
#   N_tiles → N_chain       (chain rows are captured frames consumed now;
#                             "N_tiles" no longer matches the semantics)
# Added:
#   chain_log_hash    — blake3 over chain_log.csv raw bytes. The chain walk
#                        already verifies each S_t transition; chain_log_hash
#                        commits the file as a whole so auxiliary columns
#                        and row order are also pinned.
# Retained:
#   N_captures, capture_log_hash — every capture delivered (M captures),
#                                   M == N_chain in normal operation.
_MANIFEST_KNOWN_FIELDS = frozenset({
    # Always present:
    "session_id", "session_iso_utc_start", "bundle_hash", "S_0_hex",
    "device_id", "session_status", "anchor_policy", "manifest_hash_open",
    "local_nonce_hex",              # v9: 32-byte CSPRNG at session open
    "drand_chain_hash",             # v9: drand chain pinned at session open
    "drand_public_key",             # v9: drand G2 pubkey pinned at session open
    "drand_round_at_session_open",  # v9: session-level drand anchor (int)
    "rsk_rpc_endpoints",            # v9: list of RPC URLs (primary + fallbacks)
    # Optional start-time:
    "anchor_start",
    # End-of-session:
    "session_iso_utc_end", "N_chain", "N_captures", "S_N_hex",
    "anchor_end", "anchor_log_hash",
    "capture_log_hash", "chain_log_hash",
    "camera_acquisition_start_wall_ns", "camera_acquisition_end_wall_ns",
    # v8.3 informational: runtime mitigations the async-tightened loop
    # was able to apply in the current environment (e.g. ["sched_fifo",
    # "mlockall", "gc_lockdown"]). Pinned by manifest_hash_final.
    # Present only on async-tightened sessions; null for others.
    "mitigations_applied",
    "manifest_hash_final",
})

# v4: nested schemas for manifest dict fields.
# v9 (audit-trail addition, 2026-04-25): pre_wait_tip_block_number /
# pre_wait_tip_block_hash / wait_duration_s explicitly record the
# fresh-block-wait audit trail. Verifier can check that
# parent_hash == pre_wait_tip_block_hash AND
# block_number == pre_wait_tip_block_number + 1 to prove the recorded
# fresh-block was published AFTER the pre-wait tip was observed.
# All three fields are optional (None) when --rsk-rpc was not used.
_ANCHOR_START_FIELDS = frozenset({
    "chain", "chain_id",
    "block_number", "block_hash", "block_timestamp_utc", "parent_hash",
    "observed_at_utc",  # host-reported; NOT chain-verifiable
    "pre_wait_tip_block_number",   # v9: int, the tip seen at session-start
    "pre_wait_tip_block_hash",     # v9: 64-hex, the tip's hash before waiting
    "wait_duration_s",             # v9: float, seconds spent waiting
})

# anchor_end expanded in v4 to include block info and both payload hashes.
_ANCHOR_END_FIELDS = frozenset({
    "chain", "chain_id", "tx_hash",
    "payload_S_N_hex",            # the terminal chain state
    "payload_final_root_hex",      # what's actually in tx.input
    "block_number", "block_hash", "block_timestamp_utc",
})

_ANCHOR_POLICY_ENABLED_FIELDS = frozenset({
    "enabled", "interval_s", "chain_id", "network",
})


def _validate_manifest_nested(d, path):
    """Walk manifest nested dicts against the registry; reject unknowns and
    enforce value-level rules + cross-field wallet/policy rule (v5)."""

    # anchor_policy: required. Schema depends on enabled flag.
    ap = d.get("anchor_policy")
    if not isinstance(ap, dict):
        raise ValueError(f"{path}: anchor_policy must be a dict")
    if "enabled" not in ap:
        raise ValueError(f"{path}: anchor_policy missing 'enabled' key")
    if not isinstance(ap["enabled"], bool):  # bool is int subclass; explicit type check
        raise ValueError(
            f"{path}: anchor_policy.enabled must be bool, got "
            f"{type(ap['enabled']).__name__}"
        )
    if ap["enabled"] is False:
        unknown = set(ap.keys()) - {"enabled"}
        if unknown:
            raise ValueError(
                f"{path}: anchor_policy (disabled): only 'enabled' allowed; "
                f"got extras {sorted(unknown)}"
            )
    else:
        required = _ANCHOR_POLICY_ENABLED_FIELDS
        missing = required - set(ap.keys())
        if missing:
            raise ValueError(
                f"{path}: anchor_policy (enabled): missing {sorted(missing)}"
            )
        unknown = set(ap.keys()) - required
        if unknown:
            raise ValueError(
                f"{path}: anchor_policy (enabled): unknown {sorted(unknown)}"
            )
        # v5 value-level: chain_id, interval_s, network.
        _validate_chain_id(ap["chain_id"], f"{path}: anchor_policy.chain_id")
        iv = ap["interval_s"]
        if not isinstance(iv, (int, float)) or isinstance(iv, bool):
            raise ValueError(
                f"{path}: anchor_policy.interval_s must be a number, got "
                f"{type(iv).__name__}"
            )
        if not (1.0 <= float(iv) <= 60.0):
            raise ValueError(
                f"{path}: anchor_policy.interval_s must be in [1.0, 60.0], "
                f"got {iv}"
            )
        if ap["network"] not in ("testnet", "mainnet"):
            raise ValueError(
                f"{path}: anchor_policy.network must be 'testnet' or "
                f"'mainnet', got {ap['network']!r}"
            )

    # anchor_start: optional but if present must conform.
    if "anchor_start" in d and d["anchor_start"] is not None:
        nested = d["anchor_start"]
        if not isinstance(nested, dict):
            raise ValueError(f"{path}: anchor_start must be a dict")
        unknown = set(nested.keys()) - _ANCHOR_START_FIELDS
        if unknown:
            raise ValueError(
                f"{path}: anchor_start: unknown fields {sorted(unknown)}"
            )
        # v5 value-level on anchor_start.
        _validate_chain_id(nested.get("chain_id"),
                            f"{path}: anchor_start.chain_id")
        bn = nested.get("block_number")
        if not isinstance(bn, int) or isinstance(bn, bool) or bn < 0:
            raise ValueError(
                f"{path}: anchor_start.block_number must be a non-negative "
                f"int, got {bn!r}"
            )
        _validate_hex(nested.get("block_hash"), 64,
                      f"{path}: anchor_start.block_hash")
        _validate_hex(nested.get("parent_hash"), 64,
                      f"{path}: anchor_start.parent_hash")
        _validate_iso8601_utc(nested.get("block_timestamp_utc"),
                              f"{path}: anchor_start.block_timestamp_utc")
        _validate_iso8601_utc(nested.get("observed_at_utc"),
                              f"{path}: anchor_start.observed_at_utc")
        # v9 audit trail: optional pre-wait tip + wait duration. If any
        # of the three is present, all three must be present and well-
        # formed (atomic record).
        ats_audit_keys = ("pre_wait_tip_block_number",
                          "pre_wait_tip_block_hash",
                          "wait_duration_s")
        present = [k for k in ats_audit_keys if k in nested]
        if present:
            missing = [k for k in ats_audit_keys if k not in nested]
            if missing:
                raise ValueError(
                    f"{path}: anchor_start audit-trail fields are "
                    f"all-or-nothing; got {present} but missing {missing}"
                )
            pre_bn = nested["pre_wait_tip_block_number"]
            if not isinstance(pre_bn, int) or isinstance(pre_bn, bool) or pre_bn < 0:
                raise ValueError(
                    f"{path}: anchor_start.pre_wait_tip_block_number "
                    f"must be non-negative int, got {pre_bn!r}"
                )
            _validate_hex(nested["pre_wait_tip_block_hash"], 64,
                          f"{path}: anchor_start.pre_wait_tip_block_hash")
            wd = nested["wait_duration_s"]
            if not isinstance(wd, (int, float)) or isinstance(wd, bool) or wd < 0:
                raise ValueError(
                    f"{path}: anchor_start.wait_duration_s must be "
                    f"non-negative number, got {wd!r}"
                )

    # anchor_end: optional but if present must conform completely.
    if "anchor_end" in d and d["anchor_end"] is not None:
        nested = d["anchor_end"]
        if not isinstance(nested, dict):
            raise ValueError(f"{path}: anchor_end must be a dict")
        unknown = set(nested.keys()) - _ANCHOR_END_FIELDS
        if unknown:
            raise ValueError(
                f"{path}: anchor_end: unknown fields {sorted(unknown)}"
            )
        # v5: anchor_end is "all-or-nothing" — every field MUST be populated.
        # No null block_hash / null block_timestamp_utc allowed.
        missing = _ANCHOR_END_FIELDS - set(nested.keys())
        if missing:
            raise ValueError(
                f"{path}: anchor_end missing required fields {sorted(missing)}"
            )
        for k, v in nested.items():
            if v is None:
                raise ValueError(
                    f"{path}: anchor_end.{k} must not be null (v5 "
                    f"requires fully-populated anchor_end OR omitted entirely)"
                )
        _validate_chain_id(nested["chain_id"],
                            f"{path}: anchor_end.chain_id")
        _validate_hex(nested["payload_S_N_hex"], 64,
                      f"{path}: anchor_end.payload_S_N_hex")
        _validate_hex(nested["payload_final_root_hex"], 64,
                      f"{path}: anchor_end.payload_final_root_hex")
        _validate_iso8601_utc(nested["block_timestamp_utc"],
                              f"{path}: anchor_end.block_timestamp_utc")
        bn = nested["block_number"]
        if not isinstance(bn, int) or isinstance(bn, bool) or bn < 0:
            raise ValueError(
                f"{path}: anchor_end.block_number must be non-negative int"
            )

    # v5: top-level value-level checks.
    _validate_iso8601_utc(d["session_iso_utc_start"],
                          f"{path}: session_iso_utc_start")
    if "session_iso_utc_end" in d and d["session_iso_utc_end"] is not None:
        _validate_iso8601_utc(d["session_iso_utc_end"],
                              f"{path}: session_iso_utc_end")
    _validate_hex(d["bundle_hash"], 64, f"{path}: bundle_hash")
    _validate_hex(d["S_0_hex"], 64, f"{path}: S_0_hex")
    _validate_hex(d["manifest_hash_open"], 64, f"{path}: manifest_hash_open")
    _validate_hex(d["local_nonce_hex"], 64, f"{path}: local_nonce_hex")
    # v9: drand anchor fields. In corroborating-evidence mode drand
    # may be unavailable at session open; the writer records chain_hash
    # and public_key as the PLANNED beacon (from config), and
    # drand_round_at_session_open as 0 if no round was verified by
    # start-of-capture. Chain_hash / public_key are still required to be
    # well-formed hex so the verifier knows which beacon to cross-check.
    _validate_hex(d["drand_chain_hash"], 64, f"{path}: drand_chain_hash")
    _validate_hex(d["drand_public_key"], 192, f"{path}: drand_public_key")
    dro = d["drand_round_at_session_open"]
    if not isinstance(dro, int) or isinstance(dro, bool) or dro < 0:
        raise ValueError(
            f"{path}: drand_round_at_session_open must be non-negative "
            f"int (0 sentinel = drand unavailable at session open), "
            f"got {dro!r}"
        )
    # v9: rsk_rpc_endpoints is a list of strings (URLs). Empty list is
    # valid for unanchored sessions.
    rpc_eps = d.get("rsk_rpc_endpoints", [])
    if not isinstance(rpc_eps, list):
        raise ValueError(
            f"{path}: rsk_rpc_endpoints must be a list, got "
            f"{type(rpc_eps).__name__}"
        )
    for i, u in enumerate(rpc_eps):
        if not isinstance(u, str) or not u:
            raise ValueError(
                f"{path}: rsk_rpc_endpoints[{i}] must be a non-empty "
                f"string, got {u!r}"
            )
    if d.get("S_N_hex") is not None:
        _validate_hex(d["S_N_hex"], 64, f"{path}: S_N_hex")
    if d.get("anchor_log_hash") is not None:
        _validate_hex(d["anchor_log_hash"], 64, f"{path}: anchor_log_hash")
    if d.get("capture_log_hash") is not None:
        _validate_hex(d["capture_log_hash"], 64, f"{path}: capture_log_hash")
    if d.get("chain_log_hash") is not None:
        _validate_hex(d["chain_log_hash"], 64, f"{path}: chain_log_hash")
    # v8.3: mitigations_applied is a list of short strings if present.
    if d.get("mitigations_applied") is not None:
        mits = d["mitigations_applied"]
        if not isinstance(mits, list):
            raise ValueError(
                f"{path}: mitigations_applied must be a list, got "
                f"{type(mits).__name__}"
            )
        for i, m in enumerate(mits):
            if not isinstance(m, str) or not m:
                raise ValueError(
                    f"{path}: mitigations_applied[{i}] must be a "
                    f"non-empty string, got {m!r}"
                )
    if d.get("manifest_hash_final") is not None:
        _validate_hex(d["manifest_hash_final"], 64,
                      f"{path}: manifest_hash_final")
    # v0.5.1: camera_acquisition_*_wall_ns are non-negative ints if present.
    for k in ("camera_acquisition_start_wall_ns",
              "camera_acquisition_end_wall_ns"):
        if k in d and d[k] is not None:
            v = d[k]
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise ValueError(
                    f"{path}: {k} must be non-negative int, got {v!r}"
                )
    # v0.8: N_chain must be positive int when present (chain has at least
    # one row to be finalized). N_captures must be non-negative — in normal
    # operation N_captures == N_chain.
    if d.get("N_chain") is not None:
        v = d["N_chain"]
        if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
            raise ValueError(f"{path}: N_chain must be positive int, got {v!r}")
    if d.get("N_captures") is not None:
        v = d["N_captures"]
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            raise ValueError(f"{path}: N_captures must be non-negative int, got {v!r}")

# Fields hashed into manifest_hash_open. S_0_hex NOT in this set (S_0 is
# derived from manifest_hash_open; including it would be circular).
# session_status is always hardcoded to "open" in the hash input.
# v3 adds anchor_policy so the anchor configuration is pinned from session start.
_OPEN_SNAPSHOT_FIELDS = (
    "session_id",
    "session_iso_utc_start",
    "bundle_hash",
    "device_id",
    "anchor_policy",
    # session_status handled specially — see _compute_open_hash_input.
    "anchor_start",                  # optional; stripped if None.
    "local_nonce_hex",               # v9: always present; part of opening snapshot.
    "drand_chain_hash",              # v9: session-level drand anchor
    "drand_public_key",              # v9: pinned drand pubkey
    "drand_round_at_session_open",   # v9: drand round live at session open
    "rsk_rpc_endpoints",             # v9: RPC URL list committed pre-session
)

# End-of-session fields. All required on a finalized manifest.
# v0.8: N_tiles → N_chain; chain_log_hash added.
_MANIFEST_END_FIELDS = (
    "session_iso_utc_end",
    "N_chain",
    "N_captures",
    "S_N_hex",
    "capture_log_hash",
    "chain_log_hash",
)


def _hash_dict_present(source, fields):
    """Build a dict from source containing only fields whose value is not None.
    source: a dict (loaded JSON) OR a SessionManifest instance.
    """
    out = {}
    if isinstance(source, dict):
        for f in fields:
            if f in source and source[f] is not None:
                out[f] = source[f]
    else:
        for f in fields:
            v = getattr(source, f, None)
            if v is not None:
                out[f] = v
    return out


def _compute_open_hash_input(source):
    """Build the canonical dict hashed into manifest_hash_open.

    session_status is HARDCODED to "open" — manifest_hash_open is a snapshot
    of the opening state, regardless of the current session_status on disk.
    """
    d = _hash_dict_present(source, _OPEN_SNAPSHOT_FIELDS)
    d["session_status"] = SESSION_STATUS_OPEN
    return d


@dataclass
class SessionManifest:
    session_id: str
    session_iso_utc_start: str
    bundle_hash: str
    S_0_hex: str
    device_id: str
    # v9: 64-char hex of 32 random bytes generated by `os.urandom(32)` at
    # session open. Published in manifest (not secret), covered by
    # manifest_hash_open via _OPEN_SNAPSHOT_FIELDS, and explicitly
    # re-committed as an input to derive_s0. Makes S_0 unpredictable to
    # anyone without rig access at session-open time, even when the other
    # S_0 inputs (RSK block hash, session_id, etc.) are publicly known.
    local_nonce_hex: str = ""
    # v9: drand session-level anchor. drand_chain_hash and
    # drand_public_key pin the drand beacon the writer used; the
    # verifier refuses to accept any round value whose verification
    # would use a different chain or key. drand_round_at_session_open
    # is the round number active at session-open; the verifier
    # independently confirms this round was live at session_iso_utc_start
    # as a session-level bounded-lag check (parallel to rsk_block_hash).
    drand_chain_hash: str = ""
    drand_public_key: str = ""
    drand_round_at_session_open: int = 0
    # v9: RSK RPC endpoints used for this session's anchor broadcasts.
    # Primary first; fallbacks follow. Pinned by manifest_hash_open so
    # the operator cannot retroactively claim a different endpoint set
    # was used. Optional (empty list) for unanchored sessions.
    rsk_rpc_endpoints: list = field(default_factory=list)
    # v3: anchor_policy is required — even for unanchored sessions. Captures
    # {enabled, interval_s, chain_id, network} when interior-anchor is on;
    # {enabled: False} when off. Pinned by manifest_hash_open so the policy
    # cannot be retroactively claimed after the fact.
    anchor_policy: dict = field(default_factory=lambda: {"enabled": False})
    session_status: str = SESSION_STATUS_OPEN
    # Computed once at session start, NEVER recomputed. Snapshot of the
    # opening state (session_status="open" hardcoded in its hash input).
    manifest_hash_open: str = ""

    # Optional start-time field: RSK fresh-block anchor. Included in
    # manifest_hash_open (transitively into S_0) when present; stripped
    # from JSON when None. An unanchored session is valid for offline demos.
    anchor_start: dict | None = None

    # End-of-session fields. Populated by finalize_end() when the session
    # completes cleanly; remain None on aborted/open sessions.
    # v0.8 rename: N_tiles → N_chain (chain rows are captured frames now).
    # v0.8 add:   chain_log_hash.
    session_iso_utc_end: str | None = None
    N_chain: int | None = None
    N_captures: int | None = None
    S_N_hex: str | None = None
    capture_log_hash: str | None = None
    chain_log_hash: str | None = None
    # v8.3: list of short labels describing runtime mitigations that
    # actually landed (e.g. "sched_fifo:prio=80", "affinity:[0]",
    # "mlockall", "gc_lockdown", "cpu_dma_latency:0us"). None on all
    # non-async-tightened sessions.
    mitigations_applied: list | None = None
    # Optional terminal anchor: {chain, tx_hash, payload_S_hex}. Populated
    # only if the final tx confirmed within the stop-wait window AND
    # payload_S_hex == S_N_hex.
    anchor_end: dict | None = None
    # v3: blake3 of the raw bytes of anchor_txs.csv (when interior-anchor
    # was used). Binds the pulse-tx log into manifest_hash_final so rows
    # cannot be stripped without detection.
    anchor_log_hash: str | None = None
    # v0.5.1: host monotonic_ns at Aravis start_acquisition() / stop_acquisition().
    # Both informational — a host-clock bracket on when the camera was
    # actively streaming. Verifier MUST NOT treat as authoritative time.
    # Hashed into manifest_hash_final so the value is pinned end-of-session.
    camera_acquisition_start_wall_ns: int | None = None
    camera_acquisition_end_wall_ns: int | None = None
    # Covers ALL current fields (including manifest_hash_open as a pinned
    # reference to the opening state) EXCEPT itself. Populated only by
    # finalize_end; remains None on open/aborted sessions.
    manifest_hash_final: str | None = None

    def compute_manifest_hash_open(self):
        return canonical_json_hash_hex(_compute_open_hash_input(self))

    def compute_manifest_hash_final(self):
        d = self.to_dict()
        d.pop("manifest_hash_final", None)  # exclude self-hash
        return canonical_json_hash_hex(d)

    def finalize_open(self):
        """Compute manifest_hash_open ONCE at session start. Call exactly once
        after all open-state fields (including anchor_start and S_0_hex) are set.
        Do not call again — manifest_hash_open is a permanent snapshot."""
        if self.manifest_hash_open:
            raise RuntimeError("finalize_open() called twice")
        self.manifest_hash_open = self.compute_manifest_hash_open()
        return self.manifest_hash_open

    def finalize_end(self, session_iso_utc_end, N_chain, N_captures,
                     S_N_hex, capture_log_hash, chain_log_hash,
                     anchor_end=None, anchor_log_hash=None,
                     camera_acquisition_start_wall_ns=None,
                     camera_acquisition_end_wall_ns=None,
                     mitigations_applied=None):
        """Mark session finalized. Sets session_status, populates end fields,
        computes manifest_hash_final. manifest_hash_open is NOT touched.

        v0.8 signature: N_chain + N_captures + capture_log_hash + chain_log_hash
        are required positional. The chain has N_chain rows (one per captured
        frame consumed); the capture log has N_captures rows. chain_log_hash
        is blake3 over the raw bytes of chain_log.csv.

        anchor_end (optional): dict with {chain, tx_hash, payload_S_hex}
        from the interior-anchor publisher's final tx. Stripped from JSON
        when None.

        anchor_log_hash (v3, optional): hex blake3 of the raw bytes of
        anchor_txs.csv. Binds the pulse log into manifest_hash_final.
        Stripped from JSON when None.

        camera_acquisition_start_wall_ns / camera_acquisition_end_wall_ns
        (v0.5.1, optional): host monotonic_ns at Aravis start/stop.
        Stripped from JSON when None. Informational; pinned by
        manifest_hash_final.
        """
        if not self.manifest_hash_open:
            raise RuntimeError("finalize_end() before finalize_open()")
        self.session_status = SESSION_STATUS_FINALIZED
        self.session_iso_utc_end = session_iso_utc_end
        self.N_chain = int(N_chain)
        self.N_captures = int(N_captures)
        self.S_N_hex = S_N_hex
        self.capture_log_hash = capture_log_hash
        self.chain_log_hash = chain_log_hash
        if mitigations_applied is not None:
            # Defensive copy; caller may mutate its own list.
            self.mitigations_applied = list(mitigations_applied)
        if anchor_end is not None:
            self.anchor_end = anchor_end
        if anchor_log_hash is not None:
            self.anchor_log_hash = anchor_log_hash
        if camera_acquisition_start_wall_ns is not None:
            self.camera_acquisition_start_wall_ns = int(camera_acquisition_start_wall_ns)
        if camera_acquisition_end_wall_ns is not None:
            self.camera_acquisition_end_wall_ns = int(camera_acquisition_end_wall_ns)
        self.manifest_hash_final = self.compute_manifest_hash_final()
        return self.manifest_hash_final

    def update_open_state_anchor_start(self, anchor_start):
        """v6: while still in open state, set anchor_start and recompute
        manifest_hash_open. Caller MUST then re-derive S_0 from the new
        manifest_hash_open and update self.S_0_hex BEFORE the chain starts.

        Used by tb_loop's reordered _build_and_write_session_artefacts:
          1) write open manifest WITHOUT anchor_start (so a findable
             aborted manifest exists on disk if the anchor wait fails)
          2) do the RSK fresh-block wait (may take ≤30s)
          3) call update_open_state_anchor_start with the result
          4) re-derive S_0 and write the manifest again

        manifest_hash_open is conceptually "computed once" but in the
        anchored case it's computed twice. Only the second value is hashed
        into S_0 and only the second written manifest is the one the chain
        commits to.
        """
        if self.session_status != SESSION_STATUS_OPEN:
            raise RuntimeError(
                f"update_open_state_anchor_start: session_status is "
                f"{self.session_status!r}, not 'open' — cannot update "
                f"anchor_start once the session has transitioned"
            )
        self.anchor_start = anchor_start
        self.manifest_hash_open = self.compute_manifest_hash_open()
        return self.manifest_hash_open

    def mark_aborted(self, reason=SESSION_STATUS_ABORTED):
        """Mark session aborted. Sets session_status to one of the abort
        reason values. Does NOT change manifest_hash_open (opening snapshot)
        and does NOT compute manifest_hash_final.

        v5: reason can be one of:
          - SESSION_STATUS_ABORTED (generic)
          - SESSION_STATUS_ABORTED_PUBLISHER_STUCK
          - SESSION_STATUS_ABORTED_SETUP_FAILED
          - SESSION_STATUS_ABORTED_SIGNAL_DURING_FINALIZE
        """
        if not self.manifest_hash_open:
            raise RuntimeError("mark_aborted() before finalize_open()")
        if reason not in SESSION_STATUS_ABORT_REASONS:
            raise ValueError(f"invalid abort reason: {reason!r}; expected one "
                             f"of {sorted(SESSION_STATUS_ABORT_REASONS)}")
        self.session_status = reason
        return self.manifest_hash_open

    def to_dict(self):
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}

    def validate(self):
        """v6: run reader-equivalent validation BEFORE writing."""
        SessionManifest.validate_dict(self.to_dict(),
                                       path="<unwritten manifest>")

    @staticmethod
    def validate_dict(d, path):
        """v6: full schema validation against the v0.5 manifest schema.

        Called by SessionManifest.write() before any write, and by
        load_and_verify() after parse. Reader/writer symmetry: the writer
        cannot produce an artefact load_and_verify() would reject.

        Does NOT include the bundle-pair cross-field check (wallet_address
        ↔ anchor_policy.enabled); that's validate_bundle_manifest_pair().
        """
        # Top-level closed schema.
        unknown = set(d.keys()) - _MANIFEST_KNOWN_FIELDS
        if unknown:
            raise ValueError(
                f"{path}: unknown manifest fields {sorted(unknown)} — "
                f"closed schema for protocol_version {PROTOCOL_VERSION}"
            )
        missing = [f for f in _MANIFEST_REQUIRED_FIELDS if f not in d]
        if missing:
            raise ValueError(
                f"{path}: missing required manifest fields: {missing}"
            )
        # session_status must be a known value.
        if d["session_status"] not in SESSION_STATUS_VALUES:
            raise ValueError(
                f"{path}: invalid session_status {d['session_status']!r}; "
                f"expected one of {sorted(SESSION_STATUS_VALUES)}"
            )
        # Nested schema + value-level + cross-field anchor_policy structure.
        _validate_manifest_nested(d, path)
        # Finalized-only field consistency. v0.8: N_chain + N_captures +
        # capture_log_hash + chain_log_hash all required end-fields.
        if d["session_status"] == SESSION_STATUS_FINALIZED:
            for k in ("session_iso_utc_end", "N_chain", "N_captures",
                       "S_N_hex", "capture_log_hash", "chain_log_hash",
                       "manifest_hash_final"):
                if d.get(k) is None:
                    raise ValueError(
                        f"{path}: finalized manifest missing required "
                        f"end-of-session field {k!r}"
                    )

    def write(self, path):
        if not self.manifest_hash_open:
            raise RuntimeError("SessionManifest.write() before finalize_open()")
        if self.session_status not in SESSION_STATUS_VALUES:
            raise RuntimeError(
                f"SessionManifest.write(): invalid session_status "
                f"{self.session_status!r}; expected one of {sorted(SESSION_STATUS_VALUES)}"
            )
        # v6: validate BEFORE write — same rules the reader applies. No path
        # in the recording loop can produce a schema-invalid artefact.
        try:
            self.validate()
        except ValueError as e:
            raise ValueError(
                f"manifest fails schema validation before write: {e}"
            ) from e
        path = Path(path)
        d = self.to_dict()
        # v5: atomic write (tmp → fsync → rename → dir fsync).
        _atomic_write_bytes(path, canonical_json_bytes(d))
        pretty_path = path.parent / (path.stem + ".pretty.json")
        _atomic_write_bytes(pretty_path,
                             (pretty_json_str(d) + "\n").encode("utf-8"))

    @staticmethod
    def load_and_verify(path):
        """Load JSON (strict), verify closed schema, recompute manifest_hash_open.

        v3:
          - Duplicate JSON keys rejected.
          - Closed schema: unknown top-level fields rejected.
          - Required fields enforced (including anchor_policy).
          - manifest_hash_open recomputed with session_status="open" HARDCODED.

        Does NOT verify manifest_hash_final — use verify_final_hash() for that.

        Returns (dict, manifest_hash_open_hex). Raises ValueError on any failure.
        """
        path = Path(path)
        d = _strict_json_load(path)
        # v6: full schema validation via validate_dict — same rules the
        # writer ran before producing this file. Reader/writer symmetry.
        SessionManifest.validate_dict(d, str(path))
        claimed = d["manifest_hash_open"]
        recomputed = canonical_json_hash_hex(_compute_open_hash_input(d))
        if claimed != recomputed:
            raise ValueError(
                f"{path}: manifest_hash_open mismatch — "
                f"claimed {claimed[:16]}…, recomputed {recomputed[:16]}…"
            )
        return d, claimed

    @staticmethod
    def verify_final_hash(d):
        """Recompute manifest_hash_final from a finalized manifest dict.
        Returns the recomputed hex digest. Caller compares against d['manifest_hash_final'].

        Raises if any required end field is missing.
        """
        if d.get("session_status") != SESSION_STATUS_FINALIZED:
            raise ValueError(
                f"manifest is not finalized "
                f"(session_status={d.get('session_status')!r})"
            )
        missing = [f for f in _MANIFEST_END_FIELDS
                   if f not in d or d[f] is None]
        if missing:
            raise ValueError(
                f"manifest is not finalized (missing end fields: {missing})"
            )
        if "manifest_hash_final" not in d or d["manifest_hash_final"] is None:
            raise ValueError("manifest_hash_final missing or null")
        # Hash all current non-None fields except manifest_hash_final itself.
        hash_input = {k: v for k, v in d.items()
                      if k != "manifest_hash_final" and v is not None}
        return canonical_json_hash_hex(hash_input)

# Truth Beam v10 session bundle — reader's guide

This document is the authoritative description of what a v10 session
bundle contains, how to verify it, and what it claims versus what it
does not. Paired with `CLAIMS.md` at the bundle root.

v10 extends the v9 protocol in exactly two ways: the per-row chain
transition length-prefixes the prior state and folds an additional
32-byte `ai_payload_root` (domain tag `TB:ROW:v10`), and the chain
log gains two AI-payload columns. Everything else is shared with v9.

## Target audience

- Human reviewers (research peers, operators, auditors).
- Automated readers (pipeline consumers, future language-model
  reviewers who ingest the bundle as structured context).

Everything in this document should be readable by both, without
either having to guess at conventions.

## Bundle layout

A v10 session directory contains:

    <session_id>/
      manifest.json, manifest.pretty.json       — session identity + pinned inputs
      verification_bundle.json, *.pretty.json   — algorithm contract (pinned pre-session)
      chain_log.csv                             — authoritative chain row log (15 cols)
      capture_log.csv                           — per-captured-frame log (8 cols)
      anchor_txs.csv                            — RSK tx log (15 cols; present iff --interior-anchor)
      Recordings/frame_NNNNNN.raw               — authoritative raw Bayer captures
      derived/
        Emissions/tile_NNNNNN.png               — per-row emission pixel bytes (E_t)
        Recordings_previews/*.png               — post-finalize debayered previews (optional)
      ai_payloads/included/<row>/*.payload      — AI-agent response payloads (v10)
      CLAIMS.md                                 — machine-parseable claim tiers
      README_BUNDLE.md                          — this file

## Convention: `null` in JSON

For any field in `manifest.json` that can be absent (e.g. `anchor_end`
when the final tx did not confirm within the window, or `anchor_start`
when the session was unanchored), the JSON emits literal `null`. A
`null` value means *this field was not applicable or not available*,
not that it was *zero*, *empty*, or *unset-by-accident*.

The writer never fabricates a placeholder. It writes `null` when a
thing didn't happen; the verifier reports status tiers when it
couldn't confirm something at verify time.

(Exception: scalar sentinel values are used INSIDE chain_log rows
where the CSV format doesn't carry a null concept. Specifically:
`drand_round_number=0` and `drand_round_value_hex="00"*64` when drand
was unavailable at that row. This is documented in the chain_log
column vocabulary below.)

## Chain log columns (`chain_log.csv`, v10, 15 cols)

| column | meaning |
|---|---|
| `t` | chain row index, starts at 0, strictly +1 |
| `S_t_hex` | chain state *before* processing this row; `== S_0_hex` for row 0 |
| `bayer_blake3_hex` | blake3 of the raw bayer bytes in `Recordings/frame_{t:06d}.raw` |
| `capture_frame_id` | camera-side frame counter (informational) |
| `aravis_device_timestamp_ns` | camera-device clock at trigger |
| `capture_wall_ns` | host wall time at buffer-pop (bug-1 guard) |
| `emission_live_pixel_blake3_hex` | blake3 of the RGB pixel bytes of E_t, the emission that was LIVE on the DMD during cap_t |
| `emission_live_queued_wall_ns` | host wall time at which E_t was queued to the DMD |
| `meta_hex` | 28-byte capture meta struct (see `chain.META_STRUCT`) |
| `emission_png_path` | relative path from session root to `derived/Emissions/tile_{t:06d}.png` |
| `drand_round_number` | drand quicknet round active at this row's capture, or `0` if drand was unavailable |
| `drand_round_value_hex` | 64-hex (32 bytes) drand randomness for that round, or all-zeros if unavailable |
| `drand_staleness_ms` | age of the cached drand round when this row fired (ms); `-1` if never fetched |
| `ai_payload_count` | (v10) number of AI-agent payloads committed at this row; `0` for most rows |
| `ai_payload_root_hex` | (v10) 64-hex domain-separated commitment over the row's payload digests; 32 zero bytes (the empty-root sentinel) when `ai_payload_count=0` |

## Session-level fields in `manifest.json`

Beyond standard session identity (`session_id`, `session_iso_utc_start`,
`device_id`, `bundle_hash`, `session_status`, `manifest_hash_open`,
`S_0_hex`), v10 carries the same fields v9 added:

- `local_nonce_hex` — 64-hex, 32 random bytes from `os.urandom` at
  session open. Published, not secret. Covered by `manifest_hash_open`
  and re-committed explicitly in `S_0` derivation.
- `drand_chain_hash`, `drand_public_key` — drand beacon pinned
  pre-session.
- `drand_round_at_session_open` — session-level drand anchor. `0` if
  drand was unreachable at session open (corroborating-evidence
  policy, not mandatory).
- `rsk_rpc_endpoints` — list of RPC URLs (primary + fallbacks) used
  for anchor tx broadcasts. Empty list if unanchored.
- `anchor_start` — RSK fresh-block record (mainnet block_hash +
  number + parent_hash + timestamps), or `null` if unanchored.
- `anchor_end` — RSK final-anchor tx record, or `null` if the tx
  did not confirm within the window. Fully-populated or absent; no
  partial `anchor_end`.

## `S_0` derivation (shared with v9, `TB:S0:v1`)

```
S_0 = blake3(
    b"TB:S0:v1"
    || len32(rsk_block_hash)              || rsk_block_hash   (32 B)
    || len32(rsk_block_height_be8)        || rsk_block_height (8  B)
    || len32(local_nonce)                 || local_nonce      (32 B)
    || len32(session_id_utf8)             || session_id_utf8  (variable)
    || len32(manifest_hash_open)          || manifest_hash_open (32 B)
)
```

Each segment has a 4-byte big-endian length prefix. `rsk_block_hash`
uses 32 zero bytes when `anchor_start` is null; `rsk_block_height`
uses 0. The `TB:S0:v1` domain tag is a fresh version for the layered
primitive, distinct from the per-row `TB:ROW:v10` chain tag.

## Per-row chain advance (v10)

```
S_{t+1} = blake3(
    b"TB:ROW:v10"
    || len32(S_t)            || S_t                (32 B)
    || len32(bayer_hash)     || bayer_hash         (32 B)
    || len32(meta_bytes)     || meta_bytes         (28 B)
    || len32(drand_round_be8)|| drand_round_be8    (8  B)
    || len32(drand_value)    || drand_value        (32 B)
    || len32(ai_payload_root)|| ai_payload_root    (32 B)
)
```

Note the two differences from v9: the prior state `S_t` is
length-prefixed like every other field, and the row's
`ai_payload_root` is folded in. The distinct domain tag means a v9
log cannot be re-walked under v10 or vice versa.

The `ai_payload_root` itself is a domain-separated commitment
(`TB:AI_PAYLOAD_ROOT:v10`) over the row's ordered payload digests,
or the 32-zero-byte sentinel for an empty payload list. The root is
cryptographically committed — it is folded into every `S_{t+1}`, so
the chain walk verifies it row by row. Recomputing the root from the
payload *files* requires the writer-side digest serialization, which
is not part of the published verifier snapshot; the verifier instead
checks the sentinel invariant (`count=0` ⇔ sentinel root).

Drand round=0 + value=zeros is a valid input pattern signalling drand
unavailability; the chain still advances deterministically.

## Pulse commitment (shared with v9, `TB:PULSE:v1`)

```
pulse_commitment = blake3(
    b"TB:PULSE:v1"
    || len32 || session_id_utf8
    || len32 || struct(">I", pulse_index)
    || len32 || struct(">I", frame_start)
    || len32 || struct(">I", frame_end)
    || len32 || prev_pulse_commitment   (32 B; zeros for first pulse)
    || len32 || S_t                     (32 B)
)
```

The on-chain RSK tx's `data` field is this 32-byte commitment, NOT
raw S_t. Raw S_t is recorded in `anchor_txs.csv.payload_S_hex` for
local verification convenience. The verifier recomputes the
commitment and confirms it matches both the CSV field and the
on-chain tx's calldata.

## Running the verifier

### Dependencies (fresh clone)

```
pip install blake3 Pillow numpy py_ecc web3 eth_account
```

Only `blake3`, `Pillow`, and `numpy` are strictly required for
offline chain-integrity verification. `py_ecc` is needed for drand
BLS re-verify (`--online` mode). `web3` is needed for RSK tx
inclusion checks (`--online` mode). `eth_account` is needed by
`web3`.

### Logs-only verification (fastest; ~MBs, no bulk data)

```
python3 code/recording/verify/verify_v10.py \
    --session-dir /path/to/<session_id> --logs-only
```

Verifies the chain math alone — S_0 derivation, every per-row chain
advance, the terminal-state-vs-manifest match, and every pulse
commitment — from `chain_log.csv` + `manifest.json` +
`verification_bundle.json` + `anchor_txs.csv`, without the raw
frames or emission PNGs. `chain_integrity` reports `LOGS_ONLY`
(never PASS) in this mode, because the media hashes were not
exercised.

### Offline verification (default)

```
python3 code/recording/verify/verify_v10.py \
    --session-dir /path/to/<session_id>
```

(The verifier ships in the public TruthBeam repository,
github.com/poliebotics/truthbeam, under `code/recording/verify/`.)

Offline verification checks the cryptographic chain (the hard claim)
without any network calls. Reports `chain_integrity: PASS|FAIL`,
per-row invariants, and pulse-commitment math. RSK inclusion status
for each pulse is `NOT_CHECKED`; drand BLS re-verify is skipped.

### Online verification

```
python3 code/recording/verify/verify_v10.py \
    --session-dir /path/to/<session_id> --online
```

Online mode additionally queries the manifest's
`rsk_rpc_endpoints` to fetch each pulse tx's inclusion status and
refetches + BLS-verifies every unique drand round that appears in
the chain log.

### Reading the output

The verifier writes `verify_report.json` in the session directory
and prints a human-readable summary to stderr. The JSON report
contains all per-row and per-pulse data for machine consumers; the
summary is for humans.

Key session-level fields:

- `chain_integrity: "PASS" | "FAIL" | "LOGS_ONLY"` — gates overall session
  validity. In v10, PASS additionally requires every `ai_payload_count=0`
  row to carry the 32-zero-byte empty-root sentinel AND every
  `ai_payload_count>0` row to carry a non-sentinel root (reported as
  `ai_sentinel_check: "PASS"`); `--logs-only` reports `LOGS_ONLY` when the
  chain math and sentinel hold but the media hashes were not exercised.
  validity. PASS means every chain_log row's transition recomputed,
  every `bayer_blake3_hex` matches its raw file, every
  `emission_live_pixel_blake3_hex` matches its PNG. No other checks
  are allowed to fail silently.
- `cycle_closure_verdict: "PASS" | "FAIL" | "INCONCLUSIVE"` —
  PASS iff chain_integrity passed AND every pulse commitment
  recomputed AND the computed terminal state matches the manifest
  (`terminal_state_matches_manifest: PASS`). INCONCLUSIVE if the
  chain passed but some pulse commitment couldn't be checked (e.g.
  no anchor_txs.csv in an unanchored session).
- `terminal_S_N_hex` / `terminal_state_matches_manifest` — the
  verifier recomputes the post-final-transition terminal state
  `S_N = advance(last row)` and compares it to `manifest.S_N_hex`
  (the value carried in the final RSK anchor). `last_logged_S_t_hex`
  is the final row's *pre*-state (S_{N-1}). **Erratum:** verifier
  versions before 2026-06-10 mislabelled that pre-state as
  `terminal_S_N_hex`; a `verify_report.json` produced by an older
  verifier therefore shows S_{N-1} under that name. Both released
  sessions' true terminal states have been re-verified to match
  their manifests with the current verifier.
- `drand_availability: "FULL" | "PARTIAL" | "MISSING"` — reports
  what drand data is present in the chain log. Does NOT gate PASS.
- Commitment checks (verifier 2026-06-10 or later; each
  `PASS | FAIL | UNCHECKED`, all gating `cycle_closure_verdict`):
  `manifest_hash_final_matches` and `bundle_hash_matches` recompute
  the canonical-JSON self-hashes of `manifest.json` and
  `verification_bundle.json` (plus the manifest↔bundle linkage);
  `chain_log_hash_matches_manifest` / `capture_log_hash_matches_manifest`
  compare the recomputed log hashes to the values pinned in the
  manifest; `anchor_log_hash_matches_manifest` checks
  `anchor_txs.csv` against `manifest.anchor_log_hash` (snapshotted at
  finalize time, before the trailing `final_root` row is appended, so
  a match after stripping trailing `final_root` rows also passes);
  `pulse_chain_continuity` enforces sequential pulse indices, the
  `prev_pulse_commitment` linkage (32 zero bytes for the first
  pulse), and contiguous frame ranges — catching deletion or
  reordering of interior pulse rows.
- Exit code: `0` only when the chain walk passed AND
  `cycle_closure_verdict` is not `FAIL`. A vacuously empty pulse set
  (`n_pulses = 0`) can never yield `cycle_closure_verdict: PASS`, and
  a missing `anchor_txs.csv` on an anchored session
  (`manifest.anchor_log_hash` set) is a failure.

Per-pulse fields:

- `payload_commitment_verified: bool`
- `rsk_inclusion_status: "NOT_FOUND" | "PENDING" | "INCLUDED_1_CONF" | "INCLUDED_6_CONF" | "STABLE_100_CONF" | "NOT_CHECKED"`
- `block_depth: int | null`
- `block_explorer_url: str` — derived deterministically from
  `(chain_id, tx_hash)` via the function below.

## Block-explorer URL derivation

One-line function. Feed this any `(chain_id, tx_hash)` pair from
`anchor_txs.csv` to get the public explorer URL.

```python
def block_explorer_url(chain_id: int, tx_hash: str) -> str:
    base = {
        30: "https://explorer.rsk.co",          # RSK mainnet
        31: "https://explorer.testnet.rsk.co",  # RSK testnet
    }.get(int(chain_id))
    return f"{base}/transactions/{tx_hash}" if base else ""
```

This function lives in `code/recording/verify/verify_v10.py`
and is imported by the verifier when it populates pulse reports.
Reviewers can construct the same URL on any `(chain_id, tx_hash)`
pair without running the verifier.

## Shape of an honest verifier report

Four lessons a reader should take from the structure:

1. **No single "authenticated" badge.** The report is status-tiered;
   a consumer that wants a binary answer should read
   `chain_integrity`. Anything more ambitious (anchor confirmations,
   drand BLS re-verify) is status-reported at a finer grain.
2. **Offline mode produces a complete, honest report.** The only
   things that change in online mode are RSK inclusion tiers and
   drand re-verification counts.
3. **The writer never lies.** Soft-fail sources (drand, individual
   pulse RPC errors) are recorded with sentinels in chain_log or
   skip-counts in anchor_txs; they don't mutate into fake success.
4. **Null in JSON means not-applicable-or-not-available.** Never
   "zero" or "unknown."

## Recorder-environment residue

`verify_report.json` records the absolute `session_dir` path of the
machine the verifier ran on, and `manifest.json` /
`verification_bundle.json` record the recorder hostname/device id
(`g1a`) and host configuration. These fields are provenance residue
of the recording/verification environment; they are not part of any
cryptographic claim and carry no secret material.

# Truth Beam v9 claims manifest

Machine-parseable and honest. Paired with `README_BUNDLE.md`.

This document replaces marketing-style framing ("authenticated,"
"proven," "blockchain-secured") with a structured three-tier claim
statement. Each claim is labelled with exactly one tier; readers
(human or automated) can filter by tier to get precisely the
confidence level they want.

The three tiers:

- **DEMONSTRATED** — the session's on-disk data, verified by
  `verify_v9.py`, makes this claim true. Reproducible from the
  bundle alone.
- **ASSERTED** — the protocol design makes this claim true, but
  verifying it requires work beyond Phase A (v9) that hasn't
  shipped yet. Listed as a design promise, not a demonstration.
- **FUTURE** — not in the protocol today. Listed to prevent the
  ambiguity of an unasserted future ambition getting mistaken for
  a current claim.

---

## DEMONSTRATED (reproducible from this bundle + `verify_v9.py`)

### Chain cryptographic integrity

Every row `t` in `chain_log.csv` satisfies:

- `bayer_blake3_hex` equals `blake3(Recordings/frame_{t:06d}.raw)`.
- For `t > 0`: `S_t_hex` equals `compute_chain_row(S_{t-1},
  bayer_{t-1}, meta_{t-1}, drand_round_{t-1}, drand_value_{t-1})`
  under domain tag `TB:ROW:v9`.
- For `t == 0`: `S_t_hex` equals `S_0` recomputed from the 5-input
  layered derivation (see README_BUNDLE.md, §"S_0 derivation"),
  which binds the chain to local CSPRNG nonce, RSK block hash (if
  anchored), session_id, and `manifest_hash_open`.

Verification: offline. `chain_integrity: PASS`.

### Emission live attribution (bug-2 fix)

For every row `t`, `emission_live_pixel_blake3_hex` equals
`blake3(decode(derived/Emissions/tile_{t:06d}.png))`. That is: the
PNG at index `t` contains the RGB pixel bytes of the emission that
was live on the DMD during `cap_t`, not the next emission as under
v8.

Verification: offline (PNG decode + hash). Reported per-row as
`emission_png_hash_ok`.

### Pulse commitment derivation (interior-anchor sessions only)

Every pulse tx in `anchor_txs.csv` (rows the verifier selects by
`payload_kind == "state"`; the single trailing `payload_kind == "final_root"`
row, `is_final=1`, is excluded)
satisfies `payload_commitment_hex ==
compute_pulse_commitment(session_id, pulse_index, frame_start,
frame_end, prev_pulse_commitment, S_t)` under domain tag
`TB:PULSE:v1`. Chains via `prev_pulse_commitment`.

Verification: offline. Reported per-pulse as
`payload_commitment_verified`.

### Local-nonce unpredictability

The local CSPRNG nonce at `manifest.local_nonce_hex` was generated
by `os.urandom(32)` at session open and published in the manifest.
A verifier reading the session cannot prove the rig generated it
honestly, but the nonce's presence in the S_0 derivation means any
party without access to the rig at session-open time cannot have
predicted S_0.

Verification: offline (structural — the value is present and has
the right length). The *honesty* of its generation is not
verifiable from the bundle alone (see ASSERTED below).

### No optical lookahead from the pulse tx

Because the pulse tx's on-chain `data` field is a domain-separated
BLAKE3 commitment, not raw S_t, an observer watching the RSK mempool
cannot infer the live emission from pulse payloads alone. They can
still infer it from the chain_log itself if they have access to the
session bundle; but the pulse stream does not carry S_t in the clear.

Verification: offline (structural — reconstructable per pulse).

---

## ASSERTED (protocol design promises; Phase-B work required for full demonstration)

> **Update (2026-06-20):** for the two released sessions (D2, V10) these online time-bound
> checks have been performed and pass — RSK mainnet anchor blocks and the final-root tx confirmed
> on-chain, drand quicknet rounds BLS-verified, commit rate ~2.5 Hz. See
> [`../../../TEMPORAL_VERIFICATION.md`](../../../TEMPORAL_VERIFICATION.md).


### RSK timestamp-corroborated session-open time

Per protocol design: `manifest.anchor_start.block_hash` did not
exist before `manifest.anchor_start.block_timestamp_utc`. Therefore
the session cannot have been committed before that time.

Verification: requires the `--online` mode to fetch the RSK block
and confirm the block_hash matches at that height (and that the
block isn't orphaned/reorged). The verifier reports this as part
of online output; offline mode reports `ASSERTED` only.

### RSK pulse tx inclusion bounds session end time

Per protocol design: each pulse tx's inclusion in an RSK block
places an upper bound on when the pulse's `S_t` could have been
chosen. A sequence of pulses therefore bounds the session's
duration from both ends.

Verification: requires online RPC access and an archival RSK node
for post-hoc block confirmation. The verifier reports per-pulse
`rsk_inclusion_status` in online mode. Deeper bound-derivation
(e.g. "these 5 pulses span at most 15 seconds") is a Phase-B
analysis.

### Drand BLS signature corroborates timing at per-round granularity

Per protocol design: for every `drand_round_number` appearing in
`chain_log.csv` with a non-zero value, the round's BLS signature
can be verified against the drand quicknet public key. Round N was
published at `genesis_time + N * period_s`; therefore any row
committing round N cannot have been committed earlier.

Verification: requires `--online` mode (drand refetch + py_ecc BLS
verify). Drand is corroborating-evidence tier in v9, NOT gating.

### Cycle-closure

Per protocol design: the chain log's terminal `S_N` should match
`manifest.S_N_hex` (the same value carried on-chain as the final anchor's
`payload_S_N_hex`; the offline verifier compares the recomputed terminal
state to the manifest field, and — when present — also authenticates the
trailing `final_root` row against `manifest.anchor_end`) (if an anchor_end
confirmed). Pulse-commitment chain (via `prev_pulse_commitment`)
connects every pulse back to the session's start.

Verification: offline for both the pulse-commitment chain and the
terminal state — the verifier (2026-06-10 or later) recomputes
`S_N = advance(last row)` and compares it to `manifest.S_N_hex`
(`terminal_state_matches_manifest`). Full closure (final anchor tx
inclusion at sufficient depth) additionally requires an online RSK
query.

---

## FUTURE (not in the v9 protocol; reserved names)

- **OpenTimestamps archival anchor** — calendar-server submission at
  session end for long-horizon archival attestation. Not in v9. A
  later protocol may add `manifest.ots_proof_path` pointing
  to an `.ots` file in the session bundle.

- **RFC 3161 TSA receipt layer** — per-pulse or per-bucket TSA
  timestamps for bounded-lag claims under drand-unavailable
  conditions. Not in v9. Listed as future scope in the protocol
  design space. Would add `timestamp_authority_certs/` directory to
  the bundle.

- **NIST beacon corroboration** — second fast-challenge source for
  drand redundancy. Not in v9.

- **Per-frame drand** — under the current protocol a drand round
  covers several chain rows (the released sessions commit at
  ~2.5 Hz while drand quicknet publishes at ~0.33 Hz, so one round
  spans ~7–8 rows). A future per-frame drand would require waiting
  on drand rounds to publish, destroying capture rate. Listed as
  explicitly not a goal.

- **RSK mainnet operational production use** — v9 enables mainnet
  submissions with explicit operator confirmation; this is suitable
  for preprint-grade demo sessions. Treating mainnet anchoring as
  production-grade infrastructure for sustained operation is future
  work (key-rotation procedures, HSM integration, formal audit).

- **Multi-chain anchoring** — simultaneous RSK + Bitcoin (via OTS)
  + other. Future.

- **Phase-B bucket aggregation** — batching C_t commitments into
  Merkle buckets, TSA-timestamping bucket roots, RSK-anchoring
  5-bucket superroots. Designed to fit on top of the v9 pulse
  commitment primitive without format changes; reserved for a later
  protocol (not present in v10 either — v10 adds only the
  AI-payload commitment).

- **Capture-path attestation** — secure-boot + camera firmware
  signing to rule out sensor-side tampering. Future, out-of-scope
  for Phase A.

---

## What the bundle does NOT claim

A non-exhaustive list of things sometimes read INTO projects like
this that are not claimed by v9:

- Not a claim that the projected emission is a unique unforgeable
  artifact. The XOF-fBm emissions are deterministic functions of
  S_t; anyone with S_t can re-generate them. The cryptographic
  chain binds when and by whom they were generated, not that they
  are one-of-a-kind.

- Not a claim that the rig is tamper-evident at the photon layer.
  If an attacker has physical access to the projector+camera, they
  can substitute inputs. v9 claims are cryptographic integrity of
  the recorded chain against post-hoc tampering, plus external
  anchoring to bound session time — not sensor-level attestation.

- Not a claim of zero-knowledge or privacy preservation. The bundle
  publishes S_0, every S_t, every raw bayer hash, and every
  emission. Anyone with the bundle can fully reconstruct the
  projected sequence.

- Not a claim of real-time verification. The verifier runs post-hoc
  on a completed bundle. In-session assurance is bounded by the
  writer's own checks (chain math, drand BLS, RSK RPC response).

- Not a claim of cross-provider consensus. Online mode queries
  whichever RPC endpoints the manifest pins; it does not
  cross-verify across independent RSK infrastructure operators.
  Adding independent multi-provider consensus is Phase-B work.

---

## Reading this file programmatically

Machine consumers: all claim items in each tier are prefixed by a
`###` heading. Within DEMONSTRATED claims, verification mode is
noted explicitly ("Verification: offline" or "Verification: online").
Within ASSERTED claims, the prerequisite for full verification is
noted.

A future JSON-formatted version of this manifest is possible but
not currently shipped; the Markdown form is chosen for human
reviewer primacy.

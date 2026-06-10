# Golden-diff field policy

`tools/golden_diff.py` classifies every field in each session artefact
into one of three buckets. This document explains the classification
and the evidence the diff can (and cannot) provide.

## Use case

Two recordings on the same rig, back-to-back, with identical config:
one on the pre-refactor backup (`_archive/pre_refactor_20260421_182758/`),
one on the current code. Differ is invoked

    python3 tools/golden_diff.py <phase0_golden> <phaseA_reference>

and reports whether the refactor changed anything a re-recording can
observe.

## The three buckets

### MUST_MATCH

A difference is a regression. These fields are deterministic in the
code — they do not depend on wall-clock values, randomness, or sensor
noise.

- `chain_log.csv`: `t`, `emission_png_path`, column set
- `capture_log.csv`: `capture_idx`, column set
- `manifest.json`: `bundle_hash` (commits to rig config; identical by
  construction across two same-config runs)
- `verification_bundle.json`: **every top-level key** — the bundle is a
  pure function of rig config + generator source + protocol version

### WILL_DIFFER

These fields MUST differ across two real recordings. They are
wall-clock, device-clock, or random-UUID-based. A *match* on these would
indicate either a fake recording or a time source that wasn't advancing.

- `chain_log.csv`: `capture_wall_ns`, `aravis_device_timestamp_ns`,
  `tile_queued_wall_ns`
- `capture_log.csv`: `capture_wall_ns`, `aravis_device_timestamp_ns`
- `manifest.json`: `session_id`, `session_iso_utc_start`,
  `manifest_hash_open`, `manifest_hash_final`, `anchor_start`,
  `anchor_end`

### CONDITIONAL

These fields are deterministic in the code PATH but their *value*
depends on a WILL_DIFFER input. So they differ across two real
recordings even though the code is identical. Matching is not evidence
of correctness; differing is not evidence of a regression.

The cleanest example: `S_t_hex`. The v8 chain transition is

    S_{t+1} = blake3("TB:ROW:v8" || S_t || len32(bh) || bh || len32(m) || m)

where `m` (meta_bytes) embeds `capture_wall_ns` and
`aravis_device_timestamp_ns`. Since those differ between runs, so does
`m`, so does `S_{t+1}`, so does every downstream xof_seed, every
tile_pixel_sha, and so on. In rows t >= 1, every chain-derived column
differs purely from wall-clock drift.

For row t == 0, `S_t_hex == S_0`, and `S_0 = blake3(...||manifest_hash_open)`
where `manifest_hash_open` is a function of `session_id` (random) and
`session_iso_utc_start`. So S_0 also differs per run.

Upshot: the chain log diff provides **schema-level** evidence only —
column names, row counts, data types, emission-path format. It cannot
provide bit-level equivalence without pinning the timestamp sources,
which the current recording code does not support.

- `chain_log.csv`: `S_t_hex`, `bayer_blake3_hex`, `capture_frame_id`,
  `xof_seed_R_hex`, `xof_seed_G_hex`, `xof_seed_B_hex`,
  `tile_pixel_sha_hex`, `meta_hex`
- `capture_log.csv`: `capture_frame_id`, `bayer_blake3_hex`, `raw_path`,
  `consumed_into_chain`, `consumed_as_t`
- `manifest.json`: `S_0_hex`, `S_N_hex`, `final_root_hex`,
  `session_status`

## Stronger tests that do NOT require two live recordings

The limitation above is structural: the v8 chain is wall-clock-coupled
by design (meta_bytes carries capture timestamps into the hash). Two
ways to get bit-level evidence:

1. **Synthetic-input equivalence** — feed a fixed sequence of
   `(bayer_hash, ts_ns, capture_wall_ns, exposure_us, fourcc)` tuples
   into the pre-refactor and post-refactor `compute_chain_row` +
   `pack_meta` + CSV writers + manifest assembly code, compare outputs.
   Fully deterministic, no hardware. Not yet scaffolded — forward
   work item.

2. **AST-equivalence of moved functions** — `tools/ast_equivalence_check.py`.
   Proves every moved function's BODY is AST-identical to the pre-
   refactor source (modulo a leading docstring). This is strictly
   weaker than a synthetic-input test but strictly stronger than the
   434-literals check from the Phase-A summary. Runnable today.

## What `golden_diff.py` PASS / FAIL means

A PASS says: "no MUST_MATCH field disagrees; WILL_DIFFER fields
disagree as expected; every top-level key in every artefact is
classified and all unclassified keys agree." That rules out:

- Swapped fields between CSVs
- A silently renamed column
- A rig-config divergence in the bundle
- A code path that populates `bundle_hash`-affecting state differently

It does NOT rule out:

- A chain-math bug that produces a different `S_t_hex` sequence —
  because `S_t_hex` is CONDITIONAL anyway
- A mismatch in how meta_bytes is packed (same reason)
- A mismatch in bayer hash computation (differs from sensor noise too)

For those, use AST equivalence + synthetic-input tests.

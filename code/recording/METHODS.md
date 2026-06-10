# Methods

## 1. Overview

Truth Beam is a recording protocol for tamper-evident optical sessions. A
projector-camera rig emits a deterministic pseudorandom image tile at 60 Hz
on a DMD panel; a hardware-triggered global-shutter camera captures each
frame; the BLAKE3 hash of every captured bayer frame, together with its
device timestamps and exposure metadata, is folded into a BLAKE3 hash
chain whose opening state is optionally derived from an external block
hash (RSK mainnet or testnet) and whose terminal state is optionally
anchored back on-chain at session end. Anchoring is optional; sessions
are cryptographically self-consistent without it, and unanchored sessions
derive `S_0` purely from the internal manifest-hash. The recorded
artefacts are a raw bayer sequence, a per-step chain log, a verification
bundle pinning every configuration constant, and a session manifest that
commits to all of the above.

The protocol claims *chain-verified* provenance: the sequence of captures
in a session is cryptographically tied to the session-specific generator
(so a bayer frame cannot be silently substituted), to the exact rig
configuration (so a different camera / projector / exposure setting
cannot be mixed in), and to an external timeline anchor (so the session's
opening and closing times are bounded by independently observable
block-chain events). It does **not** claim that the projected content is
optically bound to the scene — that is a separate reconstructor-level
claim evaluated by the verifier in `verify/` (`verify_v9.py`) and the
broader verification stack in `../verifier/`.

## 2. Chain construction

### Seed derivation

The chain is rooted in a 32-byte opening state `S_0`:

    S_0 = blake3(DS("TB:S0:v8") || len32(manifest_hash_open) || manifest_hash_open)

where `DS` is a literal ASCII domain-separation tag (`b"TB:S0:v8"`),
`len32` is a 4-byte big-endian length prefix, and `manifest_hash_open`
is the BLAKE3-over-canonical-JSON hash of the session manifest's
*open-state* snapshot (see `protocol/session_schema.py:derive_s0`).

`manifest_hash_open` itself is a hash over the following fields
(`_OPEN_SNAPSHOT_FIELDS` in `protocol/session_schema.py`):

- `session_id` (UUID4, generated at session start)
- `session_iso_utc_start`
- `bundle_hash` (see below)
- `device_id` (host identifier)
- `anchor_policy` (`{"enabled": bool, ...}`; if interior-anchor is on,
  also records the interval, chain_id and network)
- `anchor_start` (optional; the RSK fresh-block record if the session
  is anchored)
- `session_status` — hardcoded to the literal string `"open"` during
  hashing, so that the opening-state snapshot is stable even after the
  session finalizes or aborts.

`bundle_hash` is a BLAKE3-over-canonical-JSON hash of the entire
`VerificationBundle` (`_BUNDLE_KNOWN_FIELDS`), excluding only the
bundle_hash field itself. The bundle includes:

- `protocol_version`
- `generator_code_hash` (BLAKE3 of the concatenated source of the active
  tile-generator functions; see §3)
- `generator_config`, `camera_config`, `projector_config`, `tile_config`,
  `chain_config`, `metadata_schema`, `host_config`
- `wallet_address`, `chain_id`
- `session_mode` (`blocking`, `async`, or `async-tightened`)
- `rig_pipeline_calibration` — the entire calibration JSON, including
  the measured `stable_white_mean`, `stable_black_mean`, per-wait
  correctness fractions, rig hash, and the recommended wait. Null when
  `session_mode != "blocking"`.
- `rig_pipeline_calibration_hash` — redundant BLAKE3 of that JSON.

The full pipeline calibration contents are therefore folded transitively
into `S_0` through `bundle_hash`. A verifier recomputes `bundle_hash`
from the on-disk `verification_bundle.json`, recomputes
`manifest_hash_open` over the manifest's open fields, and derives `S_0`
from it; any tampering with the calibration, wallet, host config, or
generator source changes `S_0` and the chain fails to walk.

`~/.tb/pipeline_calibration_<rig_hash>.json` (the operator's machine-
local calibration store) is a convenience cache only: at verify time
the calibration is read *out of the session's own bundle*, not from the
operator machine. Sessions are therefore self-contained for verification.

### State transition

Per-step advancement is

    S_{t+1} = blake3(b"TB:ROW:v8"
                    || S_t
                    || len32(bayer_hash) || bayer_hash
                    || len32(meta_bytes) || meta_bytes)

(`protocol/session_schema.py:compute_chain_row`). `bayer_hash` is the
BLAKE3 digest of the raw bayer payload for capture `t`. `meta_bytes` is
a 28-byte big-endian pack of `(t, aravis_device_timestamp_ns,
capture_wall_ns, exposure_us, fourcc)` (`META_STRUCT = ">IQQI4s"` in
`protocol/tb_loop.py`; the leading `>` selects big-endian).

Each `t` therefore commits to the capture's payload, the camera's
hardware timestamp, the host receive time, the exposure setting, and
the pixel format, ordered inside a fixed protocol tag.

### Session finalization

At session end the chain commits additionally to:

- `N_chain`, `N_captures`
- `S_N_hex` (terminal chain state)
- `capture_log_hash`, `chain_log_hash` (BLAKE3 over the raw CSV bytes)
- `anchor_end` (if the interior-anchor publisher landed a terminal tx)
- `anchor_log_hash` (BLAKE3 over `anchor_txs.csv` when interior-anchor
  was active)
- `camera_acquisition_start_wall_ns`, `camera_acquisition_end_wall_ns`
- `mitigations_applied` (runtime tweaks that actually landed; only set
  on async-tightened sessions)

These fields are hashed into `manifest_hash_final`. The terminal anchor
transaction, when enabled, commits to a derived `final_root` rather
than to `manifest_hash_final` directly:

    final_root = blake3(
                    b"TB:FINAL:v8"
                 || S_N                  (32 bytes)
                 || bundle_hash          (32 bytes)
                 || manifest_hash_open   (32 bytes)
                 || anchor_log_hash      (32 bytes, zero-filled if
                                          interior-anchor was not used)
               )

(`protocol/session_schema.py:compute_final_root`). `final_root` is
what the publisher posts in the closing transaction's data field, and
what the recorded `anchor_end.payload_final_root_hex` matches against.
`manifest_hash_final` is committed on disk and verifiable locally;
`final_root` is committed on-chain.

## 3. Emission generation

Each chain step projects a deterministic 1920×1080×3 RGB tile derived
entirely from `S_{t+1}`. The pixel bytes are a function of the chain
state and nothing else.

### XOF expansion

Per-channel XOF seeds are domain-separated off `S_{t+1}`:

    xof_seed_c = blake3(b"TB:SEED:" || c || b":v8" || S_{t+1})    c ∈ {R,G,B}

(`protocol/session_schema.py:compute_xof_seeds`). Each 32-byte XOF seed
is then expanded to a deterministic stream:

    stream_c = blake3(xof_seed_c).digest(length=TOTAL_XOF_BYTES_PER_CHANNEL)

`TOTAL_XOF_BYTES_PER_CHANNEL` is the sum of the per-octave grid sizes
defined in `protocol/tile_params.py`. For the current configuration
(`NUM_OCTAVES=4`, `GRID_H_TABLE=[17,34,68,135]`,
`GRID_W_TABLE=[30,60,120,240]`), this is 17·30 + 34·60 + 68·120 +
135·240 = **43,110 bytes per channel** (≈ 129 KB for RGB combined).

### Multi-octave grids

The XOF stream is split into four sub-grids (`NUM_OCTAVES = 4`). The
slice boundaries are precomputed as cumulative sums of the per-octave
grid sizes (see `_OFFSETS` in `protocol/tile_gpu.py` and the
channel-local `offset` variable in `protocol/tile_cpu.py`). Each octave
yields a `(GRID_H_TABLE[o], GRID_W_TABLE[o])` grid of centred signed
bytes (raw uint8 value minus 128).

### Integer bilinear upsample

Each grid is upsampled to `(TILE_H, TILE_W) = (1080, 1920)` using a
fixed-point integer-bilinear kernel (`_integer_bilinear_upsample` in
CPU code, `_upsample_bilinear_int_cuda` in GPU code). The fractional
representation uses a 16-bit shift (`SB = 16`, `S = 1 << SB`), so
coordinates on the source grid are computed as

    xs = (arange(out_w, dtype=int64) * (gw - 1) * S) // max(out_w - 1, 1)
    ix = xs >> SB
    fx = xs & (S - 1)
    ix_n = min(ix + 1, gw - 1)

and the bilinear mix is done in two passes — horizontal in int32, then
vertical in int64 to keep the full `2^40` intermediate — ending in an
arithmetic right-shift by 16 bits to recover an int32 upsampled value.
The CPU and GPU implementations are bit-exact by construction; tb_loop
runs a determinism check at session start that verifies CPU and GPU
tiles for the same seeds are byte-identical.

### fBm composition

The four octave contributions are summed with geometric 1/2^oct
weighting in integer arithmetic:

    frame = v[0]
    for oct in 1..NUM_OCTAVES-1:
        frame += v[oct] >> oct          # arithmetic shift, not divide

Output is then centred and clipped back to uint8:

    out_u8 = clip((frame >> 16) + 128, 0, 255)

This is a discrete integer-fBm texture whose spatial frequency content
spans ~8 pixel features (finest octave, 1080/135 ≈ 8) to ~64 pixel
blobs (coarsest octave, 1080/17 ≈ 64), weighted 1/2^oct. The resulting
tile has coherent low-frequency structure (visible projected patterning,
robust under camera integration) and high-frequency detail (fine-scale
verification signal), unlike per-pixel uniform noise.

### Generator source binding

`generator_code_hash` is BLAKE3 over the concatenated source text of the
generator's core functions: `tile_cpu._tables`, `tile_cpu.gen_channel_v2`,
`tile_cpu.gen_rgb_v2` (CPU backend) or `tile_gpu._tables_torch`,
`tile_gpu._upsample_bilinear_int_cuda`, `tile_gpu.gen_rgb_tile_cuda`
(GPU backend). Because the source is committed into the bundle, the
exact implementation — not just its shape — is pinned for verification.

## 4. Recording modes

Three modes are defined in `protocol/tb_loop.py`; one is selected per
session via `--mode`.

### Blocking mode (`--mode blocking`)

Single-threaded synchronous chain loop. Each iteration:

1. Accept the next capture from the camera whose trigger time is at or
   after `prev_queue_draw_wall_ns + pipeline_wait_ms` (see §5).
2. Compute `bayer_hash`; write the raw bayer file with `fsync` +
   directory fsync.
3. Advance the chain (`S_{t+1} = compute_chain_row(S_t, bayer_hash,
   meta_bytes)`).
4. Generate the next emission tile from `S_{t+1}`.
5. Queue the tile for projection; record `tile_queued_wall_ns`.
6. Append chain and capture log rows; `fsync` both.

Mechanical correspondence: every chain step has exactly one capture,
and every capture's exposure is provably after the previous queue_draw
plus the calibrated wait. Requires a valid pipeline calibration for
the current `(exposure_us, gain_db, projector_connector, EDID, camera)`
tuple.

### Async mode (`--mode async`)

Five-thread pipeline that consumes captures at the camera rate without
mechanical per-step wait. Chain advances are decoupled from capture
delivery: a capture may be projected-against-while-still-transitioning
and still enter the chain; the reconstructor at verify time is
responsible for assessing when the DMD actually settled. No calibration
is required.

### Async-tightened mode (`--mode async-tightened`)

Async with best-effort runtime mitigations applied by
`protocol/async_mitigations.py`:

- `SCHED_FIFO` with high priority on the capture and chain threads.
- CPU affinity pinning to isolate the pipeline from scheduler noise.
- `mlockall` to prevent paging of generator buffers.
- `/dev/cpu_dma_latency = 0` to block c-state transitions.
- Garbage-collection lockdown during the session.
- Pre-allocated capture ring with fixed buffer count.

Each mitigation is capability-probed (`probe_capabilities`) and falls
back silently on permission denial. Mitigations that actually landed
are recorded in `manifest.mitigations_applied`.

## 5. Timing correspondence

### Hardware triggering

The camera is operated in `TriggerMode=On`, `TriggerSource=Line1`,
`TriggerSelector=FrameStart`, `TriggerActivation=RisingEdge` (see
`_setup_camera` in `protocol/tb_loop.py`). `Line1` is wired to the
projector's VSYNC output, so every HDMI vertical blanking interval
emits a capture-start pulse. Exposure is fixed via `ExposureTime` in
microseconds; gain fixed via `Gain` in dB.

### Global shutter rationale

The rig uses a global-shutter IMX540 sensor. Rolling-shutter sensors
would inject row-wise exposure skew into each capture, which would
appear in the bayer hash but not in any deterministic metadata, making
verifier reconstruction of the projected-vs-captured timing ambiguous.
Global shutter gives a single integration window per frame, which is
what `meta_bytes.exposure_us` claims.

### Pipeline calibration

`tools/calibrate_pipeline_delay.py` empirically measures how many
milliseconds the DMD needs, post-`queue_draw`, before a new capture's
exposure window reliably sees the new tile. It projects alternating
solid-white and solid-black tiles with varying `wait_ms` and classifies
each trial by the mean intensity of the capture relative to
`(stable_white_mean + stable_black_mean) / 2`:

- `p99_wait_ms`: smallest wait at which ≥99 % of trials transitioned
  cleanly in both W→B and B→W directions.
- `p100_wait_ms`: smallest wait at which all trials did.
- `recommended_wait_ms = max(W→B.p100, B→W.p100) + 20`, with a hardcoded
  20 ms safety margin.

The calibration JSON (per-wait correctness fractions, stable-phase
intensity distributions, rig config hash, derivation timestamp) is
what gets folded into `bundle_hash` on blocking-mode sessions, so any
future claim about the timing of that specific session is pinned to
these measurements.

### Accept-after-threshold rule (blocking, v8.2.1)

The v8.2.1 accept rule maps each frame's device timestamp (or host
receive time, minus a conservative readout estimate `READOUT_NS`) into
the host monotonic-ns clock and compares against
`prev_queue_draw_wall_ns + pipeline_wait_ms * 1e6`. Frames whose
trigger lands before that threshold are pushed back to the camera's
buffer pool and the loop re-pops. This replaced the earlier "drain +
sleep + wait for fresh capture" which was correct but ceded an extra
camera-delivery cycle on every step.

## 6. Blockchain anchoring

### Fresh-block seed

When `--rsk-rpc <url>` is passed, tb_loop at session start calls
`rsk_anchor.fetch_current_tip` to obtain the current block number and
then `wait_for_next_block` to obtain the *next* block. The
newly-landed block's hash, number, parent hash, block timestamp, and
host observation time are written to `manifest.anchor_start`:

    {
      "chain": "rootstock",
      "chain_id": 30,      # 30 = mainnet, 31 = testnet
      "block_number": ...,
      "block_hash": "a1b2c3d4e5f6...",   # bare 64-char hex, no "0x" prefix
      "block_timestamp_utc": "...",
      "parent_hash": "...",              # bare 64-char hex, no "0x" prefix
      "observed_at_utc": "..."
    }

All seven fields are folded into `manifest_hash_open` and therefore
into `S_0`. Consequently, the session's opening state cannot have been
committed before `anchor_start.block_timestamp_utc`, because the block
hash did not exist before that time.

### End anchor

When `--interior-anchor` is passed, a background publisher
(`protocol/anchor_publisher.py`) periodically broadcasts RSK
transactions carrying the latest committed `S_t` in the data field.
At session end, the publisher posts a terminal transaction whose
payload commits to `manifest_hash_final` (or, more precisely, to the
final root derived in `compute_final_root`, which chains
`bundle_hash`, `manifest_hash_open`, `S_N`, and `anchor_log_hash`
together under a domain tag).

### Chain-id verification

Both seeding and end-anchor paths call `rsk_anchor.fetch_chain_id` and
compare against the expected value in `NETWORK_CHAIN_IDS` (30 for
mainnet, 31 for testnet). A mismatch aborts the session with a clear
error; a wrong-network RPC cannot silently stand in for the correct
one.

## 7. Artifact format

### Session directory

    <session_dir>/
      manifest.json, manifest.pretty.json
      verification_bundle.json, verification_bundle.pretty.json
      chain_log.csv                  # one row per captured frame, consumed
      capture_log.csv                # one row per delivered frame
      anchor_txs.csv                 # when --interior-anchor; else absent
      Recordings/frame_NNNNNN.raw    # authoritative bayer captures
      derived/
        Emissions/tile_NNNNNN.png    # best-effort tile encode
        Recordings_previews/         # post-finalize debayered preview

### manifest.json

A JSON-serialised `SessionManifest`. Schema is enforced by
`SessionManifest.validate_dict`; unknown fields are rejected on load.
Fields fall into three groups:

- Opening-state (pinned into `manifest_hash_open`): `session_id`,
  `session_iso_utc_start`, `bundle_hash`, `device_id`,
  `anchor_policy`, `anchor_start`, with `session_status` hardcoded
  to `"open"` during hashing.
- Disk-only: `S_0_hex`, `session_status` (the live value, post-
  finalize/abort).
- Ending-state (pinned into `manifest_hash_final`):
  `session_iso_utc_end`, `N_chain`, `N_captures`, `S_N_hex`,
  `capture_log_hash`, `chain_log_hash`, `anchor_end`,
  `anchor_log_hash`, `camera_acquisition_*_wall_ns`,
  `mitigations_applied`.

### verification_bundle.json

A pinned snapshot of every non-session-specific configuration. It is
identical across sessions with identical rig + protocol + generator
code. `bundle_hash` = BLAKE3-over-canonical-JSON over every non-
`bundle_hash` field.

### chain_log.csv / capture_log.csv

Both files begin with a single comment line `# session_iso_utc=...`
(the verifier asserts this exact prefix) and continue with a
DictWriter-produced CSV body. `chain_log` rows commit to the
per-step chain state, XOF seeds, tile PNG path, and the 28-byte
meta struct (hex-encoded). `capture_log` rows record every delivered
frame, whether or not it entered the chain, with a `consumed_as_t`
back-pointer to the chain row.

### Raw captures

Raw bayer payloads are stored in `Recordings/` as
`frame_NNNNNN.raw` at sensor dimensions (e.g. 5320×4600 BayerRG8 ≈
24.5 MB each). These are the authoritative capture artefacts; the
preview PNGs are regenerable and can be discarded.

## Package layout (reference)

    truth_beam_recording/
      protocol/        # recording code; flat scripts, directly executable
        tb_main.py          # CLI entry point + session-lifecycle main()
        tb_loop.py          # TBLoopV8 GTK application + class-owned helpers;
                            # also executable as a legacy shim that defers
                            # to tb_main.main()
        chain.py            # S_0, S_{t+1}, XOF seeds, META struct
        camera.py           # Aravis setup / probes / acquire
        projector.py        # GTK/Cairo tile projection + EDID probe
        session_logs.py     # CSV column definitions and writers
        session_finalize.py # post-session previews + reconcile + PNG save
        rsk_integration.py  # preflight, chain_id verification, wallet guard
        host_info.py        # host-identity probe for the bundle
        tile_backend.py     # select tile_cpu vs tile_gpu, hash generator src
        session_schema.py
        tile_params.py
        tile_cpu.py, tile_gpu.py
        rsk_anchor.py, rsk_wallet.py, anchor_publisher.py
        async_mitigations.py
      verify/          # third-party verifier (ships with this release)
        verify_v9.py               # recompute S_0, walk the chain row-by-row
        verify_generator_hash.py   # recompute the tile-generator code hash (no GPU)
        README_BUNDLE.md, CLAIMS.md
      tools/           # dev utilities; reach into protocol/ via sys.path
        calibrate_pipeline_delay.py
        feasibility_capture.py
        analyze_blocking_timing.py
        ast_equivalence_check.py    # post-refactor behaviour-preservation check
        golden_diff.py              # session-pair differ (see GOLDEN_DIFF.md)
        GOLDEN_DIFF.md
      README.md, CITATION.cff, METHODS.md

The third-party verifier ships in `verify/` (chain walk + generator-hash check);
the broader verification stack (Phase G/F/H) is in `../verifier/`.

A flat `__init__.py` is present in each subdirectory for future-proofing
(`python -m truth_beam_recording.protocol.tb_main` will work once a
packaging boundary is desired), but the shipping form is still
directly-executable scripts: `python3 protocol/tb_main.py <args>`.
`protocol/tb_loop.py` is retained as a legacy entry-point shim so
existing automation continues to work; new callers should use
`tb_main.py`.

## Verification

Verification procedure is specified in a separate document accompanying
the `truth_beam_verification` project. The verifier reconstructs `S_0`
from the session directory alone (`manifest.json` +
`verification_bundle.json` self-contain all inputs), walks the chain
from `S_0` to `S_N`, re-hashes every capture to confirm `bayer_hash`
inclusion, and optionally re-submits `anchor_end` to an RSK node to
confirm the end-of-session anchor.

# Truth Beam

Hardware-bound tamper-evident optical recording: a projector-camera rig
emits a deterministic pseudorandom image chain and records the resulting
captures under a BLAKE3 state chain, with opening and closing states
optionally anchored on-chain (RSK). Sessions run unanchored by default;
anchoring is enabled with `--rsk-rpc` and `--network` flags.

## Overview

Each session projects a sequence of pseudorandom RGB tiles generated
from a hash chain `{S_t}`. Captures from a hardware-triggered global-
shutter camera are folded back into the chain's state transition, so
every bayer frame is cryptographically committed under the exact rig
configuration, generator source, exposure settings, and (optionally)
a fresh external block hash. See **METHODS.md** for the full protocol
specification.

## Hardware requirements

- A DMD-based projector on an HDMI output, sync-locked at 60 Hz (the
  implementation has been developed against a single-connector setup
  using HDMI-1 as the tile output).
- A global-shutter industrial camera exposing GenICam / Aravis with
  hardware triggering via a Line1 input wired to the projector VSYNC.
  Reference configuration: The Imaging Source IMX540, BayerRG8.
- GPU with CUDA (optional but strongly preferred; CPU fallback is
  bit-exact but ~10× slower per tile).
- Host storage capable of ~73 MB/s sustained writes for the raw bayer
  capture rate at full sensor.

## Software requirements

- Python 3.10+
- `blake3`, `numpy`, `torch` (for GPU tile generator), `opencv-python`,
  `Pillow`
- `PyGObject` / `gi` bindings for GTK 4, GDK 4, Aravis 0.8
- `eth_account`, `web3` (for RSK anchoring)

Developed and tested on Linux. The async-tightened mode requires
Linux-specific facilities (SCHED_FIFO, mlockall, /dev/cpu_dma_latency).
Recording has not been tested on macOS or Windows.

## Installation

    git clone <repo>
    cd truth_beam
    pip install blake3 numpy torch opencv-python Pillow eth-account web3

For GTK / Aravis on Debian/Ubuntu:

    sudo apt install python3-gi gir1.2-gtk-4.0 gir1.2-aravis-0.8

## Quick start

1. Calibrate pipeline delay for the current rig (one-off per
   exposure/gain/monitor/camera combination):

        python3 tools/calibrate_pipeline_delay.py --connector HDMI-1 \
            --exposure-us 64000 --gain-db 24.0

   This writes `~/.tb/pipeline_calibration_<rig_hash>.json`.

2. Record a blocking-mode session:

        python3 protocol/tb_main.py --mode blocking --connector HDMI-1 \
            --duration 60 --exposure-us 64000 --gain 24.0 \
            --session-dir ./sessions/my_session

   (`protocol/tb_loop.py` remains executable as a legacy entry-point
   shim and defers to `tb_main.main()`; new callers should use
   `tb_main.py` directly.)

3. (Optional) run the timing diagnostic against the session:

        python3 tools/analyze_blocking_timing.py ./sessions/my_session

4. (Verification is handled by a separate project; see Scope below.)

See **METHODS.md** for the on-chain fresh-block seeding flags
(`--rsk-rpc`, `--network`), `--interior-anchor` for end-of-session
tx emission, the async / async-tightened modes, and the artefact
layout.

## Directory structure

    recording/
    ├── protocol/           # recording core (tb_loop, tile generators,
    │                       # session schema, RSK anchoring,
    │                       # async mitigations)
    ├── tools/              # calibration + diagnostic scripts
    ├── verify/             # third-party verifier: verify_v9.py,
    │                       # verify_generator_hash.py, README_BUNDLE.md,
    │                       # CLAIMS.md
    ├── README.md           # this file
    ├── METHODS.md          # protocol specification
    └── CITATION.cff

Licensing is governed by the repository-root `LICENSE` (all rights reserved).

## Citation

See `CITATION.cff`.

## Licensing

All rights reserved; no open-source or patent license is granted (see the
repository-root `LICENSE`). Contact the author for any reuse.

## Scope

This directory covers the recording protocol. The verifier ships alongside it
in `verify/` (`verify_v9.py` walks the chain; `verify_generator_hash.py`
recomputes the tile-generator code hash) and the broader verification stack is
in `../verifier/`.

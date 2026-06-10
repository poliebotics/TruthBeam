"""tb_main_patterns.py — pattern-emission variant of tb_main.

Monkey-patches the tile-generator loader to `load_tile_generator_patterns`
before TBLoopV8 resolves it, then runs the normal main(). Session output,
chain, manifest, and calibration discipline are unchanged; only the emission
pixels differ.

Intended for short visual-timing-verification sessions. See generator_patterns.py.
"""
from __future__ import annotations

import tile_backend  # noqa: E402
import tb_loop  # noqa: E402
from generator_patterns import load_tile_generator_patterns  # noqa: E402

tile_backend.load_tile_generator = load_tile_generator_patterns
tb_loop._load_tile_generator = load_tile_generator_patterns

import tb_main  # noqa: E402


if __name__ == "__main__":
    tb_main.main()

#!/usr/bin/env python3
"""verify_generator_hash.py — verify a session's generator_code_hash against
the published tile-generator source code, with no GPU required.

This closes the code→chain verification link: verify_v9.py proves the chain
is internally consistent (S_0 derivation, row-by-row state transitions), but
the chain only attests to the *generator that ran* via the BLAKE3 digest
pinned in verification_bundle.json["generator_code_hash"]. This script
recomputes that digest from the source files in this repository and compares.

The recording rig computes the digest as (see protocol/tile_backend.py):

    blake3( inspect.getsource(_tables_torch)
          + inspect.getsource(_upsample_bilinear_int_cuda)
          + inspect.getsource(gen_rgb_tile_cuda) )      # GPU backend

    blake3( inspect.getsource(_tables)
          + inspect.getsource(gen_channel_v2)
          + inspect.getsource(gen_rgb_v2) )             # CPU backend

inspect.getsource() requires importing the module, and protocol/tile_gpu.py
initialises CUDA at import time. To allow verification on machines without a
GPU, this script extracts the identical source text via the `ast` module
instead of importing. The two methods are byte-identical for undecorated
top-level functions (validated against the rig's own CPU-backend digest).

Usage:
    python3 verify/verify_generator_hash.py <session_dir>
        where <session_dir> contains verification_bundle.json
    python3 verify/verify_generator_hash.py --hash <hex_digest>
        compare against an explicit digest (e.g. read off the video/paper)

Exit code 0 = MATCH, 1 = MISMATCH or error.

Known-good values for the published TruthBeam sessions (D2, V10):
    generator_code_hash = 154be9dd75e0586df456a7eae1528b7334a415f3a977a107c91c6b0751bfc540
    (GPU backend; both sessions were recorded with the same generator source)
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

try:
    from blake3 import blake3
except ImportError:
    sys.exit("ERROR: pip install blake3")

PROTOCOL_DIR = Path(__file__).resolve().parent.parent / "protocol"

# (source file, [function names in hashing order]) — must mirror tile_backend.py
GPU_SPEC = ("tile_gpu.py", ["_tables_torch", "_upsample_bilinear_int_cuda", "gen_rgb_tile_cuda"])
CPU_SPEC = ("tile_cpu.py", ["_tables", "gen_channel_v2", "gen_rgb_v2"])


def extract_function_sources(path: Path, names: list[str]) -> str:
    """Return the concatenated source text of top-level functions `names` in
    `path`, in the given order, byte-identical to inspect.getsource()."""
    src = path.read_text()
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    found: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            start = (node.decorator_list[0].lineno if node.decorator_list else node.lineno) - 1
            found[node.name] = "".join(lines[start : node.end_lineno])
    missing = [n for n in names if n not in found]
    if missing:
        raise RuntimeError(f"{path.name}: functions not found: {missing}")
    return "".join(found[n] for n in names)


def backend_hash(spec: tuple[str, list[str]]) -> str:
    fname, funcs = spec
    return blake3(extract_function_sources(PROTOCOL_DIR / fname, funcs).encode("utf-8")).hexdigest()


def main(argv: list[str]) -> int:
    if len(argv) != 2 and not (len(argv) == 3 and argv[1] == "--hash"):
        print(__doc__)
        return 1

    if argv[1] == "--hash":
        target = argv[2].strip().lower()
        source = "(command line)"
    else:
        bundle_path = Path(argv[1]) / "verification_bundle.json"
        if not bundle_path.exists():
            print(f"ERROR: {bundle_path} not found")
            return 1
        target = json.loads(bundle_path.read_text())["generator_code_hash"].lower()
        source = str(bundle_path)

    gpu = backend_hash(GPU_SPEC)
    cpu = backend_hash(CPU_SPEC)

    print(f"target  ({source}):\n  {target}")
    print(f"recomputed from {PROTOCOL_DIR}:\n  GPU backend: {gpu}\n  CPU backend: {cpu}")

    if target == gpu:
        print("\nMATCH (GPU backend) — the published generator source is the code "
              "committed in this session's genesis hash.")
        return 0
    if target == cpu:
        print("\nMATCH (CPU backend) — the published generator source is the code "
              "committed in this session's genesis hash.")
        return 0
    print("\nMISMATCH — the published source does NOT hash to the session's "
          "generator_code_hash. Either the source differs from what ran, or the "
          "bundle is from a different code version.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

"""tile_backend.py — select and load the active tile generator backend.

Scope: picking between `tile_cpu` and `tile_gpu` at session start, computing
`generator_code_hash` (which is pinned into the verification bundle), and
returning the chosen generator's `gen(xof_r, xof_g, xof_b) -> ndarray`
function ready to be called on the chain thread.

Adapt this module for:
- A third backend (e.g. a shader-based or vectorised-SIMD implementation)
  — add it as an `elif` branch and extend the `inspect.getsource`
  concatenation to cover the new backend's bit-exact surface
- A different source-hash discipline (currently: concatenated source text
  of the generator's three core functions under BLAKE3)

Do NOT put here:
- Tile math itself — that's tile_cpu.py / tile_gpu.py
- XOF seed derivation — chain.py / session_schema.py
- Session state — the class in tb_loop.py owns backend_name, gen_tile_fn,
  generator_code_hash as attributes
"""
from __future__ import annotations

import inspect
from concurrent.futures import ThreadPoolExecutor

from blake3 import blake3


def load_tile_generator(cpu_only: bool):
    """Load the active tile generator. Returns (backend_name, gen_func,
    source_hash_hex, cpu_pool). `gen_func` signature: (xof_r, xof_g,
    xof_b) -> uint8 (TILE_H, TILE_W, 3) array.

    `gen_func` must be called from a single thread (the chain thread);
    the GPU backend reuses a set of pinned/device buffers that are not
    thread-safe. The returned `cpu_pool` is the ThreadPoolExecutor the
    CPU backend uses for per-channel parallelism (or None on the GPU
    path); the caller is responsible for shutting it down at session
    end."""
    if cpu_only:
        import tile_cpu
        pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="tb-chan-v8")
        def gen(xr, xg, xb):
            return tile_cpu.gen_rgb_v2(xr, xg, xb, pool)
        src = (
            inspect.getsource(tile_cpu._tables)
            + inspect.getsource(tile_cpu.gen_channel_v2)
            + inspect.getsource(tile_cpu.gen_rgb_v2)
        )
        return "tile_cpu", gen, blake3(src.encode("utf-8")).hexdigest(), pool
    import tile_gpu
    # Touch the module-level pre-allocations. They initialize on first
    # import; this is here for explicitness.
    _ = tile_gpu._TABLES
    _ = tile_gpu._OFFSETS
    _ = tile_gpu._stream_pinned
    _ = tile_gpu._out_pinned
    _ = tile_gpu._stream_dev
    def gen(xr, xg, xb):
        return tile_gpu.gen_rgb_tile_cuda(xr, xg, xb)
    src = (
        inspect.getsource(tile_gpu._tables_torch)
        + inspect.getsource(tile_gpu._upsample_bilinear_int_cuda)
        + inspect.getsource(tile_gpu.gen_rgb_tile_cuda)
    )
    return "tile_gpu", gen, blake3(src.encode("utf-8")).hexdigest(), None

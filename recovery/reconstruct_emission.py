#!/usr/bin/env python3
"""reconstruct_emission.py — regenerate a 2023 TruthBeam emission frame from its
BLAKE3 chain hash, and reconstruct a lost storage block from its byte-exact
neighbours.

Background
----------
The "Truth Beam PoliePals Trailer" pin (IPFS root QmejyJWognSYn7UhygsHuQzkDK5vY4i
zU9SsCL785NsHCN) is the *original TruthBeam recording* made for the 2023 trailer.
One 262,144-byte IPFS block of one emission frame became unavailable from Pinata
("no providers found"), block index 32 of frame 000511
(file bytes 8388608..8650751; CID QmRyXZyDbVaE6V7vKBL873DwhA8Xb5VSX7ujcoM1kp3Hpx).

The byte-exact original resurfaced on a retained backup on 2026-07-27 and
reproduces both pre-recorded CIDs. See RECOVERY_RECEIPT.md. This script is kept
because it documents and reproduces the earlier approximation without requiring
the recovered bytes.

Each emission is a *deterministic* function of its chain hash (the filename hash =
the chain's emission seed `Ep`). The 2023 generator (from the TruthBeam slide of
the PolieBotics whitepaper video):

    def emission_from_hash(input_hash):
        output_hash = blake3(input_hash.encode('utf-8')).hexdigest(length=1536)
        complex_array = [complex(int(output_hash[i:i+2],16),
                                 int(output_hash[i+2:i+4],16))
                         for i in range(0, len(output_hash), 4)]          # 768 values
        emission = 512*cv2.resize(numpy.abs(tf.signal.ifft3d(tf.signal.ifft3d(
                       numpy.reshape(complex_array,(16,16,3))))), (1024,1024))
        return emission

So the data is recoverable from the hash. **Bit-exactness caveat:** the result
runs through an FFT + bilinear resize in floating point. Those are NOT
bit-reproducible across different hardware/library builds (cuFFT/Eigen sum in
different orders), and float32 rounding makes the inverse non-unique. So this
reconstruction matches the original to ~float32 precision (max error ~2e-5 on a
0..255 scale; ~60-75% of values bit-identical) but does NOT reproduce the exact
original bytes, so its CID is deliberately distinct. Comparison with the later
byte-exact recovery measures 76.2085% of the block values as bit-identical and a
maximum absolute error of 1.52587890625e-05. This is itself the argument for
committing to a canonical/quantised representation (or hashing the seed, not the
raw float bytes), which the 2026 protocol's integer/fixed-point tile generator
does.

Two reconstruction paths are provided:
  1. emission_from_hash(seed)        — full frame from the hash (the XOF demo).
  2. reconstruct_block(frame, ...)   — recover a lost interior block from the
                                       frame's own byte-exact neighbours via the
                                       bilinear-band structure of cv2.resize
                                       (~2e-5, the better path when neighbours
                                       survive, as here).

Usage:
    python3 reconstruct_emission.py from-hash <seed_hex> <out.npy>
    python3 reconstruct_emission.py fill-block <partial_frame.npy> <out.npy> \
            [byte_start byte_end npy_header_len]
"""
from __future__ import annotations
import sys
import numpy as np
import cv2
try:
    from blake3 import blake3
except ImportError:
    blake3 = None


def emission_from_hash(input_hash: str) -> np.ndarray:
    """Exact transcription of the 2023 generator. tf.signal.ifft3d is replaced by
    numpy.fft.ifftn over the 3 axes — mathematically identical; they differ only
    at the float-ULP level (which is not bit-reproducible across hardware anyway).
    complex64 matches the original single-precision FFT most closely."""
    if blake3 is None:
        raise RuntimeError("pip install blake3")
    output_hash = blake3(input_hash.encode("utf-8")).hexdigest(length=1536)  # 1536 B -> 3072 hex
    arr = []
    i = 0
    while i < len(output_hash):
        arr.append(complex(int(output_hash[i:i + 2], 16), int(output_hash[i + 2:i + 4], 16)))
        i += 4
    c = np.array(arr, dtype=np.complex64).reshape(16, 16, 3)
    x = np.fft.ifftn(np.fft.ifftn(c, axes=(0, 1, 2)), axes=(0, 1, 2))
    m = np.abs(x).astype(np.float32)
    return (np.float32(512.0) * cv2.resize(m, (1024, 1024), interpolation=cv2.INTER_LINEAR)).astype(np.float32)


def reconstruct_block(frame: np.ndarray, miss_idx_start: int, miss_idx_end: int) -> np.ndarray:
    """Recover the flat-index range [miss_idx_start, miss_idx_end] of a (1024,1024,3)
    emission from its byte-exact neighbours. cv2.resize upsamples 16->1024 as
    src=(Y+0.5)*16/1024-0.5; within a band where iy=floor(src) is constant the
    output is exactly linear in Y, so the missing interior rows are determined by
    the surviving rows in the same band. Only the requested indices are replaced;
    all byte-exact values are preserved untouched."""
    H = 1024
    scale = 16.0 / 1024.0

    def band(y):
        s = (y + 0.5) * scale - 0.5
        s = min(max(s, 0.0), 15.0)
        iy = min(int(np.floor(s)), 14)
        return iy, s - iy

    rows = sorted(set((np.arange(miss_idx_start, miss_idx_end + 1) // (1024 * 3)).tolist()))
    flat = frame.reshape(-1).copy()
    for y in rows:
        iy, fy = band(y)
        known = [yy for yy in range(H) if band(yy)[0] == iy and yy not in rows]
        f = np.array([band(yy)[1] for yy in known])
        A = np.stack([1 - f, f], 1)
        Y = frame[known, :, :].reshape(len(known), -1).astype(np.float64)
        sol, *_ = np.linalg.lstsq(A, Y, rcond=None)            # H10, H11 per (col,ch)
        rec = ((1 - fy) * sol[0] + fy * sol[1]).reshape(1024, 3).astype(np.float32)
        r0 = y * 1024 * 3
        flat_rec = rec.reshape(-1)
        lo = max(miss_idx_start, r0)
        hi = min(miss_idx_end, r0 + 1024 * 3 - 1)
        if lo <= hi:
            flat[lo:hi + 1] = flat_rec[lo - r0:hi - r0 + 1]    # splice only missing indices
    return flat.reshape(1024, 1024, 3)


def main(argv):
    mode = argv[1]
    if mode == "from-hash":
        seed, out = argv[2], argv[3]
        em = emission_from_hash(seed)
        np.save(out, em)
        print(f"wrote {out}  shape={em.shape} dtype={em.dtype} range=({em.min():.4f},{em.max():.4f})")
    elif mode == "fill-block":
        partial, out = argv[2], argv[3]
        # default: block 32 of frame 000511, .npy header 128 bytes
        bstart = int(argv[4]) if len(argv) > 4 else 8388608
        bend = int(argv[5]) if len(argv) > 5 else 8650751
        hdr = int(argv[6]) if len(argv) > 6 else 128
        frame = np.load(partial)
        i0 = (bstart - hdr) // 4
        i1 = (bend - hdr) // 4
        rec = reconstruct_block(frame, i0, i1)
        np.save(out, rec)
        print(f"wrote {out}  filled flat idx {i0}..{i1} ({i1 - i0 + 1} float32) "
              f"shape={rec.shape} dtype={rec.dtype}")
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

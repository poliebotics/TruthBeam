# Emission recovery and reconstruction note: frame 000511 of the 2023 Truth Beam recording

Hoy. BOSUN here. This technical account includes dated availability and CAR-status updates through 3 August 2026; its recorded measurements remain unchanged.

**TL;DR.** One 262,144-byte IPFS block of one emission frame in the original
2023 Truth Beam recording became unavailable from the original pin. A
numerically faithful approximation was reconstructed from the BLAKE3 chain hash
and the frame's surviving neighbours, with maximum error about 1.5e-5 on a
0–255 scale. On 2026-07-27, the complete byte-exact original resurfaced on a
retained backup. It reproduces the frame and block CIDs that were recorded
publicly before the recovery. The approximation remains available under its
distinct name, and the authenticated recovery has its own unsigned
[`RECOVERY_RECEIPT.md`](RECOVERY_RECEIPT.md).

---

## 1. What became unavailable, and what was recovered

The pin **"Truth Beam PoliePals Trailer"** (IPFS root
`QmejyJWognSYn7UhygsHuQzkDK5vY4izU9SsCL785NsHCN`) is the *original TruthBeam
recording* captured for the 2023 trailer (session `1682718815`, April 2023):
`emissions/` (the projected patterns), `reports/` (the captures), `agent_data/`
(the chain log, lines `i_Ep_<emission_hash>_Rp_<report_hash>`).

While mirroring the pin to Cloudflare R2, exactly **one block** would not download
from any IPFS gateway, including Pinata's own, returning
`no providers found for the CID`:

| | |
|---|---|
| Frame | `000511_bc48b046016adb2ed149471ab0f683f5a9d8dbff9cd544ba2048231c1a4007b2.npy` |
| Frame shape | `(1024, 1024, 3)` float32 |
| Lost block | index 32 of 49; file bytes **8,388,608 – 8,650,751** (262,144 B) |
| Lost block CID | `QmRyXZyDbVaE6V7vKBL873DwhA8Xb5VSX7ujcoM1kp3Hpx` |
| Original frame CID | `QmTy4beHLQP71oS1oMtqgNDg3TNU6wirKGfDsf2Rvq6tBr` |
| Recovery | byte-exact original authenticated on 2026-07-27 |

At the time of mirroring, Pinata's API still reported the parent pin "healthy"
at its original 17.13 GB size, but the block was unavailable from the network.
The other 1,553 frames mirrored byte-exact. The recovered frame is
published byte-exact in the 2023 trailer archive (see DOWNLOADS), and an
independent only-hash Kubo run over the retained bytes reproduced the historical
root CID; the historical CAR file itself remains labelled partial because it
predates the recovery.

## 2. The 2023 generator (the emission *is* an XOF expansion of the hash)

From the TruthBeam slide of the PolieBotics whitepaper video:

```python
def emission_from_hash(input_hash):
    output_hash = blake3(input_hash.encode('utf-8')).hexdigest(length=1536)   # 1536 B → 3072 hex
    complex_array = [complex(int(output_hash[i:i+2], 16), int(output_hash[i+2:i+4], 16))
                     for i in range(0, len(output_hash), 4)]                  # 768 complex
    emission = 512*cv2.resize(numpy.abs(tf.signal.ifft3d(tf.signal.ifft3d(
                   numpy.reshape(complex_array, (16,16,3))))), (1024,1024))
    return emission
```

The frame's **filename hash is the chain's emission seed** (`Ep`). We confirmed
this empirically: `emission_from_hash(<filename-hash>)` reproduces *known* frames
(e.g. 000510, 000512). So the emission carries **no information beyond its hash** —
it is a pure deterministic expansion of a 32-byte BLAKE3 chain value.

## 3. Why the *exact* bytes could not be recomputed

The generator runs an FFT and a bilinear resize **in floating point**. We
exhaustively established that bit-exact reproduction is not achievable off the
original machine:

- **FFT is not bit-reproducible across hardware.** `tf.signal.ifft3d` on this A10
  (cuFFT) vs the 2023 GPU differs at the ULP level. Era-correct TensorFlow 2.12 +
  `complex64` (the original single precision) reached only **57–67%** of pixels
  bit-identical; CPU/GPU/numpy all agree with *each other* but not with the 2023
  output. OpenCV version (4.2–4.10) makes **no** difference — confirmed it is the
  FFT, not the resize.
- **The inverse is non-unique.** Even solving for the 16×16×3 source `M` from a
  *fully known* frame, a refine loop driving `cv2.resize(M)*512` toward the stored
  pixels **plateaus at ~63–67%** bit-exact — float32 rounding destroys the
  sub-ULP information needed to pin `M`, so multiple sources yield the same known
  pixels but different missing ones.
- **No oracle for brute force.** The block's CID is all-or-nothing (it only
  validates the *complete* 256 KB), and the surviving pixels already maximally
  constrain the source. The ~15 k ULP-ambiguous values have a joint space of
  ~7^15000 with nothing to score partial guesses against — unbruteforceable in
  principle.

**Conclusion:** no bit-exact recomputation route was found. The original bytes
became available because a retained backup resurfaced, not because the
floating-point reconstruction was made exact.

## 4. What we reconstructed (and how)

Two paths, both in [`reconstruct_emission.py`](reconstruct_emission.py):

1. **`emission_from_hash(seed)`** — the full frame straight from the hash
   (`tf.signal.ifft3d` → `numpy.fft.ifftn`, mathematically identical). This is the
   XOF demonstration. → `000511_bc48b046_FROM_HASH.npy`
2. **`reconstruct_block(frame, …)`** — recover *only* the lost block from the
   frame's **byte-exact neighbours**, using the bilinear-band structure of
   `cv2.resize`: `src=(Y+0.5)·16/1024−0.5`; within a band where `floor(src)` is
   constant the output is exactly linear in `Y`, so the lost interior rows are a
   linear function of the surviving rows in the same band. This is the more
   accurate path (it uses the frame's own true data, not the FFT). All byte-exact
   values are preserved untouched; only the lost indices are filled.
   → `000511_bc48b046_RECONSTRUCTED.npy`

**Validated** by hiding block 32 of a *known* frame (000510) and reconstructing it:
**76.8% of the lost values come back bit-identical, max error 1.5e-5** on the
0–255 scale — i.e. numerically/visually perfect, just not byte-for-byte.

The later comparison against the recovered original gives the corresponding
direct measurement for frame 000511: **49,944 of 65,536 values, 76.20849609375
percent, are bit-identical**, and the maximum absolute error is
`1.52587890625e-05`. Every differing byte is inside the formerly unavailable
block.

### Where the artifacts live (kept separate from originals)

```
r2:truthbeam/pinata/RECONSTRUCTED_truth_beam_poliepals_trailer/
    000511_bc48b046_RECONSTRUCTED.npy   neighbours (byte-exact) + reconstructed block 32
    000511_bc48b046_FROM_HASH.npy       full frame regenerated from the hash (XOF demo)
    reconstruct_emission.py             the code
```

The original (incomplete) frame and the other 1,553 byte-exact frames are
untouched under `r2:truthbeam/pinata/truth_beam_poliepals_trailer/`.

## 5. Why this matters — XOF justification and a design lesson

- **It justifies the XOF design.** The recording's *content* is a deterministic
  XOF expansion of the hash chain, so a whole frame is regenerable from its
  BLAKE3 chain hash to numerical precision (Section 4, path 1); the
  neighbour-assisted path recovered 76.8% of a hidden block's values
  bit-identically with maximum error 1.5e-5 on the 0–255 scale (Section 4,
  path 2). Byte-exact regeneration was out of reach only because the 2023
  pipeline committed to raw float output (next bullet); the byte-exact bytes
  came from a retained backup (Section 6).
- **It is also a reproducibility lesson.** Because the 2023 pipeline committed to
  *raw float32 FFT/resize output*, exact-byte reproduction is fragile across
  hardware/library builds. A verifiable protocol should commit to a **canonical,
  quantised, integer/fixed-point representation** (or hash the *seed*, not the
  float bytes). The 2026 protocol's bit-exact integer tile generator
  (`code/recording/`, and `code/verifier/.../bitexact_renderer.py`) is precisely
  that fix.

## 6. The byte-exact recovery

The complete original frame resurfaced on a retained backup on 2026-07-27. It is
a valid `(1024, 1024, 3)` little-endian float32 NPY, 12,583,040 bytes. File bytes
`[8,388,608, 8,650,752)` are byte-identical to the recovered standalone block.

Hash-only IPFS addition reproduces the previously recorded targets exactly:

| Artifact | SHA-256 | CIDv0 |
|---|---|---|
| Complete original frame | `6b7a6bc9f052d78dd161d20c22a19a6341f1447af6721a96f72425ce4bce9c1a` | `QmTy4beHLQP71oS1oMtqgNDg3TNU6wirKGfDsf2Rvq6tBr` |
| Recovered block 32 | `f035c6ef0f25ca20d1a9bda8d88cce591be70abb9b209d7c6841f43293a59041` | `QmRyXZyDbVaE6V7vKBL873DwhA8Xb5VSX7ujcoM1kp3Hpx` |

Those CIDs were present in public repository history before the recovery. This
authenticates the bytes independently of the backup narrative. The exact
comparison, custody boundary, and machine-readable record are in the
[`byte-exact recovery receipt`](RECOVERY_RECEIPT.md).

The exact original and the reconstruction serve different purposes. The exact
frame restores the historical byte record. The reconstruction remains a useful,
reproducible demonstration of what the deterministic design and the surviving
neighbours could recover without that backup.

— BOSUN ⚓

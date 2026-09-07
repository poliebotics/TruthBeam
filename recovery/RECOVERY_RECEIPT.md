# Byte-exact recovery receipt: frame 000511

**Status:** authenticated locally on 2026-07-27. This is metadata-only staging
for the unsigned v1.1.0 source and site update.

One emission frame from the April 2023 Truth Beam PoliePals Trailer record had a
single 262,144-byte UnixFS block that could no longer be retrieved from the
original pin. A numerically faithful reconstruction was published and kept every
surviving byte unchanged. The complete byte-exact frame has now been recovered
from a retained backup.

## Recovered artifacts

| Artifact | Bytes | SHA-256 | CIDv0 |
|---|---:|---|---|
| `000511_bc48b046016adb2ed149471ab0f683f5a9d8dbff9cd544ba2048231c1a4007b2.npy` | 12,583,040 | `6b7a6bc9f052d78dd161d20c22a19a6341f1447af6721a96f72425ce4bce9c1a` | `QmTy4beHLQP71oS1oMtqgNDg3TNU6wirKGfDsf2Rvq6tBr` |
| `lost_block_32.bin` | 262,144 | `f035c6ef0f25ca20d1a9bda8d88cce591be70abb9b209d7c6841f43293a59041` | `QmRyXZyDbVaE6V7vKBL873DwhA8Xb5VSX7ujcoM1kp3Hpx` |

The NPY is a valid C-order array with shape `(1024, 1024, 3)`, dtype
little-endian `float32`, and a 128-byte header. The recovered block is exactly
file bytes `[8,388,608, 8,650,752)` of that frame.

## Authentication

The result is authenticated by measurements that do not depend on the backup
story:

1. `ipfs add --only-hash --cid-version=0` reproduces both CIDs above. Those
   target CIDs were already present in public repository history at commit
   `d294b0d53f29f31e309e3f1fa8fc4083a3f16029`, before this recovery.
2. Against the published reconstruction, every byte before and after the missing
   block is identical. All 15,658 differing bytes are confined to that block.
3. Inside the block, 49,944 of 65,536 float32 values are already bit-identical
   in the reconstruction, or 76.20849609375 percent. The maximum absolute error
   is `1.52587890625e-05`, matching the published reconstruction account.
4. The recovered standalone block is byte-identical to the corresponding slice
   of the recovered full frame.

The complete machine-readable record is
[`RECOVERY_RECEIPT.json`](RECOVERY_RECEIPT.json).

## Availability

At the time of this receipt (27 July 2026) the exact NPY and recovered block were
retained in the private owner archive and were absent from Git and from this
site overlay. Update, 30 July 2026: the byte-exact frame and recovered UnixFS
block were published under `pinata/RECOVERED_truth_beam_poliepals_trailer/` and
indexed on DOWNLOADS. Update, 3 August 2026: the same byte-exact frame was
published in the immutable 2023 trailer archive
(`archive/2023/truth_beam_poliepals_trailer/v1/1682718815/emissions/000511_bc48b046016adb2ed149471ab0f683f5a9d8dbff9cd544ba2048231c1a4007b2.npy`,
12,583,040 bytes, SHA-256
`6b7a6bc9f052d78dd161d20c22a19a6341f1447af6721a96f72425ce4bce9c1a`). This receipt
records the recovery; the archive publication has its own receipt in the
archive's `_control/` prefix. The bytes remain outside Git.

The JSON twin, `RECOVERY_RECEIPT.json`, is the receipt as issued on 27 July
2026; its `publication` and `car_status` fields describe that date and are left
as issued.

## Preservation and scope

The byte-exact frame enables repair of the decomposed public mirror. The earlier
reconstruction remains published under its distinct name because it documents a
useful deterministic-recovery result and the float reproducibility lesson.

The historical root CID remains
`QmejyJWognSYn7UhygsHuQzkDK5vY4izU9SsCL785NsHCN`. Its historical CAR file remains
labelled **partial** because it was recovered before the block was; an
independent only-hash Kubo run over the restored bytes has since reproduced the
historical wrapped root, the inner session directory and the frame-511 CID
exactly (the trailer archive's `CID_REPRODUCTION.json`). No historical signed manifest,
signature, CID, or release byte is changed by this unsigned recovery record.

## Reproduce the file checks

```sh
sha256sum 000511_bc48b046016adb2ed149471ab0f683f5a9d8dbff9cd544ba2048231c1a4007b2.npy
sha256sum lost_block_32.bin
ipfs add --only-hash --cid-version=0 -Q 000511_bc48b046016adb2ed149471ab0f683f5a9d8dbff9cd544ba2048231c1a4007b2.npy
ipfs add --only-hash --cid-version=0 -Q lost_block_32.bin
```

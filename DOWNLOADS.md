---
version: "2.0"
date: "2026-08-03"
status: "public"
author: "Xathal"
---

# Truth Beam downloads

Current release: D2 and V10. Verified historical records follow. Each link
opens a file or control record.

<a id="current-d2-v10"></a>

## Current D2 and V10 release

29,266 files, 406,155,874,045 bytes. Same rig, two sessions, one performer.
Scores and samples are available separately. The complete corpus is 378.262
GiB.

| Session | File list | Files | Bytes | Size | IPFS CID |
|---|---|---:|---:|---:|---|
| D2 | [per-file URLs](https://data.poliebotics.com/downloads/d2_files.txt) | 17,987 | 249,282,572,059 | 232.162 GiB | `bafybeicrssbic35534es3sbwyzhlw7reboh6wy75htmo53ke5mfsphkmwi` |
| V10 | [per-file URLs](https://data.poliebotics.com/downloads/v10_files.txt) | 11,279 | 156,873,301,986 | 146.100 GiB | `bafybeier2sfcjrrgw7amne3lwogise6umyeyf6qgivgrmvx3to4vsdsbcm` |

[`download.sh`](download.sh) offers scores, models, session samples, the 2023
film, individual full sessions and the complete corpus. The
[PolieBotics index](https://poliebotics.com/DOWNLOADS.html) carries the session
logs, model files, films, patent documents and historical IPFS register.

Claim ceiling: finite held-out tests from two sessions on one rig. AUROC 1.000
is the observed estimate on those samples. F-A v1 is the trained attacker. F-A
v2 remains a design; adaptive-attacker work remains open. Start with
[Reproduce](REPRODUCE.html) or [Verify](VERIFY_FAST.html).

<a id="historical-truthbeam"></a>

## April 2023 record

<a id="trailer-2023"></a>

### Complete trailer session

Session `1682718815`, captured 28 April 2023. 777 emissions, 777 matching
camera reports, one chain log. Frame 000511 is present byte-for-byte.

| Receipt | Value |
|---|---|
| Data | 1,555 files, 17,122,505,309 bytes |
| Array validation | 1,554 checked, zero invalid, complete paired sequences |
| Public prefix | `archive/2023/truth_beam_poliepals_trailer/v1/` |
| Manifest SHA-256 | `4feaedaf31162c434684450ebb06e746a87d7e32cfea4a40ab5d81f290a026cc` |
| Publication receipt SHA-256 | `5842699805c83338131522b7737132ca978d6991a560a0efd909193c802692dd` |
| Historical root CID | `QmejyJWognSYn7UhygsHuQzkDK5vY4izU9SsCL785NsHCN` |
| Session-directory CID | `QmWhmuYTywuwRcmsPtxrNnpJjBJZmBTkxAYy2j2vteuqp3` |

Controls: [reader's guide](https://data.truthbeam.com/archive/2023/truth_beam_poliepals_trailer/v1/_control/README.md) · [manifest](https://data.truthbeam.com/archive/2023/truth_beam_poliepals_trailer/v1/_control/MANIFEST.jsonl) · [SHA-256 list](https://data.truthbeam.com/archive/2023/truth_beam_poliepals_trailer/v1/_control/SHA256SUMS) · [NumPy validation](https://data.truthbeam.com/archive/2023/truth_beam_poliepals_trailer/v1/_control/NPY_VALIDATION.json) · [CID reproduction](https://data.truthbeam.com/archive/2023/truth_beam_poliepals_trailer/v1/_control/CID_REPRODUCTION.json) · [release record](https://data.truthbeam.com/archive/2023/truth_beam_poliepals_trailer/v1/_control/RELEASE.json) · [publication receipt](https://data.truthbeam.com/archive/2023/truth_beam_poliepals_trailer/v1/_control/PUBLICATION_RECEIPT.json).

Frame 000511: [download the manifest copy](https://data.truthbeam.com/archive/2023/truth_beam_poliepals_trailer/v1/1682718815/emissions/000511_bc48b046016adb2ed149471ab0f683f5a9d8dbff9cd544ba2048231c1a4007b2.npy), 12,583,040 bytes, SHA-256
`6b7a6bc9f052d78dd161d20c22a19a6341f1447af6721a96f72425ce4bce9c1a`.

Fetch the archive from its manifest, resume partial transfers, then verify every
file:

```sh
curl -fsS https://data.truthbeam.com/archive/2023/truth_beam_poliepals_trailer/v1/_control/MANIFEST.jsonl \
  -o MANIFEST.jsonl
jq -r '"https://data.truthbeam.com/" + .object_key' MANIFEST.jsonl \
  > truth_beam_poliepals_trailer_urls.txt
wget -c --no-host-directories --cut-dirs=4 \
  -P truth_beam_poliepals_trailer_v1 \
  -i truth_beam_poliepals_trailer_urls.txt
curl -fsS https://data.truthbeam.com/archive/2023/truth_beam_poliepals_trailer/v1/_control/SHA256SUMS \
  -o truth_beam_poliepals_trailer_v1/SHA256SUMS
(cd truth_beam_poliepals_trailer_v1 && sha256sum -c SHA256SUMS)
```

### Frame 000511 recovery

One 262,144-byte UnixFS block became unavailable from the original
Pinata-hosted IPFS pin. A deterministic reconstruction supplied a numerically
faithful frame. The byte-exact original was later recovered from retained
custody and reproduced both historical CIDs.

| Record | Link |
|---|---|
| Reconstruction and recovery account | [read](recovery/index.html) |
| Byte-exact frame | [download](https://data.truthbeam.com/pinata/RECOVERED_truth_beam_poliepals_trailer/000511_bc48b046016adb2ed149471ab0f683f5a9d8dbff9cd544ba2048231c1a4007b2.npy) |
| Recovered UnixFS block 32 | [download](https://data.truthbeam.com/pinata/RECOVERED_truth_beam_poliepals_trailer/lost_block_32.bin) |
| Recovery receipt | [open](recovery/RECOVERY_RECEIPT.html) |
| Reconstruction code | [source](recovery/reconstruct_emission.py) |
| Recorder provenance | [open](recovery/recorder-source/SOURCE_PROVENANCE.html) |
| Recorder used in April 2023 | [redacted source](recovery/recorder-source/truth_beam_2023_REDACTED.py) |

The public recorder copy replaces the signing key with an environment-variable
lookup. Recording, BLAKE3-XOF, projection, capture, chain-log and Rootstock
anchoring logic retain the archived implementation.

*Recovery footnote. The 427-object R2 extraction and historical CAR preserve
the incomplete Pinata recovery. The complete 1,555-file archive above came
from retained custody.*

### Complete `old_truth_beams` archive

Seven April 2023 session trees. 5,255 data objects, 57,851,074,892 data bytes,
5,246 checked NumPy files, zero invalid. With its five controls and completion
receipt, the archive contains 5,261 objects and 57,854,448,591 bytes.

Manifest SHA-256:
`33a91c67db91849887004674f8b875be13ea01ba47c26adc9790b159863213db`.

Controls: [reader's guide](https://data.truthbeam.com/archive/2023/old_truth_beams/v1/_control/README.md) · [release record](https://data.truthbeam.com/archive/2023/old_truth_beams/v1/_control/RELEASE.json) · [publication receipt](https://data.truthbeam.com/archive/2023/old_truth_beams/v1/_control/PUBLICATION_RECEIPT.json) · [manifest](https://data.truthbeam.com/archive/2023/old_truth_beams/v1/_control/MANIFEST.jsonl) · [SHA-256 list](https://data.truthbeam.com/archive/2023/old_truth_beams/v1/_control/SHA256SUMS) · [NumPy validation](https://data.truthbeam.com/archive/2023/old_truth_beams/v1/_control/NPY_VALIDATION.json).

```sh
curl -fsS https://data.truthbeam.com/archive/2023/old_truth_beams/v1/_control/MANIFEST.jsonl \
  -o MANIFEST.jsonl
jq -r '"https://data.truthbeam.com/" + .object_key' MANIFEST.jsonl \
  > old_truth_beams_urls.txt
wget -c --no-host-directories --cut-dirs=4 \
  -P old_truth_beams_v1 -i old_truth_beams_urls.txt
curl -fsS https://data.truthbeam.com/archive/2023/old_truth_beams/v1/_control/SHA256SUMS \
  -o old_truth_beams_v1/SHA256SUMS
(cd old_truth_beams_v1 && sha256sum -c SHA256SUMS)
```

<a id="december-2024"></a>

## Seven cleared captures from 19 December 2024

Seven physical projector-camera sessions, recorded 19 December 2024. Each HDF5
contains 64 emissions, 64 camera recordings and 64 stored chain hashes. Total
raw data: 35,677,358,080 bytes.

Session `20241219_050046` completed Rootstock anchoring. Its transaction input
is the stored terminal hash. The remaining six are local-chain captures made
with `dummy_run=True`; final Rootstock submission was disabled for those runs.

| Session | Download | Chain status | Bytes |
|---|---|---|---:|
| `20241219_044052` | [HDF5](https://data.truthbeam.com/archive/2024/truth_beam_20241219_unanchored/v1/20241219_044052/data.h5) | local chain, `dummy_run=True` | 5,096,765,440 |
| `20241219_044529` | [HDF5](https://data.truthbeam.com/archive/2024/truth_beam_20241219_unanchored/v1/20241219_044529/data.h5) | local chain, `dummy_run=True` | 5,096,765,440 |
| `20241219_050046` | [HDF5](https://data.truthbeam.com/pinata/20241219_050046_TB.h5) · [transaction](https://explorer.rootstock.io/tx/0xfc17756ee8232a1b76876b87e33d63ee96ebb18f91ca60176e680c32d0876947) | Rootstock-anchored | 5,096,765,440 |
| `20241219_050648` | [HDF5](https://data.truthbeam.com/archive/2024/truth_beam_20241219_unanchored/v1/20241219_050648/data.h5) | local chain, `dummy_run=True` | 5,096,765,440 |
| `20241219_051150` | [HDF5](https://data.truthbeam.com/archive/2024/truth_beam_20241219_unanchored/v1/20241219_051150/data.h5) | local chain, `dummy_run=True` | 5,096,765,440 |
| `20241219_051629` | [HDF5](https://data.truthbeam.com/archive/2024/truth_beam_20241219_unanchored/v1/20241219_051629/data.h5) | local chain, `dummy_run=True` | 5,096,765,440 |
| `20241219_052040` | [HDF5](https://data.truthbeam.com/archive/2024/truth_beam_20241219_unanchored/v1/20241219_052040/data.h5) | local chain, `dummy_run=True` | 5,096,765,440 |

Anchored session receipt: SHA-256
`f4792a200cffcb70471a03c6ee6212737cefd4b71630ed5e72f654f445b8496a`,
terminal hash
`436c37a9c0d13bdae3d456f0f95a9c867fcae7ae8534af30de5373fc86a5479a`,
Rootstock block 7,034,554 at 2024-12-19 05:03:39 UTC, HDF5 CID
`bafybeiaqbbacosf2evzxiezp4jlrbylgrhe7vw6bwfthuq74fh34bi27bq`.

The manifest and controls below cover the six unanchored files, totalling
30,580,592,640 bytes. Controls: [reader's guide](https://data.truthbeam.com/archive/2024/truth_beam_20241219_unanchored/v1/_control/README.md) · [manifest](https://data.truthbeam.com/archive/2024/truth_beam_20241219_unanchored/v1/_control/MANIFEST.jsonl) · [SHA-256 list](https://data.truthbeam.com/archive/2024/truth_beam_20241219_unanchored/v1/_control/SHA256SUMS) · [HDF5 validation](https://data.truthbeam.com/archive/2024/truth_beam_20241219_unanchored/v1/_control/HDF5_VALIDATION.json) · [recorder provenance](https://data.truthbeam.com/archive/2024/truth_beam_20241219_unanchored/v1/_control/CODE_PROVENANCE.json) · [release record](https://data.truthbeam.com/archive/2024/truth_beam_20241219_unanchored/v1/_control/RELEASE.json) · [publication receipt](https://data.truthbeam.com/archive/2024/truth_beam_20241219_unanchored/v1/_control/PUBLICATION_RECEIPT.json).

Fetch all six manifest-bound files while retaining one directory per session:

```sh
curl -fsS https://data.truthbeam.com/archive/2024/truth_beam_20241219_unanchored/v1/_control/MANIFEST.jsonl \
  -o MANIFEST.jsonl
jq -r '"https://data.truthbeam.com/" + .object_key' MANIFEST.jsonl \
  > truth_beam_20241219_urls.txt
wget -c --no-host-directories --cut-dirs=4 \
  -P truth_beam_20241219_v1 -i truth_beam_20241219_urls.txt
curl -fsS https://data.truthbeam.com/archive/2024/truth_beam_20241219_unanchored/v1/_control/SHA256SUMS \
  -o truth_beam_20241219_v1/SHA256SUMS
(cd truth_beam_20241219_v1 && sha256sum -c SHA256SUMS)
```

## Likeness and rights

The recordings depict Xathal alone. He authorised publication of his likeness
on 29 July 2026.

Rights: all rights reserved. Publication permits inspection and independent
verification under rights arising by law. Open-source licence: none. Patent
licence: none.

Start with the manifest. It says exactly what you are downloading.

## Log

| Version | Date | Change |
|---|---|---|
| 2.0 | 2026-08-03 | Simplified the index and recorded all seven December 2024 captures, including the anchored session. |
| 1.4 | 2026-08-03 | Corrected the release hierarchy and retained complete historical records. |
| 1.3 | 2026-08-02 | Added the six-file December 2024 release and resumable recipes. |
| 1.2 | 2026-07-30 | Published the complete 2023 controls, frame-511 recovery and recorder provenance. |

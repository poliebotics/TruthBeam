---
version: "2.1"
date: "2026-09-07"
status: "public"
author: "Xathal"
---

# Truth Beam downloads

Current release: D2 and V10. The September 2026 results and proofs shelf and the
verified historical records follow. Each link opens a file, a repository or a
control record.

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
[Reproduce](REPRODUCE.md) or [Verify](VERIFY_FAST.md).

<a id="results-2026-09"></a>

## Results and proofs, September 2026

Every entry below is a positive, measured result of the programme, published with its scope. Each links to the
repository or package that carries the bytes, the checks and the audit record. Nothing in this table is a held-out
estimate for another rig, room or person unless it says so, and none of it establishes liveness or adversarial
resistance. Read the row, then read the package.

| Result | In plain English | Boundary | Where |
|---|---|---|---|
| **ZeeBeam: The Zero-Knowledge Beam** (paper and proofs) | For each of 259 anchored rows of a 712-row projector-camera session recorded on 22 August 2026, one zero-knowledge proof (356-byte Groth16, with a 1,101-byte public statement that carries no pixels) shows that the drand beacon signatures for the row and its predecessor verify, that the BLAKE3 chain advanced as specified, that the projected pattern came from that chain, that two frozen networks produced the published scores on the committed frame, and that the row sits in the session tree; a second proof covers the whole 712-row chain log. | One session, one rig, one performer. The proof establishes that the recorded computation happened on committed bytes; the physical meaning of the pose class is confounded with time on this take. | [github.com/poliebotics/zeebeam](https://github.com/poliebotics/zeebeam), manuscript at [paper/zeebeam.md](https://github.com/poliebotics/zeebeam/blob/main/paper/zeebeam.md) |
| **One Look Is All You Get** (the one look, PASS) | The frozen coupling verifier proved inside ZeeBeam was scored exactly once on a sealed 288-row take it had never seen, under criteria fixed in advance: AUROC 0.999879 and 0.997782 on the two donor maps, 1 and 7 false accepts, 4 and 4 false rejects of 288, 144 of 144 reciprocal pairs ordered correctly in both directions, shuffled-emission controls at 0.4986 and 0.4960. The opening was authorised by one recorded owner sentence and the evaluation program refused to run twice. | Same rig, subject and room as the development take, recorded minutes after it. Nothing about another rig, room or subject, liveness or adversaries; the zero-knowledge proof covers the development take, not this one. | [data.truthbeam.com/results/one_look_20260906/v1/](https://data.truthbeam.com/results/one_look_20260906/v1/README.md) |
| **Dark Lantern: Zero-Knowledge Light, Shuttered by Design** (programme record) | The curated public record of the privacy and zero-knowledge programme behind ZeeBeam: standalone proofs, diagnostics, notes, audit prompts and verdicts, published under the rule of positive, patent-supporting results disclosed as fully as the record allows. | A publication subset; its notices identify what was omitted and what cannot be rebuilt from the tree. | [github.com/poliebotics/dark-lantern](https://github.com/poliebotics/dark-lantern) |
| **No Training Required** (train-free coupling statistic) | A statistic with no fitted parameters separates matched projector-camera frames from mismatched ones on the 975 held-out tail frames of the two public sessions, with pooled AUROC between 0.683 and 0.756 at each of the five grid sizes tested (4, 8, 16, 32 and 64), and holds under five recorded within-session shuffles and four hard-negative families registered before scoring. | Retrospective, on previously inspected frames of trained-on sessions; a threshold that transfers between sessions was not tested; no liveness or adversarial claim. | [results/train_free_coupling_20260906](https://github.com/poliebotics/dark-lantern/tree/main/results/train_free_coupling_20260906) |
| **Sixteen Cells, One Threshold** (exact grid proof) | Two PLONK proofs certify, in exact integer arithmetic over committed 4 by 4 cell-sum grids of a real public capture, that the matched emission correlates above one eighth and a recorded mismatch below it. | Two public examples with a threshold chosen after inspection: a demonstration of the proof system, not of confidentiality or generalisation. | [proofs/train_free_grid_correlation_20260906](https://github.com/poliebotics/dark-lantern/tree/main/proofs/train_free_grid_correlation_20260906) |
| **Two Proofs, One Commitment** (conditional micro-proofs) | Two PLONK proofs over one committed collection of 4 by 4 tensors establish a frozen discriminator margin and a fixed-noise residual inequality; both verify, and every recorded mutation and malformed artefact rejects. | Arithmetic feasibility of frozen conditional computation over shared hidden inputs, not a two-sided verifier; each aggregate passes while one of its two directions fails. | [proofs/conditional_micro](https://github.com/poliebotics/dark-lantern/tree/main/proofs/conditional_micro) |
| **The pose proofs** | Two Groth16 proofs of a frozen eleven-class pose classifier's verdict over committed frame bytes, cropped and uncropped, with their receipts and public values. | Computational integrity over committed pixels; what the pose class means physically is confounded with time on the only take. | [proofs/pose_cropped](https://github.com/poliebotics/dark-lantern/tree/main/proofs/pose_cropped) and [proofs/pose_uncropped](https://github.com/poliebotics/dark-lantern/tree/main/proofs/pose_uncropped) |
| **Following the Light** (image-translation results) | Eight pix2pixHD runs on the two public sessions, scored on their 975 held-out tail frames. Six static-condition discriminators separate matched from mismatched pairs with AUROC between 0.692 and 0.965 over 400 recorded targets. Two temporal generators, given the recorded emission against a recorded alternative under the intended fixed context (the previous emission and the previous real capture reused), produced outputs closer to the real capture with the recorded emission on 972 of 973 and 973 of 973 frames. | Exploratory evidence consistent with current-emission correspondence, as the package states: the record does not bind the two generation runs to identical checkpoint bytes, history inputs or runtime versions; held-out frames of trained-on sessions; no unseen-session or liveness claim. | [data.truthbeam.com/results/p2pv2v_20260906/v1/](https://data.truthbeam.com/results/p2pv2v_20260906/v1/RESULTS.md) |
| **pix2pixHD checkpoints, January 2025** | The full earlier training run on Truth Beam emission-to-recording pairs: 75 generator and 75 discriminator checkpoints, the training logs and the epoch gallery, published for inspection with a model card. | A model card without performance claims; the training corpus was not recovered. | [data.truthbeam.com/models/truth_beam_pix2pixhd_2048_1024/v1/](https://data.truthbeam.com/models/truth_beam_pix2pixhd_2048_1024/v1/_control/README.md) |

Patent position: the matter above is the subject of Irish patent applications filed on 25 August 2026 and
6 September 2026, in addition to the filings listed on the [PolieBotics repository](https://github.com/poliebotics/PolieBotics).
They are unpublished applications, not grants. Publication of the results permits inspection and independent
verification; it grants no licence under any of them.

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
| Reconstruction and recovery account | [read](https://data.truthbeam.com/release/recovery/index.html) |
| Byte-exact frame | [download](https://data.truthbeam.com/pinata/RECOVERED_truth_beam_poliepals_trailer/000511_bc48b046016adb2ed149471ab0f683f5a9d8dbff9cd544ba2048231c1a4007b2.npy) |
| Recovered UnixFS block 32 | [download](https://data.truthbeam.com/pinata/RECOVERED_truth_beam_poliepals_trailer/lost_block_32.bin) |
| Recovery receipt | [open](recovery/RECOVERY_RECEIPT.md) |
| Reconstruction code | [source](recovery/reconstruct_emission.py) |
| Recorder provenance | [open](https://data.truthbeam.com/release/recovery/recorder-source/SOURCE_PROVENANCE.html) |
| Recorder used in April 2023 | [redacted source](https://data.truthbeam.com/release/recovery/recorder-source/truth_beam_2023_REDACTED.py) |

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
| 2.1 | 2026-09-07 | Added the September 2026 results and proofs shelf: ZeeBeam, the one look, Dark Lantern, the image-translation results, the January 2025 checkpoints, and the patent position. Links that resolved only on the site now resolve in the repository too. |
| 2.0 | 2026-08-03 | Simplified the index and recorded all seven December 2024 captures, including the anchored session. |
| 1.4 | 2026-08-03 | Corrected the release hierarchy and retained complete historical records. |
| 1.3 | 2026-08-02 | Added the six-file December 2024 release and resumable recipes. |
| 1.2 | 2026-07-30 | Published the complete 2023 controls, frame-511 recovery and recorder provenance. |

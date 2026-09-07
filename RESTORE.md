# TruthBeam — Data Restore & Verification Guide

Hoy. BOSUN here. This is the restore and mirror guide for the release data, kept as the release ships it.

This repository holds the code, paper, and small results. The **large artifacts** (raw capture
sessions, model weights, full evidence/eval trees) live in **Cloudflare R2** (bucket `truthbeam`),
content-addressed and verifiable against the CIDs in `CID_MANIFEST.json` (at the R2 root; the
release-unit CID table is also reproduced inline below for offline reference).

The Lambda training box that originally held this data was decommissioned on 2026-06-02 after a
checksum-verified evacuation; R2 is now the system of record. This document is how you get it back.

---

## 1. R2 bucket layout (`r2:truthbeam/...`)

```
sessions/d2/                      raw D2 capture session  (17,987 files, 249.3 GB)
sessions/v10/                     raw V10 capture session (11,279 files, 156.9 GB)
                                  each session = raw frames + chain_log.csv, capture_log.csv,
                                  anchor_txs.csv, manifest.json, verification_bundle.json,
                                  verify_report.json

models/                           28 curated .pt weights (7.64 GB):
                                    phase_g_verifier/  (main + shuffled + synthetic_positive controls)
                                    fa_v1_forger/      (step 5k/25k/70k/100k)
                                    fa_v2_surrogate_binders/
                                    emission_binders_phase_e/  (~13 best_by_psnr)

lambda/poliebotics_phase_b/       full protocol + training source tree (the code)
lambda/experiments/<name>/        evidence/eval trees (paper_analyses, cross_session_ablation,
                                    phase_e, phase_f, phase_g_diffusion_diagnostic, fa_v2,
                                    phase_h_supervised_baseline, stage_0_*, exp001*, etc.)
                                  NOTE: the 3 render-heavy trees (projection_cleaner_temporal_v1,
                                    projection_cleaner_v1, visualization_design) hold ONLY
                                    weights + metadata + the .py render-recipe scripts; the bulk
                                    rendered frames/tensors were intentionally not copied (rederivable
                                    from sessions + weights + code + these recipes).
lambda/{backups,sanity,scripts_runtime,orchestration}/   small operational dirs
lambda/*.md                       STATUS.md, HANDOFF_TO_REMOTE.md, QUESTIONS.md

ipfs/car/<cid>.car                Pinata pins as CARs (exact DAG, preserves original CID)
pinata/<name>/                    Pinata pins as directly-usable plain files
                                    (truth_beam_poliepals_trailer, 20241219_050046,
                                     20241219_050046_TB.h5, PolieBotics.mp4, PolieBotics)

CID_MANIFEST.json                 CID + bytes + file count for every release unit
logs/                             transfer + ipfs-prep logs
```

**NOT in R2** (intentionally skipped, rederivable): `visualization_renders_20260507/` (11 TB demo-video
renders), `cache/` (packed-CFA normalization cache), `checkpoints/` (public MaskRCNN pretrained),
`visual_grids/` (mirrored locally), and per-run `tb/`, `tb_logs/`, `__pycache__/` subpaths.

---

## 2. Access

The R2 API bucket remains private. Public release objects are already served read-only through
`https://data.truthbeam.com/`; use the indexed HTTPS links in `ARTIFACTS.md` and `DOWNLOADS.md`. Rclone
access requires R2 credentials and covers private or request-only objects.

rclone remote config (`~/.config/rclone/rclone.conf`):
```ini
[r2]
type = s3
provider = Cloudflare
access_key_id = <ACCESS_KEY_ID>
secret_access_key = <SECRET>
endpoint = https://<ACCOUNT_ID>.r2.cloudflarestorage.com
acl = private
no_check_bucket = true
```

Download examples:
```bash
rclone copy r2:truthbeam/sessions/d2        ./d2        --transfers 16   # whole session
rclone copy r2:truthbeam/models             ./models                     # all weights
rclone copyto r2:truthbeam/sessions/d2/manifest.json ./manifest.json     # one file
rclone cat  r2:truthbeam/CID_MANIFEST.json                               # the manifest
```

---

## 3. Verifying data against its CID

Every unit's CID in `CID_MANIFEST.json` is reproducible with these exact flags:
```bash
ADD_FLAGS="--only-hash --recursive --cid-version=1 --raw-leaves --chunker=size-1048576"
# download a unit, then recompute its CID and compare to the manifest:
rclone copy r2:truthbeam/sessions/d2 ./d2
ipfs add $ADD_FLAGS --quieter ./d2     # must equal manifest units.sessions_d2.cid
```

Release-unit CIDs (from `CID_MANIFEST.json`):

| Unit | CID |
|---|---|
| sessions_d2 | `bafybeicrssbic35534es3sbwyzhlw7reboh6wy75htmo53ke5mfsphkmwi` |
| sessions_v10 | `bafybeier2sfcjrrgw7amne3lwogise6umyeyf6qgivgrmvx3to4vsdsbcm` |
| models | `bafybeihffzc7fn5q3u7tf3k5hqkcpaoozdnx7pdm2lw4fmfwkgfwjnzzpm` |
| code_src_full | `bafybeiguoiy24zqup7pp7wkgjjnhoyojuzlabzwizuabgbrbejz4jgun3i` |
| evidence_paper_analyses | `bafybeibrnz5mmz53fz3vw7xpthvaiefm4j4h2yscv2bnhftgorrbjbnnre` |
| evidence_cross_session_ablation | `bafybeibs2xyyyevdnpy2a2rr4yipmpht3iyek6fnlzhpcmty7woydipecq` |
| evidence_phase_e | `bafybeibpgty4iu355or6c2d27djkfyudwdc6rz7i3bjiw7ekcr4tvc3t5i` |
| evidence_phase_g_verifier | `bafybeibfqaw4wtpzo46a767udltt3vupii66lttrlyiheeajjsffgf2z7a` |
| evidence_phase_h_supervised_baseline | `bafybeid6ibcxgh2najibjoyqyth2qhryx57ufnzasfpf27krsnpkndvqba` |
| evidence_phase_f | `bafybeiebxm5p4uhh5nm5n2dfzpzykecawamlzwmmzhq6pm6lncplygmdmy` |
| evidence_fa_v2 | `bafybeifpfdgyycg6swe6bj3oonfpn7n3zhikoapjx3vr2znejaevogjvpy` |
| evidence_stage_0_cross_verifier | `bafybeieomhoxmevp2rol4pvy2z6rzhtzmktzl6yjvdlfhcie7sahdgjvd4` |
| evidence_closure_package | `bafybeiavyfm7vqrvg6mbg5sc7p7jg73n34oodvjfyyy6n5aw6ns2atn7ye` |

Pinata-pinned media (content-addressed; live on the IPFS network via Pinata, mirrored to
`r2:truthbeam/ipfs/car/` and `r2:truthbeam/pinata/`):

| Name | CID |
|---|---|
| Truth Beam PoliePals Trailer | `QmejyJWognSYn7UhygsHuQzkDK5vY4izU9SsCL785NsHCN` |
| 20241219_050046_TB.h5 | `bafybeiaqbbacosf2evzxiezp4jlrbylgrhe7vw6bwfthuq74fh34bi27bq` |
| 20241219_050046 | `bafybeibbapmogu2bro3ettoilge6bp5lic3u2mdohvbutgixnee463kmga` |
| PolieBotics.mp4 | `QmP8JDfeBCunq4VQ8f6XUbiLJK55dG9jLav7k5q2HpnmxS` |
| FULL_128_TEST_v108.mp4 (launch video) | `QmQqDVntpNw8gLaiMuFAJSe3r7g2UzDAQNch1ZErcZ2b5Y` |
| PolieBotics | `QmQ2BTcVWBEZqL7pBJNsbfjc37AwT18byknVvPmmQwgWZa` |

The 2023 trailer's exact frame 000511 and its formerly unavailable block were
recovered from a retained backup and authenticated on 2026-07-27. The recovered
frame and UnixFS block were published under the recovered `pinata/` prefix on
2026-07-30, and the frame was published byte-exact in the immutable 2023 trailer
archive on 2026-08-03 (see `DOWNLOADS.md`). Their pre-recorded CIDs, full hashes,
and direct links are in [`recovery/RECOVERY_RECEIPT.md`](recovery/RECOVERY_RECEIPT.md).
An independent only-hash Kubo run over the retained bytes reproduced the
historical root `QmejyJWognSYn7UhygsHuQzkDK5vY4izU9SsCL785NsHCN` exactly; the
historical CAR file itself stays labelled partial because it predates the recovery.

---

## 4. The recording → genesis-hash → chain verification loop

The two sessions are tamper-evident. Each frame is committed under a BLAKE3 state chain whose genesis
hash `S_0` commits to the **tile-generator source code** (`generator_code_hash`). Verify from this repo:

```bash
# 1. code -> hash: recompute generator_code_hash from the published source (no GPU needed)
python3 code/recording/verify/verify_generator_hash.py ./d2
#    -> 154be9dd75e0586df456a7eae1528b7334a415f3a977a107c91c6b0751bfc540  (matches d2 + v10)

# 2. hash -> chain: recompute S_0 from the manifest and walk the chain
python3 code/recording/verify/verify_v9.py ./d2
```

---

## 5. Integrity check used at evacuation (for reference)

Content was verified bit-for-bit before the source was decommissioned:
```bash
rclone check <source> r2:truthbeam/<dest> --checksum --one-way   # 0 differences = MD5-identical
```
On 2026-06-02 this reported `0 differences` for sessions/d2 (17,987), sessions/v10 (11,279),
cross_session_ablation (75), and the phase_g verifier (13).

---

## 6. Re-pinning to IPFS (optional)

R2 is an HTTP store, not an IPFS node — the R2-hosted CIDs are verification hashes, not network
addresses. To make any unit natively resolvable on the IPFS network, download it and re-add/pin:
```bash
rclone copy r2:truthbeam/models ./models
ipfs add -r --cid-version=1 --raw-leaves --chunker=size-1048576 ./models   # reproduces the CID, then `ipfs pin`
```
or pin via a service (Pinata / web3.storage). The Pinata media pins above are already live on IPFS.

— BOSUN ⚓

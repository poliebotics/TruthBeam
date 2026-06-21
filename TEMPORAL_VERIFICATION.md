# Temporal verification — the time-bound, established on-chain

This is the **online temporal check** for the two released sessions: it looks the anchors up **live on RSK
mainnet**, **BLS-verifies** the drand quicknet rounds, and derives the **commit rate**. It moves the
external time-window claim from *asserted* (protocol design promise; see
[`code/recording/verify/CLAIMS.md`](code/recording/verify/CLAIMS.md)) to **demonstrated for these
sessions** — anyone can re-run it (commands at the bottom).

> The offline, GPU-free chain re-walk proves **ordering + tamper-evidence** and that the recomputed
> terminal state `S_N` equals the committed anchor value. Binding that ordered record to an **external
> time window** is what *this* online check adds: the **RSK** block timestamps give the window, and the
> **drand** BLS signatures give a publicly-verifiable per-round freshness floor inside it.

**What this is — and is not.** A session-level **time *window*** bound, **not a per-frame UTC
attestation**. Rootstock block timestamps are coarse (~30 s) *consensus* timestamps, not precise clock
readings; drand freshness is per-**round** (~3 s, ≈ 7–8 frames), not per-frame; the window is no tighter
than the anchor cadence; and none of it speaks to the semantic truth of what was staged.

## Verify by browsing — no code, no clone

Anything with a web browser (a person, or an AI assistant that can open links) can confirm the on-chain
window for **D2** from four independent public pages — no tools, no trust:

1. **Session-open block** — <https://explorer.rootstock.io/block/8768852> → hash
   `2c85d0a2…aa42a718`, timestamp **2026-04-25 02:07:48 UTC**. The genesis state `S_0` folds in this
   freshly-waited block, so the session could not have been recorded earlier.
2. **Session-end block** — <https://explorer.rootstock.io/block/8768945> → timestamp **02:48:47 UTC**.
3. **Final-root transaction** — <https://explorer.rootstock.io/tx/0x9952d22288978d18e18a832d90da80bc729ab2b236a4516472e844f112e12c8f>
   → it is *in* block 8768945, and its input data carries the session's final root
   `1f45e6596b5d…2d22c2da6` (the `S_N` commitment in `manifest.json`).
4. **A drand round** — <https://api.drand.sh/52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971/public/28093180>
   → the BLS-signed beacon value folded in at session-open (published **02:08:24 UTC**; chain params at
   `…/info`).

Those four links alone establish the **[02:07:48 → 02:48:47 UTC] window** from independent sources. (V10
is identical, with blocks **8769289 / 8769357**.) The scripted checks below add per-transaction calldata,
per-frame BLAKE3, and bit-exact emission re-derivation — but the window itself needs only a browser.

## Results

| | **D2** (session `61700096…`) | **V10** |
|---|---|---|
| RSK network | mainnet (chain_id 30) | mainnet |
| **anchor_start** (session-open lower bound) | block **8768852**, hash `2c85d0a2…a718` ✓, **2026-04-25 02:07:48 UTC** | block **8769289**, hash ✓, **05:10:29 UTC** |
| fresh-block wait (freshness mechanism) | tip 8768851 → waited 3.57 s for next block | (next-block wait) |
| **anchor_end** (session-end upper bound) | block **8768945**, hash `0x19b20a60…` ✓, **02:48:47 UTC** | block **8769357**, hash ✓, **05:35:53 UTC** |
| final-root tx (commits `S_N`'s final root) | `0x9952d222…12c8f` — **in anchor_end block** ✓, commitment present in input ✓ | `0x42293125…` — in-block ✓, commitment in input ✓ |
| RSK anchor txs (state-pulse + 1 final-root) | **161/161 included** ✓ (84 blocks, monotonic); **161/161 commit the expected state in calldata** ✓ | **102/102 included** (57 blocks); **102/102 calldata-confirmed** ✓ |
| pulse fired→inclusion latency | median ~12 s (max ~75 s ≈ 2–3 blocks) | median ~11 s (max ~49 s) |
| drand chain | quicknet `52db9ba7…e971`, pubkey ✓, period 3 s | same |
| drand rounds folded | **476**, rounds 28093180 → 28093983 | **324**, 28096824 → 28097325 |
| drand publication times | open **02:08:24** → close **02:48:33 UTC** | open **05:10:36** → close **05:35:39 UTC** |
| drand BLS pairing-verify | **all 476 rounds valid** (0 fail) | **all 324 valid** (0 fail) |
| **on-chain time window** | **[02:07:48 → 02:48:47] = 2459 s** | **[05:10:29 → 05:35:53] = 1524 s** |
| captures `N_chain` | 5992 | 3743 |
| camera acquisition span | 2401.0 s | 1500.7 s |
| **commit rate (fps)** | **2.496 Hz** | **2.494 Hz** |
| rows per drand round | ~7.5 (2.5 Hz commits / 0.33 Hz beacon) | ~7.5 |

**What this establishes.** Each session is bound to a real on-chain interval: it could not have been
produced before its `anchor_start` block existed (RSK timestamp + drand round publication time), and it
demonstrably existed by its `anchor_end` block, whose transaction commits the session's final state.
Within that window the projection ran at a steady **~2.5 Hz**, with a BLS-verifiable drand challenge
refreshed every 3 s (≈ one round per 7–8 frames).

## Full per-pulse + all-rounds download

Beyond the endpoints, **every** RSK pulse and **every** drand round was downloaded and checked:

- **RSK pulses — all included, calldata-confirmed, continuous, monotonic.** D2 **161/161** and V10
  **102/102** pulse transactions are confirmed on-chain, landing in monotonically-increasing mainnet
  blocks across the whole session (D2: 84 blocks 8768854→8768945). Crucially — because *inclusion in a
  block is not the same as committing a specific state* — **every** tx's input data was also checked to
  carry its **expected commitment** (per-pulse `payload_commitment_hex`, or `payload_final_root_hex` for
  the final-root tx): **161/161 and 102/102 match**. Their committed `state_index` values rise
  monotonically and cover the full capture range (D2 9→5992 of 5992). Median fired→inclusion latency ~12 s
  (max ~75 s ≈ 2–3 RSK blocks) — the chain is anchored **continuously through** the session, not only at
  its ends. (`temporal_analysis.py` performs all of these checks.)
- **drand — every round BLS-verified.** All **476** (D2) and **324** (V10) folded quicknet rounds were
  refetched and pass the BLS pairing-check — **zero failures**. The rounds advance monotonically at the
  3 s cadence from the session-open round to the session-close round, bracketing the window.
- **Commit rate is rock-steady.** Inter-frame intervals: **median 0.400 s, p95 0.401 s** (max 0.501 s = an
  occasional single-frame gap) → a near-constant **2.50 Hz**, ≈ one BLS-verifiable drand challenge per
  7–8 frames.

**Scope.** This demonstrates the *time window* and *commit rate* for the two released sessions. The
freshness floor is established at **drand-round granularity** (~3 s / ~7–8 frames): every folded round is
genuine (BLS) and published at a known time, and the rounds advance through the session — per-*frame*
drand is an explicit protocol non-goal (it would throttle capture), and a precise per-frame wall-clock
comparison is limited by the camera's monotonic-clock-to-UTC offset, so none is claimed. The window is
not tightened below the anchor cadence, and — like everything in this release — this says nothing about
the semantic truth of what was staged. The RSK fresh-block folded into `S_0` is the primary session-open
bound; drand is corroborating-evidence tier (non-gating in the offline verifier).

## Reproduce it

```bash
# 1. fetch the two sessions' metadata (small; no raw frames needed)
for s in d2 v10; do mkdir -p "$s"; for f in manifest.json anchor_txs.csv chain_log.csv; do
  curl -fsSL -o "$s/$f" "https://data.truthbeam.com/sessions/$s/$f"; done; done

# 2. authoritative path — the project's own verifier, online mode
python3 code/recording/verify/verify_v9.py --session-dir ./d2 --online   # RSK incl. + drand BLS
python3 code/recording/verify/verify_v10.py --session-dir ./v10 --online

# 2b. the full itemised per-pulse + all-rounds timing view (the numbers in this report):
python3 code/recording/verify/temporal_analysis.py ./d2

# 3. or look the anchors up directly: eth_getBlockByNumber against any RSK
#    mainnet RPC (public-node.rsk.co) for anchor_start/anchor_end in manifest.json,
#    and GET https://api.drand.sh/52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971/public/<round>
#    for each drand_round_number in chain_log.csv, BLS-verified with code/recording/protocol/drand_client.py.
```

*Checked against RSK mainnet and the drand network on 2026-06-20. Block hashes, transaction inclusion,
and drand BLS signatures all verified; figures above are the live results.*

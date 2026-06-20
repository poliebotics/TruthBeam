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

## Results

| | **D2** (session `61700096…`) | **V10** |
|---|---|---|
| RSK network | mainnet (chain_id 30) | mainnet |
| **anchor_start** (session-open lower bound) | block **8768852**, hash `2c85d0a2…a718` ✓, **2026-04-25 02:07:48 UTC** | block **8769289**, hash ✓, **05:10:29 UTC** |
| fresh-block wait (freshness mechanism) | tip 8768851 → waited 3.57 s for next block | (next-block wait) |
| **anchor_end** (session-end upper bound) | block **8768945**, hash `0x19b20a60…` ✓, **02:48:47 UTC** | block **8769357**, hash ✓, **05:35:53 UTC** |
| final-root tx (commits `S_N`'s final root) | `0x9952d222…12c8f` — **in anchor_end block** ✓, commitment present in input ✓ | `0x42293125…` — in-block ✓, commitment in input ✓ |
| RSK pulse txs (state commitments) | 160 | (per-pulse, online-checkable) |
| drand chain | quicknet `52db9ba7…e971`, pubkey ✓, period 3 s | same |
| drand rounds folded | **476**, rounds 28093180 → 28093983 | **324**, 28096824 → 28097325 |
| drand publication times | open **02:08:24** → close **02:48:33 UTC** | open **05:10:36** → close **05:35:39 UTC** |
| drand BLS pairing-verify | **5/5 sampled rounds valid** | **5/5 valid** |
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

**Scope.** This demonstrates the *time window* and *commit rate* for the two released sessions. It does
not tighten the window below the anchor cadence, and — like everything in this release — says nothing
about the semantic truth of what was staged. The drand layer is corroborating-evidence tier (non-gating
in the offline verifier); the RSK fresh-block folded into `S_0` is the primary session-open bound.

## Reproduce it

```bash
# 1. fetch the two sessions' metadata (small; no raw frames needed)
for s in d2 v10; do for f in manifest.json anchor_txs.csv chain_log.csv; do
  curl -sO "https://data.truthbeam.com/sessions/$s/$f" --create-dirs -o "$s/$f"; done; done

# 2. authoritative path — the project's own verifier, online mode
python3 code/recording/verify/verify_v9.py --session-dir ./d2 --online   # RSK incl. + drand BLS
python3 code/recording/verify/verify_v10.py --session-dir ./v10 --online

# 3. or look the anchors up directly: eth_getBlockByNumber against any RSK
#    mainnet RPC (public-node.rsk.co) for anchor_start/anchor_end in manifest.json,
#    and GET https://api.drand.sh/52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971/public/<round>
#    for each drand_round_number in chain_log.csv, BLS-verified with code/recording/protocol/drand_client.py.
```

*Checked against RSK mainnet and the drand network on 2026-06-20. Block hashes, transaction inclusion,
and drand BLS signatures all verified; figures above are the live results.*

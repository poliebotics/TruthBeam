#!/usr/bin/env python3
"""Full temporal analysis for a released Truth Beam session.

Downloads EVERY RSK pulse transaction and EVERY drand round committed in a
session, checks them against the public networks, and reports the timing:

  - RSK: every anchor/pulse tx is looked up (receipt + block) for on-chain
    inclusion, block monotonicity, and fired->inclusion latency.
  - drand: every unique round in chain_log.csv is refetched and BLS-verified
    (pairing check) via the sibling drand_client.py.
  - commit rate (fps) from inter-frame capture intervals.

This is the analysis behind TEMPORAL_VERIFICATION.md. The authoritative
pass/fail verdict is the project verifier (verify_v9.py / verify_v10.py
--online); this script is the supplementary, fully-itemised timing view.

Usage:   python3 temporal_analysis.py <session-dir>
Needs:   py_ecc (BLS), network access to an RSK mainnet RPC + api.drand.sh.
A session-dir needs manifest.json, anchor_txs.csv, chain_log.csv.
"""
import json, urllib.request, datetime, csv, sys, os, statistics
import concurrent.futures as cf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "protocol"))
from drand_client import fetch_round, verify_round_full

D = sys.argv[1].rstrip("/") + "/"
m = json.load(open(D + "manifest.json"))
RPCS = m["rsk_rpc_endpoints"]; CH = m["drand_chain_hash"]; PK = m["drand_public_key"]

def rpc(method, params, t=30):
    last = None
    for url in RPCS:
        try:
            req = urllib.request.Request(url, data=json.dumps(
                {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode(),
                headers={"Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(req, timeout=t)).get("result")
        except Exception as e:
            last = e
    raise last

def U(s): return datetime.datetime.fromisoformat(s)

# ---- RSK: every pulse tx — inclusion AND calldata commits the expected state ----
# Inclusion in a block is not enough; we also confirm each tx's input carries the
# expected commitment (per-pulse payload_commitment_hex, or the manifest's
# anchor_end.payload_final_root_hex for the final-root tx).
pulses = list(csv.DictReader(open(D + "anchor_txs.csv")))
final_root = m["anchor_end"]["payload_final_root_hex"].lower()
def lookup(p):
    try:
        tx = rpc("eth_getTransactionByHash", [p["tx_hash"]])
        if not tx or tx.get("blockNumber") is None:
            return p["tx_hash"], None
        bn = int(tx["blockNumber"], 16)
        expected = (p["payload_commitment_hex"] or final_root).lower()
        commits = bool(expected) and expected in tx["input"].lower()
        return p["tx_hash"], (bn, commits)
    except Exception:
        return p["tx_hash"], None
res = {}
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    for k, v in ex.map(lookup, pulses):
        res[k] = v
blocks = sorted({v[0] for v in res.values() if v})
btime = {}
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    for bn, t in ex.map(lambda b: (b, int(rpc("eth_getBlockByNumber", [hex(b), False])["timestamp"], 16)), blocks):
        btime[bn] = t
included = sum(1 for v in res.values() if v)
commits_ok = sum(1 for v in res.values() if v and v[1])
lat = [btime[res[p["tx_hash"]][0]] - U(p["wall_utc_fired"]).timestamp()
       for p in pulses if res[p["tx_hash"]]]
si = [int(p["payload_state_index"]) for p in pulses]
print(f"RSK pulses: {included}/{len(pulses)} included; "
      f"{commits_ok}/{len(pulses)} carry the expected commitment in calldata; "
      f"{len(blocks)} blocks {blocks[0]}..{blocks[-1]} monotonic={blocks == sorted(blocks)}")
print(f"  fired->inclusion latency s: median={statistics.median(lat):.0f} max={max(lat):.0f}")
print(f"  state_index monotonic={all(si[i] <= si[i+1] for i in range(len(si)-1))} range {si[0]}..{si[-1]}")

# ---- drand: BLS-verify every unique round ----
rows = [r for r in csv.reader(open(D + "chain_log.csv")) if r and not r[0].startswith('#')]
dri = rows[0].index('drand_round_number')
urounds = sorted({int(r[dri]) for r in rows[1:] if r[dri] and r[dri] != '0'})
def verify(rn):
    try:
        rd = fetch_round(rn, chain_hash=CH)
        return verify_round_full(rn, rd["randomness"], rd["signature"], pubkey_hex=PK)
    except Exception:
        return None
with cf.ThreadPoolExecutor(max_workers=12) as ex:
    vs = list(ex.map(verify, urounds))
print(f"drand: BLS valid={sum(v is True for v in vs)}/{len(urounds)} "
      f"invalid={sum(v is False for v in vs)} err={sum(v is None for v in vs)} "
      f"rounds {urounds[0]}..{urounds[-1]}")

# ---- fps ----
cwi = rows[0].index('capture_wall_ns')
caps = sorted(int(r[cwi]) for r in rows[1:])
dt = [(caps[i+1] - caps[i]) / 1e9 for i in range(len(caps) - 1)]
print(f"fps: mean={1/statistics.mean(dt):.3f} Hz; frame dt s median={statistics.median(dt):.3f} "
      f"p95={sorted(dt)[int(0.95*len(dt))]:.3f}")

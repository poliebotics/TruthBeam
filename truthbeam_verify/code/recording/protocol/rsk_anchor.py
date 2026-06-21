#!/usr/bin/env python3
"""RSK fresh-block anchor for Truth Beam session start.

Polls an RSK JSON-RPC endpoint for the current block tip, then waits for the
NEXT block to land. The newly-landed block is the session's "fresh anchor":
its existence after wall-time T proves the session could not have been
produced before T (modulo block-arrival jitter and reorg risk — see status
report on the latter).

Mechanism: JSON-RPC 2.0 over HTTP POST, method `eth_getBlockByNumber` with
params `["latest", false]`. Stdlib only (urllib.request + json), no web3 /
requests / eth_account dependencies.

Public RSK mainnet RPC for the demo: https://public-node.rsk.co
Set TB_RSK_RPC env var or pass --rpc to point elsewhere; the env var wins
if no explicit --rpc is given. NEVER hardcode a default URL — callers must
opt in.

CLI:
    python3 rsk_anchor.py --rpc https://public-node.rsk.co
        Fetches current tip, waits for next block, prints both as JSON.
"""

import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.request
from urllib.error import HTTPError, URLError


DEFAULT_POLL_INTERVAL_S = 2.0
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_RPC_TIMEOUT_S = 10.0


def _strip_0x(h):
    if isinstance(h, str) and h.startswith("0x"):
        return h[2:]
    return h


def _rpc_call(rpc_url, method, params, timeout=DEFAULT_RPC_TIMEOUT_S):
    """One JSON-RPC 2.0 call. Returns the 'result' field. Raises on RPC error."""
    payload = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params,
    }).encode("utf-8")
    req = urllib.request.Request(
        rpc_url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    obj = json.loads(body.decode("utf-8"))
    if "error" in obj and obj["error"] is not None:
        raise RuntimeError(f"RPC error from {rpc_url}: {obj['error']}")
    if "result" not in obj:
        raise RuntimeError(f"RPC response missing 'result': {obj!r}")
    return obj["result"]


def _block_to_dict(b):
    """Normalize an eth_getBlockByNumber result into the anchor schema."""
    if b is None:
        raise RuntimeError("RPC returned null block")
    block_number = int(b["number"], 16)
    block_hash = _strip_0x(b["hash"])
    parent_hash = _strip_0x(b["parentHash"])
    ts_unix = int(b["timestamp"], 16)
    iso_ts = dt.datetime.fromtimestamp(ts_unix, dt.timezone.utc).isoformat()
    return {
        "block_number": block_number,
        "block_hash": block_hash,
        "timestamp_utc": iso_ts,
        "parent_hash": parent_hash,
    }


def fetch_current_tip(rpc_url, timeout=DEFAULT_RPC_TIMEOUT_S):
    """Fetch the current chain tip via eth_getBlockByNumber('latest', false).

    Returns dict with: block_number (int), block_hash (hex no 0x),
    timestamp_utc (ISO-8601 UTC), parent_hash (hex no 0x).
    """
    b = _rpc_call(rpc_url, "eth_getBlockByNumber", ["latest", False],
                  timeout=timeout)
    return _block_to_dict(b)


def fetch_block_by_number(rpc_url, block_number, timeout=DEFAULT_RPC_TIMEOUT_S):
    """Fetch a specific block by height via eth_getBlockByNumber('0xN', false).

    Returns the same dict shape as fetch_current_tip. Returns None if the
    node does not have a block at that height (e.g., reorged out or pruned).
    """
    hex_num = hex(int(block_number))  # '0x...' with lowercase letters
    b = _rpc_call(rpc_url, "eth_getBlockByNumber", [hex_num, False],
                  timeout=timeout)
    if b is None:
        return None
    return _block_to_dict(b)


def fetch_balance(rpc_url, address, timeout=DEFAULT_RPC_TIMEOUT_S):
    """eth_getBalance for a hex address. Returns int (wei). Raises on RPC error.

    Used by tb_loop.py preflight to verify wallet balance BEFORE creating
    the session directory. Stdlib only (no Web3 dependency).
    """
    addr_hex = address if address.startswith("0x") else "0x" + address
    result = _rpc_call(rpc_url, "eth_getBalance", [addr_hex, "latest"],
                       timeout=timeout)
    if isinstance(result, int):
        return result
    return int(result, 16)


def fetch_chain_id(rpc_url, timeout=DEFAULT_RPC_TIMEOUT_S):
    """Query eth_chainId. Returns int. Raises on RPC error.

    Used by tb_loop.py and anchor_publisher.py to verify that the RPC
    endpoint is serving the chain the session was configured for. A
    mismatch between --network and the RPC's actual chain is a hard error
    — silently labelling mainnet blocks as testnet (or vice versa) would
    make the artefact's anchor claims false.
    """
    result = _rpc_call(rpc_url, "eth_chainId", [], timeout=timeout)
    # result is a 0x-prefixed hex string per EIP-695.
    if isinstance(result, int):
        return result
    return int(result, 16)


def wait_for_next_block(rpc_url, current_tip,
                         poll_interval_s=DEFAULT_POLL_INTERVAL_S,
                         timeout_s=DEFAULT_TIMEOUT_S,
                         rpc_timeout=DEFAULT_RPC_TIMEOUT_S,
                         log_fn=print):
    """Poll the RPC until block_number advances by ≥ 1. Return the new tip dict.

    Transient RPC errors are logged and retried; only a hard timeout aborts.
    Raises TimeoutError if no new block arrives within timeout_s.
    """
    target = int(current_tip["block_number"]) + 1
    deadline = time.monotonic() + timeout_s
    poll_count = 0
    last_seen = current_tip["block_number"]
    while time.monotonic() < deadline:
        poll_count += 1
        try:
            tip = fetch_current_tip(rpc_url, timeout=rpc_timeout)
        except (URLError, HTTPError, RuntimeError, TimeoutError, OSError) as e:
            log_fn(f"[rsk] poll #{poll_count}: transient error: {e!r}; retrying")
            time.sleep(poll_interval_s)
            continue
        if tip["block_number"] >= target:
            return tip
        if tip["block_number"] != last_seen:
            log_fn(f"[rsk] poll #{poll_count}: tip moved to "
                   f"#{tip['block_number']} (waiting for ≥ #{target})")
            last_seen = tip["block_number"]
        time.sleep(poll_interval_s)
    raise TimeoutError(
        f"no new block within {timeout_s:.0f}s "
        f"(started at #{current_tip['block_number']}, target ≥ #{target}, "
        f"polled {poll_count} times)"
    )


def resolve_rpc_url(cli_arg):
    """CLI arg wins; fall back to TB_RSK_RPC env. Returns None if neither set."""
    if cli_arg:
        return cli_arg
    return os.environ.get("TB_RSK_RPC") or None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rpc", type=str, default=None,
                   help="JSON-RPC URL. Wins over TB_RSK_RPC env. "
                        "For demo: https://public-node.rsk.co")
    p.add_argument("--poll-interval-s", type=float,
                   default=DEFAULT_POLL_INTERVAL_S)
    p.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    args = p.parse_args()
    rpc_url = resolve_rpc_url(args.rpc)
    if rpc_url is None:
        print("ERROR: no RPC URL — pass --rpc or set TB_RSK_RPC", file=sys.stderr)
        sys.exit(2)

    print(f"rpc_url = {rpc_url}", flush=True)
    print("fetching current tip…", flush=True)
    t0 = time.monotonic()
    cur = fetch_current_tip(rpc_url)
    print(f"current_tip ({time.monotonic() - t0:.2f}s):")
    print(json.dumps(cur, indent=2))

    print(f"waiting for next block (poll every {args.poll_interval_s}s, "
          f"timeout {args.timeout_s}s)…", flush=True)
    t1 = time.monotonic()
    new = wait_for_next_block(rpc_url, cur,
                               poll_interval_s=args.poll_interval_s,
                               timeout_s=args.timeout_s)
    delay = time.monotonic() - t1
    print(f"new_tip ({delay:.2f}s wait):")
    print(json.dumps(new, indent=2))
    print(f"parent_hash matches old tip's block_hash: "
          f"{new['parent_hash'] == cur['block_hash']}")


if __name__ == "__main__":
    main()

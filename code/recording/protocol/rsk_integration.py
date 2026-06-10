"""rsk_integration.py — Truth Beam's RSK-side glue.

Scope: the TB-specific wiring around RSK JSON-RPC and the interior-anchor
publisher. Wraps rsk_anchor.py (low-level RPC) and anchor_publisher.py
(background tx sender) with session-lifecycle-aware helpers.

Adapt this module for:
- Different blockchain anchor (Bitcoin Core, Ethereum mainnet, Signet) —
  replace fetch_fresh_block_seed() and preflight() internals; keep the
  same function signatures so tb_main.py doesn't have to change
- Stricter pre-session wallet checks (balance threshold, gas price)
- A different end-anchor payload shape (currently `final_root` as a
  32-byte commitment under the `TB:FINAL:v8` domain)

Do NOT put here:
- Low-level JSON-RPC calls — those live in rsk_anchor.py
- The AnchorPublisher background-thread machinery — that's
  anchor_publisher.py
- Manifest construction — that's session_schema.py; this module only
  supplies the `anchor_start` / `anchor_end` dict shapes for the
  manifest to fold in
"""
from __future__ import annotations

import os
import sys
from typing import Optional

from rsk_anchor import (
    fetch_balance, fetch_chain_id,
)
from anchor_publisher import MIN_BALANCE_WEI as PUBLISHER_MIN_BALANCE_WEI
import rsk_wallet


INTERIOR_ANCHOR_RPC_DEFAULT = "https://public-node.testnet.rsk.co"
INTERIOR_ANCHOR_RPC_DEFAULTS_BY_NETWORK = {
    "testnet": "https://public-node.testnet.rsk.co",
    "mainnet": "https://public-node.rsk.co",
}
NETWORK_CHAIN_IDS = {"testnet": 31, "mainnet": 30}


def preflight(args, rsk_rpc_url: Optional[str]) -> dict:
    """Pre-session RSK checks. Validates network / chain_id agreement, and
    (when --interior-anchor is on) that a funded wallet exists for the
    selected network.

    Returns {wallet, wallet_address, interior_rpc_url, expected_chain_id}.
    On any failure, prints an error and calls sys.exit(1). This is the
    intended contract — preflight aborts the whole process rather than
    letting a session start with broken RSK state.
    """
    expected_chain_id = NETWORK_CHAIN_IDS[args.network]
    result = {"wallet": None, "wallet_address": None,
              "interior_rpc_url": None,
              "expected_chain_id": expected_chain_id}
    if args.interior_anchor:
        # v9: mainnet is supported. tb_main.py enforces interactive
        # operator confirmation (TTY) or --rsk-mainnet-confirm
        # (non-TTY) before any AnchorPublisher is constructed.
        wallet = rsk_wallet.load_wallet(network=args.network)
        if wallet is None:
            print(f"ERROR: --interior-anchor requires a funded {args.network} "
                  f"wallet.", file=sys.stderr)
            sys.exit(1)
        try:
            from eth_account import Account
            recovered = Account.from_key(wallet["private_key"]).address
        except Exception as e:
            print(f"ERROR: wallet private_key load failed: {e}",
                  file=sys.stderr)
            sys.exit(1)
        if recovered.lower() != wallet["address"].lower():
            print(f"ERROR: wallet private_key does not derive recorded address.",
                  file=sys.stderr)
            sys.exit(1)
        # v9: network-aware default RPC. Env overrides still win, but
        # scoped to the matching network so a stale testnet env var
        # cannot silently target testnet during a mainnet invocation.
        if args.network == "mainnet":
            interior_rpc_url = (
                os.environ.get("TB_RSK_MAINNET_RPC")
                or INTERIOR_ANCHOR_RPC_DEFAULTS_BY_NETWORK["mainnet"]
            )
        else:
            interior_rpc_url = (
                os.environ.get("TB_RSK_TESTNET_RPC")
                or INTERIOR_ANCHOR_RPC_DEFAULTS_BY_NETWORK["testnet"]
            )
        try:
            actual = fetch_chain_id(interior_rpc_url)
        except Exception as e:
            print(f"ERROR: interior-publisher RPC {interior_rpc_url} "
                  f"unreachable: {e}", file=sys.stderr)
            sys.exit(1)
        if actual != expected_chain_id:
            print(f"ERROR: interior RPC chain_id {actual} != expected "
                  f"{expected_chain_id}", file=sys.stderr)
            sys.exit(1)
        try:
            balance = fetch_balance(interior_rpc_url, wallet["address"])
        except Exception as e:
            print(f"ERROR: balance fetch failed: {e}", file=sys.stderr)
            sys.exit(1)
        if balance < PUBLISHER_MIN_BALANCE_WEI:
            print(f"ERROR: wallet balance {balance} < required "
                  f"{PUBLISHER_MIN_BALANCE_WEI}", file=sys.stderr)
            sys.exit(1)
        result["wallet"] = wallet
        result["wallet_address"] = wallet["address"]
        result["interior_rpc_url"] = interior_rpc_url

    if rsk_rpc_url:
        try:
            actual = fetch_chain_id(rsk_rpc_url)
        except Exception as e:
            print(f"ERROR: start-anchor RPC unreachable: {e}",
                  file=sys.stderr)
            sys.exit(1)
        if actual != expected_chain_id:
            print(f"ERROR: start-anchor RPC chain_id {actual} != expected "
                  f"{expected_chain_id}", file=sys.stderr)
            sys.exit(1)
    return result

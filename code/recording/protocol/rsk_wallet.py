#!/usr/bin/env python3
"""Truth Beam RSK wallet: key management for testnet and mainnet.

Subcommands:
  import         Import an existing private key (file, CLI, or interactive).
  setup          Check balance if key exists; show import options otherwise.
                 Use --generate-fresh to create a new key explicitly.
  check          Check balance (errors if no key).
  status         Read-only; print on-disk state, no network call.

On first run: prompts the user to import an existing key (recommended) or
explicitly request fresh-key generation with --generate-fresh. Same flow for
both --network mainnet and --network testnet.

Storage:
  ~/.tb/                         dir, 700 perms
  ~/.tb/{testnet,mainnet}_key.json   file, 600 perms

Key file schema:
  {
    "network":               "rsk-testnet" | "rsk_mainnet",
    "address":               "0x...",
    "private_key":           "0x...",           # 32-byte hex, 0x-prefixed
    "created_at_utc":        "2026-...",
    "last_balance_check_utc": "2026-..." | null,
    "last_balance_wei":       "<decimal str>" | null,
  }

RPC defaults:
  testnet: https://public-node.testnet.rsk.co
  mainnet: https://public-node.rsk.co
Override: --rpc <url> OR TB_RSK_{TESTNET,MAINNET}_RPC / TB_RSK_RPC env var.

Dependencies: web3 + eth_account (pip install web3).
"""

import argparse
import datetime as dt
import json
import os
import sys
import tempfile
from pathlib import Path

try:
    from eth_account import Account
    from web3 import Web3
except ImportError as e:
    print(f"ERROR: missing Python dependency: {e}", file=sys.stderr)
    print("Install with:  pip install web3 --break-system-packages",
          file=sys.stderr)
    sys.exit(3)


# v3: aligned with anchor_publisher.MIN_BALANCE_WEI (0.0001 tRBTC =
# 1e14 wei). The publisher's threshold is the operative one — sessions
# can still start below the wallet's `setup` default but will be refused
# by the publisher. Matching avoids the confusing "setup says NEEDS
# FUNDING but tb_loop says READY" divergence.
DEFAULT_MIN_BALANCE_WEI = 100_000_000_000_000  # 0.0001 tRBTC
FAUCET_URL = "https://faucet.rootstock.io/"

# secp256k1 group order — used to reject a pathological imported private key
# outside [1, n-1]. Vanishingly unlikely for randomly-generated keys but
# easy to check for hand-constructed inputs.
SECP256K1_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

# EIP-1191 checksum chain_ids (RSK-specific). chain_id is mixed into the
# keccak input ahead of the lowercase hex address; each letter is cased
# based on the first nibble of the hash at that position. Mainnet = 30,
# testnet = 31. Used only by the import path — the fresh-generate path
# still stores eth_account's EIP-55 casing for backward compatibility.
_EIP1191_CHAIN_IDS = {"mainnet": 30, "testnet": 31}

# Supported networks. testnet is the only one exercised in this RC; mainnet
# is the parallel code path with a separate key file — same rules, different
# chainId and RPC default. Not tested end-to-end.
NETWORKS = {
    "testnet": {
        "tag": "rsk-testnet",
        "default_rpc": "https://public-node.testnet.rsk.co",
        "key_filename": "testnet_key.json",
    },
    "mainnet": {
        "tag": "rsk-mainnet",
        "default_rpc": "https://public-node.rsk.co",
        "key_filename": "mainnet_key.json",
    },
}
DEFAULT_NETWORK = "testnet"

TB_DIR = Path.home() / ".tb"

# Perms
DIR_MODE = 0o700
FILE_MODE = 0o600


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _iso_now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds")


def _key_path(network):
    return TB_DIR / NETWORKS[network]["key_filename"]


def _resolve_rpc(cli_arg, network):
    if cli_arg:
        return cli_arg
    env_name = "TB_RSK_MAINNET_RPC" if network == "mainnet" else "TB_RSK_TESTNET_RPC"
    env = os.environ.get(env_name) or os.environ.get("TB_RSK_RPC")
    if env:
        return env
    return NETWORKS[network]["default_rpc"]


def _ensure_tb_dir():
    """Create ~/.tb/ with 700 perms if missing; fix perms if wrong."""
    if not TB_DIR.exists():
        TB_DIR.mkdir(mode=DIR_MODE, parents=False, exist_ok=False)
        print(f"[tb]   created {TB_DIR} (perms {oct(DIR_MODE)})")
        return
    mode = TB_DIR.stat().st_mode & 0o777
    if mode != DIR_MODE:
        print(f"[tb]   WARN {TB_DIR} perms {oct(mode)} != {oct(DIR_MODE)}; fixing")
        os.chmod(TB_DIR, DIR_MODE)


def _atomic_write_key(data, network):
    """Write {network}_key.json atomically with 600 perms.
    tempfile in the same dir → chmod → os.replace. chmod BEFORE rename so the
    final file is never world-readable even briefly.
    """
    _ensure_tb_dir()
    key_path = _key_path(network)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{NETWORKS[network]['key_filename']}.",
        suffix=".tmp", dir=str(TB_DIR))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        os.chmod(tmp_name, FILE_MODE)
        os.replace(tmp_name, key_path)
    except Exception:
        try:
            os.remove(tmp_name)
        except OSError:
            pass
        raise


def _load_key(network):
    """Return dict or None if file is missing."""
    key_path = _key_path(network)
    if not key_path.exists():
        return None
    mode = key_path.stat().st_mode & 0o777
    if mode != FILE_MODE:
        print(f"[key]  WARN {key_path} perms {oct(mode)} != {oct(FILE_MODE)}; fixing")
        os.chmod(key_path, FILE_MODE)
    with open(key_path) as f:
        return json.load(f)


def load_wallet(network=DEFAULT_NETWORK):
    """Public API: return the on-disk wallet dict for the given network, or
    None if missing. Used by tb_loop / anchor_publisher to fetch the funded
    key for the current --network.
    """
    if network not in NETWORKS:
        raise ValueError(f"unknown network: {network!r}")
    return _load_key(network)


def _generate_fresh_key(network):
    """Generate a new secp256k1 key and return the {network}_key.json dict."""
    acct = Account.create()
    return {
        "network": NETWORKS[network]["tag"],
        "address": acct.address,
        "private_key": "0x" + acct.key.hex() if not acct.key.hex().startswith("0x") else acct.key.hex(),
        "created_at_utc": _iso_now(),
        "last_balance_check_utc": None,
        "last_balance_wei": None,
    }


def _query_balance_wei(rpc_url, address):
    """One eth_getBalance call. Returns int (wei). Raises on RPC error."""
    w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 15}))
    if not w3.is_connected():
        raise RuntimeError(f"RPC unreachable: {rpc_url}")
    return int(w3.eth.get_balance(Web3.to_checksum_address(address)))


def _wei_to_trbtc(wei):
    # Integer division for the whole part, then fractional via Decimal for display.
    from decimal import Decimal
    return Decimal(int(wei)) / Decimal(10 ** 18)


def _print_funding_instructions(address, current_wei, needed_wei):
    current_trbtc = _wei_to_trbtc(current_wei)
    needed_trbtc = _wei_to_trbtc(needed_wei)
    print()
    print("=" * 72)
    print("NEEDS FUNDING")
    print("=" * 72)
    print(f"current balance: {current_wei} wei ({current_trbtc} tRBTC)")
    print(f"minimum needed:  {needed_wei} wei ({needed_trbtc} tRBTC)")
    print()
    print("Visit the RSK testnet faucet, solve the CAPTCHA in a browser, and")
    print("paste this address:")
    print()
    print(f"    {address}")
    print()
    print(f"Faucet URL:   {FAUCET_URL}")
    print()
    print("After the faucet transaction confirms (~30s), re-run:")
    print()
    print("    python3 rsk_wallet.py setup")
    print()
    print("=" * 72)


def _print_ready(address, balance_wei):
    trbtc = _wei_to_trbtc(balance_wei)
    print(f"READY: balance={balance_wei} wei ({trbtc} tRBTC)  address={address}")


def _eip1191_checksum(address_0x, chain_id):
    """Return the EIP-1191 (RSK-style) checksum-cased address for chain_id."""
    from eth_utils import keccak
    addr = address_0x.lower()
    if addr.startswith("0x"):
        addr = addr[2:]
    if len(addr) != 40:
        raise ValueError(f"bad address length: {address_0x!r}")
    h = keccak(text=f"{chain_id}0x{addr}").hex()
    out = ["0x"]
    for i, c in enumerate(addr):
        if c in "0123456789":
            out.append(c)
        else:
            out.append(c.upper() if int(h[i], 16) >= 8 else c)
    return "".join(out)


def _print_three_options_message(network, key_path):
    print(f"No {network} wallet found at {key_path}.")
    print()
    print("You have three options:")
    print()
    print("  1. Import from file (recommended for replicators with existing keys):")
    print()
    print(f"     python3 rsk_wallet.py import --network {network} --private-key-file PATH")
    print()
    print("  2. Import interactively (hidden prompt):")
    print()
    print(f"     python3 rsk_wallet.py import --network {network}")
    print()
    print("  3. Generate a fresh key (you'll need to fund and back it up):")
    print()
    print(f"     python3 rsk_wallet.py setup --network {network} --generate-fresh")
    print()
    print("Choose one and re-run.")
    if network == "testnet":
        print()
        print(f"Faucet for testnet: {FAUCET_URL}")


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_setup(args):
    network = args.network
    _ensure_tb_dir()
    key_path = _key_path(network)
    network_tag = NETWORKS[network]["tag"]

    key_data = _load_key(network)
    newly_generated = False
    if key_data is None:
        if not getattr(args, "generate_fresh", False):
            _print_three_options_message(network, key_path)
            return 2
        print(f"[key]  no key at {key_path}; generating fresh {network} key…")
        key_data = _generate_fresh_key(network)
        _atomic_write_key(key_data, network)
        newly_generated = True
        print(f"[key]  generated and written to {key_path} (perms {oct(FILE_MODE)})")
        print()
        print("=" * 72)
        print("NEW KEY GENERATED — BACK THIS UP NOW")
        print("=" * 72)
        print(json.dumps(key_data, indent=2, sort_keys=True))
        print("=" * 72)
        print()
    else:
        print(f"[key]  loaded existing key from {key_path}")

    if key_data.get("network") != network_tag:
        print(f"ERROR: stored key's network is {key_data.get('network')!r}, "
              f"expected {network_tag!r}. Refusing to use a key from the wrong "
              f"network. Move it aside or run "
              f"`python3 rsk_wallet.py nuke --network {network}`.",
              file=sys.stderr)
        sys.exit(4)

    address = key_data["address"]
    print(f"[key]  address: {address}")

    rpc_url = _resolve_rpc(args.rpc, network)
    print(f"[rpc]  network={network}  url={rpc_url}")
    print(f"[rpc]  querying eth_getBalance…")
    try:
        balance_wei = _query_balance_wei(rpc_url, address)
    except Exception as e:
        print(f"ERROR: balance query failed: {type(e).__name__}: {e}",
              file=sys.stderr)
        sys.exit(5)

    key_data["last_balance_check_utc"] = _iso_now()
    key_data["last_balance_wei"] = str(balance_wei)
    _atomic_write_key(key_data, network)

    if balance_wei >= args.min_balance_wei:
        _print_ready(address, balance_wei)
        return 0
    _print_funding_instructions(address, balance_wei, args.min_balance_wei)
    if newly_generated:
        print()
        print(f"(Key was just generated; private_key is inline in "
              f"{key_path} — back it up before funding.)")
    return 1


def cmd_import(args):
    import getpass
    network = args.network
    _ensure_tb_dir()
    key_path = _key_path(network)
    network_tag = NETWORKS[network]["tag"]

    if key_path.exists() and not args.force:
        print(f"ERROR: {key_path} already exists. Pass --force to overwrite "
              f"(after you are sure the existing key is backed up).",
              file=sys.stderr)
        return 4

    if args.private_key_file:
        path = Path(args.private_key_file).expanduser()
        if not path.exists():
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            return 2
        raw = path.read_text().strip()
    elif args.private_key:
        if network == "mainnet":
            print("WARNING: --private-key on the CLI is visible in shell "
                  "history. Prefer --private-key-file or the interactive "
                  "prompt for mainnet keys.", file=sys.stderr)
        raw = args.private_key.strip()
    else:
        try:
            raw = getpass.getpass("Private key (hex, input hidden): ").strip()
        except EOFError:
            print("ERROR: stdin closed; aborting.", file=sys.stderr)
            return 2

    if raw.lower().startswith("0x"):
        raw = raw[2:]
    if len(raw) != 64:
        print(f"ERROR: private key must be 64 hex characters (got "
              f"{len(raw)}).", file=sys.stderr)
        return 2
    try:
        priv_bytes = bytes.fromhex(raw)
    except ValueError as e:
        print(f"ERROR: invalid hex: {e}", file=sys.stderr)
        return 2
    n = int.from_bytes(priv_bytes, "big")
    if n == 0 or n >= SECP256K1_ORDER:
        print("ERROR: private key value outside secp256k1 valid range [1, n-1].",
              file=sys.stderr)
        return 2

    acct = Account.from_key(priv_bytes)
    chain_id = _EIP1191_CHAIN_IDS[network]
    address = _eip1191_checksum(acct.address, chain_id)

    key_data = {
        "network": network_tag,
        "address": address,
        "private_key": "0x" + raw,
        "created_at_utc": _iso_now(),
        "last_balance_check_utc": None,
        "last_balance_wei": None,
    }
    _atomic_write_key(key_data, network)
    print(f"[key]  imported to {key_path} (perms {oct(FILE_MODE)})")
    print(f"[key]  address: {address}  network={network}")

    rpc_url = _resolve_rpc(args.rpc, network)
    print(f"[rpc]  network={network}  url={rpc_url}")
    print(f"[rpc]  querying eth_getBalance…")
    try:
        balance_wei = _query_balance_wei(rpc_url, address)
    except Exception as e:
        print(f"ERROR: balance query failed: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 5

    key_data["last_balance_check_utc"] = _iso_now()
    key_data["last_balance_wei"] = str(balance_wei)
    _atomic_write_key(key_data, network)

    if balance_wei >= args.min_balance_wei:
        _print_ready(address, balance_wei)
        return 0
    _print_funding_instructions(address, balance_wei, args.min_balance_wei)
    return 1


def cmd_check(args):
    network = args.network
    key_path = _key_path(network)
    key_data = _load_key(network)
    if key_data is None:
        print(f"ERROR: no key at {key_path}. "
              f"Run: python3 rsk_wallet.py setup --network {network}",
              file=sys.stderr)
        sys.exit(2)
    address = key_data["address"]
    print(f"[key]  address: {address}")

    rpc_url = _resolve_rpc(args.rpc, network)
    print(f"[rpc]  network={network}  url={rpc_url}")
    try:
        balance_wei = _query_balance_wei(rpc_url, address)
    except Exception as e:
        print(f"ERROR: balance query failed: {type(e).__name__}: {e}",
              file=sys.stderr)
        sys.exit(5)

    key_data["last_balance_check_utc"] = _iso_now()
    key_data["last_balance_wei"] = str(balance_wei)
    _atomic_write_key(key_data, network)

    if balance_wei >= args.min_balance_wei:
        _print_ready(address, balance_wei)
        return 0
    _print_funding_instructions(address, balance_wei, args.min_balance_wei)
    return 1


def cmd_status(args):
    network = args.network
    key_path = _key_path(network)
    key_data = _load_key(network)
    if key_data is None:
        print(f"no key at {key_path}", file=sys.stderr)
        sys.exit(2)
    mode_dir = TB_DIR.stat().st_mode & 0o777
    mode_file = key_path.stat().st_mode & 0o777
    print(f"network                = {key_data.get('network')}")
    print(f"address                = {key_data.get('address')}")
    print(f"created_at_utc         = {key_data.get('created_at_utc')}")
    print(f"last_balance_check_utc = {key_data.get('last_balance_check_utc')}")
    lbw = key_data.get("last_balance_wei")
    if lbw is not None:
        trbtc = _wei_to_trbtc(int(lbw))
        print(f"last_balance_wei       = {lbw} ({trbtc} tRBTC)")
    else:
        print(f"last_balance_wei       = (never checked)")
    print(f"dir  perms ({TB_DIR})    = {oct(mode_dir)}")
    print(f"file perms ({key_path})  = {oct(mode_file)}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_common_args(sp):
    sp.add_argument("--rpc", type=str, default=None,
                    help="RSK JSON-RPC URL. Wins over TB_RSK_{TESTNET,MAINNET}_RPC "
                         "and TB_RSK_RPC env vars.")
    sp.add_argument("--network", type=str, choices=sorted(NETWORKS.keys()),
                    default=DEFAULT_NETWORK,
                    help=f"Network selector. Default: {DEFAULT_NETWORK}.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_setup = sub.add_parser("setup", help="Create key if missing, check balance")
    _add_common_args(sp_setup)
    sp_setup.add_argument("--min-balance-wei", type=int,
                          default=DEFAULT_MIN_BALANCE_WEI,
                          help=f"default: {DEFAULT_MIN_BALANCE_WEI}")
    sp_setup.add_argument("--generate-fresh", action="store_true",
                          help="Generate a new private key if none exists "
                               "(otherwise, import is required)")

    sp_import = sub.add_parser("import", help="Import an existing private key")
    _add_common_args(sp_import)
    sp_import.add_argument("--private-key-file", type=str, default=None,
                           help="Read private key from file (hex, one line)")
    sp_import.add_argument("--private-key", type=str, default=None,
                           help="Private key as hex string (WARNING: visible "
                                "in shell history)")
    sp_import.add_argument("--force", action="store_true",
                           help="Overwrite existing key file")
    sp_import.add_argument("--min-balance-wei", type=int,
                           default=DEFAULT_MIN_BALANCE_WEI)

    sp_check = sub.add_parser("check", help="Check balance (errors if no key)")
    _add_common_args(sp_check)
    sp_check.add_argument("--min-balance-wei", type=int,
                          default=DEFAULT_MIN_BALANCE_WEI)

    sp_status = sub.add_parser("status", help="Print on-disk state (no network)")
    sp_status.add_argument("--network", type=str, choices=sorted(NETWORKS.keys()),
                           default=DEFAULT_NETWORK)

    args = p.parse_args()
    dispatch = {
        "setup": cmd_setup,
        "import": cmd_import,
        "check": cmd_check,
        "status": cmd_status,
    }
    rc = dispatch[args.cmd](args)
    sys.exit(rc if isinstance(rc, int) else 0)


if __name__ == "__main__":
    main()

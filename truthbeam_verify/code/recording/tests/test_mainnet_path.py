"""Mainnet-path unit tests for v9 AnchorPublisher.

Verifies without touching the real network:

  (T1) Constructing with network='mainnet', mainnet_confirmed=False
       raises RuntimeError (the operator-safety gate).
  (T2) Constructing with network='mainnet', mainnet_confirmed=True
       succeeds and records chain_id=30.
  (T3) _sign_and_broadcast produces a signed tx whose chainId == 30
       on mainnet, 31 on testnet.
  (T4) Multi-RPC failover: if the primary URL raises, the secondary
       is tried; _active_rpc_url records the winner.
  (T5) All-RPC-fail: raises aggregate RuntimeError.

No real HTTP calls. Web3.HTTPProvider is monkey-patched to return a
stub client; eth_account signs locally (no network).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

HERE = Path(__file__).resolve().parent
PROTOCOL_DIR = HERE.parent / "protocol"
sys.path.insert(0, str(PROTOCOL_DIR))

from eth_account import Account                          # noqa: E402


def _make_wallet(seed: bytes = b"test-mainnet-wallet"):
    """Deterministic test wallet. 32-byte seed → secp256k1 private key."""
    # Use blake3 to map seed → 32 bytes (same derivation shape the test
    # tools use elsewhere in this project).
    from blake3 import blake3
    pk = blake3(seed).digest()
    acct = Account.from_key(pk)
    return {
        "private_key": "0x" + pk.hex(),
        "address": acct.address,
        "network": "rsk-mainnet",
    }


def _make_mock_web3(send_returns=None, balance_wei=10**18,
                    chain_id=30, nonce=0,
                    send_raises: Exception | None = None):
    """Return a MagicMock that looks like web3.Web3() for AnchorPublisher."""
    w3 = MagicMock()
    w3.is_connected.return_value = True
    w3.eth.chain_id = chain_id
    w3.eth.get_balance.return_value = balance_wei
    w3.eth.get_transaction_count.return_value = nonce
    w3.eth.gas_price = 60_000_000
    if send_raises is not None:
        w3.eth.send_raw_transaction.side_effect = send_raises
    else:
        w3.eth.send_raw_transaction.return_value = (
            send_returns if send_returns is not None
            else bytes.fromhex("ab" * 32))
    w3.to_checksum_address = lambda a: Account.from_key(
        _make_wallet()["private_key"]).address if not a.startswith("0x") else a
    return w3


# --- T1 --------------------------------------------------------------


def test_mainnet_without_confirm_raises(tmp_path):
    from anchor_publisher import AnchorPublisher
    wallet = _make_wallet()
    with pytest.raises(RuntimeError, match="mainnet_confirmed"):
        AnchorPublisher(
            wallet=wallet,
            rpc_url="https://public-node.rsk.co",
            session_dir=tmp_path,
            network="mainnet",
            mainnet_confirmed=False,
        )


# --- T2 --------------------------------------------------------------


def test_mainnet_with_confirm_constructs(tmp_path):
    from anchor_publisher import AnchorPublisher
    wallet = _make_wallet()
    p = AnchorPublisher(
        wallet=wallet,
        rpc_url="https://public-node.rsk.co",
        session_dir=tmp_path,
        network="mainnet",
        mainnet_confirmed=True,
    )
    assert p.chain_id == 30
    assert p.network == "mainnet"
    assert p.mainnet_confirmed is True
    assert p.rpc_urls == ["https://public-node.rsk.co"]


# --- T3 --------------------------------------------------------------


def _extract_tx_chain_id_from_signed(signed_tx_raw: bytes) -> int | None:
    """Decode an RLP-encoded legacy tx (eth) and read chainId from v.

    RSK uses legacy tx format; chainId is encoded in v as
    `v = chain_id * 2 + 35 + recovery_id` per EIP-155. Recover
    chain_id from the signed tx's v field.
    """
    import rlp
    # Legacy tx: [nonce, gasPrice, gas, to, value, data, v, r, s]
    decoded = rlp.decode(signed_tx_raw)
    v = int.from_bytes(decoded[6], "big") if isinstance(decoded[6], bytes) else int(decoded[6])
    # v = chain_id*2 + 35 + recovery_id  ⇒ chain_id = (v - 35) // 2
    if v >= 35:
        return (v - 35) // 2
    return None


@pytest.mark.parametrize("network,expected_chain_id", [
    ("mainnet", 30),
    ("testnet", 31),
])
def test_signed_tx_has_expected_chain_id(tmp_path, network, expected_chain_id):
    from anchor_publisher import AnchorPublisher
    from web3 import Web3

    wallet = _make_wallet()

    # Intercept Web3 HTTPProvider construction so no real network happens.
    with patch.object(Web3, "HTTPProvider") as mock_provider:
        mock_provider.return_value = MagicMock()
        p = AnchorPublisher(
            wallet=wallet,
            rpc_url="https://stub.invalid/",
            session_dir=tmp_path,
            network=network,
            mainnet_confirmed=(network == "mainnet"),
        )
        # Bypass start(); inject a mock Web3 cache entry so
        # _sign_and_broadcast's _rpc_try lands on our mock.
        mock_w3 = _make_mock_web3(chain_id=expected_chain_id)
        p._w3_cache["https://stub.invalid/"] = mock_w3

    # _sign_and_broadcast: pipes signed raw bytes through our mock;
    # capture the raw bytes it would have sent.
    captured_raw = {}
    def capture_send(raw):
        captured_raw["raw"] = raw
        return bytes.fromhex("cd" * 32)
    mock_w3.eth.send_raw_transaction.side_effect = capture_send

    tx_hash = p._sign_and_broadcast(
        nonce=0, gas_price_wei=60_000_000,
        tx_data_hex="ab" * 32,
    )
    assert tx_hash.startswith("0x")
    assert "raw" in captured_raw, "send_raw_transaction was never called"
    actual_chain_id = _extract_tx_chain_id_from_signed(captured_raw["raw"])
    assert actual_chain_id == expected_chain_id, (
        f"signed tx has chain_id={actual_chain_id}, expected "
        f"{expected_chain_id} for {network}"
    )


# --- T4 --------------------------------------------------------------


def test_multi_rpc_failover_primary_fails_fallback_succeeds(tmp_path):
    from anchor_publisher import AnchorPublisher
    from web3 import Web3

    wallet = _make_wallet()
    with patch.object(Web3, "HTTPProvider") as mock_provider:
        mock_provider.return_value = MagicMock()
        p = AnchorPublisher(
            wallet=wallet,
            rpc_url="https://primary.invalid/",
            rpc_url_fallbacks=["https://secondary.invalid/"],
            session_dir=tmp_path,
            network="mainnet",
            mainnet_confirmed=True,
        )

    # Primary raises on send; secondary succeeds.
    primary_mock = _make_mock_web3(
        send_raises=ConnectionError("primary down"))
    secondary_mock = _make_mock_web3()
    p._w3_cache["https://primary.invalid/"] = primary_mock
    p._w3_cache["https://secondary.invalid/"] = secondary_mock

    tx = p._sign_and_broadcast(
        nonce=0, gas_price_wei=60_000_000,
        tx_data_hex="ab" * 32,
    )
    assert tx.startswith("0x")
    # Fallback URL must be recorded as active.
    assert p._active_rpc_url == "https://secondary.invalid/"
    primary_mock.eth.send_raw_transaction.assert_called_once()
    secondary_mock.eth.send_raw_transaction.assert_called_once()


# --- T5 --------------------------------------------------------------


def test_all_rpc_fail_raises_aggregate(tmp_path):
    from anchor_publisher import AnchorPublisher
    from web3 import Web3

    wallet = _make_wallet()
    with patch.object(Web3, "HTTPProvider") as mock_provider:
        mock_provider.return_value = MagicMock()
        p = AnchorPublisher(
            wallet=wallet,
            rpc_url="https://primary.invalid/",
            rpc_url_fallbacks=["https://secondary.invalid/"],
            session_dir=tmp_path,
            network="mainnet",
            mainnet_confirmed=True,
        )

    for url in p.rpc_urls:
        p._w3_cache[url] = _make_mock_web3(
            send_raises=ConnectionError(f"{url} unreachable"))

    with pytest.raises(RuntimeError, match="all .* RPC endpoints failed"):
        p._sign_and_broadcast(
            nonce=0, gas_price_wei=60_000_000,
            tx_data_hex="ab" * 32,
        )


# --- T6: CSV schema includes rpc_endpoint_used ----------------------


def test_csv_schema_has_rpc_endpoint_used():
    from anchor_publisher import ANCHOR_TX_FIELDS
    assert "rpc_endpoint_used" in ANCHOR_TX_FIELDS
    assert "payload_commitment_hex" in ANCHOR_TX_FIELDS
    assert "pulse_index" in ANCHOR_TX_FIELDS

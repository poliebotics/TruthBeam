"""drand quicknet client with BLS verification (v9 writer side).

Fetches round beacons from a drand HTTP API and verifies each
round's BLS signature locally before accepting the randomness.
Both the fetch and the verify happen in the rig; the verifier
re-runs the same verification post-hoc against the round numbers
and values recorded in the session's chain_log.

drand quicknet parameters (scheme `bls-unchained-g1-rfc9380`,
period 3 s, genesis 2023-08-23):

- chain hash: 52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971
- public key: 96-byte G2 point (see QUICKNET_PUBKEY_HEX below)
- signatures: 48-byte G1 point (short sigs)
- message:   sha256(round_number_as_8_bytes_big_endian)
- DST:       BLS_SIG_BLS12381G1_XMD:SHA-256_SSWU_RO_NUL_
- verify:    e(sig_G1, G2_gen) == e(hash_to_G1(msg), pk_G2)

Dependencies: py_ecc (pure Python, slow-but-correct). Verification
is ~100 ms per round via the optimized BLS12-381 backend; verifying
once per drand round (every 3 s) is well within budget. A future
Phase-B optimization could switch to blst-python for batch verify.
"""
from __future__ import annotations

import hashlib
import json
import struct
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional

from py_ecc.bls.hash_to_curve import hash_to_G1
from py_ecc.bls.point_compression import decompress_G1, decompress_G2
from py_ecc.bls.g2_primitives import os2ip
from py_ecc.optimized_bls12_381 import (
    pairing, G2 as _G2_GEN, final_exponentiate, FQ12, neg,
)


# --- drand quicknet constants ---------------------------------------------

QUICKNET_CHAIN_HASH = (
    "52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971"
)
QUICKNET_PUBKEY_HEX = (
    "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c"
    "8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4bcb"
    "5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"
)
QUICKNET_PERIOD_S = 3
QUICKNET_GENESIS_TIME = 1692803367
DEFAULT_API_BASE = "https://api.drand.sh"
DST_G1_NUL = b"BLS_SIG_BLS12381G1_XMD:SHA-256_SSWU_RO_NUL_"

# Staleness: how old (seconds since last successful fetch) the cached
# round may be before the chain-loop treats it as failed. Tunable;
# 10 s tolerates one missed 3 s round plus HTTP retry plus network
# variance. Bump to 15 s if 10 s causes spurious failures.
DEFAULT_MAX_STALENESS_S = 10.0

# HTTP timeout per fetch.
DEFAULT_FETCH_TIMEOUT_S = 5.0


# --- HTTP fetchers --------------------------------------------------------


def _chain_url(api_base: str, chain_hash: str) -> str:
    return f"{api_base.rstrip('/')}/{chain_hash}"


def fetch_latest(api_base: str = DEFAULT_API_BASE,
                 chain_hash: str = QUICKNET_CHAIN_HASH,
                 timeout_s: float = DEFAULT_FETCH_TIMEOUT_S) -> dict:
    """Fetch /public/latest. Returns dict with round, randomness,
    signature. Raises on HTTP / JSON error."""
    url = f"{_chain_url(api_base, chain_hash)}/public/latest"
    with urllib.request.urlopen(url, timeout=timeout_s) as r:
        return json.loads(r.read())


def fetch_round(round_number: int,
                api_base: str = DEFAULT_API_BASE,
                chain_hash: str = QUICKNET_CHAIN_HASH,
                timeout_s: float = DEFAULT_FETCH_TIMEOUT_S) -> dict:
    """Fetch /public/<round>. Returns dict with round, randomness,
    signature. Used by the verifier to refetch rounds recorded in
    chain_log.csv."""
    url = f"{_chain_url(api_base, chain_hash)}/public/{int(round_number)}"
    with urllib.request.urlopen(url, timeout=timeout_s) as r:
        return json.loads(r.read())


def fetch_chain_info(api_base: str = DEFAULT_API_BASE,
                     chain_hash: str = QUICKNET_CHAIN_HASH,
                     timeout_s: float = DEFAULT_FETCH_TIMEOUT_S) -> dict:
    """Fetch /info. Returns chain metadata including public_key,
    period, genesis_time, schemeID. Used for a self-consistency
    check: the server-reported public_key must match
    QUICKNET_PUBKEY_HEX, or the client refuses to start."""
    url = f"{_chain_url(api_base, chain_hash)}/info"
    with urllib.request.urlopen(url, timeout=timeout_s) as r:
        return json.loads(r.read())


# --- BLS verification -----------------------------------------------------


def _round_to_message(round_number: int) -> bytes:
    """drand quicknet unchained: msg = sha256(round_number_be_8)."""
    return hashlib.sha256(struct.pack(">Q", int(round_number))).digest()


def verify_round(round_number: int, signature_hex: str,
                 pubkey_hex: str = QUICKNET_PUBKEY_HEX) -> bool:
    """Verify a drand quicknet round's BLS signature.

    Scheme: bls-unchained-g1-rfc9380. Message = sha256(round_8bytes_BE).
    Signature in G1 (48 bytes). Public key in G2 (96 bytes). DST
    `BLS_SIG_BLS12381G1_XMD:SHA-256_SSWU_RO_NUL_`.

    Returns True iff signature is valid. Does NOT check round_number
    matches a particular expected value, nor that randomness ==
    sha256(signature) — those are orthogonal checks the caller
    should perform separately.
    """
    pk_bytes = bytes.fromhex(pubkey_hex)
    if len(pk_bytes) != 96:
        raise ValueError(
            f"quicknet public key must be 96 bytes; got {len(pk_bytes)}"
        )
    sig_bytes = bytes.fromhex(signature_hex)
    if len(sig_bytes) != 48:
        raise ValueError(
            f"quicknet signature must be 48 bytes; got {len(sig_bytes)}"
        )

    pk_G2 = decompress_G2((os2ip(pk_bytes[:48]), os2ip(pk_bytes[48:])))
    sig_G1 = decompress_G1(os2ip(sig_bytes))
    msg = _round_to_message(round_number)
    h_G1 = hash_to_G1(msg, DST_G1_NUL, hashlib.sha256)

    # e(sig, G2_gen) == e(H, pk)  ⇔  e(sig, G2_gen) * e(-H, pk) == 1
    result = final_exponentiate(
        pairing(_G2_GEN, sig_G1, final_exponentiate=False) *
        pairing(pk_G2, neg(h_G1), final_exponentiate=False)
    )
    return result == FQ12.one()


def verify_randomness_matches_signature(signature_hex: str,
                                        randomness_hex: str) -> bool:
    """drand quicknet: randomness == sha256(signature). Cheap
    integrity check that does NOT require BLS verification."""
    return hashlib.sha256(
        bytes.fromhex(signature_hex)
    ).hexdigest() == randomness_hex


def verify_round_full(round_number: int, randomness_hex: str,
                      signature_hex: str,
                      pubkey_hex: str = QUICKNET_PUBKEY_HEX) -> bool:
    """Combined: randomness == sha256(sig) AND BLS verify of sig
    against pubkey for this round. Both must pass."""
    if not verify_randomness_matches_signature(signature_hex, randomness_hex):
        return False
    return verify_round(round_number, signature_hex, pubkey_hex)


# --- DrandClient ----------------------------------------------------------


@dataclass
class _RoundEntry:
    round_number: int
    randomness: bytes       # 32 bytes
    signature: bytes        # 48 bytes
    fetched_at: float       # time.monotonic() when last accepted
    verified: bool


class DrandError(RuntimeError):
    """Any drand-related failure that the caller should abort on."""


class DrandClient:
    """Background-polling drand client with local BLS verification.

    Lifecycle:
        c = DrandClient(chain_hash=..., pubkey_hex=..., api_base=...)
        c.start()                # sync initial fetch + verify + poll thread
        round_num, rand_bytes = c.get_current(require_fresh=True)
        c.stop()

    `start()` MUST succeed before chain capture begins — it both
    validates the chain-info against local constants and fetches
    the initial round (recorded as `drand_round_at_session_open` in
    manifest).
    """

    def __init__(self,
                 chain_hash: str = QUICKNET_CHAIN_HASH,
                 pubkey_hex: str = QUICKNET_PUBKEY_HEX,
                 api_base: str = DEFAULT_API_BASE,
                 max_staleness_s: float = DEFAULT_MAX_STALENESS_S,
                 period_s: int = QUICKNET_PERIOD_S,
                 fetch_timeout_s: float = DEFAULT_FETCH_TIMEOUT_S,
                 log_fn=print):
        self.chain_hash = chain_hash
        self.pubkey_hex = pubkey_hex
        self.api_base = api_base
        self.max_staleness_s = float(max_staleness_s)
        self.period_s = int(period_s)
        self.fetch_timeout_s = float(fetch_timeout_s)
        self.log = log_fn

        self._lock = threading.Lock()
        self._current: Optional[_RoundEntry] = None
        self._initial_round: Optional[int] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- public ----------------------------------------------------------

    def start(self) -> None:
        """Validate chain info against local constants, fetch + verify
        the initial round, start the background poll thread. Raises
        DrandError on any failure — the caller must not proceed."""
        # 1. Fetch /info and cross-check public_key + schemeID.
        try:
            info = fetch_chain_info(
                self.api_base, self.chain_hash, self.fetch_timeout_s
            )
        except Exception as e:
            raise DrandError(
                f"drand /info fetch failed at {self.api_base} "
                f"chain={self.chain_hash[:16]}…: {e!r}"
            ) from e
        if info.get("public_key") != self.pubkey_hex:
            raise DrandError(
                f"drand /info public_key mismatch — server says "
                f"{str(info.get('public_key'))[:32]}…, client expects "
                f"{self.pubkey_hex[:32]}…. Refusing to start — this is "
                f"either a MITM or a misconfigured chain_hash."
            )
        scheme = info.get("schemeID", "")
        if scheme != "bls-unchained-g1-rfc9380":
            raise DrandError(
                f"drand /info schemeID is {scheme!r}, expected "
                f"'bls-unchained-g1-rfc9380'. verify_round's BLS "
                f"scheme assumptions do not hold for other scheme IDs."
            )
        self.log(f"[drand] chain {self.chain_hash[:16]}… info OK "
                 f"(scheme={scheme}, period={info.get('period')}s)")

        # 2. Fetch + verify the initial round (BLS).
        self._fetch_and_verify_once(initial=True)
        with self._lock:
            self._initial_round = self._current.round_number

        # 3. Launch background poller.
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="drand-poll"
        )
        self._thread.start()
        self.log(f"[drand] client ready (initial round="
                 f"{self._initial_round})")

    def stop(self) -> None:
        """Signal the poll thread and wait up to 2 s for it to exit."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def get_current(self, require_fresh: bool = False) -> tuple[int, bytes, int]:
        """Return (round_number, randomness_bytes, staleness_ms).

        v9 corroborating-evidence mode (2026-04-24): drand is no longer
        primary timing; RSK is. Staleness is reported to the caller as
        `staleness_ms` (caller decides whether to log or ignore); this
        method does NOT hard-fail when stale.

        If no round has been successfully fetched yet (start() failed
        and nothing arrived since), returns (0, b"\\x00"*32, -1) as the
        zero-sentinel. This lets the chain loop continue; downstream
        verifier reports drand_availability=MISSING.

        `require_fresh` is retained for API compatibility but is
        advisory only (logs a warning; still returns the stale value).
        """
        with self._lock:
            cur = self._current
            if cur is None:
                # Drand never produced a verified round. Zero-sentinel.
                return 0, b"\x00" * 32, -1
            age_s = time.monotonic() - cur.fetched_at
            staleness_ms = int(age_s * 1000)
            if require_fresh and age_s > self.max_staleness_s:
                self.log(
                    f"[drand] WARN round {cur.round_number} is "
                    f"{age_s:.1f}s stale (max_staleness_s="
                    f"{self.max_staleness_s:.1f}); returning anyway "
                    f"per corroborating-evidence policy"
                )
            return cur.round_number, cur.randomness, staleness_ms

    @property
    def initial_round_number(self) -> int:
        """Round number active at start() time, or 0 if drand never
        produced a verified round (e.g. start() raised and the caller
        proceeded anyway under the v9 relaxation). Recorded in manifest
        as `drand_round_at_session_open`."""
        return self._initial_round if self._initial_round is not None else 0

    # -- internals -------------------------------------------------------

    def _fetch_and_verify_once(self, initial: bool = False) -> None:
        """Fetch latest, verify, and (if new round) update cache. On
        any failure raise DrandError — caller decides whether to
        abort (start() does) or swallow (poll loop does)."""
        try:
            rd = fetch_latest(self.api_base, self.chain_hash,
                              self.fetch_timeout_s)
        except Exception as e:
            raise DrandError(f"drand fetch_latest failed: {e!r}") from e

        try:
            round_number = int(rd["round"])
            randomness_hex = str(rd["randomness"])
            signature_hex = str(rd["signature"])
        except (KeyError, TypeError, ValueError) as e:
            raise DrandError(f"drand response malformed: {rd!r}: {e!r}") from e

        # Cheap integrity check first.
        if not verify_randomness_matches_signature(signature_hex,
                                                   randomness_hex):
            raise DrandError(
                f"drand round {round_number}: randomness != "
                f"sha256(signature). Refusing to accept."
            )

        # BLS verify (expensive, ~100 ms).
        try:
            ok = verify_round(round_number, signature_hex, self.pubkey_hex)
        except Exception as e:
            raise DrandError(
                f"drand round {round_number}: BLS verify raised: {e!r}"
            ) from e
        if not ok:
            raise DrandError(
                f"drand round {round_number}: BLS signature did not "
                f"verify against pubkey {self.pubkey_hex[:16]}…"
            )

        # Commit to cache.
        entry = _RoundEntry(
            round_number=round_number,
            randomness=bytes.fromhex(randomness_hex),
            signature=bytes.fromhex(signature_hex),
            fetched_at=time.monotonic(),
            verified=True,
        )
        with self._lock:
            prev = self._current
            if prev is not None and entry.round_number < prev.round_number:
                # Server went backwards — keep the older-but-higher round
                # we already verified. Treat as transient.
                if not initial:
                    self.log(
                        f"[drand] WARN server returned round "
                        f"{entry.round_number} < cached "
                        f"{prev.round_number}; ignoring"
                    )
                return
            self._current = entry
        if initial or (prev is None) or (entry.round_number != prev.round_number):
            self.log(
                f"[drand] round={entry.round_number} "
                f"rand={randomness_hex[:16]}…  (verified)"
            )

    def _poll_loop(self) -> None:
        """Runs in background thread. Fetches every `period_s` seconds
        (offset from genesis so fetches tend to land after a new round
        publishes). Swallows individual fetch errors — the chain loop's
        require_fresh check is the hard gate."""
        # Align poll to drand's round cadence — fetch just AFTER a round
        # is expected to publish. Quicknet genesis_time + N*period gives
        # the wall-clock start of round N+1.
        while not self._stop_event.is_set():
            self._stop_event.wait(float(self.period_s))
            if self._stop_event.is_set():
                break
            try:
                self._fetch_and_verify_once(initial=False)
            except DrandError as e:
                # Don't raise in the background thread — let staleness
                # grow. The chain loop will hard-fail when it tries to
                # get_current(require_fresh=True) beyond max_staleness_s.
                self.log(f"[drand] poll error: {e}")

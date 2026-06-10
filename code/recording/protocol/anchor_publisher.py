"""Phase-2 interior-anchor publisher: periodic pulse, no per-block state.

Background thread that fires one RSK testnet tx every `anchor_interval_s`
seconds (default 5.0). Each tx carries the latest committed S_t from the
Truth Beam chain in its data field. No replacement. No gas bumping. No per-
block scheduling. Each tx is independent: fresh nonce, fixed gas price
(max(eth_gasPrice, 60_000_000)), fire-and-forget.

Whichever block picks up a given tx, picks it up. We don't try to land in
any particular block. The audit trail is the CSV of broadcasts; a verifier
later resolves each tx hash to its block via eth_getTransactionByHash.

Lifecycle:
  start()                — RPC + balance check + nonce init + thread launch
  update_chain_state(t, S_t_hex)  — main thread call (lock-protected, ~µs)
  stop()                 — stops loop, fires ONE final tx with current S_t,
                           waits up to 30s for confirmation, records final_tx_info

Failure handling:
  RPC unreachable mid-session → log + skip this pulse. No retry, no backoff;
                                 next pulse in `anchor_interval_s` anyway.
  Nonce race                  → catch broadcast, refresh nonce, retry once.
                                 On second failure, log + skip.
  Wallet balance < threshold at start() → raise. Caller refuses to start session.
"""

import datetime as dt
import os
import threading
import time
from pathlib import Path

from eth_account import Account
from web3 import Web3

from session_schema import (
    compute_pulse_commitment,
    UNANCHORED_PREV_PULSE_COMMITMENT,
)


# --- Tunable constants -----------------------------------------------------

DEFAULT_ANCHOR_INTERVAL_S = 5.0

# Floor to keep cheap-testnet gas prices from underflowing replacement rules.
GAS_PRICE_FLOOR_WEI = 60_000_000  # 0.06 gwei

# 21000 base + 32-byte data + margin. Round up; testnet is cheap.
GAS_LIMIT = 25000

TESTNET_CHAIN_ID = 31
MAINNET_CHAIN_ID = 30  # reserved; publisher is testnet-only in this RC

NETWORK_CHAIN_IDS = {
    "testnet": TESTNET_CHAIN_ID,
    "mainnet": MAINNET_CHAIN_ID,
}

# Refuse to start if balance < this. ~hundreds of pulses worth.
MIN_BALANCE_WEI = 100_000_000_000_000  # 0.0001 tRBTC

# stop() waits this long for the final tx to confirm before giving up.
# Bumped 30→90 in v8: two consecutive sessions observed tx confirmation
# 15-30 s after the 30 s deadline expired on RSK testnet, where block
# times routinely span 30-45 s. 90 s covers the observed tail without
# adding dead time in the common case (the wait loop exits on the first
# successful get_transaction(...).blockNumber, so sessions where the tx
# lands quickly still return promptly). Tunable per session via the
# AnchorPublisher constructor's final_tx_wait_s parameter.
STOP_FINAL_WAIT_S = 90.0


# --- CSV schema ------------------------------------------------------------

ANCHOR_TX_FIELDS = (
    "wall_utc_fired",
    "chain_id",
    "nonce",
    "gas_price_wei",
    "tx_hash",
    "payload_state_index",
    "payload_S_hex",
    "payload_kind",  # v4: "state" for periodic, "final_root" for is_final=1
    "is_final",
    # v9 additions for state pulses (payload_kind="state"):
    # - payload_commitment_hex: 64-hex of the commitment-digest tx data
    # - pulse_index: 0-based within-session counter
    # - frame_start, frame_end: inclusive chain-row range this pulse covers
    # - prev_pulse_commitment_hex: previous pulse's commitment (or 32
    #   zero-bytes hex for first pulse)
    # For final_root rows these four columns are empty strings.
    "payload_commitment_hex",
    "pulse_index",
    "frame_start",
    "frame_end",
    "prev_pulse_commitment_hex",
    # v9 multi-RPC: which URL was used to broadcast this tx. Commas
    # and whitespace in URLs would break _append_csv_row's
    # canonical-form check; URLs of the shape "https://foo.bar/path"
    # pass.
    "rpc_endpoint_used",
)

# Publisher stop status codes. Used by tb_loop to decide whether it is
# safe to compute anchor_log_hash and fire the final anchor.
STOP_STATUS_EXITED_CLEAN = "exited_clean"
STOP_STATUS_EXITED_AFTER_WAIT = "exited_after_wait"
STOP_STATUS_STILL_ALIVE = "still_alive_at_timeout"


# --- Helpers ---------------------------------------------------------------

def _iso_now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds")


def _create_csv_exclusive(path, fields):
    """v4: canonical CSV writer. Exclusive-creates with the header line.

    Uses raw bytes + LF terminator so the on-disk bytes are byte-exact
    reproducible by any implementation. No csv.DictWriter — that module
    uses CRLF by default and has quoting rules that are hard to reproduce
    across languages. Field values in v4 are restricted to ASCII hex,
    ISO-8601 timestamps, integers, and known string literals — no commas
    or quotes ever appear in values, so unquoted CSV is unambiguous.

    v6: also fsyncs the parent directory after creation, matching the
    raw-frame write pattern.
    """
    path = Path(path)
    header = ",".join(fields) + "\n"
    with open(path, "xb") as f:
        f.write(header.encode("ascii"))
        f.flush()
        os.fsync(f.fileno())
    # v6: durable directory entry.
    try:
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


def _append_csv_row(path, fields, row):
    """Append one row in canonical CSV form. Flushes + fsyncs."""
    values = [str(row[f]) for f in fields]
    # Defensive: reject any value that would break canonical form.
    for v in values:
        if "," in v or "\n" in v or "\r" in v or "\"" in v:
            raise ValueError(
                f"anchor_txs.csv value contains reserved character: {v!r}"
            )
    line = ",".join(values) + "\n"
    with open(path, "ab") as f:
        f.write(line.encode("ascii"))
        f.flush()
        os.fsync(f.fileno())


def _normalize_tx_hash(h):
    """eth.send_raw_transaction returns HexBytes; normalize to '0x...'."""
    if hasattr(h, "hex"):
        s = h.hex()
    else:
        s = str(h)
    if not s.startswith("0x"):
        s = "0x" + s
    return s


# --- AnchorPublisher -------------------------------------------------------

class AnchorPublisher:
    def __init__(self, wallet, rpc_url, session_dir,
                 anchor_interval_s=DEFAULT_ANCHOR_INTERVAL_S,
                 network="testnet", log_fn=None,
                 final_tx_wait_s=STOP_FINAL_WAIT_S,
                 rpc_url_fallbacks=None,
                 mainnet_confirmed=False):
        """v9 constructor extensions:
          - rpc_url_fallbacks: optional list of additional RPC URLs to
            try when rpc_url fails. All must point to the same chain_id.
          - mainnet_confirmed: must be True when network='mainnet' —
            caller (tb_main) is responsible for obtaining interactive
            confirmation from the operator before instantiating.
        """
        if network not in NETWORK_CHAIN_IDS:
            raise ValueError(f"network must be one of "
                             f"{sorted(NETWORK_CHAIN_IDS)}, got {network!r}")
        # v9: mainnet gate replaced. Caller must pass mainnet_confirmed=True,
        # obtained via tb_main's interactive confirm (or --rsk-mainnet-confirm
        # in non-TTY mode). This prevents a stray --rsk-network mainnet
        # invocation from submitting real-RBTC transactions without
        # explicit operator sign-off.
        if network == "mainnet" and not mainnet_confirmed:
            raise RuntimeError(
                "AnchorPublisher(network='mainnet', mainnet_confirmed=False) "
                "— mainnet submissions require explicit operator confirmation. "
                "In TTY: tb_main.py prompts for 'confirm'. Non-TTY: pass "
                "--rsk-mainnet-confirm. AnchorPublisher refuses to bind a "
                "mainnet wallet without it."
            )
        if not isinstance(wallet, dict) or "private_key" not in wallet \
                or "address" not in wallet:
            raise ValueError("wallet must be a dict with 'private_key' and 'address'")
        if anchor_interval_s <= 0:
            raise ValueError(f"anchor_interval_s must be > 0, got {anchor_interval_s}")
        if final_tx_wait_s <= 0:
            raise ValueError(f"final_tx_wait_s must be > 0, got {final_tx_wait_s}")

        self.wallet = wallet
        # v9: multi-RPC. `rpc_url` is the primary; `rpc_url_fallbacks`
        # (optional list) are tried in order on primary failure. All
        # must point to the same chain_id (validated at first use).
        self.rpc_url = rpc_url
        self.rpc_url_fallbacks = list(rpc_url_fallbacks or [])
        self.rpc_urls = [rpc_url] + self.rpc_url_fallbacks
        self._w3_cache = {}          # url → Web3 instance (lazy)
        self._active_rpc_url = None  # URL of the most-recent successful call
        self.session_dir = Path(session_dir)
        self.anchor_interval_s = float(anchor_interval_s)
        self.final_tx_wait_s = float(final_tx_wait_s)
        self.network = network
        self.chain_id = NETWORK_CHAIN_IDS[network]
        self.mainnet_confirmed = bool(mainnet_confirmed)
        self.log = log_fn if log_fn is not None else print

        self.acct = Account.from_key(wallet["private_key"])
        if self.acct.address.lower() != wallet["address"].lower():
            raise ValueError("wallet address does not match private_key")
        self.address_cs = Web3.to_checksum_address(self.acct.address)

        self.csv_path = self.session_dir / "anchor_txs.csv"

        # Shared chain state — updated by main thread, read by publisher.
        # latest_chain_state holds (state_index:int, S_hex:str) such that
        # S_hex == S[state_index]. Chain rows are indexed 0..N-1 and states
        # are indexed 0..N; S_0 is the initial state, S_k is the state after
        # row (k-1) commits. So when chain row t has just been written, the
        # publisher is updated with state_index = t+1, S_hex = S_{t+1}.
        self._chain_state_lock = threading.Lock()
        self._latest_chain_state = None  # (state_index:int, S_hex:str)

        # v9 pulse-commitment state. session_id must be set via
        # set_session_id() before the first pulse fires (the publisher
        # start()s before tb_loop has generated the session_id, so we
        # defer this binding to a separate setter). Pulses skip cleanly
        # until session_id is known.
        self._session_id = None
        self._prev_pulse_commitment = UNANCHORED_PREV_PULSE_COMMITMENT
        self._pulse_index_counter = 0
        # First pulse covers rows [0 .. last_state_index-1]. Initialise
        # to -1 so frame_start = self._last_pulse_frame_end + 1 = 0.
        self._last_pulse_frame_end = -1

        # Internal state.
        self._w3 = None
        self._stop_event = threading.Event()
        self._thread = None
        self._nonce = None
        # v3: serializes nonce read + sign + broadcast + CSV append. Held
        # by both _fire_pulse and _fire_final_tx so concurrency in stop()'s
        # wait-window cannot produce duplicate nonces or missing CSV rows.
        self._broadcast_lock = threading.Lock()

        # Accounting (set during start/stop).
        self.starting_balance_wei = None
        self.ending_balance_wei = None
        self.broadcast_count = 0
        self.skip_count = 0

        # Populated by stop()'s final-tx flow. Caller (tb_loop) reads this
        # to build manifest.anchor_end.
        self.final_tx_info = None

    # --- Lifecycle ---------------------------------------------------------

    def start(self):
        """Validate RPC + wallet balance + nonce, pre-create anchor_txs.csv
        in exclusive mode, spawn the pulse thread. Raises on any failure;
        caller refuses to start the session."""
        self.log(f"[anchor] starting AnchorPublisher (periodic pulse)")
        self.log(f"[anchor]   network:           {self.network}  "
                 f"(chainId={self.chain_id})")
        self.log(f"[anchor]   rpc(s):            {self.rpc_urls}")
        self.log(f"[anchor]   address:           {self.acct.address}")
        self.log(f"[anchor]   anchor_interval_s: {self.anchor_interval_s}")
        # v9: multi-RPC. Chain-id + balance + nonce queries use the
        # failover path; the first successfully-responding URL becomes
        # the active endpoint. All endpoints in self.rpc_urls must agree
        # on chain_id; if any returns a mismatch, hard-fail (can't let
        # a wrong-network RPC silently stand in for a correct one).
        def _verify_chain_id(w3, url):
            cid = int(w3.eth.chain_id)
            if cid != self.chain_id:
                raise RuntimeError(
                    f"RPC {url} returned chain_id={cid}, expected "
                    f"{self.chain_id}"
                )
            return cid
        actual_chain_id, used_url = self._rpc_try(
            _verify_chain_id, op_label="chain_id")
        self.log(f"[anchor]   chain_id verified: {actual_chain_id} "
                 f"(via {used_url})")
        # Keep a "legacy" self._w3 pointing at the currently active
        # endpoint for code paths that still expect it (confirmation
        # polling, end-balance read). Every call that can trigger
        # a broadcast or a critical read uses _rpc_try instead.
        self._w3 = self._w3_cache[used_url]

        bal, _ = self._rpc_try(
            lambda w3, _u: int(w3.eth.get_balance(self.address_cs)),
            op_label="get_balance")
        self.starting_balance_wei = bal
        if self.starting_balance_wei < MIN_BALANCE_WEI:
            raise RuntimeError(
                f"wallet balance {self.starting_balance_wei} wei "
                f"< required {MIN_BALANCE_WEI} wei. "
                f"Run: python3 rsk_wallet.py check"
            )

        nonce_val, _ = self._rpc_try(
            lambda w3, _u: int(w3.eth.get_transaction_count(
                self.address_cs, "pending")),
            op_label="get_transaction_count")
        self._nonce = nonce_val
        self.log(f"[anchor]   balance:           {self.starting_balance_wei} wei")
        self.log(f"[anchor]   next_nonce:        {self._nonce}")

        # Exclusive-create anchor_txs.csv. Fails if the session dir is stale.
        _create_csv_exclusive(self.csv_path, ANCHOR_TX_FIELDS)

        self._thread = threading.Thread(
            target=self._run, daemon=True, name="anchor-pub")
        self._thread.start()

    def update_chain_state(self, state_index, S_hex):
        """Called by main thread after each chain row write. Non-blocking.

        state_index is the state index k such that S_hex == S_k. When row t
        has just been logged, the caller passes (t+1, S_{t+1}_hex) here.
        """
        with self._chain_state_lock:
            self._latest_chain_state = (int(state_index), str(S_hex))

    def set_session_id(self, session_id):
        """v9: bind the publisher to a session_id. Must be called before
        the first pulse fires. The publisher thread skips pulses cleanly
        until this is set — they'd fail the commitment-derivation check
        otherwise.

        Called by tb_loop._build_and_write_session_artefacts as soon as
        the SessionManifest carries a session_id.
        """
        if self._session_id is not None and self._session_id != session_id:
            raise RuntimeError(
                f"AnchorPublisher.set_session_id: attempt to rebind "
                f"session_id from {self._session_id!r} to {session_id!r}; "
                f"refusing — each publisher instance serves one session."
            )
        self._session_id = str(session_id)

    def _get_latest_state(self):
        with self._chain_state_lock:
            return self._latest_chain_state

    def stop(self):
        """v4: stop the pulse loop ONLY. No final tx here.

        Returns one of:
          STOP_STATUS_EXITED_CLEAN — pulse thread exited within timeout
          STOP_STATUS_STILL_ALIVE  — pulse thread did not exit within timeout
                                     (the caller MUST abort; calling fire_final_anchor
                                     after this is unsafe due to potential pulse-thread
                                     writes to the CSV)

        The caller (tb_loop) then reads anchor_txs.csv to compute
        anchor_log_hash, computes final_root, and if appropriate calls
        fire_final_anchor(final_root_hex). Splitting these phases lets the
        caller bind anchor_log_hash INTO the final tx's payload.
        """
        self.log(f"[anchor] stop() — signalling pulse thread")
        self._stop_event.set()
        if self._thread is None:
            return STOP_STATUS_EXITED_CLEAN
        self._thread.join(timeout=self.anchor_interval_s + 20.0)
        if self._thread.is_alive():
            self.log("[anchor] WARN pulse thread did not exit within "
                     "timeout — caller MUST treat this as aborted_publisher_stuck")
            return STOP_STATUS_STILL_ALIVE
        return STOP_STATUS_EXITED_CLEAN

    def read_ending_balance(self):
        """Read current wallet balance for accounting. Called by tb_loop
        after all broadcasts are done."""
        try:
            self.ending_balance_wei = int(
                self._w3.eth.get_balance(self.address_cs))
            delta = self.starting_balance_wei - self.ending_balance_wei
            self.log(f"[anchor] balance delta: {delta} wei "
                     f"(start={self.starting_balance_wei}, "
                     f"end={self.ending_balance_wei})")
        except Exception as e:
            self.log(f"[anchor] failed to read ending balance: {e!r}")
        self.log(f"[anchor] summary: broadcasts={self.broadcast_count}, "
                 f"skips={self.skip_count}")

    def fire_final_anchor(self, final_root_hex, state_index, S_N_hex):
        """v4: fire the final anchor tx with data = bytes.fromhex(final_root_hex).

        Must be called AFTER stop() returned STOP_STATUS_EXITED_CLEAN AND
        after the caller has computed anchor_log_hash and final_root.

        Args:
            final_root_hex: hex str of the 32-byte final_root (tx.input)
            state_index:    the state index N = N_frames; recorded in CSV
            S_N_hex:        hex str of the terminal chain state; recorded
                            into final_tx_info so anchor_end can expose
                            both payload_S_N_hex and payload_final_root_hex.

        On confirmed landing, populates self.final_tx_info with block info
        and both payload hashes. On timeout, final_tx_info stays None and
        the caller omits anchor_end from the manifest.
        """
        if len(final_root_hex) != 64:
            raise ValueError(f"final_root_hex must be 64 hex chars, got {len(final_root_hex)}")

        with self._broadcast_lock:
            try:
                gas_price = self._gas_price_now()
            except Exception as e:
                self.log(f"[anchor] final: gas price query failed: {e!r}")
                return

            nonce = self._nonce
            try:
                tx_hash = self._sign_and_broadcast(nonce, gas_price, final_root_hex)
            except Exception as e:
                err_low = repr(e).lower()
                if "nonce" in err_low and ("too low" in err_low or "already known" in err_low):
                    try:
                        new_nonce, _ = self._rpc_try(
                            lambda w3, _u: int(
                                w3.eth.get_transaction_count(
                                    self.address_cs, "pending")),
                            op_label="nonce_reprobe")
                        self.log(f"[anchor] final: nonce drift {nonce}->{new_nonce}; retrying once")
                        nonce = new_nonce
                        self._nonce = new_nonce
                        tx_hash = self._sign_and_broadcast(nonce, gas_price, final_root_hex)
                    except Exception as e2:
                        self.log(f"[anchor] final: retry after nonce drift FAILED: {e2!r}")
                        return
                else:
                    self.log(f"[anchor] final: broadcast failed: {e!r}")
                    return

            self._nonce = nonce + 1
            self.broadcast_count += 1
            self.log(f"[anchor] final tx: nonce={nonce} chain_id={self.chain_id} "
                     f"tx={tx_hash[:18]}… payload_state_index={state_index} "
                     f"final_root={final_root_hex[:16]}… "
                     f"S_N={S_N_hex[:16]}…")
            _append_csv_row(self.csv_path, ANCHOR_TX_FIELDS, {
                "wall_utc_fired": _iso_now(),
                "chain_id": self.chain_id,
                "nonce": nonce,
                "gas_price_wei": gas_price,
                "tx_hash": tx_hash,
                "payload_state_index": state_index,
                "payload_S_hex": final_root_hex,  # v4: on-chain data is final_root
                "payload_kind": "final_root",  # v4
                "is_final": 1,
                # v9 commitment fields are NOT populated for final_root
                # rows — final anchor uses the final_root digest directly
                # (already a domain-separated BLAKE3 commitment under
                # TB:FINAL:v8). Empty strings preserve CSV column count.
                "payload_commitment_hex": "",
                "pulse_index": "",
                "frame_start": "",
                "frame_end": "",
                "prev_pulse_commitment_hex": "",
                "rpc_endpoint_used": self._active_rpc_url or self.rpc_url,
            })

        # Wait for confirmation. final_tx_info is populated ONLY on
        # confirmation within the window. Callers omit anchor_end otherwise.
        # Window is per-publisher-instance (self.final_tx_wait_s) so sessions
        # can tune it; default STOP_FINAL_WAIT_S is 90 s in v8.
        deadline = time.monotonic() + self.final_tx_wait_s
        landed_block = None
        while time.monotonic() < deadline:
            try:
                tx = self._w3.eth.get_transaction(tx_hash)
            except Exception:
                tx = None
            if tx is not None and tx.get("blockNumber") is not None:
                landed_block = int(tx["blockNumber"])
                break
            time.sleep(2.0)

        if landed_block is None:
            self.log(f"[anchor] final tx BROADCAST BUT NOT CONFIRMED within "
                     f"{self.final_tx_wait_s}s; anchor_end will be OMITTED "
                     f"from manifest. tx_hash={tx_hash}")
            self.final_tx_info = None
            return

        # Fetch full block info. v5: if any required field is missing, do
        # NOT populate final_tx_info — anchor_end is "all-or-nothing" by
        # schema, never partial. Honest verdict becomes "final-root-absent".
        try:
            blk = self._w3.eth.get_block(landed_block)
        except Exception as e:
            self.log(f"[anchor] final: block info fetch failed: {e!r}; "
                     f"anchor_end will be OMITTED")
            self.final_tx_info = None
            return
        try:
            raw_hash = blk["hash"]
            block_hash = raw_hash.hex() if hasattr(raw_hash, "hex") else str(raw_hash)
            if not block_hash.startswith("0x"):
                block_hash = "0x" + block_hash
            block_ts = dt.datetime.fromtimestamp(
                int(blk["timestamp"]), dt.timezone.utc).isoformat()
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            self.log(f"[anchor] final: block info had unexpected shape: "
                     f"{e!r}; anchor_end will be OMITTED")
            self.final_tx_info = None
            return

        # All required fields present and well-formed.
        self.log(f"[anchor] final tx LANDED in block {landed_block}")
        self.final_tx_info = {
            "tx_hash": tx_hash,
            "state_index": state_index,
            "payload_S_N_hex": S_N_hex,
            "payload_final_root_hex": final_root_hex,
            "block_number": landed_block,
            "block_hash": block_hash,
            "block_timestamp_utc": block_ts,
        }

        try:
            self.ending_balance_wei = int(
                self._w3.eth.get_balance(self.address_cs))
            delta = self.starting_balance_wei - self.ending_balance_wei
            self.log(f"[anchor] balance delta: {delta} wei "
                     f"(start={self.starting_balance_wei}, "
                     f"end={self.ending_balance_wei})")
        except Exception as e:
            self.log(f"[anchor] failed to read ending balance: {e!r}")

        self.log(f"[anchor] summary: broadcasts={self.broadcast_count}, "
                 f"skips={self.skip_count}")

    # --- Background loop ---------------------------------------------------

    def _run(self):
        """Fire one tx every anchor_interval_s. Skip on transient errors."""
        # Fire the first pulse immediately so we don't waste the first interval
        # waiting for chain state to appear.
        next_fire_at = time.monotonic()
        while not self._stop_event.is_set():
            now = time.monotonic()
            if now < next_fire_at:
                self._stop_event.wait(min(next_fire_at - now, 1.0))
                continue
            next_fire_at = now + self.anchor_interval_s

            latest = self._get_latest_state()
            if latest is None:
                # No chain state yet — nothing to commit. Don't burn a nonce
                # on a meaningless tx.
                continue
            t, S_t_hex = latest
            self._fire_pulse(t, S_t_hex)

    def _fire_pulse(self, payload_state_index, payload_S_hex):
        """One periodic broadcast. Catches RPC errors. One nonce-race retry.

        v3: entire nonce read → sign → broadcast → CSV append is serialized
        under _broadcast_lock. Prevents duplicate nonces or missing CSV
        rows when the pulse loop and stop()'s final-tx path overlap.

        v9: the on-chain tx `data` field is a commitment-digest derived
        from (session_id, pulse_index, frame_range, prev_commitment, S_t),
        NOT raw S_t. See session_schema.compute_pulse_commitment. Raw
        S_t is still recorded in `payload_S_hex` for local verification
        convenience; the commitment goes to the chain as
        `payload_commitment_hex`.
        """
        with self._broadcast_lock:
            # v9 gate: publisher cannot fire a pulse until tb_loop has
            # bound a session_id. Skip cleanly and return — next interval
            # tries again. In practice set_session_id fires within a few
            # hundred ms of publisher.start(), so this window is tiny.
            if self._session_id is None:
                self.log("[anchor] pulse: session_id not yet bound; "
                         "skipping this interval")
                self.skip_count += 1
                return

            # Derive v9 commitment from the current pulse's range + state.
            # state_index = payload_state_index = t+1 where t is the last
            # row logged. Frame range covered by this pulse is
            # [self._last_pulse_frame_end + 1 .. payload_state_index - 1].
            pulse_index = self._pulse_index_counter
            frame_start = self._last_pulse_frame_end + 1
            frame_end = int(payload_state_index) - 1
            if frame_end < frame_start:
                # No new frames since last pulse. Skip (would produce a
                # degenerate zero-range commitment).
                self.log(f"[anchor] pulse: no new frames since last pulse "
                         f"(frame_start={frame_start}, "
                         f"frame_end={frame_end}); skipping")
                self.skip_count += 1
                return

            try:
                commitment = compute_pulse_commitment(
                    session_id=self._session_id,
                    pulse_index=pulse_index,
                    frame_start=frame_start,
                    frame_end=frame_end,
                    prev_pulse_commitment=self._prev_pulse_commitment,
                    S_t=bytes.fromhex(payload_S_hex),
                )
            except Exception as e:
                self.log(f"[anchor] pulse: commitment derivation failed: "
                         f"{e!r}; skipping")
                self.skip_count += 1
                return
            commitment_hex = commitment.hex()
            prev_pulse_hex = self._prev_pulse_commitment.hex()

            try:
                gas_price = self._gas_price_now()
            except Exception as e:
                self.log(f"[anchor] pulse: gas price query failed: {e!r}; skipping")
                self.skip_count += 1
                return

            nonce = self._nonce
            # v9: tx data is the commitment, not raw S_t.
            try:
                tx_hash = self._sign_and_broadcast(nonce, gas_price, commitment_hex)
            except Exception as e:
                err_low = repr(e).lower()
                if "nonce" in err_low and ("too low" in err_low or "already known" in err_low):
                    try:
                        new_nonce, _ = self._rpc_try(
                            lambda w3, _u: int(
                                w3.eth.get_transaction_count(
                                    self.address_cs, "pending")),
                            op_label="nonce_reprobe")
                        self.log(f"[anchor] pulse: nonce drift {nonce}->{new_nonce}; retrying once")
                        self._nonce = new_nonce
                        nonce = new_nonce
                        tx_hash = self._sign_and_broadcast(nonce, gas_price, commitment_hex)
                    except Exception as e2:
                        self.log(f"[anchor] pulse: retry after nonce drift FAILED: {e2!r}; skipping")
                        self.skip_count += 1
                        return
                else:
                    self.log(f"[anchor] pulse: broadcast FAILED: {e!r}; skipping")
                    self.skip_count += 1
                    return

            self._nonce = nonce + 1
            self.broadcast_count += 1
            _append_csv_row(self.csv_path, ANCHOR_TX_FIELDS, {
                "wall_utc_fired": _iso_now(),
                "chain_id": self.chain_id,
                "nonce": nonce,
                "gas_price_wei": gas_price,
                "tx_hash": tx_hash,
                "payload_state_index": payload_state_index,
                "payload_S_hex": payload_S_hex,     # raw S_t, for local verify
                "payload_kind": "state",
                "is_final": 0,
                # v9 commitment fields:
                "payload_commitment_hex": commitment_hex,
                "pulse_index": pulse_index,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "prev_pulse_commitment_hex": prev_pulse_hex,
                "rpc_endpoint_used": self._active_rpc_url or self.rpc_url,
            })
            # v9: advance pulse-chain state ONLY on successful broadcast.
            # Failed pulses retain the prior prev_pulse_commitment and
            # pulse_index_counter so the next attempt continues the chain
            # linearly (with a wider frame range if frames accumulated in
            # the meantime).
            self._prev_pulse_commitment = commitment
            self._pulse_index_counter = pulse_index + 1
            self._last_pulse_frame_end = frame_end
            self.log(f"[anchor] pulse: nonce={nonce} "
                     f"chain_id={self.chain_id} gas={gas_price} "
                     f"tx={tx_hash[:18]}… "
                     f"pulse_index={pulse_index} "
                     f"frames=[{frame_start},{frame_end}] "
                     f"commitment={commitment_hex[:16]}…")

    # --- Sign + broadcast (v9: multi-RPC failover) -------------------------

    def _get_w3(self, url):
        """Lazy-cache a Web3 instance per URL."""
        w3 = self._w3_cache.get(url)
        if w3 is None:
            w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 15}))
            self._w3_cache[url] = w3
        return w3

    def _rpc_try(self, fn, op_label="rpc"):
        """Iterate self.rpc_urls in order. Call fn(w3, url); return
        (result, successful_url). On each exception, log the fallover
        and try the next URL. Raises RuntimeError if all fail.
        Updates self._active_rpc_url on success."""
        errors = []
        for url in self.rpc_urls:
            try:
                w3 = self._get_w3(url)
                result = fn(w3, url)
                self._active_rpc_url = url
                return result, url
            except Exception as e:
                errors.append((url, repr(e)[:120]))
                self.log(f"[anchor] {op_label} via {url} failed: "
                         f"{e!r}; failing over")
                continue
        raise RuntimeError(
            f"{op_label}: all {len(self.rpc_urls)} RPC endpoints failed: "
            f"{errors}"
        )

    def _gas_price_now(self):
        median, _ = self._rpc_try(
            lambda w3, _u: int(w3.eth.gas_price),
            op_label="gas_price")
        return max(median, GAS_PRICE_FLOOR_WEI)

    def _sign_and_broadcast(self, nonce, gas_price_wei, tx_data_hex):
        """Sign once (chain_id pinned), broadcast via failover. Returns
        the normalized tx_hash. tx_data_hex is the 64-hex payload
        (pulse commitment OR final_root)."""
        tx = {
            "to": self.address_cs,
            "value": 0,
            "nonce": int(nonce),
            "gasPrice": int(gas_price_wei),
            "gas": GAS_LIMIT,
            "data": bytes.fromhex(tx_data_hex),
            "chainId": self.chain_id,
        }
        signed = self.acct.sign_transaction(tx)
        raw = signed.raw_transaction
        h, _url = self._rpc_try(
            lambda w3, _u: w3.eth.send_raw_transaction(raw),
            op_label="send_raw_transaction")
        return _normalize_tx_hash(h)

"""tb_main.py — CLI entry point and session-lifecycle orchestration.

Scope: argument parsing and the top-level main() that wires together
camera.py, projector.py (via the Gtk.Application in tb_loop.py),
rsk_integration.py, session_finalize.py, and chain.py for a single
recording session.

Adapt this module for:
- Adding / renaming CLI flags
- Changing the default session directory naming scheme
- Reordering the pre-session / post-session phases
- Plugging in a different orchestrator class (future Phase-B: when
  TBLoopV8 decomposes into free loop functions, main() here creates
  a Gtk.Application and calls `capture_blocking.run(...)` directly
  rather than going through the monolithic class)

Do NOT put here:
- Camera / projector / chain logic — those are their own modules
- Long-running threads — currently owned by TBLoopV8 in tb_loop.py;
  a later refactor moves them to dedicated capture_blocking / capture_async
  modules.
- Hash math — chain.py / session_schema.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import os
import shutil
import signal
import statistics
import sys
import time
from pathlib import Path

from blake3 import blake3

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import GLib  # noqa: E402

from session_schema import (
    compute_final_root,
    compute_capture_log_hash, compute_chain_log_hash,
    SESSION_MODE_ASYNC_TIGHTENED,
    SESSION_STATUS_ABORTED,
    SESSION_STATUS_ABORTED_PUBLISHER_STUCK,
    SESSION_STATUS_ABORTED_SETUP_FAILED,
    SESSION_STATUS_ABORTED_SIGNAL_DURING_FINALIZE,
)
from rsk_anchor import resolve_rpc_url
from anchor_publisher import AnchorPublisher, STOP_STATUS_STILL_ALIVE

from projector import pick_monitor
from rsk_integration import preflight, NETWORK_CHAIN_IDS
from session_finalize import offline_encode_previews, reconcile_capture_log
from tb_loop import TBLoopV8, DEFAULT_EXPOSURE_US, DEFAULT_GAIN_DB


# ---- CLI ----

def _anchor_interval_s_type(s):
    try:
        v = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--anchor-interval-s must be a number, got {s!r}"
        )
    if not (1.0 <= v <= 60.0):
        raise argparse.ArgumentTypeError(
            f"--anchor-interval-s must be in [1.0, 60.0], got {v}"
        )
    return v


def _final_tx_wait_s_type(s):
    try:
        v = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--final-tx-wait-s must be a number, got {s!r}"
        )
    if not (10.0 <= v <= 600.0):
        raise argparse.ArgumentTypeError(
            f"--final-tx-wait-s must be in [10.0, 600.0], got {v}"
        )
    return v


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--monitor", type=int)
    g.add_argument("--connector", type=str)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--exposure-us", type=float, default=DEFAULT_EXPOSURE_US)
    p.add_argument("--gain", type=float, default=DEFAULT_GAIN_DB)
    p.add_argument("--activation", type=str, default="RisingEdge",
                   choices=["RisingEdge", "FallingEdge"])
    p.add_argument("--format", type=str, default="BayerRG8",
                   choices=["BayerRG8"])
    p.add_argument("--session-dir", type=str, default=None)
    p.add_argument("--cpu-only", action="store_true",
                   help="Force CPU tile generator (tile_cpu). Default is "
                        "GPU (tile_gpu); CPU is the fallback.")
    p.add_argument("--mode", type=str, default="blocking",
                   choices=("blocking", "async", "async-tightened"),
                   help="Mode of operation. "
                        "blocking (default): chain advances once per "
                        "capture-emission pair with empirical pipeline-"
                        "delay wait; mechanical correspondence between "
                        "captures and emissions; chain rate ~3-4 Hz; "
                        "requires a pipeline calibration at "
                        "~/.tb/pipeline_calibration_<rig_hash>.json. "
                        "async: chain advances at camera rate (~15 Hz) "
                        "via five-thread pipeline; statistical "
                        "correspondence; no calibration required. "
                        "async-tightened (v8.3): same five-thread "
                        "pipeline as async with added runtime "
                        "mitigations (SCHED_FIFO, CPU affinity, "
                        "mlockall, GC lockdown, pre-allocated capture "
                        "ring, /dev/cpu_dma_latency = 0 µs) to cut "
                        "async slip rate. Mitigations are best-effort "
                        "with graceful fallback to nice/unlocked on "
                        "permission denial.")
    p.add_argument("--calibration-dir", type=str, default=None,
                   help="Directory containing pipeline_calibration_*.json. "
                        "Default: ~/.tb/.")
    p.add_argument("--rsk-rpc", type=str, default=None,
                   help="RSK JSON-RPC URL for fresh-block anchor at start.")
    p.add_argument("--interior-anchor", action="store_true",
                   help="Run background AnchorPublisher broadcasting S_t.")
    p.add_argument("--anchor-interval-s", type=_anchor_interval_s_type,
                   default=5.0,
                   help="Seconds between interior-anchor pulses, in [1.0, 60.0].")
    p.add_argument("--final-tx-wait-s", type=_final_tx_wait_s_type,
                   default=90.0,
                   help="Seconds to wait for the final anchor tx to confirm "
                        "on-chain before omitting anchor_end from the "
                        "manifest. In [10.0, 600.0]. Default 90 (bumped from "
                        "30 in v8 after RSK testnet sessions observed "
                        "confirmation 15-30 s after the 30 s deadline). The "
                        "publisher exits promptly when the tx lands early, "
                        "so a larger window does not cost dead time in the "
                        "common case.")
    p.add_argument("--network", type=str,
                   choices=("testnet", "mainnet"), default="testnet",
                   help="RSK network for interior anchor + anchor_start label. "
                        "mainnet submissions require --rsk-mainnet-confirm "
                        "(or interactive 'confirm' prompt in TTY mode).")
    p.add_argument("--rsk-mainnet-confirm", action="store_true",
                   help="Non-TTY bypass for the mainnet confirmation prompt. "
                        "Required when --network=mainnet and stdin is not a "
                        "TTY. In TTY mode the interactive prompt is used "
                        "regardless of this flag. IMPORTANT: mainnet "
                        "submissions spend real RBTC; this flag is the "
                        "robot-proof gate against accidental invocations.")
    p.add_argument("--rsk-rpc-fallbacks", type=str, default="",
                   help="Comma-separated list of fallback RSK RPC URLs. "
                        "Tried in order if the primary (--rsk-rpc or network "
                        "default) fails. Example: "
                        "https://mycrypto.rsk.co,https://rpc.mainnet.rootstock.io")
    return p.parse_args()


def _mainnet_confirm_prompt(args, wallet_address: str,
                            starting_balance_wei: int,
                            rpc_urls: list) -> bool:
    """v9: interactive safety gate for mainnet submissions.

    Returns True if the operator confirmed. Returns False if they
    declined. Non-TTY mode REQUIRES --rsk-mainnet-confirm; absence
    refuses without prompting.
    """
    is_tty = sys.stdin.isatty() and sys.stdout.isatty()
    pulse_count = max(1, int(args.duration / args.anchor_interval_s)) + 1  # +1 final
    # Minimum-cost estimate at the RSK mainnet gas floor (60 gwei × 25000).
    # Real gas price at tx-time may be higher.
    min_cost_wei = pulse_count * 25000 * 60_000_000
    bal_rbtc = starting_balance_wei / 1e18
    min_cost_rbtc = min_cost_wei / 1e18
    banner = (
        "\n" + "=" * 72 + "\n"
        "MAINNET SUBMISSION CONFIRMATION\n"
        + "=" * 72 + "\n"
        f"  network:           rsk-mainnet (chain_id=30)\n"
        f"  RPC endpoints:     {rpc_urls}\n"
        f"  wallet:            {wallet_address}\n"
        f"  balance:           {starting_balance_wei} wei "
        f"({bal_rbtc:.8f} RBTC)\n"
        f"  duration:          {args.duration} s\n"
        f"  anchor_interval_s: {args.anchor_interval_s} s\n"
        f"  expected pulses:   {pulse_count}  (incl. final)\n"
        f"  estimated MIN fee: {min_cost_wei} wei "
        f"({min_cost_rbtc:.8f} RBTC)  [at 60 gwei gas floor; real "
        f"price may be higher]\n"
        + "=" * 72 + "\n"
    )
    print(banner, flush=True)
    if not is_tty:
        if args.rsk_mainnet_confirm:
            print("[mainnet] non-TTY + --rsk-mainnet-confirm set; "
                  "confirmation accepted.", flush=True)
            return True
        print("[mainnet] stdin/stdout is not a TTY. This session would "
              "submit real-RBTC transactions. Re-invoke with "
              "--rsk-mainnet-confirm to bypass the interactive prompt, "
              "OR run from a TTY terminal. Refusing.", flush=True)
        return False
    print("Type exactly 'confirm' (lower-case, no quotes) to proceed. "
          "Anything else aborts.", flush=True)
    try:
        answer = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("[mainnet] aborted by operator (EOF/Ctrl-C).", flush=True)
        return False
    if answer != "confirm":
        print(f"[mainnet] did not receive 'confirm' (got {answer!r}); "
              f"aborting.", flush=True)
        return False
    print("[mainnet] confirmation accepted.", flush=True)
    return True


# ---- main ----

def main():
    args = parse_args()
    mon = pick_monitor(args)
    print(f"target monitor: idx={mon['index']} connector={mon['connector']} "
          f"geo={mon['geometry']}")

    if args.session_dir is None:
        args.session_dir = (
            f"tb_session_v8_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    rsk_rpc_url = resolve_rpc_url(args.rsk_rpc)
    pf = preflight(args, rsk_rpc_url)
    wallet = pf["wallet"]
    wallet_address = pf["wallet_address"]
    interior_rpc_url = pf["interior_rpc_url"]

    app = TBLoopV8(
        monitor=mon, duration=args.duration,
        exposure_us=args.exposure_us, gain_db=args.gain,
        activation=args.activation, format_str=args.format,
        session_dir=args.session_dir, cpu_only=args.cpu_only,
        session_mode=args.mode, calibration_dir=args.calibration_dir,
        rsk_rpc_url=rsk_rpc_url, interior_publisher=None,
        network=args.network, wallet_address=wallet_address,
    )

    interior_publisher = None
    if args.interior_anchor:
        # v9: mainnet gate. Probe balance first so the confirmation
        # prompt shows accurate numbers; then prompt; then construct.
        rpc_fallbacks = [u.strip() for u in args.rsk_rpc_fallbacks.split(",")
                         if u.strip()]
        rpc_urls_for_display = [interior_rpc_url] + rpc_fallbacks

        mainnet_confirmed = False
        if args.network == "mainnet":
            # Pre-prompt balance probe (best-effort; the AnchorPublisher
            # does its own strict check later).
            try:
                from rsk_anchor import fetch_balance
                starting_balance = fetch_balance(
                    interior_rpc_url, wallet_address)
            except Exception as e:
                print(f"WARN: pre-prompt balance probe failed: {e!r}; "
                      f"proceeding to prompt with balance=0", file=sys.stderr)
                starting_balance = 0
            mainnet_confirmed = _mainnet_confirm_prompt(
                args, wallet_address, starting_balance, rpc_urls_for_display)
            if not mainnet_confirmed:
                try:
                    shutil.rmtree(args.session_dir)
                except Exception:
                    pass
                print("[mainnet] session aborted before any tx submission.",
                      file=sys.stderr)
                sys.exit(3)

        try:
            interior_publisher = AnchorPublisher(
                wallet=wallet, rpc_url=interior_rpc_url,
                session_dir=args.session_dir,
                anchor_interval_s=args.anchor_interval_s,
                network=args.network,
                final_tx_wait_s=args.final_tx_wait_s,
                rpc_url_fallbacks=rpc_fallbacks,
                mainnet_confirmed=mainnet_confirmed,
            )
            interior_publisher.start()
        except Exception as e:
            print(f"ERROR: AnchorPublisher.start() failed: {e}",
                  file=sys.stderr)
            try:
                shutil.rmtree(args.session_dir)
            except Exception:
                pass
            sys.exit(2)
        app.interior_publisher = interior_publisher

    def _abort_signal(*_):
        print("\n[abort signal] quitting", flush=True)
        app.signal_caught = True
        app.stop_event.set()
        app.capture_stop_event.set()
        app.quit()
        return False

    signal.signal(signal.SIGINT, _abort_signal)
    signal.signal(signal.SIGTERM, _abort_signal)
    GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGINT, _abort_signal)
    GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGTERM, _abort_signal)

    t0 = time.monotonic()
    rc = app.run([])
    session_wall_s = time.monotonic() - t0

    # Re-enable GC and release any async-tightened mitigations.
    try:
        gc.enable()
    except Exception:
        pass
    try:
        app._release_async_tightened_mitigations()
    except Exception:
        pass

    # Stop pools / durability.
    for pool_attr in ("hash_pool", "disk_pool", "png_pool"):
        pool = getattr(app, pool_attr, None)
        if pool is not None:
            try:
                pool.shutdown(wait=True, cancel_futures=False)
            except Exception:
                pass
    if getattr(app, "durability_thread", None) is not None:
        try:
            app.durability_thread.shutdown()
            app.durability_thread.join(timeout=5.0)
        except Exception:
            pass
    if getattr(app, "_cpu_pool", None) is not None:
        try:
            app._cpu_pool.shutdown(wait=True)
        except Exception:
            pass

    chain_thread_alive = False
    if app._chain_thread is not None:
        app.stop_event.set()
        app._chain_thread.join(timeout=30.0)
        chain_thread_alive = app._chain_thread.is_alive()
        if chain_thread_alive:
            print(f"[finalize] WARN chain thread still alive after 30s",
                  flush=True)
    if app._capture_thread is not None:
        app._capture_thread.join(timeout=5.0)

    # Reconcile capture_log with chain_log (rewrite consumed_into_chain/_as_t).
    try:
        if app.chain_log_path.exists() and app.capture_log_path.exists():
            M, _ = reconcile_capture_log(
                args.session_dir, app.chain_log_path, app.capture_log_path
            )
            print(f"[finalize] capture_log reconciled: {M} rows", flush=True)
    except Exception as e:
        print(f"[finalize] WARN capture_log reconcile failed: {e!r}",
              flush=True)

    # Determine outcome.
    n_rows = app.N_chain
    if app.manifest is None:
        outcome = "no-manifest"
    elif getattr(app, "setup_error", None) is not None:
        outcome = "aborted_setup_failed"
    elif app.signal_caught:
        outcome = "abort-signal"
    elif chain_thread_alive:
        outcome = "abort-thread-stuck"
    elif app.chain_error:
        outcome = "abort-chain-exception"
    elif n_rows == 0 or app.last_S_next is None:
        outcome = "abort-no-rows"
    else:
        outcome = "finalize"

    publisher_stop_status = None
    if interior_publisher is not None:
        try:
            publisher_stop_status = interior_publisher.stop()
        except Exception as e:
            print(f"[anchor] WARN stop raised: {e!r}", flush=True)
            publisher_stop_status = STOP_STATUS_STILL_ALIVE
        if publisher_stop_status == STOP_STATUS_STILL_ALIVE:
            print(f"[finalize] WARN publisher still alive; anchor_log_hash NOT "
                  f"computed", flush=True)
            if outcome == "finalize":
                outcome = "aborted_publisher_stuck"

    print(f"chain_log: {n_rows} rows → {app.chain_log_path}")
    print(f"capture_log: {app.N_captures} rows → {app.capture_log_path}")
    if app.chain_step_ms:
        med = statistics.median(app.chain_step_ms)
        p95 = sorted(app.chain_step_ms)[int(0.95 * (len(app.chain_step_ms) - 1))]
        print(f"chain step: median={med:.2f}ms  p95={p95:.2f}ms  "
              f"(N={len(app.chain_step_ms)})")
    if app.cuda_gen_ms:
        med = statistics.median(app.cuda_gen_ms)
        p95 = sorted(app.cuda_gen_ms)[int(0.95 * (len(app.cuda_gen_ms) - 1))]
        print(f"tile gen:   median={med:.2f}ms  p95={p95:.2f}ms  "
              f"backend={app.backend_name}")
    if app.hash_latency_ms:
        med = statistics.median(app.hash_latency_ms)
        p95 = sorted(app.hash_latency_ms)[int(0.95 * (len(app.hash_latency_ms) - 1))]
        print(f"hash lat:   median={med:.2f}ms  p95={p95:.2f}ms")
    if app.session_mode == "async-tightened":
        total_caps = app.N_captures or 0
        slow = app.tightened_slow_iter_count
        rate = (slow / total_caps) if total_caps else 0.0
        print(f"tightened slow-iter count (capture handler > 50 ms): "
              f"{slow}/{total_caps} = {rate:.2%}")
        applied = sorted(set(app.mitigations_applied))
        print(f"tightened mitigations ({len(applied)} unique): {applied}")

    print()
    if outcome == "no-manifest":
        print(f"[abort] no manifest — aborted before artefact build")
    elif outcome == "finalize":
        iso_end = dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="microseconds")
        S_N_hex = app.last_S_next.hex()
        anchor_log_hash_hex = None
        anchor_end = None
        finalize_aborted_by_signal = False
        if interior_publisher is not None:
            if app.signal_caught:
                finalize_aborted_by_signal = True
            else:
                anchor_csv = Path(args.session_dir) / "anchor_txs.csv"
                if anchor_csv.exists():
                    csv_bytes = anchor_csv.read_bytes()
                    anchor_log_hash_hex = blake3(csv_bytes).hexdigest()
                    print(f"[finalize] anchor_log_hash="
                          f"{anchor_log_hash_hex[:32]}…")
                final_root_bytes = compute_final_root(
                    S_N_hex, app.manifest.bundle_hash,
                    app.manifest.manifest_hash_open, anchor_log_hash_hex,
                )
                final_root_hex = final_root_bytes.hex()
                print(f"[finalize] final_root={final_root_hex[:32]}…  "
                      f"(TB:FINAL:v8 || S_N || bundle_hash || "
                      f"manifest_hash_open || anchor_log_hash)")
                if app.signal_caught:
                    finalize_aborted_by_signal = True
                else:
                    try:
                        interior_publisher.fire_final_anchor(
                            final_root_hex, state_index=n_rows,
                            S_N_hex=S_N_hex
                        )
                    except Exception as e:
                        print(f"[anchor] fire_final_anchor raised: {e!r}",
                              flush=True)
                    if app.signal_caught:
                        finalize_aborted_by_signal = True
                    else:
                        fti = interior_publisher.final_tx_info
                        if fti is None:
                            print(f"[anchor] final tx not confirmed — "
                                  f"anchor_end OMITTED", flush=True)
                        else:
                            anchor_end = {
                                "chain": f"rsk-{args.network}",
                                "chain_id": NETWORK_CHAIN_IDS[args.network],
                                "tx_hash": fti["tx_hash"],
                                "payload_S_N_hex": fti["payload_S_N_hex"],
                                "payload_final_root_hex": fti["payload_final_root_hex"],
                                "block_number": fti["block_number"],
                                "block_hash": fti["block_hash"],
                                "block_timestamp_utc": fti["block_timestamp_utc"],
                            }
                        interior_publisher.read_ending_balance()

        manifest_path = Path(args.session_dir) / "manifest.json"
        if finalize_aborted_by_signal:
            app.manifest.mark_aborted(SESSION_STATUS_ABORTED_SIGNAL_DURING_FINALIZE)
            app.manifest.write(manifest_path)
            outcome = "aborted_signal_during_finalize"
        else:
            if app.signal_caught:
                app.manifest.mark_aborted(SESSION_STATUS_ABORTED_SIGNAL_DURING_FINALIZE)
                app.manifest.write(manifest_path)
                outcome = "aborted_signal_during_finalize"
            else:
                capture_log_hash_hex = compute_capture_log_hash(
                    app.capture_log_path
                )
                chain_log_hash_hex = compute_chain_log_hash(
                    app.chain_log_path
                )
                print(f"[finalize] capture_log_hash="
                      f"{capture_log_hash_hex[:32]}…")
                print(f"[finalize] chain_log_hash="
                      f"{chain_log_hash_hex[:32]}…")
                mits = (list(dict.fromkeys(app.mitigations_applied))
                        if app.session_mode
                        == SESSION_MODE_ASYNC_TIGHTENED
                        else None)
                app.manifest.finalize_end(
                    iso_end, n_rows, app.N_captures, S_N_hex,
                    capture_log_hash_hex, chain_log_hash_hex,
                    anchor_end=anchor_end,
                    anchor_log_hash=anchor_log_hash_hex,
                    camera_acquisition_start_wall_ns=app.camera_acquisition_start_wall_ns,
                    camera_acquisition_end_wall_ns=app.camera_acquisition_end_wall_ns,
                    mitigations_applied=mits,
                )
                app.manifest.write(manifest_path)
                try:
                    dfd = os.open(args.session_dir, os.O_RDONLY)
                    try:
                        os.fsync(dfd)
                    finally:
                        os.close(dfd)
                except OSError:
                    pass
                print(f"[finalize] session_id={app.manifest.session_id}  "
                      f"N_chain={n_rows}  N_captures={app.N_captures}  "
                      f"S_N={S_N_hex[:16]}…  "
                      f"manifest_hash_final={app.manifest.manifest_hash_final[:16]}…")
                if anchor_end is not None:
                    print(f"[finalize] anchor_end.tx_hash="
                          f"{anchor_end['tx_hash']}  "
                          f"block={anchor_end['block_number']}")
    else:
        manifest_path = Path(args.session_dir) / "manifest.json"
        outcome_to_status = {
            "abort-signal": SESSION_STATUS_ABORTED,
            "abort-thread-stuck": SESSION_STATUS_ABORTED,
            "abort-chain-exception": SESSION_STATUS_ABORTED,
            "abort-no-rows": SESSION_STATUS_ABORTED,
            "aborted_setup_failed": SESSION_STATUS_ABORTED_SETUP_FAILED,
            "aborted_publisher_stuck": SESSION_STATUS_ABORTED_PUBLISHER_STUCK,
        }
        status = outcome_to_status.get(outcome, SESSION_STATUS_ABORTED)
        if app.manifest is not None:
            app.manifest.mark_aborted(status)
            app.manifest.write(manifest_path)
        print(f"[abort] outcome={outcome}  manifest status={status}")

    # Best-effort preview encoding.
    if app.real_roi is not None:
        try:
            n_png, enc_s = offline_encode_previews(args.session_dir, app.real_roi)
            print(f"offline encode: {n_png} PNGs in {enc_s:.1f}s")
        except Exception as e:
            print(f"[preview] WARN non-fatal: {e!r}", flush=True)

    sys.exit(rc)


if __name__ == "__main__":
    main()

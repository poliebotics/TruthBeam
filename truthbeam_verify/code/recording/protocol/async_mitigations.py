"""Runtime latency mitigations for v8.3 async-tightened mode.

All helpers follow the same contract:

  apply_X(...) -> (success: bool, label: str, detail: str)

`success` is True iff the mitigation was applied in the current
environment. `label` is a stable short string suitable for inclusion
in `manifest.mitigations_applied`. `detail` is a human-readable
description, including why a fallback was chosen when applicable.

None of these helpers abort on failure — the async-tightened loop
must continue running whether or not each individual mitigation
landed. The capability summary (below) records what the loop
actually got and the session report publishes it.

Designed for Linux. Non-Linux hosts will see every probe fail with
`unsupported`.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import gc
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


# --- Probes ---------------------------------------------------------

@dataclass
class Capabilities:
    """What the current process + host can actually do."""
    can_sched_fifo: bool = False
    sched_fifo_max: int = 0           # max FIFO priority allowed
    can_mlockall: bool = False
    mlockall_rlimit_bytes: int = 0
    can_set_affinity: bool = True     # universally available on Linux 3.x+
    governor_per_cpu: dict[int, str] = field(default_factory=dict)
    can_cpu_dma_latency: bool = False
    cpu_count_online: int = 0
    compositor: str = "unknown"       # mutter, gnome-shell, kwin, weston, ...
    session_type: str = "unknown"     # wayland, x11
    sibling_cpu_hotness: float = 0.0  # fraction-of-core aggregate non-us load


def _rlimit_max(name_attr):
    try:
        import resource
        limit = getattr(resource, name_attr)
        soft, hard = resource.getrlimit(limit)
        return soft, hard
    except Exception:
        return (0, 0)


def probe_capabilities() -> Capabilities:
    cap = Capabilities()
    cap.cpu_count_online = os.cpu_count() or 0

    # Session + compositor.
    cap.session_type = os.environ.get("XDG_SESSION_TYPE", "unknown")
    cap.compositor = os.environ.get("XDG_CURRENT_DESKTOP", "unknown")

    # SCHED_FIFO: rlimit-controlled. We can see RLIMIT_RTPRIO (soft).
    # RLIM_INFINITY == -1 ⇒ unlimited, not "zero".
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_RTPRIO)
        cap.sched_fifo_max = int(soft)
        cap.can_sched_fifo = (soft == resource.RLIM_INFINITY
                               or soft > 0
                               or os.geteuid() == 0)
    except Exception:
        cap.can_sched_fifo = (os.geteuid() == 0)

    # mlockall: RLIMIT_MEMLOCK. A non-zero soft limit is enough for a
    # modest session; root can lock anything. `unlimited` presents as
    # RLIM_INFINITY (= -1) in the Python resource module.
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        cap.mlockall_rlimit_bytes = int(soft)
        cap.can_mlockall = (soft == resource.RLIM_INFINITY
                             or soft > 0
                             or os.geteuid() == 0)
    except Exception:
        cap.can_mlockall = (os.geteuid() == 0)

    # CPU governor per cpu (read-only; we don't try to set it).
    base = Path("/sys/devices/system/cpu")
    if base.exists():
        for cpu_dir in sorted(base.glob("cpu[0-9]*")):
            gov = cpu_dir / "cpufreq" / "scaling_governor"
            if gov.exists():
                try:
                    idx = int(cpu_dir.name[3:])
                    cap.governor_per_cpu[idx] = gov.read_text().strip()
                except (OSError, ValueError):
                    pass

    # /dev/cpu_dma_latency is root-owned; detect via open-for-write.
    p = Path("/dev/cpu_dma_latency")
    if p.exists():
        try:
            fd = os.open(p, os.O_WRONLY)
            os.close(fd)
            cap.can_cpu_dma_latency = True
        except OSError:
            cap.can_cpu_dma_latency = False

    # Sibling-process CPU hotness: 2-second top sample.
    try:
        r = subprocess.run(
            ["top", "-b", "-n", "2", "-d", "0.5"],
            capture_output=True, text=True, timeout=3,
        )
        # Crude: find "%Cpu(s):" line, sum us + sy + ni.
        total_usage = 0.0
        samples = 0
        for line in r.stdout.splitlines():
            if "%Cpu(s):" in line:
                # e.g. "%Cpu(s):  3.2 us,  1.1 sy,  0.0 ni, 95.6 id, ..."
                try:
                    us = float(line.split("us")[0].split(":")[-1].strip().rstrip(","))
                    sy = float(line.split("us")[1].split("sy")[0].strip().rstrip(","))
                    total_usage += (us + sy) / 100.0
                    samples += 1
                except (ValueError, IndexError):
                    pass
        if samples:
            cap.sibling_cpu_hotness = total_usage / samples
    except (subprocess.SubprocessError, FileNotFoundError,
            subprocess.TimeoutExpired):
        pass

    return cap


def _fmt_rlimit(v: int) -> str:
    try:
        import resource
        if v == resource.RLIM_INFINITY:
            return "unlimited"
    except Exception:
        pass
    return str(v)


def summarize_capabilities(cap: Capabilities) -> str:
    lines = [
        f"compositor        : {cap.compositor} ({cap.session_type})",
        f"cpu_count_online  : {cap.cpu_count_online}",
        f"sched_fifo        : {'available' if cap.can_sched_fifo else 'DENIED'}"
        f"  (rlimit_rtprio_soft={_fmt_rlimit(cap.sched_fifo_max)})",
        f"mlockall          : {'available' if cap.can_mlockall else 'DENIED'}"
        f"  (rlimit_memlock_soft={_fmt_rlimit(cap.mlockall_rlimit_bytes)})",
        f"cpu_dma_latency   : {'available' if cap.can_cpu_dma_latency else 'DENIED'}",
        f"governor          : {','.join(sorted(set(cap.governor_per_cpu.values()))) or '(none)'}",
        f"sibling_cpu       : {cap.sibling_cpu_hotness*100:.1f}% avg",
    ]
    return "\n".join(lines)


# --- Per-thread helpers (init callbacks for ThreadPoolExecutor) ----

def try_sched_fifo(priority: int) -> tuple[bool, str, str]:
    """Try SCHED_FIFO at the given priority on the current thread.
    Fall back to os.nice(-20) on failure."""
    try:
        param = os.sched_param(int(priority))
        os.sched_setscheduler(0, os.SCHED_FIFO, param)
        return True, f"sched_fifo:prio={priority}", (
            f"SCHED_FIFO at priority {priority} applied")
    except (PermissionError, OSError) as e:
        # Fall back to nice.
        try:
            os.nice(-20 - os.nice(0))  # best-effort, may still fail
            return False, "sched_fifo:nice_fallback", (
                f"SCHED_FIFO denied ({e}); fell back to nice(-20). "
                f"For unprivileged SCHED_FIFO, edit /etc/security/limits.conf"
                f" to grant rtprio.")
        except OSError as e2:
            return False, "sched_fifo:denied", (
                f"SCHED_FIFO denied ({e}); nice fallback also denied"
                f" ({e2}).")


def try_cpu_affinity(cpus) -> tuple[bool, str, str]:
    """Pin the current thread to the given CPU set."""
    try:
        os.sched_setaffinity(0, set(int(c) for c in cpus))
        return True, f"affinity:{sorted(cpus)}", (
            f"CPU affinity set to {sorted(cpus)}")
    except (PermissionError, OSError) as e:
        return False, "affinity:denied", (
            f"CPU affinity denied ({e}); thread will float.")


def thread_tighten(cpus, priority) -> list[tuple[bool, str, str]]:
    """Convenience for ThreadPoolExecutor initializer. Applies
    affinity then SCHED_FIFO on the current thread. Returns the list
    of (success, label, detail) tuples — the caller can collect them
    into the mitigations list."""
    results = []
    results.append(try_cpu_affinity(cpus))
    results.append(try_sched_fifo(priority))
    return results


# --- Session-wide helpers ------------------------------------------

# MCL_CURRENT | MCL_FUTURE for glibc on Linux.
MCL_CURRENT = 0x01
MCL_FUTURE = 0x02


def try_mlockall() -> tuple[bool, str, str, object]:
    """mlockall(MCL_CURRENT|MCL_FUTURE). Returns the libc handle so
    callers can munlockall at session end. On failure, handle is None."""
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6",
                           use_errno=True)
        ret = libc.mlockall(MCL_CURRENT | MCL_FUTURE)
        if ret == 0:
            return True, "mlockall", (
                "mlockall(MCL_CURRENT|MCL_FUTURE) applied; pages "
                "pinned for the duration of the session."
            ), libc
        errno = ctypes.get_errno()
        return False, "mlockall:denied", (
            f"mlockall failed: errno={errno} "
            f"({os.strerror(errno)}). RLIMIT_MEMLOCK or CAP_IPC_LOCK "
            f"required."), None
    except (OSError, AttributeError) as e:
        return False, "mlockall:unsupported", (
            f"mlockall unavailable: {e}"), None


def munlockall(libc) -> None:
    if libc is None:
        return
    try:
        libc.munlockall()
    except Exception:
        pass


def try_cpu_dma_latency_zero() -> tuple[bool, str, str, int | None]:
    """Pin the CPU dma-latency QoS to 0 µs for the session duration.
    Returns the open fd (or None). Close it at session end to release."""
    try:
        fd = os.open("/dev/cpu_dma_latency", os.O_WRONLY)
        os.write(fd, (0).to_bytes(4, "little"))
        return True, "cpu_dma_latency:0us", (
            "Wrote 0 µs to /dev/cpu_dma_latency; C-states effectively "
            "disabled for the duration the fd is open."), fd
    except (PermissionError, OSError) as e:
        return False, "cpu_dma_latency:denied", (
            f"/dev/cpu_dma_latency not writable: {e}. Root or "
            f"CAP_SYS_ADMIN required."), None


def close_cpu_dma_latency(fd) -> None:
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass


def gc_lockdown() -> tuple[bool, str, str]:
    """Collect + freeze + disable GC. Meant to be called once, right
    before the session loop begins."""
    try:
        n0 = len(gc.get_objects())
        for _ in range(3):
            gc.collect()
        gc.freeze()
        gc.disable()
        n1 = len(gc.get_objects())
        n_frozen = gc.get_freeze_count()
        return True, "gc_lockdown", (
            f"gc.collect()×3 + freeze + disable applied; "
            f"objects {n0}→{n1}, frozen={n_frozen}. GC will not run "
            f"during the session.")
    except Exception as e:
        return False, "gc_lockdown:failed", f"gc lockdown failed: {e}"


def gc_release() -> None:
    try:
        gc.unfreeze()
    except Exception:
        pass
    try:
        gc.enable()
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass


def governor_summary(cap: Capabilities) -> tuple[bool, str, str]:
    """Not a mitigation — we don't auto-set the governor. Report
    current state so the operator knows whether to fix it before the
    session."""
    if not cap.governor_per_cpu:
        return False, "governor:unknown", "no governor info available"
    vals = set(cap.governor_per_cpu.values())
    if vals == {"performance"}:
        return True, "governor:performance", (
            "cpufreq governor is 'performance' on all cores")
    if "performance" in vals:
        return False, f"governor:mixed:{sorted(vals)}", (
            f"cpufreq governors are mixed: {sorted(vals)}. "
            f"Run: sudo cpupower frequency-set -g performance")
    return False, f"governor:{sorted(vals)[0]}", (
        f"cpufreq governor is {sorted(vals)[0]}; expected "
        f"'performance' for low-jitter. Run: "
        f"sudo cpupower frequency-set -g performance")


def service_warnings(cap: Capabilities) -> list[str]:
    """Return short warning strings for background work that could
    disturb the session. Best-effort; silent on missing commands."""
    warns = []
    if cap.sibling_cpu_hotness > 0.20:
        warns.append(
            f"sibling CPU usage averaging "
            f"{cap.sibling_cpu_hotness*100:.1f}% across cores; "
            f"consider stopping background work before the session")
    # Probe known troublemakers.
    for unit in ("apt-daily.timer", "apt-daily-upgrade.timer",
                 "packagekit.service", "snapd.service",
                 "updatedb.timer", "man-db.timer"):
        try:
            r = subprocess.run(
                ["systemctl", "is-active", unit],
                capture_output=True, text=True, timeout=1,
            )
            if r.stdout.strip() == "active":
                warns.append(
                    f"{unit} is active; may trigger mid-session "
                    f"(stop/mask before collection if concerned)")
        except (subprocess.SubprocessError, FileNotFoundError,
                subprocess.TimeoutExpired):
            pass
    return warns

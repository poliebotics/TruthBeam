"""host_info.py — host-identity probe for inclusion in the bundle.

Scope: collecting hostname / OS / kernel / CPU / Python-version info
that the VerificationBundle commits to. One function, one purpose.

Adapt this module for:
- Additional host fields (e.g. PCI device list, board-level info, CI
  fingerprints) — add to the returned dict and update the schema in
  session_schema.py to accept the new keys
- Redacting specific fields for public-artefact releases

Do NOT put here:
- Camera / projector / Aravis probes — camera.py / projector.py
- Anything that changes `bundle_hash` semantics without a matching
  update to the schema validator
"""
from __future__ import annotations

import os
import socket
import sys


def probe_host_config() -> dict:
    """Return a dict suitable for `VerificationBundle.host_config`.
    All fields fall back to "unknown" on probe failure — no raising."""
    os_release = "unknown"
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    os_release = line.split("=", 1)[1].strip().strip('"')
                    break
    except OSError:
        pass
    if os_release == "unknown":
        try:
            os_release = " ".join(os.uname()[:3])
        except OSError:
            pass
    cpu_model = "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {
        "hostname": socket.gethostname() or "unknown",
        "os_release": os_release,
        "kernel_version": (os.uname().release if hasattr(os, "uname") else "unknown"),
        "python_version": sys.version.split()[0],
        "cpu_model": cpu_model,
    }

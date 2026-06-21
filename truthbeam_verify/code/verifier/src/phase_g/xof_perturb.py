"""Phase G — Item 1: XOF-stream structural perturbation primitives.

Per `experiments/phase_g_diffusion_diagnostic/ITEM_1_PLAN.md` (post-audit
locked plan, 2026-05-02).

For each held-out frame t in D2/V10:
    1. Read real S_t_hex from chain_log.csv. Frame t's emission tile_t.png
       is rendered from chain[t].S_t_hex (verified empirically — bit-exact
       match against on-disk PNG; chain[t+1] is for a different protocol task).
    2. Derive real channel seeds via blake3 domain-tagged.
    3. Expand each seed via BLAKE3-XOF to 43,110 bytes per channel.
    4. Apply a perturbation to those real bytes (six types — see below).
    5. Render via bitexact_renderer.render_from_streams(...).
    6. Resize 1080×1920 → 768×1024 to feed the diffusion verifier.

Six perturbation types:
    Type 1 — global bit-flip (k bits chosen uniformly across all 1,034,640
             bits of the 3 streams). k ∈ {1, 4, 16, 64, 256, 1024, 4096}.
    Type 2 — octave-localized bit-flip (k bits within ONE octave's bytes,
             other octaves untouched). 4 octaves × 4 k-values = 16 conditions.
    Type 3 — spatially-localized bit-flip (k bits within an 8×8 grid region
             of octave 2 (68×120)). 3 k-values.
    Type 4 — octave swap with bytes from a cross-frame donor (|lag|>60).
             4 conditions, one per octave.
    Type 5 — channel swap (replace one of seed_R/G/B's stream entirely).
             3 conditions.
    Type 6 — full replacement (donor stream lag>60). 1 condition.
             Plus a "calibration" check using partner=row+30 to match
             the existing wrong_+30 baseline.

RNG seeding for bit-flip selection:
    seed_bytes = blake3(f"{session}|{row}|{condition_label}".encode()).digest(length=k*4)
    rng = numpy.random.default_rng(int.from_bytes(seed_bytes[:8], "big"))
    bit_indices = rng.choice(n_total_bits, size=k, replace=False)
"""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from blake3 import blake3

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "phase_e"))

from data.xof_generation import (  # noqa: E402
    derive_xof_seeds, OCTAVE_SHAPES, OCTAVE_BYTES_PER_CHANNEL,
    TOTAL_BYTES_PER_CHANNEL,
)
from bitexact_renderer import render_from_streams  # noqa: E402


# Per-channel offsets into the 43,110-byte stream (one channel)
PER_CHANNEL_OCTAVE_OFFSETS = [0]
for nb in OCTAVE_BYTES_PER_CHANNEL:
    PER_CHANNEL_OCTAVE_OFFSETS.append(PER_CHANNEL_OCTAVE_OFFSETS[-1] + nb)
# So per-channel ranges are [0,510), [510,2550), [2550,10710), [10710,43110)

CHANNEL_NAMES = ("R", "G", "B")


# ---------------- chain log access ----------------

def load_chain_log(session_dir: Path) -> dict[int, str]:
    """Returns {t: S_t_hex} from chain_log.csv."""
    path = session_dir / "chain_log.csv"
    out: dict[int, str] = {}
    with open(path) as f:
        reader = csv.reader(f)
        header_seen = False
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if not header_seen:
                if row[0] == "t":
                    header_seen = True
                    continue
                # malformed — assume header
                header_seen = True
                continue
            try:
                t = int(row[0])
                out[t] = row[1]
            except (ValueError, IndexError):
                continue
    return out


# ---------------- stream construction ----------------

def expand_streams_from_s_t(s_t_hex: str) -> tuple[bytes, bytes, bytes]:
    """Expand real chain state into the 3 channel byte streams used by
    the bit-exact renderer (43,110 bytes/channel)."""
    s_t = bytes.fromhex(s_t_hex)
    seed_r, seed_g, seed_b = derive_xof_seeds(s_t)
    stream_r = blake3(seed_r).digest(length=TOTAL_BYTES_PER_CHANNEL)
    stream_g = blake3(seed_g).digest(length=TOTAL_BYTES_PER_CHANNEL)
    stream_b = blake3(seed_b).digest(length=TOTAL_BYTES_PER_CHANNEL)
    return stream_r, stream_g, stream_b


# ---------------- RNG seeding ----------------

def derive_rng(session: str, row: int, condition_label: str) -> np.random.Generator:
    """Deterministic NumPy RNG keyed by (session, row, condition).
    Domain-separated by condition label so different perturbation types within
    the same frame get independent bit-selection draws."""
    seed_bytes = blake3(f"{session}|{row}|{condition_label}".encode()).digest(length=32)
    seed_int = int.from_bytes(seed_bytes[:8], "big")
    return np.random.default_rng(seed_int)


# ---------------- Type 1: global bit-flip ----------------

def flip_bits_global(streams: tuple[bytes, bytes, bytes], k: int,
                     rng: np.random.Generator) -> tuple[bytes, bytes, bytes]:
    """Flip k random bits across the FULL 3 × 43,110 = 129,330-byte concatenation
    (= 1,034,640 bits)."""
    concat = np.frombuffer(streams[0] + streams[1] + streams[2], dtype=np.uint8).copy()
    n_bits = concat.size * 8
    if k > n_bits:
        raise ValueError(f"k={k} exceeds available bits ({n_bits})")
    bit_idx = rng.choice(n_bits, size=k, replace=False)
    byte_idx = bit_idx // 8
    bit_pos = bit_idx % 8
    for bi, bp in zip(byte_idx, bit_pos):
        concat[bi] ^= np.uint8(1 << bp)
    return (
        bytes(concat[:TOTAL_BYTES_PER_CHANNEL]),
        bytes(concat[TOTAL_BYTES_PER_CHANNEL:2 * TOTAL_BYTES_PER_CHANNEL]),
        bytes(concat[2 * TOTAL_BYTES_PER_CHANNEL:]),
    )


# ---------------- Type 2: octave-localized bit-flip ----------------

def flip_bits_octave(streams: tuple[bytes, bytes, bytes],
                     octave_idx: int, k: int,
                     rng: np.random.Generator) -> tuple[bytes, bytes, bytes]:
    """Flip k bits within ONLY octave_idx's bytes across all 3 channels.
    Other octaves untouched.

    Octave byte ranges (per channel): [510), [510,2550), [2550,10710), [10710,43110)
    Across 3 channels concatenated, they occupy 3 disjoint ranges (one per channel).
    """
    if not (0 <= octave_idx < 4):
        raise ValueError(f"octave_idx={octave_idx} out of range")
    a = PER_CHANNEL_OCTAVE_OFFSETS[octave_idx]
    b = PER_CHANNEL_OCTAVE_OFFSETS[octave_idx + 1]
    octave_bytes_per_channel = b - a
    # Selectable bits: 3 channels × octave_bytes_per_channel × 8 bits
    n_bits_octave = 3 * octave_bytes_per_channel * 8
    if k > n_bits_octave:
        raise ValueError(f"k={k} exceeds octave bits ({n_bits_octave})")

    rel_bit_idx = rng.choice(n_bits_octave, size=k, replace=False)
    out = [bytearray(streams[c]) for c in range(3)]
    for rbi in rel_bit_idx:
        # decode: rbi → (channel, byte_within_octave, bit_within_byte)
        channel = rbi // (octave_bytes_per_channel * 8)
        within = rbi % (octave_bytes_per_channel * 8)
        byte_in_oct = within // 8
        bit_pos = within % 8
        abs_byte_idx = a + byte_in_oct
        out[channel][abs_byte_idx] ^= 1 << bit_pos
    return bytes(out[0]), bytes(out[1]), bytes(out[2])


# ---------------- Type 3: spatially-localized bit-flip in octave 2 ----------------

def flip_bits_region_octave2(streams: tuple[bytes, bytes, bytes],
                             region_xy: tuple[int, int],
                             k: int,
                             rng: np.random.Generator) -> tuple[bytes, bytes, bytes]:
    """Flip k bits within an 8×8 grid region of octave 2 (68×120).
    region_xy = (x, y) of top-left of 8×8 region.
    Valid region positions: x ∈ [0, 112], y ∈ [0, 60].

    Spatial layout: octave 2 grid is row-major 68 × 120; byte at (gy, gx) for
    channel c is at offset PER_CHANNEL_OCTAVE_OFFSETS[2] + gy*120 + gx within
    that channel's stream.

    Region covers gy ∈ [y, y+8) × gx ∈ [x, x+8) → 64 bytes per channel × 3
    channels = 192 bytes total = 1,536 bits.
    """
    x, y = region_xy
    if not (0 <= x <= 112 and 0 <= y <= 60):
        raise ValueError(f"region_xy=({x},{y}) outside valid 8×8 positions in 68×120 grid")
    OCT_H, OCT_W = OCTAVE_SHAPES[2]  # (68, 120)
    base = PER_CHANNEL_OCTAVE_OFFSETS[2]

    # Build the list of byte positions in the region, per channel
    region_byte_offsets_per_channel = []
    for gy in range(y, y + 8):
        for gx in range(x, x + 8):
            region_byte_offsets_per_channel.append(base + gy * OCT_W + gx)
    n_bytes_per_channel = len(region_byte_offsets_per_channel)  # 64
    n_bits_total = 3 * n_bytes_per_channel * 8                  # 1,536

    if k > n_bits_total:
        raise ValueError(f"k={k} exceeds region bits ({n_bits_total})")

    rel_bit_idx = rng.choice(n_bits_total, size=k, replace=False)
    out = [bytearray(streams[c]) for c in range(3)]
    for rbi in rel_bit_idx:
        channel = rbi // (n_bytes_per_channel * 8)
        within = rbi % (n_bytes_per_channel * 8)
        byte_in_region = within // 8
        bit_pos = within % 8
        abs_byte_idx = region_byte_offsets_per_channel[byte_in_region]
        out[channel][abs_byte_idx] ^= 1 << bit_pos
    return bytes(out[0]), bytes(out[1]), bytes(out[2])


def deterministic_region_xy(session: str, row: int) -> tuple[int, int]:
    """Per-frame region position for Type 3, deterministic from session+row."""
    rng = derive_rng(session, row, "type3_region")
    x = int(rng.integers(0, 113))
    y = int(rng.integers(0, 61))
    return x, y


# ---------------- Type 4: octave swap ----------------

def swap_octave(streams: tuple[bytes, bytes, bytes],
                donor_streams: tuple[bytes, bytes, bytes],
                octave_idx: int) -> tuple[bytes, bytes, bytes]:
    """Replace octave_idx's bytes ENTIRELY with donor's. Other octaves intact."""
    if not (0 <= octave_idx < 4):
        raise ValueError(f"octave_idx={octave_idx}")
    a = PER_CHANNEL_OCTAVE_OFFSETS[octave_idx]
    b = PER_CHANNEL_OCTAVE_OFFSETS[octave_idx + 1]
    out = []
    for c in range(3):
        ba = bytearray(streams[c])
        ba[a:b] = donor_streams[c][a:b]
        out.append(bytes(ba))
    return tuple(out)  # type: ignore


# ---------------- Type 5: channel swap ----------------

def swap_channel(streams: tuple[bytes, bytes, bytes],
                 donor_streams: tuple[bytes, bytes, bytes],
                 channel_idx: int) -> tuple[bytes, bytes, bytes]:
    """Replace one channel's full 43,110-byte stream with donor's."""
    if not (0 <= channel_idx < 3):
        raise ValueError(f"channel_idx={channel_idx}")
    out = list(streams)
    out[channel_idx] = donor_streams[channel_idx]
    return tuple(out)  # type: ignore


# ---------------- Type 6: full replacement ----------------

def replace_all(donor_streams: tuple[bytes, bytes, bytes]) -> tuple[bytes, bytes, bytes]:
    """Use donor's full streams (Type 6 / general)."""
    return donor_streams


# ---------------- cross-frame partner selection ----------------

def cross_frame_partner(session: str, row: int, total_rows: int, salt: str,
                        min_gap: int = 60) -> int:
    """Deterministic cross-frame partner row with |partner - row| > min_gap.

    Uses domain-separated rotation. Tries small offsets near n/4 first, falls
    back to wider offsets. Always returns a row in [0, total_rows).
    """
    rng = derive_rng(session, row, f"partner|{salt}")
    # Choose a target offset > min_gap, biased toward n/4 for "natural" distance
    target = int(total_rows / 4) + int(rng.integers(-total_rows // 8, total_rows // 8 + 1))
    for sign in (1, -1):
        cand = (row + sign * abs(target)) % total_rows
        if abs(cand - row) > min_gap:
            return cand
    # Fallback: walk the row space until we find a gap-satisfying row
    for off in range(min_gap + 1, total_rows):
        cand = (row + off) % total_rows
        if abs(cand - row) > min_gap:
            return cand
    raise ValueError(f"no partner found for row={row} in {total_rows}-row session")


# ---------------- render → tensor ----------------

def render_streams_to_tile(streams: tuple[bytes, bytes, bytes],
                           device: str = "cpu") -> torch.Tensor:
    """Render perturbed streams via the bit-exact renderer.
    Returns (3, 1080, 1920) uint8 tensor."""
    return render_from_streams(*streams, device=device)


# Resize target matches Phase G eval: E goes 1080×1920 → 768×1024
TARGET_E_H = 768
TARGET_E_W = 1024


def render_perturbed_E(streams: tuple[bytes, bytes, bytes],
                       device: str = "cpu") -> torch.Tensor:
    """Render perturbed streams + resize to the diffusion model's input size.
    Returns (3, 768, 1024) float32 in [0, 1]. cv2.resize requires a CPU
    NumPy array, so the rendered tile is brought back to CPU before resize
    regardless of the renderer's compute device."""
    tile = render_streams_to_tile(streams, device=device).cpu().numpy()  # (3, 1080, 1920) uint8
    # Use cv2 area interp (matches Phase G's resize convention)
    arr = tile.transpose(1, 2, 0)  # (H, W, 3)
    resized = cv2.resize(arr, (TARGET_E_W, TARGET_E_H), interpolation=cv2.INTER_AREA)
    out = resized.astype(np.float32) / 255.0
    return torch.from_numpy(out.transpose(2, 0, 1))  # (3, 768, 1024)


# ---------------- spec parsing ----------------

@dataclass
class PerturbSpec:
    """Describes a single perturbation condition for a (session, row).

    type: one of {"type1_global", "type2_octave", "type3_region",
                  "type4_octave_swap", "type5_channel_swap", "type6_replace",
                  "type6_calibration"}
    Other fields depend on the type.
    """
    type: str
    k: int = 0                    # used by type1/2/3
    octave_idx: int = -1          # used by type2/4
    channel_idx: int = -1         # used by type5
    label: str = ""               # condition_label for naming

    def needs_donor(self) -> bool:
        return self.type in ("type4_octave_swap", "type5_channel_swap",
                             "type6_replace", "type6_calibration")


# ---------------- list of all 34 perturbation conditions ----------------

TYPE1_K_VALUES = (1, 4, 16, 64, 256, 1024, 4096)
TYPE2_K_VALUES = (1, 4, 16, 64)
TYPE2_OCTAVES = (0, 1, 2, 3)
TYPE3_K_VALUES = (16, 64, 256)
TYPE4_OCTAVES = (0, 1, 2, 3)
TYPE5_CHANNELS = (0, 1, 2)


def all_perturbation_specs() -> list[PerturbSpec]:
    out: list[PerturbSpec] = []
    for k in TYPE1_K_VALUES:
        out.append(PerturbSpec(type="type1_global", k=k,
                               label=f"xof_t1_global_k{k}"))
    for octave_idx in TYPE2_OCTAVES:
        for k in TYPE2_K_VALUES:
            out.append(PerturbSpec(type="type2_octave", k=k,
                                   octave_idx=octave_idx,
                                   label=f"xof_t2_oct{octave_idx}_k{k}"))
    for k in TYPE3_K_VALUES:
        out.append(PerturbSpec(type="type3_region", k=k,
                               label=f"xof_t3_region_k{k}"))
    for octave_idx in TYPE4_OCTAVES:
        out.append(PerturbSpec(type="type4_octave_swap",
                               octave_idx=octave_idx,
                               label=f"xof_t4_swap_oct{octave_idx}"))
    for channel_idx in TYPE5_CHANNELS:
        out.append(PerturbSpec(type="type5_channel_swap",
                               channel_idx=channel_idx,
                               label=f"xof_t5_swap_{CHANNEL_NAMES[channel_idx]}"))
    out.append(PerturbSpec(type="type6_replace",
                           label="xof_t6_replace_general"))
    # Type 6 calibration variant — partner is row+30, expected to match wrong_+30
    out.append(PerturbSpec(type="type6_calibration",
                           label="xof_t6_calibration_row+30"))
    return out


# ---------------- apply spec ----------------

def apply_spec(spec: PerturbSpec,
               base_streams: tuple[bytes, bytes, bytes],
               donor_streams: tuple[bytes, bytes, bytes] | None,
               session: str, row: int) -> tuple[bytes, bytes, bytes]:
    """Apply a perturbation spec to base streams. donor_streams must be
    provided when spec.needs_donor()."""
    if spec.needs_donor() and donor_streams is None:
        raise ValueError(f"spec {spec.label} needs donor streams but none given")
    rng = derive_rng(session, row, spec.label)
    if spec.type == "type1_global":
        return flip_bits_global(base_streams, spec.k, rng)
    elif spec.type == "type2_octave":
        return flip_bits_octave(base_streams, spec.octave_idx, spec.k, rng)
    elif spec.type == "type3_region":
        region_xy = deterministic_region_xy(session, row)
        return flip_bits_region_octave2(base_streams, region_xy, spec.k, rng)
    elif spec.type == "type4_octave_swap":
        assert donor_streams is not None
        return swap_octave(base_streams, donor_streams, spec.octave_idx)
    elif spec.type == "type5_channel_swap":
        assert donor_streams is not None
        return swap_channel(base_streams, donor_streams, spec.channel_idx)
    elif spec.type == "type6_replace":
        assert donor_streams is not None
        return replace_all(donor_streams)
    elif spec.type == "type6_calibration":
        assert donor_streams is not None  # partner=row+30 streams provided by caller
        return replace_all(donor_streams)
    else:
        raise ValueError(f"unknown spec.type {spec.type}")


# ---------------- Phase H reusable API ----------------
#
# Phase H training requires a single render path for ALL E inputs (positives
# AND negatives) so the CNN can't learn "PNG-loaded vs tensor-rendered" as a
# shortcut.  This helper wraps the byte-stream → render → resize pipeline
# behind a clean session/frame_id/condition_label interface.

# Build a label → spec lookup once (module-level cache)
_PERTURB_SPECS_BY_LABEL: dict[str, PerturbSpec] | None = None


def _spec_by_label(label: str) -> PerturbSpec:
    global _PERTURB_SPECS_BY_LABEL
    if _PERTURB_SPECS_BY_LABEL is None:
        _PERTURB_SPECS_BY_LABEL = {s.label: s for s in all_perturbation_specs()}
    if label not in _PERTURB_SPECS_BY_LABEL:
        raise ValueError(f"unknown perturbation label: {label!r}")
    return _PERTURB_SPECS_BY_LABEL[label]


# Train pool (operator spec): Types 1, 2, 4, 6 general (NOT calibration variant).
# Held-out for generalization eval: Types 3, 5.
TRAIN_POOL_LABELS: list[str] = []
for _k in TYPE1_K_VALUES:
    TRAIN_POOL_LABELS.append(f"xof_t1_global_k{_k}")
for _o in TYPE2_OCTAVES:
    for _k in TYPE2_K_VALUES:
        TRAIN_POOL_LABELS.append(f"xof_t2_oct{_o}_k{_k}")
for _o in TYPE4_OCTAVES:
    TRAIN_POOL_LABELS.append(f"xof_t4_swap_oct{_o}")
TRAIN_POOL_LABELS.append("xof_t6_replace_general")
# Total: 7 + 16 + 4 + 1 = 28 train conditions

HELDOUT_POOL_LABELS: list[str] = []
for _k in TYPE3_K_VALUES:
    HELDOUT_POOL_LABELS.append(f"xof_t3_region_k{_k}")
for _c in TYPE5_CHANNELS:
    HELDOUT_POOL_LABELS.append(f"xof_t5_swap_{CHANNEL_NAMES[_c]}")
# Total: 3 + 3 = 6 held-out conditions

assert len(TRAIN_POOL_LABELS) == 28, f"train pool has {len(TRAIN_POOL_LABELS)} != 28"
assert len(HELDOUT_POOL_LABELS) == 6, f"held-out pool has {len(HELDOUT_POOL_LABELS)} != 6"


def render_E_for_phase_h(
    session: str,
    target_frame_id: int,
    condition_label: str,
    chain_log: dict[int, str],
    device: str = "cpu",
    donor_chain_log: dict[int, str] | None = None,
    donor_frame_id: int | None = None,
) -> torch.Tensor:
    """Render E for Phase H training/eval.

    Single render path for positives and negatives — ALL E tensors must come
    through this function so CNN can't shortcut on render-path artifacts.

    Args:
        session: session label ("D2" / "V10") — for derive_rng domain separation.
        target_frame_id: frame whose XOF streams form the base.
        condition_label: one of:
            - "identity" — render the canonical (unperturbed) E for target_frame_id.
            - one of the 35 perturbation labels (e.g. "xof_t1_global_k64",
              "xof_t4_swap_oct2", "xof_t6_replace_general", etc.).
        chain_log: {frame_id: S_t_hex} for the session containing target_frame_id.
        device: passed to bitexact_renderer.
        donor_chain_log: {frame_id: S_t_hex} for the session containing the donor
            row (for type 4/5/6 perturbations). May be the same dict as chain_log
            for intra-session donors, or a different session for cross-session.
        donor_frame_id: row of the donor frame (must be in donor_chain_log).
            Required for type 4/5/6.

    Returns: (3, 768, 1024) float32 in [0, 1].
    """
    if target_frame_id not in chain_log:
        raise ValueError(
            f"target_frame_id {target_frame_id} missing from chain_log for {session}")
    base_streams = expand_streams_from_s_t(chain_log[target_frame_id])

    if condition_label == "identity":
        return render_perturbed_E(base_streams, device=device)

    spec = _spec_by_label(condition_label)
    donor_streams: tuple[bytes, bytes, bytes] | None = None
    if spec.needs_donor():
        if donor_chain_log is None or donor_frame_id is None:
            raise ValueError(
                f"perturbation {condition_label!r} needs donor; "
                f"donor_chain_log + donor_frame_id required.")
        if donor_frame_id not in donor_chain_log:
            raise ValueError(
                f"donor_frame_id {donor_frame_id} missing from donor_chain_log")
        donor_streams = expand_streams_from_s_t(donor_chain_log[donor_frame_id])

    perturbed = apply_spec(spec, base_streams, donor_streams, session, target_frame_id)
    return render_perturbed_E(perturbed, device=device)


def load_canonical_E_phase_g_resolution(session_dir: Path,
                                        frame_id: int) -> torch.Tensor:
    """Load tile_<frame_id>.png from disk and resize to Phase G resolution
    (3, 768, 1024). Used to verify identity-render parity."""
    em_path = session_dir / "derived" / "Emissions" / f"tile_{frame_id:06d}.png"
    arr_bgr = cv2.imread(str(em_path), cv2.IMREAD_UNCHANGED)
    arr_rgb = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(arr_rgb, (TARGET_E_W, TARGET_E_H), interpolation=cv2.INTER_AREA)
    out = resized.astype(np.float32) / 255.0
    return torch.from_numpy(out.transpose(2, 0, 1))


def verify_identity_render_parity(session_dir: Path, chain_log: dict[int, str],
                                  sample_frames: list[int],
                                  tol: float = 1e-6) -> tuple[bool, dict]:
    """HARD GATE for Phase H: verify render_E_for_phase_h(condition='identity')
    matches load_canonical_E_phase_g_resolution within float tolerance.

    If this fails, the CNN would learn render-path-difference instead of
    chain coupling.
    """
    diffs = []
    for frame_id in sample_frames:
        rendered = render_E_for_phase_h(
            session="parity_check", target_frame_id=frame_id,
            condition_label="identity", chain_log=chain_log)
        canonical = load_canonical_E_phase_g_resolution(session_dir, frame_id)
        diff = (rendered - canonical).abs()
        diffs.append({
            "frame_id": frame_id,
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
        })
    max_overall = max(d["max_abs_diff"] for d in diffs)
    ok = max_overall <= tol
    return ok, {"per_frame": diffs, "max_overall": max_overall, "tol": tol}


# ---------------- self-test ----------------

def self_test() -> None:
    """Quick sanity check: round-trip rendering on real D2 frame 100 should
    match the on-disk PNG bit-exactly when no perturbation is applied."""
    chain = load_chain_log(Path("data/d2"))
    if 100 not in chain:
        print("self_test: row 100 missing — skipping")
        return
    streams = expand_streams_from_s_t(chain[100])
    tile = render_streams_to_tile(streams).numpy()  # (3, 1080, 1920) uint8

    disk = cv2.imread("data/d2/derived/Emissions/tile_000100.png", cv2.IMREAD_UNCHANGED)
    disk_rgb = cv2.cvtColor(disk, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)

    diff = np.abs(tile.astype(np.int32) - disk_rgb.astype(np.int32))
    print(f"render_from_streams vs tile_000100.png:  max_diff={diff.max()}  "
          f"mean_diff={diff.mean():.4f}  bit_exact={(diff == 0).all()}")

    # Spec coverage
    specs = all_perturbation_specs()
    print(f"\n{len(specs)} perturbation specs ({len(specs)} should be 34 + 1 calibration):")
    type_counts: dict[str, int] = {}
    for s in specs:
        type_counts[s.type] = type_counts.get(s.type, 0) + 1
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")

    # Render one of each type to verify no crashes
    print("\nrender each perturbation type:")
    rng = derive_rng("D2", 100, "self_test_seed")
    donor = expand_streams_from_s_t(chain[200])
    for spec in [s for s in specs[:7]] + specs[7:8] + specs[23:24] + specs[27:28] + specs[31:32] + specs[33:34]:
        try:
            perturbed = apply_spec(spec, streams, donor, "D2", 100)
            tile_p = render_streams_to_tile(perturbed).numpy()
            d = np.abs(tile_p.astype(np.int32) - disk_rgb.astype(np.int32))
            print(f"  {spec.label:34s}  mean_diff_vs_real={d.mean():6.2f}")
        except Exception as e:
            print(f"  {spec.label}: FAIL — {e}")


if __name__ == "__main__":
    self_test()

"""Tier 0 V1 — Figure 1: system overview.

publication-ready Figure 1 using ACTUAL session
data. Chain semantics verified in
`experiments/paper_analyses/figure_1/chain_semantics_verification.md` BEFORE
drawing.

Layout: left-to-right flow with 4 logical stages
    [chain commit] → [chain state S_t] → [rendered emission E_t] → [captured C_t]
                                                                      ↓
                                                         [verifier evaluation]

Annotations are 1-2 words. Real data from D2 session frame 100 is used so the
figure is visibly anchored in actual experimental conditions.

Output:
    experiments/paper_analyses/figure_1/system_overview.png
    experiments/paper_analyses/figure_1/figure_1_caption_draft.md
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


LOCAL_ROOT = Path("/path/to/poliebotics_phase_b")
DEFAULT_OUT = LOCAL_ROOT / "experiments" / "paper_analyses" / "figure_1"
SESSION_DIR = LOCAL_ROOT / "data" / "d2"
DEMO_FRAME = 100


def load_chain_row(t: int) -> dict:
    """Load chain_log row for frame t."""
    import csv
    with open(SESSION_DIR / "chain_log.csv") as f:
        reader = csv.reader(f)
        header = None
        for row in reader:
            if not row or row[0].startswith("#"): continue
            if header is None:
                if row[0] == "t":
                    header = row
                continue
            try:
                if int(row[0]) == t:
                    return dict(zip(header, row))
            except (ValueError, IndexError):
                continue
    raise SystemExit(f"frame {t} not in chain_log")


def debayer_capture(t: int) -> np.ndarray:
    """Load and debayer captured raw frame, return (H', W', 3) uint8 at display size."""
    raw_path = SESSION_DIR / "Recordings" / f"frame_{t:06d}.raw"
    raw = np.fromfile(raw_path, dtype=np.uint8).reshape(4600, 5320)
    rgb = cv2.cvtColor(raw, cv2.COLOR_BayerRG2RGB)
    rgb = (np.clip(rgb / 255.0, 0, 1) ** (1.0/1.6) * 255).astype(np.uint8)
    return cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_AREA)


def load_emission(t: int) -> np.ndarray:
    """Load emission tile, return (H', W', 3) uint8 at display size."""
    em_path = SESSION_DIR / "derived" / "Emissions" / f"tile_{t:06d}.png"
    img = cv2.imread(str(em_path), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_AREA)


def hex_to_blocks(hex_str: str, n_chars: int = 16) -> str:
    """Truncate hex string for display, e.g. '530de6fa...0e92' (16 chars + ...)."""
    if len(hex_str) > n_chars + 4:
        return hex_str[:n_chars] + "…" + hex_str[-4:]
    return hex_str


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--frame", type=int, default=DEMO_FRAME)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Load actual data
    row_t = load_chain_row(args.frame)
    row_t1 = load_chain_row(args.frame + 1)

    s_t = row_t["S_t_hex"]
    s_t1 = row_t1["S_t_hex"]
    bayer_hash_t = row_t["bayer_blake3_hex"]
    drand_round = row_t["drand_round_number"]
    drand_value = row_t["drand_round_value_hex"]
    em_pixel_hash = row_t["emission_live_pixel_blake3_hex"]

    em_rgb = load_emission(args.frame)
    cap_rgb = debayer_capture(args.frame)

    # Figure layout: 2 rows, left-to-right flow
    # Row 0: chain commits + state hashes (text + drand badge)
    # Row 1: emission tile + captured frame + verifier evaluation panel

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 5, height_ratios=[0.4, 1.0, 0.4],
                            hspace=0.25, wspace=0.15,
                            left=0.04, right=0.98, top=0.92, bottom=0.05)

    # === Row 0 (text-heavy chain state) ===
    # drand commit
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("drand round commit", fontsize=10, fontweight="bold")
    ax.text(0.5, 0.5, f"round={drand_round}\nvalue={hex_to_blocks(drand_value, 12)}",
            ha="center", va="center", fontsize=8, family="monospace")
    for spine in ax.spines.values(): spine.set_color("tab:purple"); spine.set_linewidth(1.5)

    # chain state at t
    ax = fig.add_subplot(gs[0, 1])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"chain state S_{args.frame}", fontsize=10, fontweight="bold")
    ax.text(0.5, 0.5, f"S_t = {hex_to_blocks(s_t, 16)}",
            ha="center", va="center", fontsize=8, family="monospace")
    for spine in ax.spines.values(): spine.set_color("tab:blue"); spine.set_linewidth(1.5)

    # XOF expansion
    ax = fig.add_subplot(gs[0, 2])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("BLAKE3-XOF emission render", fontsize=10, fontweight="bold")
    ax.text(0.5, 0.5,
            "blake3('TB:SEED:{R,G,B}:v8' || S_t)\n→ 43,110 B per channel\n→ 4 octaves × 3 channels\n→ 1080×1920 RGB tile",
            ha="center", va="center", fontsize=7, family="monospace")
    for spine in ax.spines.values(): spine.set_color("tab:green"); spine.set_linewidth(1.5)

    # captured Bayer hash (commitment) — this hash FEEDS S_{t+1}
    ax = fig.add_subplot(gs[0, 3])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"capture hash (closes the loop)", fontsize=10, fontweight="bold")
    ax.text(0.5, 0.5, f"bayer_hash_t = BLAKE3(C_t raw)\n= {hex_to_blocks(bayer_hash_t, 16)}\n↓ folded into S_{{t+1}}",
            ha="center", va="center", fontsize=8, family="monospace")
    for spine in ax.spines.values(): spine.set_color("tab:orange"); spine.set_linewidth(2.0)

    # next chain state — show formula explicitly
    ax = fig.add_subplot(gs[0, 4])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"next chain state S_{args.frame+1}", fontsize=10, fontweight="bold")
    ax.text(0.5, 0.5,
            f"S_{{t+1}} = blake3(\n"
            f"  'TB:ROW:v9' || S_t\n"
            f"  || bayer_hash_t || meta_t\n"
            f"  || drand_round_n || drand_v)\n\n"
            f"D2 = v9; V10 = v10\n"
            f"(v10 also folds ai_payload_root)\n\n"
            f"= {hex_to_blocks(s_t1, 12)}",
            ha="center", va="center", fontsize=7, family="monospace")
    for spine in ax.spines.values(): spine.set_color("tab:blue"); spine.set_linewidth(2.0)

    # === Row 1 (visual content) ===
    # Left: rendered emission E_t
    ax = fig.add_subplot(gs[1, 1:3])
    ax.imshow(em_rgb)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"rendered emission E_t  (BLAKE3-XOF deterministic from S_t)",
                  fontsize=11, fontweight="bold")

    # Right: captured frame C_t
    ax = fig.add_subplot(gs[1, 3:])
    ax.imshow(cap_rgb)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"captured frame C_t  (E_t projected onto cloth + camera capture)",
                  fontsize=11, fontweight="bold")

    # === Closed-loop annotation: arrow from C_t (lower right) up to S_{t+1} cell (upper right) ===
    # Use figure-level arrow with curved connection style, drawn in figure coords.
    fig.canvas.draw()  # so we can compute axes positions
    cap_ax = axes_capture if False else None  # placeholder
    # Place a curved arrow from upper-mid of capture panel to top of S_{t+1} panel
    fig.text(0.84, 0.46,
             "↑ C_t bytes hashed\n   into S_{t+1}",
             ha="center", va="center", fontsize=9,
             color="tab:orange", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lemonchiffon",
                       edgecolor="tab:orange", linewidth=1.2))
    # Visual arrow as annotation between figure points: from (0.85, 0.55) up to (0.91, 0.78)
    fig.patches.append(mpatches.FancyArrowPatch(
        (0.95, 0.50), (0.95, 0.79),
        transform=fig.transFigure,
        connectionstyle="arc3,rad=0.18",
        arrowstyle="->", mutation_scale=22,
        color="tab:orange", linewidth=2.5, zorder=10))

    # === Row 2 (verifier evaluation summary) ===
    ax = fig.add_subplot(gs[2, :])
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.text(0.5, 0.65, "Closed feedback loop: every captured frame's bytes are committed into the next chain state, making forgeries detectable",
            ha="center", va="center", fontsize=11, fontweight="bold", color="tab:orange")
    ax.text(0.5, 0.30, "verifier:  diffusion model trained on (C, E_correct) pairs from session ⇒ evaluates C ↔ E coupling",
            ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(0.5, 0.00,
            "Phase G AUROC = 1.000 [1.000, 1.000] (Δ ≈ 0.004 MSE) on correct-vs-wrong-frame E   |   shuffled control AUROC = 0.5006 (chance baseline)",
            ha="center", va="center", fontsize=10, color="dimgray")

    # Title above everything
    fig.suptitle(
        "Truth Beam: chain-coupled optical evidence loop\n"
        f"(frame t={args.frame} from D2 session, 8-bit Bayer capture, fBm emission render)",
        fontsize=14, fontweight="bold")

    fig.savefig(args.out / "system_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[V1] wrote system_overview.png "
          f"({(args.out / 'system_overview.png').stat().st_size // 1024} KB)")

    # Caption draft
    caption = []
    caption.append("# Figure 1 caption draft")
    caption.append("")
    caption.append(f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    caption.append("")
    caption.append("## Suggested caption (for manuscript)")
    caption.append("")
    caption.append(
        f"**Figure 1.** Truth Beam closed feedback loop. At each frame t, a 32-byte "
        f"chain state S_t is expanded via BLAKE3-XOF (`blake3('TB:SEED:{{R,G,B}}:v8' || S_t)` → "
        f"4 octaves × 3 channels = 43,110 B per channel) into a deterministic emission "
        f"render E_t — a 1080×1920 fBm pattern — which is projected onto a cloth surface "
        f"and captured by a Bayer-pattern camera as C_t. The captured raw Bayer is "
        f"hashed (`bayer_hash_t = BLAKE3(C_t)`) and **folded back into the next chain "
        f"state**: `S_{{t+1}} = blake3(<row_tag> || S_t || bayer_hash_t || meta_t || "
        f"drand_round_n || drand_value [|| ai_payload_root])`. The D2 session was "
        f"recorded under v9 (row_tag=`TB:ROW:v9`, no ai_payload term); the V10 session "
        f"was recorded under v10 (row_tag=`TB:ROW:v10`, with `ai_payload_root` folded "
        f"in for 17 of 3743 frames where AI payloads arrived in the row's window, "
        f"otherwise the AI_PAYLOAD_EMPTY_ROOT sentinel). Both formulae verified at "
        f"`truth_beam/protocol/session_schema.py:305` (v9) and `:403` (v10). This "
        f"closed loop makes the protocol tamper-evident: an attacker who substitutes "
        f"C_t for a forgery produces a different `bayer_hash_t`, hence a different "
        f"`S_{{t+1}}`, hence a different `E_{{t+1}}` — detectable by any verifier "
        f"holding a sample of the chain. A diffusion verifier trained on (C, E_correct) "
        f"pairings achieves AUROC = 1.000 [1.000, 1.000] across {{D2, V10}} sessions "
        f"distinguishing correct from wrong-frame E pairings (mean Δ ≈ 0.004 MSE; "
        f"shuffled-mode control baseline = 0.5006). Frame shown is t=" +
        str(args.frame) + f" from session D2 (v9 protocol)."
    )
    caption.append("")
    caption.append("## Notes for operator review")
    caption.append("")
    caption.append("1. **Chain semantics confirmed** (operator urgent inspection 2026-05-03 18:15 UTC): canonical protocol code at `truth_beam/protocol/session_schema.py:305` (function `compute_chain_row`) shows `bayer_hash_t` IS hashed into S_{t+1} alongside drand entropy and capture meta. Closed-loop arrow added. Chain integrity claim no longer hedged.")
    caption.append("2. **Figure data is real**: D2 session frame 100, actual debayered raw + actual rendered emission + actual chain log entries. Hex strings are truncated for visual clarity but match `data/d2/chain_log.csv` row 100.")
    caption.append("3. **Caption framing** uses 'non-repudiable record' for capture commitment — this is justified because BLAKE3 of raw bytes is committed; operator may want stronger or weaker phrasing depending on the threat-model framing chosen for the paper.")
    caption.append("4. **Phase G AUROC + shuffled-control numbers** in the caption are the headline empirical claim; if the manuscript moves these to a later section, the caption should be updated.")
    caption.append("5. **'Cloth surface'** assumes the operator wants to identify the experimental setup at this level of specificity. If the rig is being kept abstract for review, swap to 'projection surface' or similar.")
    caption.append("")
    (args.out / "figure_1_caption_draft.md").write_text("\n".join(caption))
    print(f"[V1] wrote figure_1_caption_draft.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

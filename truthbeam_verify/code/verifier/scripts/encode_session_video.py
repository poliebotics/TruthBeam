"""Session video encoder — back-to-back captured frames at 15 fps.

Reads raw .raw captures (8-bit BayerRG, 5320×4600 per frame), debayers to
RGB, gamma-corrects, and pipes to ffmpeg for H.264 encoding at 1920×1080.

Pre-flight mode (--n-frames N): encodes only the first N frames + dumps
frame_0_preview.png so a human can sanity-check Bayer pattern and color
balance before launching the full encode.

Usage:
    # Pre-flight 100 frames of D2 (writes preview PNG + 100-frame test video)
    python scripts/encode_session_video.py --session D2 --n-frames 100 \
        --out /tmp/d2_test.mp4

    # Full encode
    python scripts/encode_session_video.py --session D2 \
        --out experiments/visual_grids/session_previews/d2_full_15fps.mp4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np


SESSION_DIRS = {
    "D2":  Path("/path/to/poliebotics_phase_b/data/d2"),
    "V10": Path("/path/to/poliebotics_phase_b/data/v10"),
}

# Source raw geometry (per src/data/emission_dataset.py)
WIDTH_RAW, HEIGHT_RAW = 5320, 4600
EXPECTED_RAW_BYTES = WIDTH_RAW * HEIGHT_RAW   # 24,472,000

# Output geometry
OUT_W, OUT_H = 1920, 1080
GAMMA = 1.6   # matches scripts/visual_grids.py convention


def frame_paths(session: str) -> list[Path]:
    sd = SESSION_DIRS[session]
    rec_dir = sd / "Recordings"
    if not rec_dir.exists():
        raise SystemExit(f"missing {rec_dir}")
    paths = sorted(rec_dir.glob("frame_*.raw"))
    return paths


def debayer_and_process(raw_path: Path) -> np.ndarray:
    """Read .raw, debayer to RGB, gamma-correct, return (H, W, 3) uint8 at OUT_HxOUT_W."""
    raw_bytes = np.fromfile(raw_path, dtype=np.uint8)
    if raw_bytes.size != EXPECTED_RAW_BYTES:
        raise RuntimeError(
            f"unexpected raw size {raw_bytes.size} for {raw_path.name} "
            f"(expected {EXPECTED_RAW_BYTES})")
    raw2d = raw_bytes.reshape(HEIGHT_RAW, WIDTH_RAW)
    # cv2's BayerRG2RGB: pattern is RGGB-starting-with-RG. _split_cfa_rggb
    # confirms our layout: (0,0)=R, (0,1)=G1, (1,0)=G2, (1,1)=B → RGGB.
    rgb = cv2.cvtColor(raw2d, cv2.COLOR_BayerRG2RGB)
    # Gamma correction in [0,1] space, then back to uint8.
    rgb01 = rgb.astype(np.float32) / 255.0
    rgb01 = np.clip(rgb01, 0, 1) ** (1.0 / GAMMA)
    rgb_g = (np.clip(rgb01, 0, 1) * 255).astype(np.uint8)
    # Resize to 1920x1080 (force_aspect_distort — operator OK'd standard preview).
    return cv2.resize(rgb_g, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)


def blake3_file(path: Path) -> str:
    try:
        from blake3 import blake3
        h = blake3()
    except ImportError:
        h = hashlib.blake2b(digest_size=32)
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, choices=("D2", "V10"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-frames", type=int, default=0,
                    help="0 = full encode; >0 = pre-flight subset.")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--preview-png", type=Path, default=None,
                    help="If set, save first-frame debayer result here for "
                    "color-correctness sanity check.")
    ap.add_argument("--ffmpeg-bin", type=str, default=None,
                    help="Override ffmpeg binary; default = imageio-ffmpeg bundled.")
    args = ap.parse_args()

    if args.ffmpeg_bin is None:
        try:
            import imageio_ffmpeg
            args.ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            args.ffmpeg_bin = "ffmpeg"

    args.out.parent.mkdir(parents=True, exist_ok=True)

    paths = frame_paths(args.session)
    n_total = len(paths)
    print(f"[encode] {args.session}: {n_total} raw frames")
    if args.n_frames > 0:
        paths = paths[:args.n_frames]
        print(f"[encode] PRE-FLIGHT: encoding first {len(paths)} frames")

    # Halt: disk space safety check. Estimate output ~10 MB/min at CRF 20 1080p,
    # so a full session is ~50-100 MB. Need at least 1 GB free for safety.
    df = subprocess.run(["df", "--output=avail", "--block-size=1G",
                          str(args.out.parent)],
                         capture_output=True, text=True)
    try:
        free_gb = int(df.stdout.splitlines()[1].strip())
        if free_gb < 50:
            raise SystemExit(
                f"[encode] HALT: only {free_gb} GB free at {args.out.parent}; "
                "spec requires ≥50 GB.")
        print(f"[encode] disk free: {free_gb} GB at output dir")
    except (IndexError, ValueError):
        print(f"[encode] WARN: could not parse df output; continuing")

    # Optional preview PNG of first debayered frame
    if args.preview_png is not None:
        first_rgb = debayer_and_process(paths[0])
        # Save as BGR (cv2 imwrite convention)
        bgr = cv2.cvtColor(first_rgb, cv2.COLOR_RGB2BGR)
        args.preview_png.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.preview_png), bgr)
        print(f"[encode] preview frame → {args.preview_png}")

    # Launch ffmpeg with stdin pipe for raw RGB frames
    cmd = [
        args.ffmpeg_bin, "-y", "-loglevel", "warning",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{OUT_W}x{OUT_H}",
        "-framerate", str(args.fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(args.crf),
        "-movflags", "+faststart",
        str(args.out),
    ]
    print(f"[encode] launching ffmpeg → {args.out}")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    t0 = time.time()
    n_processed = 0
    try:
        for i, raw_path in enumerate(paths):
            rgb = debayer_and_process(raw_path)
            proc.stdin.write(rgb.tobytes())
            n_processed += 1
            if (i + 1) % 200 == 0 or i + 1 == len(paths):
                elapsed = time.time() - t0
                rate = n_processed / elapsed
                remaining = (len(paths) - n_processed) / max(rate, 1e-3)
                print(f"  frame {i+1}/{len(paths)}  rate={rate:.1f} fps  "
                      f"elapsed={elapsed:.0f}s  ETA={remaining:.0f}s",
                      flush=True)
    except (BrokenPipeError, OSError) as ex:
        proc.stdin.close()
        proc.wait()
        raise SystemExit(f"[encode] HALT: ffmpeg pipe broke at frame {n_processed}: {ex}")
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"[encode] HALT: ffmpeg exit code {rc}")

    # Manifest
    elapsed = time.time() - t0
    sz = args.out.stat().st_size
    duration_sec = n_processed / args.fps
    h = blake3_file(args.out)
    manifest = {
        "session": args.session,
        "n_frames_encoded": n_processed,
        "n_frames_total_in_session": n_total,
        "fps": args.fps,
        "duration_sec": duration_sec,
        "duration_hms": time.strftime("%H:%M:%S", time.gmtime(duration_sec)),
        "out_resolution": f"{OUT_W}x{OUT_H}",
        "crf": args.crf,
        "output_path": str(args.out),
        "size_bytes": sz,
        "size_mib": round(sz / (1024*1024), 2),
        "blake3": h,
        "encode_wall_sec": round(elapsed, 1),
        "ffmpeg_bin": args.ffmpeg_bin,
        "gamma_applied": GAMMA,
        "debayer_pattern": "BayerRG (cv2.COLOR_BayerRG2RGB) — matches RGGB layout",
        "raw_geometry": f"{WIDTH_RAW}x{HEIGHT_RAW} 8-bit Bayer",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[encode] DONE in {elapsed:.0f}s  "
          f"size={sz/(1024*1024):.1f} MiB  duration={manifest['duration_hms']}")
    print(f"[encode] manifest → {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""verify_frames.py — random-frame spot check, GPU-free, against the protocol's
own cryptographic commitments.

For N *randomly chosen* frames of a released session it confirms two things that
a staged dataset could not both satisfy:

  (1) RAW FRAME:  blake3(raw bytes) == the `bayer_blake3_hex` committed for that
      row in chain_log.csv — the canonical per-frame commitment.

  (2) EMISSION:   re-derive the projected tile E_t straight from the chain state
      S_t (compute_xof_seeds -> gen_rgb_v2, the bit-exact CPU twin of the GPU
      generator whose source hash is pinned in S_0) and confirm
      blake3(tile) == the `emission_live_pixel_blake3_hex` committed for that row.
      So the emission is the deterministic BLAKE3-XOF expansion of the state,
      not an arbitrary image.

Because the frames are chosen at random each run, a passing result can't have
been pre-arranged. Everything is fetched from public URLs; no GPU, no login.

Usage:
    python3 code/recording/verify/verify_frames.py [N] [d2|v10] [--seed S]
                                                    [--gateway https://data.truthbeam.com]
"""
import argparse, csv, os, sys, urllib.request, secrets
from multiprocessing.pool import ThreadPool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "protocol"))
from blake3 import blake3                          # noqa: E402
try:
    from session_schema import compute_xof_seeds   # noqa: E402
except Exception:
    from chain import compute_xof_seeds            # noqa: E402
from tile_cpu import gen_rgb_v2                     # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("n", nargs="?", type=int, default=5, help="how many random frames")
ap.add_argument("session", nargs="?", default="d2", choices=["d2", "v10"])
ap.add_argument("--seed", type=int, default=None, help="fix the RNG (default: truly random)")
ap.add_argument("--gateway", default=os.environ.get("TB_GATEWAY", "https://data.truthbeam.com"))
ap.add_argument("--session-dir", default=None, help="use a local session dir for chain_log.csv")
args = ap.parse_args()

base = f"{args.gateway}/sessions/{args.session}"
_UA = {"User-Agent": "truthbeam-verify/1.0 (+https://truthbeam.com)"}  # default urllib UA is 403'd by Cloudflare
def fetch(path, t=120):
    req = urllib.request.Request(f"{base}/{path}", headers=_UA)
    with urllib.request.urlopen(req, timeout=t) as r:
        return r.read()

# chain_log: local if available, else fetch
if args.session_dir and os.path.exists(os.path.join(args.session_dir, "chain_log.csv")):
    raw = open(os.path.join(args.session_dir, "chain_log.csv"), "rb").read()
else:
    print(f"fetching {base}/chain_log.csv ...")
    raw = fetch("chain_log.csv")
rows = [r for r in csv.reader(raw.decode().splitlines()) if r and not r[0].startswith("#")]
hdr = rows[0]; data = rows[1:]
Si = hdr.index("S_t_hex"); Bi = hdr.index("bayer_blake3_hex"); Ei = hdr.index("emission_live_pixel_blake3_hex")
N = len(data)

rng = (lambda: secrets.randbelow(N)) if args.seed is None else None
if args.seed is not None:
    import random; random.seed(args.seed); rng = lambda: random.randrange(N)
picks = sorted({rng() for _ in range(args.n * 3)})[: args.n]

print(f"session {args.session}: {N} frames; spot-checking {len(picks)} random frames {picks}")
pool = ThreadPool(3); ok_raw = ok_emit = 0
for t in picks:
    f = f"frame_{t:06d}.raw"
    rawb = fetch(f"Recordings/{f}")
    raw_ok = blake3(rawb).hexdigest() == data[t][Bi]
    seeds = compute_xof_seeds(bytes.fromhex(data[t][Si]))
    tile = gen_rgb_v2(*seeds, pool)
    emit_ok = blake3(tile.tobytes()).hexdigest() == data[t][Ei]
    ok_raw += raw_ok; ok_emit += emit_ok
    print(f"  frame {t:>5}: raw blake3 {'OK' if raw_ok else 'FAIL'}  |  "
          f"emission re-derived {'OK' if emit_ok else 'FAIL'}")

print(f"\nraw-frame commitment : {ok_raw}/{len(picks)}")
print(f"emission re-derivation: {ok_emit}/{len(picks)}")
if ok_raw == len(picks) and ok_emit == len(picks):
    print("PASS — random frames match their committed BLAKE3, and emissions re-derive bit-exactly.")
else:
    print("FAIL — see above"); sys.exit(1)

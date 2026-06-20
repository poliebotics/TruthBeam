#!/usr/bin/env bash
# ======================================================================
#  TruthBeam — friendly tiered downloader
#
#  You do NOT have to grab 378 GiB to look. Pick a tier:
#
#    ./download.sh                 # menu + sizes
#    ./download.sh scores          # ~2 MB    Path A inputs (recompute the AUROC)
#    ./download.sh models          # ~1.1 GB  verifier + F-A v1 forger weights
#    ./download.sh sample [d2|v10] # ~150 MB  a TASTE: one session's metadata +
#                                  #          12 preview/emission pairs + 3 raw frames
#    ./download.sh video           # ~640 MB  the hand-made 2023 video (+ 64s intro)
#    ./download.sh session d2|v10  # 232/146 GiB  a full session
#    ./download.sh all             # everything (huge)
#
#  Everything lands under ./tb_download/.  Re-runnable (curl -C - resumes).
# ======================================================================
set -euo pipefail
G="${TB_GATEWAY:-https://data.truthbeam.com}"
OUT="${TB_OUT:-./tb_download}"
get() { mkdir -p "$OUT/$(dirname "$1")"; echo "  $1"; curl -fsSL -C - -o "$OUT/$1" "$G/$1"; }

menu() {
  cat <<EOF
TruthBeam download helper — you don't need the whole 378 GiB to look.

  scores    ~2 MB     Path A inputs — recompute the headline AUROC yourself
  models    ~1.1 GB   verifier (456 MB) + F-A v1 forger checkpoints
  sample    ~150 MB   a taste of one session: metadata + 12 preview/emission
                      pairs + 3 raw frames — enough to SEE the data
  video     ~640 MB   the hand-made 2023 PolieBotics video (+ the 64 s intro)
  session   232/146   a full ground-truth session (d2 / v10)
  all       378 GiB   the complete two-session corpus

Usage:  ./download.sh <tier> [d2|v10]      (output -> $OUT/)
Tip: 'sample' is the fun one. 'scores' + the repo's verify_all.sh = full proof.
EOF
}

scores() {
  echo "[scores] Path A inputs (~2 MB)..."
  for ck in 00005000 00025000 00070000 00100000; do for s in d2 v10; do
    get "models/repro/stage_0_eval/step_$ck/stage0_${s}_raw.npz"; done; done
}
models() {
  echo "[models] verifier + forger weights (~1.1 GB)..."
  get "models/verifier/model_final.pt"
  for ck in 00005000 00025000 00070000 00100000; do get "models/fa_v1_forger/f_a_v1_step_$ck.pt"; done
}
sample() {
  local s="${1:-d2}"
  echo "[sample] a taste of session $s (metadata + 12 preview/emission pairs + 3 raws)..."
  for f in manifest.json manifest.pretty.json chain_log.csv anchor_txs.csv capture_log.csv \
           verification_bundle.json verify_report.json README_BUNDLE.md CLAIMS.md; do
    get "sessions/$s/$f"; done
  for i in 000000 000500 001000 001500 002000 002500 003000 003500 004000 004500 005000 005500; do
    get "sessions/$s/derived/Recordings_previews/frame_$i.png"
    get "sessions/$s/derived/Emissions/tile_$i.png"
  done
  for i in 000000 002500 005000; do get "sessions/$s/Recordings/frame_$i.raw"; done
  echo "  -> open the .png previews/tiles, read chain_log.csv, then run the repo's"
  echo "     code/recording/verify/temporal_analysis.py on $OUT/sessions/$s"
}
video() {
  echo "[video] hand-made 2023 video + 64 s intro (~640 MB)..."
  get "pinata/PolieBotics.mp4"; get "pinata/TruthBeam_Introduction.mp4"
}
session() {
  local s="${1:-}"; [ "$s" = d2 ] || [ "$s" = v10 ] || { echo "usage: ./download.sh session d2|v10"; exit 1; }
  echo "[session $s] full corpus — this is large (d2=232 GiB, v10=146 GiB)."
  echo "Mirroring sessions/$s/ ... (Ctrl-C to stop; re-run to resume)"
  # uses the published per-file URL list if present, else the gateway directory walk
  if [ -f "downloads/${s}_files.txt" ]; then
    ( cd "$OUT" && wget -x -c -i "$OLDPWD/downloads/${s}_files.txt" )
  else
    echo "Per-file list downloads/${s}_files.txt not found next to this script."
    echo "Get it from the PolieBotics umbrella repo (DOWNLOADS.md) and: wget -x -c -i ${s}_files.txt"
  fi
}

case "${1:-menu}" in
  scores) scores ;;
  models) models ;;
  sample) sample "${2:-d2}" ;;
  video) video ;;
  session) session "${2:-}" ;;
  all) scores; models; sample d2; video; echo "For the full 378 GiB corpus: ./download.sh session d2 && ./download.sh session v10" ;;
  *) menu ;;
esac
echo "done -> $OUT/"

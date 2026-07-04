#!/usr/bin/env bash
# ======================================================================
#  TruthBeam — friendly tiered downloader
#
#  You do NOT have to grab 378 GiB to look. Pick a tier:
#
#    ./download.sh                 # menu + sizes
#    ./download.sh scores          # ~2 MB    Path A inputs (recompute the AUROC)
#    ./download.sh models          # ~2.5 GB  verifier + F-A v1 forger weights
#    ./download.sh sample [d2|v10] # ~180 MB  a TASTE: one session metadata +
#                                  #          8 preview/emission pairs + 2 raw frames
#    ./download.sh video           # ~640 MB  the hand-made 2023 video (+ 64s intro)
#    ./download.sh session d2|v10  # 232/146 GiB  a full session
#    ./download.sh all             # everything (huge)
#
#  Everything lands under ./tb_download/.  Re-runnable (curl -C - resumes).
# ======================================================================
set -euo pipefail
G="${TB_GATEWAY:-https://data.truthbeam.com}"
OUT="${TB_OUT:-./tb_download}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUMS="$HERE/SHA256SUMS"
declare -A SHA; CHECKED=0; FAILS=0
if [ -f "$SUMS" ]; then while read -r h p; do [ -n "${p:-}" ] && SHA["$p"]="$h"; done < "$SUMS"; fi
# download a file; if its SHA-256 is published, verify it immediately (universal — no extra tools)
get() {
  local p="$1"; mkdir -p "$OUT/$(dirname "$p")"
  curl -fsSL -C - -o "$OUT/$p" "$G/$p"
  if [ -n "${SHA[$p]:-}" ] && command -v sha256sum >/dev/null 2>&1; then
    if [ "$(sha256sum "$OUT/$p" | awk '{print $1}')" = "${SHA[$p]}" ]; then echo "  ✓ $p"; CHECKED=$((CHECKED+1))
    else echo "  ✗ HASH MISMATCH: $p"; FAILS=$((FAILS+1)); fi
  else echo "    $p"; fi
}

menu() {
  cat <<EOF
TruthBeam download helper — you don't need the whole 378 GiB to look.

  scores    ~2 MB     Path A inputs — recompute the headline AUROC yourself
  models    ~2.5 GB   verifier (478 MB / 455 MiB) + F-A v1 forger checkpoints (484 MiB each)
  sample    ~180 MB   a taste of one session: metadata + 8 preview/emission
                      pairs + 2 raw frames — enough to SEE the data
                      (then: python3 code/recording/verify/verify_frames.py 5 d2)
  video     ~640 MB   the hand-made 2023 PolieBotics video (+ the 64 s intro)
  session   232/146   a full ground-truth session (d2 / v10)
  all       378 GiB   the complete two-session corpus
  verify              re-check everything already downloaded against SHA256SUMS

Usage:  ./download.sh <tier> [d2|v10]      (output -> $OUT/)
Every download is AUTO-VERIFIED (SHA-256) against SHA256SUMS as it lands.
Tip: 'sample' is the fun one. 'scores' + the repo's verify_all.sh = full proof.
EOF
}

scores() {
  echo "[scores] Path A inputs (~2 MB)..."
  for ck in 00005000 00025000 00070000 00100000; do for s in d2 v10; do
    get "models/repro/stage_0_eval/step_$ck/stage0_${s}_raw.npz"; done; done
}
models() {
  echo "[models] verifier + forger weights (~2.5 GB)..."
  get "models/verifier/model_final.pt"
  for ck in 00005000 00025000 00070000 00100000; do get "models/fa_v1_forger/f_a_v1_step_$ck.pt"; done
}
sample() {
  local s="${1:-d2}"
  echo "[sample] a taste of session $s (metadata + 8 preview/emission pairs + 2 raws)..."
  for f in manifest.json manifest.pretty.json chain_log.csv anchor_txs.csv capture_log.csv \
           verification_bundle.json verify_report.json README_BUNDLE.md CLAIMS.md; do
    get "sessions/$s/$f"; done
  # indices valid for both sessions (v10 has 3743 frames) and present in SHA256SUMS
  for i in 000000 000500 001000 001500 002000 002500 003000 003500; do
    get "sessions/$s/derived/Recordings_previews/frame_$i.png"
    get "sessions/$s/derived/Emissions/tile_$i.png"
  done
  for i in 000000 002500; do get "sessions/$s/Recordings/frame_$i.raw"; done
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
    ( cd "$OUT" && wget -x -nH -c -i "$OLDPWD/downloads/${s}_files.txt" )
  else
    echo "Per-file list downloads/${s}_files.txt not found next to this script."
    echo "Get it from the PolieBotics umbrella repo (DOWNLOADS.md) and: wget -x -nH -c -i ${s}_files.txt"
  fi
}

case "${1:-menu}" in
  scores) scores ;;
  models) models ;;
  sample) sample "${2:-d2}" ;;
  video) video ;;
  session) session "${2:-}" ;;
  all) scores; models; sample d2; video; echo "For the full 378 GiB corpus: ./download.sh session d2 && ./download.sh session v10" ;;
  verify) command -v sha256sum >/dev/null 2>&1 || { echo "need sha256sum"; exit 1; }
          [ -f "$SUMS" ] || { echo "no SHA256SUMS next to this script"; exit 1; }
          ( cd "$OUT" && awk 'NF==2' "$SUMS" | while read -r h p; do [ -f "$p" ] && printf '%s  %s\n' "$h" "$p"; done | sha256sum -c - ) ;;
  *) menu ;;
esac
if [ "${CHECKED:-0}" -gt 0 ] || [ "${FAILS:-0}" -gt 0 ]; then
  if [ "${FAILS:-0}" -eq 0 ]; then echo "integrity: $CHECKED file(s) verified against SHA256SUMS ✓"
  else echo "integrity: $FAILS MISMATCH(es) — DO NOT trust these bytes"; exit 1; fi
fi
echo "done -> $OUT/"

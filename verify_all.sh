#!/usr/bin/env bash
# ======================================================================
#  TruthBeam — one-command clean-room verification
#
#  From a bare machine with  python3 + git + curl , run:
#
#      git clone https://github.com/poliebotics/truthbeam
#      cd truthbeam
#      bash verify_all.sh
#
#  It fetches ONLY public URLs (data.truthbeam.com) — no private context,
#  no logins — and verifies:
#    (A) the headline  AUROC = 1.000  (CPU, ~2 min)
#    (B) the temporal binding: RSK-mainnet anchor blocks, that each pulse
#        tx's calldata commits the expected state, drand BLS signatures,
#        and the commit rate.
#  Everything prints to screen and to logs under .verify_run/.
# ======================================================================
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATEWAY="${TB_GATEWAY:-https://data.truthbeam.com}"
WORK="$REPO/.verify_run"

echo "=================================================================="
echo " TruthBeam clean-room verification"
echo " repo:    $REPO"
echo " gateway: $GATEWAY"
echo "=================================================================="
mkdir -p "$WORK"; cd "$WORK"

echo; echo "[0/3] isolated Python environment..."
python3 -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
pip install -q --upgrade pip >/dev/null
pip install -q numpy scikit-learn py_ecc blake3 >/dev/null
echo "      deps: numpy, scikit-learn, py_ecc, blake3 installed."

echo; echo "[1/3] Path A — recompute the headline AUROC from public scores..."
for ck in 00005000 00025000 00070000 00100000; do
  for s in d2 v10; do
    mkdir -p "stage_0_eval/step_$ck"
    curl -fsSL -o "stage_0_eval/step_$ck/stage0_${s}_raw.npz" \
      "$GATEWAY/models/repro/stage_0_eval/step_$ck/stage0_${s}_raw.npz"
  done
done
python3 "$REPO/code/verifier/scripts/decomposition_part_1.py" \
  --stage-0-root stage_0_eval --out out --seed 0 | tee pathA.log

echo; echo "[2/3] Temporal — fetch session metadata, verify RSK + drand..."
for s in d2 v10; do
  mkdir -p "$s"
  for f in manifest.json anchor_txs.csv chain_log.csv; do
    curl -fsSL -o "$s/$f" "$GATEWAY/sessions/$s/$f"
  done
  echo "--- $s ---"
  python3 "$REPO/code/recording/verify/temporal_analysis.py" "$s" | tee "temporal_$s.log"
done

echo; echo "[3/4] Random frames — BLAKE3 commitments + emission re-derivation (GPU-free)..."
for s in d2 v10; do
  echo "--- $s ---"
  python3 "$REPO/code/recording/verify/verify_frames.py" 5 "$s" | tee "frames_$s.log"
done

echo; echo "[4/4] Verdict"
PASS=1
if grep -q "AUROC combined=1.0000" pathA.log; then echo "  Path A      : AUROC = 1.0000  ✓"
else echo "  Path A      : FAIL"; PASS=0; fi
for s in d2 v10; do
  if grep -q "carry the expected commitment in calldata" "temporal_$s.log" \
     && grep -qE "invalid=0 " "temporal_$s.log"; then  # 0 BLS failures (transient fetch retries OK)
    echo "  Temporal $s  : RSK calldata + drand BLS  ✓"
  else echo "  Temporal $s  : FAIL"; PASS=0; fi
  if grep -q "^PASS — random frames" "frames_$s.log"; then echo "  Frames $s    : raw BLAKE3 + emission re-derivation  ✓"
  else echo "  Frames $s    : FAIL"; PASS=0; fi
done
echo "=================================================================="
if [ "$PASS" = 1 ]; then
  echo " ALL CHECKS PASSED — verified from public URLs, no private context."
else
  echo " SOME CHECKS FAILED — see logs under $WORK"; exit 1
fi
echo "=================================================================="

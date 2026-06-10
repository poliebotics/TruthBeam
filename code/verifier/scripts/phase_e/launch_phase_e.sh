#!/usr/bin/env bash
# Phase E training launcher — runs P2 smokes, then launches E1/E2/E3r in
# parallel, with per-experiment auto-probe chain on success.
#
# Pre-conditions (must be verified before calling):
#   - V1/V2/V3/V4/V5 all green
#   - P0 confirms exp001c reproduces (outcome A or B, NOT C)
#   - P1 oracle/null controls pass
#
# Allocation (DDP=2 per V4 / V5):
#   E1:  GPUs 0-1
#   E2:  GPUs 2-3
#   E3r: GPUs 4-5
#   E0:  GPU 6 (concurrent rigorous exp001c re-eval, 1-2 hr)
#   GPU 7: reserved for adversarial probes auto-launched on E1 success
#
# Auto-launch (per operator pre-approved interventions):
#   On STOP file with status=success in any experiment's out_dir:
#     fire adversarial_probes.py against best_by_psnr.pt on freed GPUs

set -euo pipefail
cd "$(dirname "$0")/../.."

EXP_ROOT=/path/to/poliebotics_phase_b
PHASE_E=$EXP_ROOT/experiments/phase_e
mkdir -p $PHASE_E $PHASE_E/p2

# Source venv for the orchestrator process itself.
source .venv_a100/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

# ---------- P2: overfit smokes (sequential, ~10 min total on GPU 0) ----------
# E2 uses native res 2300×2660 (4× E1 pixels) → bs=8 OOMs at 80 GB; use n_frames=2.
echo "=== P2: overfit smokes ==="
for cfg in e1 e3r e2; do
  if [ "$cfg" = "e2" ]; then
    NFRAMES=2
  else
    NFRAMES=8
  fi
  CUDA_VISIBLE_DEVICES=0 python scripts/phase_e/p2_overfit_smoke.py \
    --config scripts/phase_e/configs/${cfg}.yaml --bf16 \
    --n-frames $NFRAMES \
    --out $PHASE_E/p2/${cfg}.json 2>&1 | tee $PHASE_E/p2/${cfg}.log
  pass=$(python3 -c "import json; print(json.load(open('$PHASE_E/p2/${cfg}.json'))['PASS'])")
  if [ "$pass" != "True" ]; then
    echo "[P2 FAIL] $cfg did not pass overfit smoke — STOP, no training launch"
    exit 1
  fi
  echo "[P2 PASS] $cfg"
done

# ---------- E0: rigorous exp001c re-eval on GPU 6 (concurrent, ~1 hr) ----------
echo "=== Launch E0 rigorous re-eval on GPU 6 ==="
mkdir -p $PHASE_E/e0
screen -dmS e0_rigorous bash -lc "
cd $(pwd)
. .venv_a100/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
# Phase D-style calib/report split: tau5 from first half, FMR on second half
# Both halves of d2_phase_d split [4792, 5992); also d2_orig for completeness
for SPLIT in d2_phase_d_calib d2_phase_d_report d2_orig; do
  CUDA_VISIBLE_DEVICES=6 python scripts/phase_e/phase_e_emission_eval.py \
    --mode ckpt --ckpt $EXP_ROOT/experiments/exp001c/checkpoints/ep027.pt \
    --arch EmissionPredictor --capture-h 1150 --capture-w 1330 \
    --d2-dir $EXP_ROOT/poliebotics_phase_b/data/d2 \
    --v10-dir $EXP_ROOT/poliebotics_phase_b/data/v10 \
    --split \$SPLIT --methodology legacy,family_balanced --bf16 \
    --n-fb 200 --n-legacy 50 \
    --out $PHASE_E/e0/exp001c_\${SPLIT}.json 2>&1 | tee $PHASE_E/e0/exp001c_\${SPLIT}.log
done
echo E0_DONE
"

# ---------- Launch E1/E2/E3r in parallel ----------
launch_train() {
  local exp=$1
  local gpus=$2
  local nproc=$3
  local port=$4
  mkdir -p $PHASE_E/$exp
  screen -dmS train_$exp bash -lc "
cd $(pwd)
. .venv_a100/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
CUDA_VISIBLE_DEVICES=$gpus torchrun --standalone --nproc_per_node=$nproc --master_port=$port \
  scripts/phase_e/train_phase_e.py --config scripts/phase_e/configs/${exp}.yaml --bf16 \
  2>&1 | tee $PHASE_E/$exp/run.log
echo TRAIN_EXIT_${exp}=\$? >> $PHASE_E/$exp/run.log
"
  echo "[launch] $exp on GPUs $gpus (DDP=$nproc, port $port)"
}

echo "=== Launch E1/E2/E3r in parallel ==="
launch_train e1  0,1   2 29501
launch_train e2  2,3   2 29502
launch_train e3r 4,5   2 29503

# ---------- Auto-probe chain (one per experiment) ----------
launch_probes_on_success() {
  local exp=$1
  local probe_gpu=$2
  screen -dmS probes_$exp bash -lc "
cd $(pwd)
. .venv_a100/bin/activate
# Wait for STOP file with status=success
while true; do
  if [ -f $PHASE_E/$exp/STOP ]; then
    status=\$(python3 -c \"import json; print(json.load(open('$PHASE_E/$exp/STOP'))['status'])\")
    if [ \"\$status\" = success ]; then
      echo \"[probe-chain $exp] success detected; launching probes on GPU $probe_gpu\"
      mkdir -p $PHASE_E/probes_$exp
      mkdir -p $PHASE_E/probes_$exp
      # Chain-byte sensitivity probe (operator pre-approved, replaces pixel-perturbation probes)
      # Pixel-perturbation probes (adversarial_probes.py) moved to _deferred/, requires operator approval to re-enable.
      CUDA_VISIBLE_DEVICES=$probe_gpu python scripts/phase_e/chain_byte_probe.py \
        --ckpt $PHASE_E/$exp/checkpoints/best_by_psnr.pt \
        --config scripts/phase_e/configs/${exp}.yaml \
        --n-rows 200 --bf16 --device cuda --render-device cpu \
        --out $PHASE_E/probes_$exp 2>&1 | tee $PHASE_E/probes_$exp/chain_byte_probe.log
    else
      echo \"[probe-chain $exp] status=\$status (not success); skipping probes\"
    fi
    break
  fi
  sleep 60
done
"
  echo "[probe-chain] armed for $exp on GPU $probe_gpu"
}

echo "=== Arm auto-probe chains ==="
launch_probes_on_success e1  7   # E1 → GPU 7 (always free)
launch_probes_on_success e3r 4   # E3r → reuse one of its GPUs once E3r STOPs
launch_probes_on_success e2  2   # E2 → reuse one of its GPUs once E2 STOPs

echo "=== Phase E launch complete ==="
echo "Watch:"
echo "  tail -f $PHASE_E/{e1,e2,e3r}/run.log"
echo "  ls $PHASE_E/*/STOP  # success/plateau/broken"
echo "  ls $PHASE_E/probes_*/probes.json"

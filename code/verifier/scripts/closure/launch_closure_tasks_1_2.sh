#!/usr/bin/env bash
# Launches Tasks 1 (trained report-half eval) and 2 (zero+oracle) for all
# 5 XOF variants. Uses GPUs 1-5 (GPU 0 is occupied by A4 post-hoc).
#
# Each variant runs sequentially in its own screen:
#   trained → zero → oracle (noiseless) → oracle (noise=0.05)
# All 5 variants run in parallel.
set -euo pipefail
cd "$(dirname "$0")/../.."

EXP_ROOT=/path/to/poliebotics_phase_b
STATS_DIR=$EXP_ROOT/cache/normalization_stats
CLOSURE=$EXP_ROOT/experiments/closure_package

mkdir -p $CLOSURE

# Map variant → GPU
declare -A GPU=(
  [a0]=1
  [a1]=2
  [a2]=3
  [a6]=4
  [a7]=5
)

for v in a0 a1 a2 a6 a7; do
  gpu="${GPU[$v]}"
  expdir=$EXP_ROOT/experiments/exp001h_${v}
  ckpt=$expdir/checkpoints/final_step.pt
  outdir=$CLOSURE/exp001h_${v}
  mkdir -p $outdir
  log=$outdir/closure_xof_eval.log
  cfg=configs/exp001h_${v}.yaml

  screen -dmS "closure_${v}" bash -lc "
cd $(pwd)
. .venv_a100/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Limit per-process CPU threads — 5 closures + Task 3 + Task 4 share host;
# without this, each process scoops all cores and all of them stall.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
echo '=== ${v} TASK 1 (trained, report+calib) ===' | tee $log
CUDA_VISIBLE_DEVICES=$gpu python scripts/closure/closure_xof_eval.py \
  --config $cfg --ckpt $ckpt --stats-dir $STATS_DIR \
  --mode trained --bf16 --max-frames 600 --splits all \
  --out $outdir/final_report.json 2>&1 | tee -a $log

echo '=== ${v} TASK 2 ZERO ===' | tee -a $log
CUDA_VISIBLE_DEVICES=$gpu python scripts/closure/closure_xof_eval.py \
  --config $cfg --stats-dir $STATS_DIR \
  --mode zero --bf16 --max-frames 600 --splits all \
  --out $outdir/zero_predictor.json 2>&1 | tee -a $log

echo '=== ${v} TASK 2 ORACLE noiseless ===' | tee -a $log
CUDA_VISIBLE_DEVICES=$gpu python scripts/closure/closure_xof_eval.py \
  --config $cfg --stats-dir $STATS_DIR \
  --mode oracle --noise-std 0.0 --bf16 --max-frames 200 --splits report \
  --out $outdir/oracle_noiseless.json 2>&1 | tee -a $log

echo '=== ${v} TASK 2 ORACLE noise=0.05 ===' | tee -a $log
CUDA_VISIBLE_DEVICES=$gpu python scripts/closure/closure_xof_eval.py \
  --config $cfg --stats-dir $STATS_DIR \
  --mode oracle --noise-std 0.05 --bf16 --max-frames 200 --splits report \
  --out $outdir/oracle_noise_005.json 2>&1 | tee -a $log

echo 'EXIT='\$? >> $log
"
  echo "[launch] closure_${v} on GPU ${gpu} (cfg=$cfg)"
done
echo "[launch] all 5 variants launched; tail logs at $CLOSURE/<exp>/closure_xof_eval.log"

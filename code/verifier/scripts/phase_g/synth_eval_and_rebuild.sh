#!/usr/bin/env bash
# Phase G — synth eval + report rebuild (hardened).
#
# Runs after synth training completes (8-GPU DDP, 15000 steps).
# Replaces the original finalizer's broken last steps with:
#   1. Wait for synth/model_final.pt
#   2. Launch 8-way frame-sharded eval on the 8 GPUs (parallel single-GPU shards)
#   3. Merge shards into summary.json
#   4. Re-run build_final_report with the proper synth eval data
#
# Hardening (applied 2026-05-01 in response to Codex finding):
#   - set -euo pipefail (was just -u; led to silent UNINTERPRETABLE report
#     when the previous synth attempt was killed mid-run)
#   - Explicit artifact-existence checks before each stage
#   - FINALIZER_DONE marker only touched if every required artifact exists
#   - All stage logs preserved for post-mortem

set -euo pipefail

PROJECT=/path/to/poliebotics_phase_b/poliebotics_phase_b
DATA_D2=$PROJECT/data/d2
DATA_V10=$PROJECT/data/v10
RUNS=/path/to/poliebotics_phase_b/experiments/phase_g_diffusion_diagnostic
PYTHON=$PROJECT/.venv_a100/bin/python

LOG=$RUNS/synth_eval_finalizer.log
exec > >(tee -a $LOG) 2>&1

ts() { date +"%Y-%m-%d %H:%M:%S UTC"; }

require_file() {
    local path=$1
    local stage=$2
    if [ ! -f "$path" ]; then
        echo "[$(ts)] STAGE FAILED — $stage: required file $path does not exist."
        echo "[$(ts)] Aborting WITHOUT touching FINALIZER_DONE."
        exit 1
    fi
    echo "[$(ts)] OK — $stage: $path exists ($(stat -c%s $path) bytes)"
}

echo "==================================================================="
echo "Phase G synth-eval finalizer — started $(ts)"
echo "==================================================================="

# ---------------- step 1: wait for synth training to finish ----------------
SYN_OUT=$RUNS/synthetic_positive
echo "[$(ts)] Waiting for $SYN_OUT/model_final.pt..."
while [ ! -f $SYN_OUT/model_final.pt ]; do
    sleep 60
done
require_file $SYN_OUT/model_final.pt "synth training"

# Settle and verify GPUs free
sleep 30
echo "[$(ts)] GPUs after synth train:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

# ---------------- step 2: 8-way sharded synth eval ----------------
SYN_EVAL=$SYN_OUT/eval
mkdir -p $SYN_EVAL
echo "[$(ts)] Launching 8 parallel synth-eval shards..."
NUM_SHARDS=8
for s in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$s $PYTHON $PROJECT/scripts/phase_g/eval_diffusion_diagnostic.py \
        --ckpt $SYN_OUT/model_final.pt \
        --d2-dir $DATA_D2 --v10-dir $DATA_V10 \
        --out $SYN_EVAL --n-d2 200 --n-v10 200 --bs 4 --bf16 \
        --shard $s --num-shards $NUM_SHARDS \
        > $SYN_EVAL/run_shard${s}.log 2>&1 &
done
echo "[$(ts)] All 8 shards launched. Waiting..."
wait
echo "[$(ts)] All shards finished. Verifying shard output files..."
for s in 0 1 2 3 4 5 6 7; do
    require_file $SYN_EVAL/eval_d2_raw_shard${s}.npz "synth-eval D2 shard $s"
    require_file $SYN_EVAL/eval_v10_raw_shard${s}.npz "synth-eval V10 shard $s"
done

# ---------------- step 3: merge shards ----------------
echo "[$(ts)] Merging shards..."
$PYTHON $PROJECT/scripts/phase_g/merge_eval_shards.py \
    --eval-dir $SYN_EVAL --num-shards $NUM_SHARDS \
    > $SYN_EVAL/merge.log 2>&1
require_file $SYN_EVAL/summary.json "synth-eval merged summary"
require_file $SYN_EVAL/eval_d2_raw.npz "synth-eval D2 merged raw"
require_file $SYN_EVAL/eval_v10_raw.npz "synth-eval V10 merged raw"

# ---------------- step 4: rebuild final report ----------------
echo "[$(ts)] Rebuilding final report..."
$PYTHON $PROJECT/scripts/phase_g/build_final_report.py \
    --runs-root $RUNS --out $RUNS/report \
    > $RUNS/build_report_v2.log 2>&1
require_file $RUNS/report/diagnostic_report.md "final report rebuild"

# ---------------- step 5: completion marker ----------------
touch $RUNS/SYNTH_EVAL_FINALIZER_DONE
echo "==================================================================="
echo "Phase G synth-eval finalizer — finished $(ts)"
echo "==================================================================="
echo "Final report: $RUNS/report/diagnostic_report.md"
tail -20 $RUNS/build_report_v2.log

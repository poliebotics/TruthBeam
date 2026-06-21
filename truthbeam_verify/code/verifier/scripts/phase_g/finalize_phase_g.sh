#!/usr/bin/env bash
# Phase G — finalizer. Runs after the main + shuffled evals complete.
#
# Assumed state:
#   - main/model_final.pt and shuffled/model_final.pt already exist
#   - main/eval/summary.json and shuffled/eval/summary.json arrive
#     when the parallel evals finish (single-GPU each on GPU 0 and GPU 1)
#   - synth_positive has not been trained yet (the in-place attempt was
#     killed due to disk-I/O contention with concurrent evals)
#
# This script:
#   1. Waits for both eval summary.json files to exist
#   2. Kills the original phase_g_overnight orchestrator (its Run 3 step
#      cannot run after another process took over, and its later steps
#      would race)
#   3. Launches synth training on all 8 GPUs (DDP=8, 15000 steps)
#   4. Runs synth eval on GPU 0
#   5. Builds the final report

set -u

PROJECT=/path/to/poliebotics_phase_b/poliebotics_phase_b
DATA_D2=$PROJECT/data/d2
DATA_V10=$PROJECT/data/v10
RUNS=/path/to/poliebotics_phase_b/experiments/phase_g_diffusion_diagnostic
PYTHON=$PROJECT/.venv_a100/bin/python
TORCHRUN=$PROJECT/.venv_a100/bin/torchrun

LOG=$RUNS/finalizer.log
exec > >(tee -a $LOG) 2>&1

ts() { date +"%Y-%m-%d %H:%M:%S UTC"; }

echo "==================================================================="
echo "Phase G finalizer — started $(ts)"
echo "==================================================================="

# ---------------- step 1: wait for both evals -----------------
echo "[$(ts)] Waiting for main/eval/summary.json AND shuffled/eval/summary.json..."
while true; do
    if [ -f $RUNS/main/eval/summary.json ] && [ -f $RUNS/shuffled/eval/summary.json ]; then
        echo "[$(ts)] Both evals finished."
        ls -la $RUNS/main/eval/summary.json $RUNS/shuffled/eval/summary.json
        break
    fi
    sleep 60
done

# ---------------- step 2: kill the original orchestrator ----------
echo "[$(ts)] Killing the original phase_g_overnight orchestrator (taking over)..."
screen -S phase_g_overnight -X quit 2>&1 || true
sleep 5
echo "[$(ts)] screens after kill:"
screen -ls 2>&1

# Wait for any leftover python processes
sleep 30
echo "[$(ts)] Verifying GPUs idle:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

# Belt-and-suspenders: kill any remaining orchestrator-spawned processes
pkill -9 -f 'eval_diffusion_diagnostic' 2>&1 || true
pkill -9 -f 'mode shuffled' 2>&1 || true
sleep 10
echo "[$(ts)] After cleanup:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

# ---------------- step 3: launch synth training on all 8 GPUs -------
SYN_OUT=$RUNS/synthetic_positive
mkdir -p $SYN_OUT
echo "[$(ts)] Launching Run 3 (synthetic_positive): 15000 steps DDP=8"
cd $PROJECT
$TORCHRUN --nproc_per_node=8 scripts/phase_g/train_diffusion_diagnostic.py \
    --d2-dir $DATA_D2 --v10-dir $DATA_V10 \
    --out-dir $SYN_OUT --mode synthetic_positive \
    --max-steps 15000 --bs 2 --bf16 --T 1000 --base-ch 96 \
    --ckpt-every 5000 --log-every 50 --num-workers 2 \
    > $SYN_OUT/run.log 2>&1
echo "[$(ts)] Synth train exit=$?, last lines:"
tail -10 $SYN_OUT/run.log

# ---------------- step 4: synth eval -----------------
SYN_EVAL=$SYN_OUT/eval
mkdir -p $SYN_EVAL
echo "[$(ts)] Running synth eval..."
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/phase_g/eval_diffusion_diagnostic.py \
    --ckpt $SYN_OUT/model_final.pt \
    --d2-dir $DATA_D2 --v10-dir $DATA_V10 \
    --out $SYN_EVAL --n-d2 200 --n-v10 200 --bs 4 --bf16 \
    > $SYN_EVAL/run.log 2>&1
echo "[$(ts)] Synth eval exit=$?, last lines:"
tail -8 $SYN_EVAL/run.log

# ---------------- step 5: final report ---------------
echo "[$(ts)] Building final report..."
$PYTHON scripts/phase_g/build_final_report.py \
    --runs-root $RUNS --out $RUNS/report \
    > $RUNS/build_report.log 2>&1
echo "[$(ts)] Report build exit=$?, last lines:"
tail -10 $RUNS/build_report.log

touch $RUNS/FINALIZER_DONE
echo "==================================================================="
echo "Phase G finalizer — finished $(ts)"
echo "==================================================================="

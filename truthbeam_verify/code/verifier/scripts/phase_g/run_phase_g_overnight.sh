#!/usr/bin/env bash
# Phase G — overnight orchestrator for the diffusion diagnostic.
#
# Runs autonomously: waits for v1.5 ablation Run B to finish, runs a Lambda
# smoke test, then trains + evals all 3 modes sequentially. Each step's logs
# live in $RUNS_ROOT/<mode>/run.log; failures don't stop later steps.
#
# Operator is offline; this script must not require any human input.

set -u  # don't `set -e` — we want to continue past per-step failures

PROJECT=/path/to/poliebotics_phase_b/poliebotics_phase_b
# Data dirs: project-local symlinks → sessions/{d2,v10}, used by F-A v1 too
DATA_D2=$PROJECT/data/d2
DATA_V10=$PROJECT/data/v10
RUNS_ROOT=/path/to/poliebotics_phase_b/experiments/phase_g_diffusion_diagnostic
PYTHON=$PROJECT/.venv_a100/bin/python

mkdir -p $RUNS_ROOT
LOGFILE=$RUNS_ROOT/orchestrator.log
exec > >(tee -a $LOGFILE) 2>&1

ts() { date +"%Y-%m-%d %H:%M:%S UTC"; }

echo "==================================================================="
echo "Phase G overnight orchestrator — started $(ts)"
echo "==================================================================="

# ---------------- step 0: wait for v1.5 ablation Run B to finish ---------
ABLATION_LOG=/path/to/poliebotics_phase_b/experiments/phase_f/v1_5_ablation/v1_5_treatment/run.log
echo "[$(ts)] Waiting for v1.5 ablation Run B to finish..."
while true; do
    if [ -f "$ABLATION_LOG" ] && grep -q "^\[done\] elapsed=" "$ABLATION_LOG" 2>/dev/null; then
        echo "[$(ts)] Ablation Run B finished:"
        grep "^\[done\]" "$ABLATION_LOG" | tail -1
        break
    fi
    if [ -f "$ABLATION_LOG" ]; then
        echo "[$(ts)] Run B still going: $(tail -1 $ABLATION_LOG | head -c 120)"
    fi
    sleep 120
done
# Belt-and-suspenders: wait for nvidia-smi to settle to <30% util across all 8.
sleep 60
echo "[$(ts)] GPUs after ablation:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

# ---------------- step 1: Lambda-side smoke test (5 steps DDP=8) ---------
SMOKE_OUT=$RUNS_ROOT/data_smoke
mkdir -p $SMOKE_OUT
echo "[$(ts)] Lambda smoke test: 5 steps DDP=8 at full resolution..."
cd $PROJECT
$PROJECT/.venv_a100/bin/torchrun --nproc_per_node=8 scripts/phase_g/train_diffusion_diagnostic.py \
    --d2-dir $DATA_D2 --v10-dir $DATA_V10 \
    --out-dir $SMOKE_OUT --mode main \
    --max-steps 5 --bs 2 --bf16 --T 1000 --base-ch 96 \
    --ckpt-every 9999 --log-every 1 --num-workers 2 \
    > $SMOKE_OUT/run.log 2>&1
SMOKE_RC=$?
echo "[$(ts)] Smoke test exit code: $SMOKE_RC"
tail -15 $SMOKE_OUT/run.log
if [ $SMOKE_RC -ne 0 ]; then
    echo "[$(ts)] SMOKE TEST FAILED — aborting full pipeline. See $SMOKE_OUT/run.log for details."
    exit 1
fi

# ---------------- step 2: Run 1 — main ----------------
MAIN_OUT=$RUNS_ROOT/main
mkdir -p $MAIN_OUT
echo "[$(ts)] Launching Run 1 (main): 25000 steps DDP=8"
$PROJECT/.venv_a100/bin/torchrun --nproc_per_node=8 scripts/phase_g/train_diffusion_diagnostic.py \
    --d2-dir $DATA_D2 --v10-dir $DATA_V10 \
    --out-dir $MAIN_OUT --mode main \
    --max-steps 25000 --bs 2 --bf16 --T 1000 --base-ch 96 \
    --ckpt-every 5000 --log-every 50 --num-workers 2 \
    > $MAIN_OUT/run.log 2>&1
echo "[$(ts)] Run 1 finished, exit=$?, last lines:"
tail -10 $MAIN_OUT/run.log

echo "[$(ts)] Running eval on Run 1 (main)..."
MAIN_EVAL=$RUNS_ROOT/main/eval
mkdir -p $MAIN_EVAL
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/phase_g/eval_diffusion_diagnostic.py \
    --ckpt $MAIN_OUT/model_final.pt \
    --d2-dir $DATA_D2 --v10-dir $DATA_V10 \
    --out $MAIN_EVAL --n-d2 200 --n-v10 200 --bs 4 --bf16 \
    > $MAIN_EVAL/run.log 2>&1
echo "[$(ts)] Run 1 eval finished, exit=$?, last lines:"
tail -8 $MAIN_EVAL/run.log

# ---------------- step 3: Run 2 — shuffled ----------------
SHUF_OUT=$RUNS_ROOT/shuffled
mkdir -p $SHUF_OUT
echo "[$(ts)] Launching Run 2 (shuffled): 25000 steps DDP=8"
$PROJECT/.venv_a100/bin/torchrun --nproc_per_node=8 scripts/phase_g/train_diffusion_diagnostic.py \
    --d2-dir $DATA_D2 --v10-dir $DATA_V10 \
    --out-dir $SHUF_OUT --mode shuffled \
    --max-steps 25000 --bs 2 --bf16 --T 1000 --base-ch 96 \
    --ckpt-every 5000 --log-every 50 --num-workers 2 \
    > $SHUF_OUT/run.log 2>&1
echo "[$(ts)] Run 2 finished, exit=$?, last lines:"
tail -10 $SHUF_OUT/run.log

echo "[$(ts)] Running eval on Run 2 (shuffled)..."
SHUF_EVAL=$RUNS_ROOT/shuffled/eval
mkdir -p $SHUF_EVAL
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/phase_g/eval_diffusion_diagnostic.py \
    --ckpt $SHUF_OUT/model_final.pt \
    --d2-dir $DATA_D2 --v10-dir $DATA_V10 \
    --out $SHUF_EVAL --n-d2 200 --n-v10 200 --bs 4 --bf16 \
    > $SHUF_EVAL/run.log 2>&1
echo "[$(ts)] Run 2 eval finished, exit=$?, last lines:"
tail -8 $SHUF_EVAL/run.log

# ---------------- step 4: Run 3 — synthetic_positive ----------------
SYN_OUT=$RUNS_ROOT/synthetic_positive
mkdir -p $SYN_OUT
echo "[$(ts)] Launching Run 3 (synthetic_positive): 15000 steps DDP=8"
$PROJECT/.venv_a100/bin/torchrun --nproc_per_node=8 scripts/phase_g/train_diffusion_diagnostic.py \
    --d2-dir $DATA_D2 --v10-dir $DATA_V10 \
    --out-dir $SYN_OUT --mode synthetic_positive \
    --max-steps 15000 --bs 2 --bf16 --T 1000 --base-ch 96 \
    --ckpt-every 5000 --log-every 50 --num-workers 2 \
    > $SYN_OUT/run.log 2>&1
echo "[$(ts)] Run 3 finished, exit=$?, last lines:"
tail -10 $SYN_OUT/run.log

echo "[$(ts)] Running eval on Run 3 (synthetic_positive)..."
SYN_EVAL=$RUNS_ROOT/synthetic_positive/eval
mkdir -p $SYN_EVAL
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/phase_g/eval_diffusion_diagnostic.py \
    --ckpt $SYN_OUT/model_final.pt \
    --d2-dir $DATA_D2 --v10-dir $DATA_V10 \
    --out $SYN_EVAL --n-d2 200 --n-v10 200 --bs 4 --bf16 \
    > $SYN_EVAL/run.log 2>&1
echo "[$(ts)] Run 3 eval finished, exit=$?, last lines:"
tail -8 $SYN_EVAL/run.log

# ---------------- step 4.5: backfill any eval that didn't run ----------------
# Defensive — covers the 2026-05-01 main-eval bug where a missing mkdir caused
# eval to be skipped. If summary.json is missing for any run that has a
# model_final.pt, retry the eval from scratch.
for run_name in main shuffled synthetic_positive; do
    RUN_DIR=$RUNS_ROOT/$run_name
    EVAL_DIR=$RUN_DIR/eval
    if [ -f "$RUN_DIR/model_final.pt" ] && [ ! -f "$EVAL_DIR/summary.json" ]; then
        echo "[$(ts)] Backfilling missing eval for $run_name..."
        mkdir -p $EVAL_DIR
        CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/phase_g/eval_diffusion_diagnostic.py \
            --ckpt $RUN_DIR/model_final.pt \
            --d2-dir $DATA_D2 --v10-dir $DATA_V10 \
            --out $EVAL_DIR --n-d2 200 --n-v10 200 --bs 4 --bf16 \
            > $EVAL_DIR/run.log 2>&1
        echo "[$(ts)] Backfill eval for $run_name exit=$?:"
        tail -8 $EVAL_DIR/run.log
    fi
done

# ---------------- step 5: build the final report ----------------
echo "[$(ts)] Assembling final diagnostic_report.md..."
REPORT_OUT=$RUNS_ROOT/report
$PYTHON $PROJECT/scripts/phase_g/build_final_report.py \
    --runs-root $RUNS_ROOT --out $REPORT_OUT \
    > $RUNS_ROOT/build_report.log 2>&1
echo "[$(ts)] Report build exit=$?, last lines:"
tail -8 $RUNS_ROOT/build_report.log

# ---------------- step 6: write a "done" marker ----------------
echo "[$(ts)] All 3 runs + evals + report attempted. Touching done marker."
touch $RUNS_ROOT/ORCHESTRATOR_DONE
echo "==================================================================="
echo "Phase G overnight orchestrator — finished $(ts)"
echo "==================================================================="

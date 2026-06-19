#!/usr/bin/env bash
# F-A v1 cross-verifier eval — run F-A v1 (step 100k) outputs through D2-only
# and V10-only Phase G verifiers, in addition to the combined verifier (already
# in stage_0/eval/).
#
# Strategy: invoke the existing audited stage_0/eval.py with --diffusion-ckpt
# pointing to LOSO verifier ckpts. F-A inference is deterministic, so C_fake
# is identical across runs; only the verifier scoring differs.
#
# Output dirs:
#   experiments/stage_0_cross_verifier/d2_only_verifier/step_*/
#   experiments/stage_0_cross_verifier/v10_only_verifier/step_*/
#
# Logs to experiments/stage_0_cross_verifier/run.log
#
# Pre-flight gates:
#   1. LOSO verifier ckpts exist
#   2. F-A v1 ckpt exists
#   3. Combined Stage 0 NPZs exist (so we have a baseline to compare against)
#   4. Phase H training has completed (PHASE_H_RUNNER_DONE sentinel)

set -uo pipefail

PROJECT=/path/to/poliebotics_phase_b/poliebotics_phase_b
EXPER_ROOT=/path/to/poliebotics_phase_b/experiments
PYTHON=$PROJECT/.venv_a100/bin/python
TORCHRUN=$PROJECT/.venv_a100/bin/torchrun

DATA_D2=$PROJECT/data/d2
DATA_V10=$PROJECT/data/v10
FA_CKPT=$EXPER_ROOT/phase_f/f_a_full_v1/checkpoints/step_00100000.pt

D2_ONLY_VERIFIER=$EXPER_ROOT/cross_session_ablation/d2_only/model_final.pt
V10_ONLY_VERIFIER=$EXPER_ROOT/cross_session_ablation/v10_only/model_final.pt
OUT_ROOT=$EXPER_ROOT/stage_0_cross_verifier
LOG=$OUT_ROOT/run.log
SENTINEL=/path/to/poliebotics_phase_b/orchestration/PHASE_H_RUNNER_DONE

mkdir -p $OUT_ROOT
exec >> $LOG 2>&1
ts() { date +"%Y-%m-%d %H:%M:%S UTC"; }
log() { echo "[$(ts)] $*"; }

log "==================================================================="
log "cross_verifier_eval.sh starting"
log "==================================================================="

# ---- Gate 1: Phase H sentinel must exist (don't run while DDP=8 is training) ----
if [ ! -f $SENTINEL ]; then
    log "ERROR: PHASE_H_RUNNER_DONE sentinel missing — Phase H still running. Aborting."
    exit 1
fi
log "Phase H sentinel present — GPUs are free"

# ---- Gate 2: LOSO + F-A ckpts exist ----
for f in "$D2_ONLY_VERIFIER" "$V10_ONLY_VERIFIER" "$FA_CKPT"; do
    if [ ! -f "$f" ]; then
        log "ERROR: required ckpt missing: $f"
        exit 1
    fi
done
log "All required ckpts present"

# ---- Gate 3: Combined Stage 0 NPZs exist (baseline) ----
COMBINED_BASELINE=$EXPER_ROOT/stage_0/eval/step_00100000/stage0_d2_raw.npz
if [ ! -f $COMBINED_BASELINE ]; then
    log "ERROR: combined Stage 0 NPZ missing at $COMBINED_BASELINE — no baseline to compare"
    exit 1
fi
log "Combined Stage 0 NPZ present"

# ---- Run cross-verifier evals (8-way sharded across GPUs) ----
run_one_verifier() {
    local verifier_name=$1
    local verifier_path=$2
    local out_dir=$OUT_ROOT/${verifier_name}/step_00100000
    mkdir -p $out_dir
    log "Running with $verifier_name verifier ($verifier_path) — 8-way sharded"
    pids=()
    for shard in 0 1 2 3 4 5 6 7; do
        CUDA_VISIBLE_DEVICES=$shard \
        $PYTHON $PROJECT/scripts/stage_0/eval.py \
            --diffusion-ckpt $verifier_path \
            --fa-ckpt $FA_CKPT \
            --d2-dir $DATA_D2 \
            --v10-dir $DATA_V10 \
            --out $out_dir \
            --n-d2 200 --n-v10 200 \
            --bs 4 --bf16 \
            --shard $shard --num-shards 8 \
            > $out_dir/shard_${shard}.log 2>&1 &
        pids+=($!)
    done
    log "  launched 8 shards (pids: ${pids[*]}); waiting..."
    rc=0
    for pid in "${pids[@]}"; do
        wait $pid || rc=1
    done
    if [ $rc -ne 0 ]; then
        log "  ERROR: at least one shard failed for $verifier_name"
        return 1
    fi
    log "  all 8 shards completed; merging"
    $PYTHON $PROJECT/scripts/stage_0/merge_shards.py --eval-dir $out_dir \
        > $out_dir/merge.log 2>&1
    if [ $? -ne 0 ]; then
        log "  ERROR: merge_shards failed for $verifier_name"
        return 1
    fi
    log "  $verifier_name verifier eval merged → $out_dir/summary.json"
}

run_one_verifier "d2_only_verifier" "$D2_ONLY_VERIFIER" || exit 1
run_one_verifier "v10_only_verifier" "$V10_ONLY_VERIFIER" || exit 1

# ---- Aggregation report ----
log "Building cross-verifier comparison report"
$PYTHON $PROJECT/scripts/cross_verifier_report.py \
    --combined-eval $EXPER_ROOT/stage_0/eval \
    --d2-verifier-eval $OUT_ROOT/d2_only_verifier \
    --v10-verifier-eval $OUT_ROOT/v10_only_verifier \
    --ckpt-step 100000 \
    --out $OUT_ROOT/cross_verifier_report.md \
    > $OUT_ROOT/report.log 2>&1
if [ $? -ne 0 ]; then
    log "ERROR: report builder failed"
    exit 1
fi
log "Report → $OUT_ROOT/cross_verifier_report.md"

# Codex audit MED 2026-05-04: g1a mirror happens via the cron job ON g1a
# (every 30 min, sources Lambda → g1a). Lambda has no creds to push to g1a,
# so we cannot force the mirror from this side. Sentinel touch is the
# "Lambda-side complete" signal; the host side will trigger the mirror
# from g1a after seeing the sentinel.
log "Note: g1a cron mirror picks up new files within 30 min; one can trigger a "
log "      manual mirror from g1a side at next monitoring wakeup."

log "==================================================================="
log "cross_verifier_eval.sh complete"
log "==================================================================="
touch $OUT_ROOT/CROSS_VERIFIER_DONE

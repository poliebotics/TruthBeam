#!/usr/bin/env bash
# Auto-chains wave 2 onto wave 1: when each wave-1 experiment finishes, the
# corresponding wave-2 experiment is launched on the same GPUs.
#
#   A4 (GPUs 0,1)  → A1 on GPUs 0,1   (port 29411)
#   A0 (GPUs 2,3)  → A7 on GPUs 2,3   (port 29412)
#   A2 (GPUs 4-7)  → A6 on GPUs 4-7   (port 29413)
#
# Designed to be launched as a long-running background process AFTER
# launch_wave1.sh has been started.
set -euo pipefail
cd "$(dirname "$0")/.."

STATS_DIR=/path/to/poliebotics_phase_b/cache/normalization_stats

# Wait for an experiment's run.log to record EXIT. Returns 0 if exp exited
# successfully (EXIT= 0), non-zero otherwise.
wait_for_exit() {
  local exp="$1"
  local logf="experiments/$exp/run.log"
  until grep -qE '^EXIT=' "$logf" 2>/dev/null; do
    sleep 60
  done
  local exit_line
  exit_line=$(grep -E '^EXIT=' "$logf" | head -1)
  echo "[chain] $exp finished: $exit_line"
  local code
  code=$(echo "$exit_line" | grep -oE '[0-9]+' | head -1)
  return "${code:-1}"
}

# Launch one experiment on the specified GPUs in a screen session
launch_exp() {
  local exp="$1" gpus="$2" port="$3" nproc="$4"
  mkdir -p "experiments/$exp"
  screen -dmS "exp_$exp" bash -lc "
cd $(pwd)
. .venv_a100/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=$gpus torchrun --standalone --nproc_per_node=$nproc --master_port=$port \
  src/training/train_phase_d.py --config configs/$exp.yaml --stats-dir $STATS_DIR --bf16 \
  2>&1 | tee experiments/$exp/run.log
echo 'EXIT=' \$? >> experiments/$exp/run.log
"
  echo "[chain] launched $exp on GPUs $gpus (port $port, nproc $nproc)"
}

# Pair-launch: when wave-1 exp finishes, launch wave-2 exp on the same GPUs.
# Skip wave-2 launch if wave-1 failed (exit non-zero) — operator's "no GPU
# contention from cascading failed experiments" rule.
chain() {
  local w1="$1" w2="$2" gpus="$3" port="$4" nproc="$5"
  if wait_for_exit "$w1"; then
    launch_exp "$w2" "$gpus" "$port" "$nproc"
  else
    echo "[chain] SKIP $w2 — $w1 exited non-zero"
  fi
}

chain exp001h_a4 exp001h_a1 '0,1' 29411 2 &
chain exp001h_a0 exp001h_a7 '2,3' 29412 2 &
chain exp001h_a2 exp001h_a6 '4,5,6,7' 29413 4 &

wait
echo "[chain] all wave-2 experiments launched (or skipped if wave-1 failed)"

#!/usr/bin/env bash
# Launch wave 2 of the Phase D 6-experiment matrix.
# Run after wave 1 completes (or per-pair, as each wave-1 experiment finishes).
#
# Wave 2:
#   GPUs 0-1: A1 (Tiny + all-oct XOF, DDP=2)         — port 29411
#   GPUs 2-3: A7 (Tiny + STN + all-oct XOF, DDP=2)   — port 29412
#   GPUs 4-7: A6 (Large + D2-only XOF, DDP=4)        — port 29413
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STATS_DIR=/path/to/poliebotics_phase_b/cache/normalization_stats

mkdir -p experiments/exp001h_a1 experiments/exp001h_a7 experiments/exp001h_a6

# A1 — GPUs 0,1
screen -dmS exp_a1 bash -lc "
cd $(pwd)
. .venv_a100/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 --master_port=29411 \
  src/training/train_phase_d.py --config configs/exp001h_a1.yaml --stats-dir $STATS_DIR --bf16 \
  2>&1 | tee experiments/exp001h_a1/run.log
echo 'EXIT=' \$? >> experiments/exp001h_a1/run.log
"

# A7 — GPUs 2,3
screen -dmS exp_a7 bash -lc "
cd $(pwd)
. .venv_a100/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 --master_port=29412 \
  src/training/train_phase_d.py --config configs/exp001h_a7.yaml --stats-dir $STATS_DIR --bf16 \
  2>&1 | tee experiments/exp001h_a7/run.log
echo 'EXIT=' \$? >> experiments/exp001h_a7/run.log
"

# A6 — GPUs 4,5,6,7
screen -dmS exp_a6 bash -lc "
cd $(pwd)
. .venv_a100/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 --master_port=29413 \
  src/training/train_phase_d.py --config configs/exp001h_a6.yaml --stats-dir $STATS_DIR --bf16 \
  2>&1 | tee experiments/exp001h_a6/run.log
echo 'EXIT=' \$? >> experiments/exp001h_a6/run.log
"

sleep 3
screen -ls 2>&1 | grep -E 'exp_'

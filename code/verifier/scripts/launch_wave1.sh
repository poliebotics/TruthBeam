#!/usr/bin/env bash
# Launch wave 1 of the Phase D 6-experiment matrix on the 8×A100 instance.
# Each experiment runs in its own screen session with torchrun-managed DDP.
#
# Wave 1:
#   GPUs 0-1: A4 (Tiny + emission, DDP=2)        — port 29401
#   GPUs 2-3: A0 (Tiny + oct0+oct1 XOF, DDP=2)   — port 29402
#   GPUs 4-7: A2 (Large + all-oct XOF, DDP=4)    — port 29403
set -euo pipefail
cd "$(dirname "$0")/.."

# Common env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STATS_DIR=/path/to/poliebotics_phase_b/cache/normalization_stats

mkdir -p experiments/exp001h_a4 experiments/exp001h_a0 experiments/exp001h_a2

# A4 — GPUs 0,1
screen -dmS exp_a4 bash -lc "
cd $(pwd)
. .venv_a100/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 --master_port=29401 \
  src/training/train_phase_d.py --config configs/exp001h_a4.yaml --stats-dir $STATS_DIR --bf16 \
  2>&1 | tee experiments/exp001h_a4/run.log
echo 'EXIT=' \$? >> experiments/exp001h_a4/run.log
"

# A0 — GPUs 2,3
screen -dmS exp_a0 bash -lc "
cd $(pwd)
. .venv_a100/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 --master_port=29402 \
  src/training/train_phase_d.py --config configs/exp001h_a0.yaml --stats-dir $STATS_DIR --bf16 \
  2>&1 | tee experiments/exp001h_a0/run.log
echo 'EXIT=' \$? >> experiments/exp001h_a0/run.log
"

# A2 — GPUs 4,5,6,7
screen -dmS exp_a2 bash -lc "
cd $(pwd)
. .venv_a100/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 --master_port=29403 \
  src/training/train_phase_d.py --config configs/exp001h_a2.yaml --stats-dir $STATS_DIR --bf16 \
  2>&1 | tee experiments/exp001h_a2/run.log
echo 'EXIT=' \$? >> experiments/exp001h_a2/run.log
"

sleep 3
screen -ls 2>&1 | grep -E 'exp_'

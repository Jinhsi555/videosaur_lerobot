#!/usr/bin/env bash
set -euo pipefail

cd /mnt/workspace/wlb/videosaur_lerobot
source env_exports/activate_videosaur_ppu.sh

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/workspace/wlb/videosaur_lerobot/.cache}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-${XDG_CACHE_HOME}/torch}"
export HF_HUB_DISABLE_TELEMETRY=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# torch.distributed.run launch variables. Override these in the PAI/DLC job spec for multi-node runs.
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-${GROUP_RANK:-0}}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${LOCAL_WORLD_SIZE:-1}}"
export WORLD_SIZE="${WORLD_SIZE:-$((NNODES * NPROC_PER_NODE))}"

mkdir -p "${HUGGINGFACE_HUB_CACHE}"
mkdir -p "${TORCH_HOME}"

echo "torch.distributed.run: nnodes=${NNODES} node_rank=${NODE_RANK} nproc_per_node=${NPROC_PER_NODE} master=${MASTER_ADDR}:${MASTER_PORT} world_size=${WORLD_SIZE}"

python -m torch.distributed.run \
  --nnodes="${NNODES}" \
  --node-rank="${NODE_RANK}" \
  --nproc-per-node="${NPROC_PER_NODE}" \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT}" \
  -m videosaur.train \
  configs/videosaur/lerobot_mixed_dataset.yml \
  trainer.devices="${NPROC_PER_NODE}" \
  globals.BATCH_SIZE_PER_GPU=32 \
  globals.NUM_GPUS="${WORLD_SIZE}"

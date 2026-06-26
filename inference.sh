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

CONFIG="${CONFIG:-configs/inference/movi_c.yml}"
INPUT_PATH="${1:-data/libero}"
OUTPUT_DIR="${2:-inference_output/mix_dataset_50k}"
OVERWRITE="${OVERWRITE:-0}"
INTERACTIVE="${INTERACTIVE:-1}"
SERVE_VIEWER="${SERVE_VIEWER:-0}"
VIEWER_HOST="${VIEWER_HOST:-0.0.0.0}"
VIEWER_PORT="${VIEWER_PORT:-8000}"

mkdir -p "${HUGGINGFACE_HUB_CACHE}"
mkdir -p "${TORCH_HOME}"
mkdir -p "${OUTPUT_DIR}"

if [[ -d "${INPUT_PATH}" ]]; then
  mapfile -t VIDEOS < <(find "${INPUT_PATH}" -maxdepth 1 -type f -name '*.mp4' | sort)
elif [[ -f "${INPUT_PATH}" ]]; then
  VIDEOS=("${INPUT_PATH}")
else
  echo "Input path does not exist: ${INPUT_PATH}" >&2
  exit 1
fi

if [[ "${#VIDEOS[@]}" -eq 0 ]]; then
  echo "No .mp4 files found in: ${INPUT_PATH}" >&2
  exit 1
fi

echo "Config: ${CONFIG}"
echo "Input: ${INPUT_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Videos: ${#VIDEOS[@]}"
echo "Interactive viewer: ${INTERACTIVE}"

for video_path in "${VIDEOS[@]}"; do
  video_stem="$(basename "${video_path}")"
  video_stem="${video_stem%.*}"
  output_path="${OUTPUT_DIR%/}/${video_stem}-mask-5-slots-50000step.mp4"
  viewer_path="${OUTPUT_DIR%/}/${video_stem}-viewer"

  if [[ -e "${output_path}" && "${OVERWRITE}" != "1" ]]; then
    echo "Skip existing: ${output_path}"
    continue
  fi

  echo "Run inference: ${video_path} -> ${output_path}"
  python -m videosaur.inference \
    --config "${CONFIG}" \
    "input.path=${video_path}" \
    "output.save_path=${output_path}" \
    "output.interactive.enabled=${INTERACTIVE}" \
    "output.interactive.viewer_dir=${viewer_path}"
done

if [[ "${SERVE_VIEWER}" == "1" ]]; then
  python -m videosaur.interactive_viewer "${OUTPUT_DIR}" \
    --host "${VIEWER_HOST}" \
    --port "${VIEWER_PORT}"
fi

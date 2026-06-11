cd /mnt/workspace/wlb/videosaur

HF_ENDPOINT=https://hf-mirror.com \
TIMM_USE_OLD_CACHE=1 \
TORCH_HOME=/mnt/workspace/wlb/videosaur/.cache/torch \
NCCL_DEBUG=WARN \
/root/miniconda3/envs/lerobot/bin/python -m videosaur.train \
  configs/videosaur/lerobot_something_something_v2.yml \
  trainer.devices=1 \
  globals.BATCH_SIZE_PER_GPU=32 \
  globals.NUM_GPUS=1
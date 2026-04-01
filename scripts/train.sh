#!/usr/bin/env bash
source attn_ft/bin/activate
CONFIG_PATH=configs/llama3_2_11b_vision_attn_ft.yaml

GPU_LIST=${AVAILABLE_GPUS:-${CUDA_VISIBLE_DEVICES:-}}
if [ -n "$GPU_LIST" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_LIST"
fi

NUM_VISIBLE_GPUS=$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)

# python src/attn_ft/train.py --config configs/qwen3_vl_8B.yaml || echo "Training failed, proceeding to destroy..."
if [ "$NUM_VISIBLE_GPUS" -gt 1 ]; then
    accelerate launch --multi_gpu --num_processes "$NUM_VISIBLE_GPUS" src/attn_ft/train.py --config "$CONFIG_PATH" || echo "Training failed, proceeding to destroy..."
else
    python src/attn_ft/train.py --config "$CONFIG_PATH" || echo "Training failed, proceeding to destroy..."
fi
if [ -f "$DATA_DIRECTORY/keep_alive" ]; then
    echo "Lock file found. Staying online."
else
    vastai destroy instance $CONTAINER_ID
fi
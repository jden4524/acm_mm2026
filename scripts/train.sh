#!/usr/bin/env bash
source attn_ft/bin/activate
# python src/attn_ft/train.py --config configs/qwen3_vl_8B.yaml || echo "Training failed, proceeding to destroy..."
accelerate launch src/attn_ft/train.py --config configs/llama3_2_11b_vision_attn_ft.yaml || echo "Training failed, proceeding to destroy..."
if [ -f "$DATA_DIRECTORY/keep_alive" ]; then
    echo "Lock file found. Staying online."
else
    vastai destroy instance $CONTAINER_ID
fi
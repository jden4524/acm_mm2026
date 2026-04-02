# Mind What Matters for Reasoning: Aligning Cross-Modal Attention via Probability Mass Concentration

This project fine-tunes selected attention heads in a VLM using segmentation masks aligned to caption phrases.

## Quickstart

1. Create and activate the environment by running setup_env.sh
   - `bash scripts/setup_env.sh`
2. Edit the config as needed:
   - [configs/qwen3_vl_8b_attn_ft.yaml](configs/qwen3_vl_8b_attn_ft.yaml)
3. Launch training:
   - `bash scripts/train.sh`

## Notes

- The dataset is loaded from [data/flickr30k_sam](data/flickr30k_sam), which contains preprocessed images, captions, and segmentation masks.

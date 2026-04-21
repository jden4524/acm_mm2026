# Mind What Matters for Reasoning: Aligning Cross-Modal Attention via Probability Mass Concentration

This project tunes selected attention heads in a VLM using segmentation masks to improve baseline models' general reasoning abilities.

## Quickstart

1. Create and activate the environment by running setup_env.sh
   - `bash scripts/setup_env.sh`
2. Edit the config as needed:
   - [configs/config_template.yaml](configs/config_template.yaml)
3. Launch training:
   - `bash scripts/train.sh`

## Notes

- The dataset is loaded from huggingface (the link will be released after review due to double blind policy), which contains preprocessed images, captions, and segmentation masks.

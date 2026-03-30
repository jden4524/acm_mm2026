cd $DATA_DIRECTORY
git clone https://github.com/jden4524/attn_ft.git
cd attn_ft
bash scripts/setup_env.sh
cd eval
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
pip install -e .
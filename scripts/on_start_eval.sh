cd $DATA_DIRECTORY
git clone https://github.com/jden4524/attn_ft.git
cd attn_ft
cd eval
git clone https://github.com/hengzhan/VLMEvalKit.git
cd VLMEvalKit
uv venv .venv
uv pip install --python .venv/bin/python -e .
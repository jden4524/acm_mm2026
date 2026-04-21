#!/usr/bin/env bash
set -euo pipefail

uv venv --python 3.10 attn_ft
uv pip install --python attn_ft/bin/python -e .
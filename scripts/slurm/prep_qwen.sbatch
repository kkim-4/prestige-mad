#!/bin/bash
#SBATCH --job-name=prestige-prep
#SBATCH --partition=h100hour
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:50:00
#SBATCH --output=/local/scratch/%u/prestige-mad/logs/%x-%j.out
# CPU only: build the env and download the two models, so no GPU waits on this.
set -eo pipefail
. /local/scratch/$USER/prestige-mad/code/scripts/slurm/env_h100.sh
UV=$HOME/.local/bin/uv
[[ -x "$PROJ/venv/bin/python" ]] || "$UV" venv --python 3.11 "$PROJ/venv"
"$UV" pip install --python "$PROJ/venv/bin/python" vllm==0.33.0 pydantic sympy openai matplotlib google-genai
"$UV" pip freeze --python "$PROJ/venv/bin/python" > "$PROJ/logs/packages-$SLURM_JOB_ID.txt"
"$PROJ/venv/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
for m in ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-3B-Instruct"):
    print(m, snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "merges*", "*.jinja"]))
PY
"$PROJ/venv/bin/python" prestige_mad_v2.py --mock --n 20 --store "$PROJ/logs/mock-$SLURM_JOB_ID.jsonl"
du -sh "$PROJ"/*

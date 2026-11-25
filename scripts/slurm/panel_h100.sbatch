#!/bin/bash
#SBATCH --job-name=prestige-panel
#SBATCH --partition=h100hour
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=00:58:00
#SBATCH --gres=gpu:1
#SBATCH --signal=B:TERM@120
#SBATCH --output=/local/scratch/%u/prestige-mad/logs/%x-%j.out
# The qwen_peers / mixed stages of scripts/run_prestige_v2.sh, five models on ONE H100.
#   sbatch --job-name=prestige-peers --export=ALL,STAGE=peers panel_h100.sbatch
#   sbatch --job-name=prestige-mixed --export=ALL,STAGE=mixed panel_h100.sbatch
# Same python command as run_prestige_v2.sh. Differences: N=600, WARMUP=50 passed explicitly (its own defaults
# resolve to 120/20); servers started here, with a fixed KV cache each so five fit in 80 GB (no CUDA_VISIBLE_DEVICES).
# Round-1 pilot (n=100) runs first into its own store. Resumable: resubmit to continue.
# New weights go to a per-job directory under scratch that is removed when the job ends.
set -eo pipefail
. /local/scratch/$USER/prestige-mad/code/scripts/slurm/env_h100.sh
. "$PROJ/venv/bin/activate"
SHARED=/local/scratch/$USER/.cache/huggingface/hub       # has Qwen3-8B (read only)
KEEP="$PROJ/hf/hub"                                       # has Qwen2.5-7B, gemma-3-4b-it
JOBHF="$SCRATCH/tmp/prestige-hf-$SLURM_JOB_ID"; mkdir -p "$JOBHF/hub" runs
V="$PROJ/logs/vllm-$SLURM_JOB_ID"; mkdir -p "$V"
export HF_HOME="$JOBHF" HF_HUB_CACHE="$JOBHF/hub"
PIDS=()
cleanup() { kill "${PIDS[@]}" 2>/dev/null || true; wait 2>/dev/null || true; rm -rf "$JOBHF"; }
find "$SCRATCH/tmp" -maxdepth 1 -name 'prestige-hf-*' ! -name "*-$SLURM_JOB_ID" -exec sh -c 'for d; do squeue -h -j "${d##*-}" 2>/dev/null | grep -q . || rm -rf "$d"; done' _ {} +   # left by a killed job
trap cleanup EXIT; trap 'exit 143' TERM INT

case "${STAGE:?}" in
  peers) PANEL=panels/qwen_peers.json; NAME=prestige_qwen_peers_math45
    # 65.8 GiB of weights: KV caches are the minimum for one 8,192-token sequence, small batches
    MODELS=("Qwen/Qwen2-7B-Instruct 8011 2400M" "Qwen/Qwen2.5-7B-Instruct 8012 520M" "Qwen/Qwen2.5-Coder-7B-Instruct 8013 520M"
            "Qwen/Qwen3-4B-Instruct-2507 8014 1300M" "Qwen/Qwen3-8B 8015 1300M"); SEQS=32; BATCH=1024; EAGER=--enforce-eager ;;
  mixed) PANEL=panels/mixed_families.json; NAME=prestige_mixed_math45
    MODELS=("Qwen/Qwen3-8B 8021 2600M" "mistralai/Ministral-3-8B-Instruct-2512 8022 2600M" "nvidia/Llama-3.1-Nemotron-Nano-8B-v1 8023 2600M"
            "google/gemma-3-4b-it 8024 2600M" "microsoft/Phi-4-mini-instruct 8025 2600M"); SEQS=64; BATCH=2048 ;;
  *) echo "STAGE must be peers or mixed"; exit 1 ;;
esac
python - "${MODELS[@]}" <<'PY'
import socket, sys
for m in sys.argv[1:]:
    with socket.socket() as s: s.bind(("127.0.0.1", int(m.split()[1])))
PY
for line in "${MODELS[@]}"; do            # link what is already on disk, download the rest
  d=models--$(echo "${line%% *}" | sed 's#/#--#')
  for src in "$KEEP" "$SHARED"; do [[ -d "$src/$d" && ! -e "$JOBHF/hub/$d" ]] && ln -s "$src/$d" "$JOBHF/hub/$d"; done
done
date; nvidia-smi -L
timeout 8m python - "${MODELS[@]}" <<'PY'
import os, sys
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import snapshot_download
def get(line):
    m = line.split()[0]
    if os.path.islink(os.path.join(os.environ["HF_HUB_CACHE"], "models--" + m.replace("/", "--"))):
        return m, "on disk"
    pats = ["params.json", "tekken.json", "config.json", "generation_config.json", "SYSTEM_PROMPT.txt", "consolidated.safetensors"] \
        if m.startswith("mistralai/") else ["*.json", "model*.safetensors", "*.txt", "*.model", "*.jinja", "*.py"]
    return m, snapshot_download(m, allow_patterns=pats)
with ThreadPoolExecutor(3) as ex:
    for r in ex.map(get, sys.argv[1:]): print(*r, flush=True)
PY
export HF_HUB_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True; date
serve() {   # model port kv_bytes
  local extra=()
  case "$1" in
    mistralai/*) extra=(--tokenizer-mode mistral --config-format mistral --load-format mistral --limit-mm-per-prompt '{"image":0}') ;;
    google/*) extra=(--limit-mm-per-prompt '{"image":0}') ;;
  esac
  vllm serve "$1" --host 127.0.0.1 --port "$2" --max-model-len 8192 --gpu-memory-utilization 0.08 \
    --kv-cache-memory-bytes "$3" --max-num-seqs "$SEQS" --max-num-batched-tokens "$BATCH" $EAGER "${extra[@]}" > "$V/$2.log" 2>&1 &
  local pid=$!; PIDS+=("$pid")
  for _ in $(seq 120); do
    curl -fsS "http://127.0.0.1:$2/v1/models" 2>/dev/null | grep -q "$1" && { echo "ready $1 $(date +%T)"; return; }
    kill -0 "$pid" 2>/dev/null || { echo "server $1 died"; tail -40 "$V/$2.log"; exit 1; }
    sleep 3
  done
  echo "server $1 timed out"; tail -40 "$V/$2.log"; exit 1
}
for line in "${MODELS[@]}"; do serve $line; done
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
run() {   # n policies store
  timeout "${4:-40m}" python prestige_mad_v3.py --panel-file "$PANEL" --problems logs/math_extra.jsonl --level 4,5 \
    --n "$1" --warmup 50 --rounds 3 --workers 24 --stagger 0 --max-retries 8 --max-tokens 2048 \
    --policies "$2" --log "$PROJ/logs/calls_$STAGE.jsonl" --store "$3" &
  wait $!
}
[[ -n "$SKIP_PILOT" || -s runs/${NAME}_n100_pilot_D.png ]] || run 100 none "runs/${NAME}_n100_pilot.jsonl" 12m || echo "pilot incomplete; going on"
date
run 600 prestige,all_speak,none "runs/${NAME}_n600.jsonl" "$([[ -n "$SKIP_PILOT" ]] && echo 47m || echo 38m)"
date

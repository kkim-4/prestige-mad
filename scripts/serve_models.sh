#!/usr/bin/env bash
# serve_models.sh — start several vLLM servers (one per model) and wait until all answer.
# Each line of the spec: MODEL PORT GPU MEMORY_FRACTION
#   bash scripts/serve_models.sh "Qwen/Qwen2.5-7B-Instruct 8003 0 0.25" "Qwen/Qwen2.5-32B-Instruct 8005 1 0.90"
# Servers keep running after this script exits; stop them with:  pkill -f "vllm serve"
set -euo pipefail
cd "$(dirname "$0")/.."   # run from the repo root
MAXLEN=${MAXLEN:-8192}
mkdir -p logs/vllm
for line in "$@"; do
  read -r MODEL PORT GPU FRAC <<<"$line"
  if curl -s "localhost:$PORT/v1/models" >/dev/null 2>&1; then echo "port $PORT already serving"; continue; fi
  TAG=$(echo "$MODEL" | tr '/:' '__')
  echo "== $MODEL on GPU $GPU, port $PORT, memory $FRAC (log logs/vllm/$TAG.log)"
  CUDA_VISIBLE_DEVICES=$GPU nohup vllm serve "$MODEL" --port "$PORT" --max-model-len "$MAXLEN" \
    --gpu-memory-utilization "$FRAC" > "logs/vllm/$TAG.log" 2>&1 &
  sleep 20      # stagger start-ups so servers sharing a GPU measure free memory one at a time
done
for line in "$@"; do
  read -r MODEL PORT GPU FRAC <<<"$line"
  for i in $(seq 1 240); do
    curl -s "localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "ready: $MODEL ($PORT)"; break; }
    sleep 5
    [[ $i == 240 ]] && { echo "timed out: $MODEL; see logs/vllm/"; exit 1; }
  done
done

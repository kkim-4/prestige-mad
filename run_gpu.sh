#!/usr/bin/env bash
# run_gpu.sh — the same experiment with an open model on a GPU, served by vLLM.
#
# One-time setup on the GPU machine (Linux + NVIDIA GPU):
#   pip install vllm openai google-genai pydantic sympy datasets
#   huggingface-cli login            # only for gated models (Gemma, Llama) and GPQA
#   python3 make_math_extra.py --exclude math.jsonl --out math_extra.jsonl
#
# Run (defaults: Qwen2.5-7B-Instruct, the pre-registered 600 MATH level 4-5 problems, all ablations):
#   bash run_gpu.sh
#   MODEL=google/gemma-3-12b-it bash run_gpu.sh
#   MODEL=Qwen/Qwen3-8B EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}' bash run_gpu.sh
#   PROBLEMS=gpqa_extended.jsonl N=546 LEVEL= SEEDS=3 bash run_gpu.sh   # GPQA (python3 make_gpqa.py --subset extended)
#
# Calls cost nothing per token here; the limit is GPU time. Results are resumable: rerun the
# same command after a crash. Each model gets its own store (named after the model).
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
PORT=${PORT:-8000}
PROBLEMS=${PROBLEMS:-math_extra.jsonl}
N=${N:-600}
LEVEL=${LEVEL-4,5}
POLICIES=${POLICIES:-round_robin,dual,seq_dual,diss_only,lead_rand,dual_randdiss,rr_dual}
SEEDS=${SEEDS:-1}
WORKERS=${WORKERS:-32}
MAXLEN=${MAXLEN:-8192}
EXTRA=${EXTRA:-}
TAG=$(echo "$MODEL" | tr '/:' '__')
STORE=${STORE:-gpu_${TAG}_$(basename "$PROBLEMS" .jsonl)${LEVEL:+_L${LEVEL//,/}}.jsonl}
export OPENAI_BASE_URL="http://localhost:$PORT/v1" OPENAI_API_KEY=EMPTY

# 1. start the server unless one is already answering on PORT
if ! curl -s "localhost:$PORT/v1/models" >/dev/null 2>&1; then
  echo "== starting vLLM for $MODEL on port $PORT (log: vllm_$TAG.log)"
  vllm serve "$MODEL" --port "$PORT" --max-model-len "$MAXLEN" --gpu-memory-utilization 0.90 \
    > "vllm_$TAG.log" 2>&1 &
  VLLM_PID=$!
  trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
  for i in $(seq 1 180); do
    curl -s "localhost:$PORT/v1/models" >/dev/null 2>&1 && break
    kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM exited; see vllm_$TAG.log"; exit 1; }
    sleep 5
  done
fi

ARGS=(--backend openai --model "$MODEL" --problems "$PROBLEMS" --n "$N" --seeds "$SEEDS"
      --sc-k 3 --sc-carry --rounds 3 --conf-mix avg --stop-unanimous
      --policies "$POLICIES" --stagger 0 --workers "$WORKERS" --price-per-call 0 --store "$STORE")
[[ -n "$LEVEL" ]] && ARGS+=(--level "$LEVEL")
[[ -n "$EXTRA" ]] && ARGS+=(--openai-extra "$EXTRA")

echo "== smoke test (one call)"
python3 mad_gemini_v5.py smoke --backend openai --model "$MODEL" ${EXTRA:+--openai-extra "$EXTRA"}
echo "== plan"
python3 mad_gemini_v5.py trials "${ARGS[@]}" --dry-run
echo "== running -> $STORE"
python3 mad_gemini_v5.py trials "${ARGS[@]}"
echo "== analysis"
python3 round_metrics.py "$STORE" --contested
python3 mechanism_check.py "$STORE"
python3 signal_check.py "$STORE"
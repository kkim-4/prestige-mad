#!/usr/bin/env bash
# run_dual_sizes.sh — v7: mixed-skill panel by model SIZE (Qwen2.5 1.5B/3B/7B/14B/32B), on GPUs.
#   bash run_dual_sizes.sh                 # 2 GPUs (80 GB each): small models on GPU 0, 32B on GPU 1
#   ONE_GPU=1 bash run_dual_sizes.sh       # one 80 GB GPU: 0.5B/1.5B/3B/7B/14B
#   PROBLEMS=gpqa_extended.jsonl N=546 LEVEL= SEEDS=3 bash run_dual_sizes.sh
set -euo pipefail
PROBLEMS=${PROBLEMS:-math_extra.jsonl}; N=${N:-600}; LEVEL=${LEVEL-4,5}; SEEDS=${SEEDS:-1}
POLICIES=${POLICIES:-round_robin,prestige_x_conf,conf_only,dual,dual_shuf,dual_randdiss,rr_dual}
if [[ -n "${ONE_GPU:-}" ]]; then
  DESIGN=sizes_1gpu
  bash serve_models.sh "Qwen/Qwen2.5-0.5B-Instruct 8001 0 0.05" "Qwen/Qwen2.5-1.5B-Instruct 8002 0 0.07" \
    "Qwen/Qwen2.5-3B-Instruct 8003 0 0.10" "Qwen/Qwen2.5-7B-Instruct 8004 0 0.22" "Qwen/Qwen2.5-14B-Instruct 8005 0 0.45"
else
  DESIGN=sizes
  bash serve_models.sh "Qwen/Qwen2.5-1.5B-Instruct 8001 0 0.08" "Qwen/Qwen2.5-3B-Instruct 8002 0 0.12" \
    "Qwen/Qwen2.5-7B-Instruct 8003 0 0.25" "Qwen/Qwen2.5-14B-Instruct 8004 0 0.45" "Qwen/Qwen2.5-32B-Instruct 8005 1 0.90"
fi
STORE=${STORE:-dual_${DESIGN}_$(basename "$PROBLEMS" .jsonl)${LEVEL:+_L${LEVEL//,/}}.jsonl}
ARGS=(--design "$DESIGN" --problems "$PROBLEMS" --n "$N" --seeds "$SEEDS" --sc-k 3 --sc-carry --rounds 3
      --conf-mix avg --stop-unanimous --policies "$POLICIES" --stagger 0 --workers 24 --price-per-call 0
      --max-retries 8 --store "$STORE")
[[ -n "$LEVEL" ]] && ARGS+=(--level "$LEVEL")
python3 mad_dual.py smoke --design "$DESIGN"
python3 mad_dual.py trials "${ARGS[@]}" --dry-run
python3 mad_dual.py trials "${ARGS[@]}"
python3 mixed_report.py "$STORE"
echo "done. Stop the model servers with: pkill -f 'vllm serve'"

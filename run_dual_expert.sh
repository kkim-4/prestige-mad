#!/usr/bin/env bash
# run_dual_expert.sh — v8: one EXPERT among four weak agents.
#   bash run_dual_expert.sh                # all Gemini on Vertex, no GPU (expert gemini-3.8-flash, weak flash-lite)
#   EXPERT=gemini-3.8-flash WEAK=gemini-3.1-flash-lite bash run_dual_expert.sh
#   OPEN=1 bash run_dual_expert.sh         # open models on one GPU: Qwen2.5-14B expert + 4x Qwen2.5-1.5B
#   PROBLEMS=gpqa_extended.jsonl N=546 LEVEL= SEEDS=3 bash run_dual_expert.sh
set -euo pipefail
PROBLEMS=${PROBLEMS:-math_extra.jsonl}; N=${N:-600}; LEVEL=${LEVEL-4,5}; SEEDS=${SEEDS:-1}
POLICIES=${POLICIES:-round_robin,prestige_x_conf,conf_only,dual,dual_shuf,dual_randdiss,rr_dual}
if [[ -n "${OPEN:-}" ]]; then
  DESIGN=expert_open; EXPERT=${EXPERT:-Qwen/Qwen2.5-14B-Instruct}; WEAK=${WEAK:-Qwen/Qwen2.5-1.5B-Instruct}
  bash serve_models.sh "$EXPERT 8001 0 0.55" "$WEAK 8002 0 0.30"
  PACE=(--stagger 0 --workers 24 --price-per-call 0)
else
  DESIGN=expert; EXPERT=${EXPERT:-gemini-3.8-flash}; WEAK=${WEAK:-gemini-3.1-flash-lite}
  PACE=(--stagger 0.3 --workers 4)
fi
STORE=${STORE:-dual_${DESIGN}_$(basename "$PROBLEMS" .jsonl)${LEVEL:+_L${LEVEL//,/}}.jsonl}
ARGS=(--design "$DESIGN" --expert-model "$EXPERT" --weak-model "$WEAK" --problems "$PROBLEMS" --n "$N"
      --seeds "$SEEDS" --sc-k 3 --sc-carry --rounds 3 --conf-mix avg --stop-unanimous
      --policies "$POLICIES" --max-retries 8 --store "$STORE" "${PACE[@]}")
[[ -n "$LEVEL" ]] && ARGS+=(--level "$LEVEL")
python3 mad_dual.py smoke --design "$DESIGN" --expert-model "$EXPERT" --weak-model "$WEAK"
python3 mad_dual.py trials "${ARGS[@]}" --dry-run
read -r -p "Run it? [y/N] " ok; [[ "$ok" == "y" || "$ok" == "Y" ]] || exit 0
if command -v caffeinate >/dev/null; then caffeinate -i python3 mad_dual.py trials "${ARGS[@]}"
else python3 mad_dual.py trials "${ARGS[@]}"; fi
python3 mixed_report.py "$STORE"

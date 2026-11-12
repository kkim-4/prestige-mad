#!/usr/bin/env bash
# run_prestige.sh — the redesigned prestige-weighted debate (prestige_mad.py).
# prestige = running accuracy (warm-up W), confidence = stated. Policies: prestige, all_speak, none.
# Produces a store plus a D = confidence - prestige over-time plot (<store>_D.png).
#
#   bash scripts/run_prestige.sh mock                    # no API calls, sanity check + D plot
#   bash scripts/run_prestige.sh gemini                  # 3.5 Flash expert + 4x 3.1 Flash-Lite, GPQA-Diamond
#   bash scripts/run_prestige.sh qwen                    # open models on a GPU (Qwen 7B + 4x 3B), MATH
#   bash scripts/run_prestige.sh report <store.jsonl>    # accuracy + D plot only (free)
#
#   N=250 WARMUP=40 bash scripts/run_prestige.sh gemini
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE=${1:-mock}
N=${N:-120}; WARMUP=${WARMUP:-20}; ROUNDS=${ROUNDS:-3}
POLICIES=${POLICIES:-prestige,all_speak,none}
MAD=prestige_mad.py

case "$STAGE" in
  mock)
    python3 $MAD --mock --n "${N:-60}" --warmup "$WARMUP" --rounds "$ROUNDS" \
      --policies "$POLICIES" --store runs/prestige_mock.jsonl ;;
  gemini)
    PROBLEMS=${PROBLEMS:-gpqa_diamond.jsonl}
    [[ -f "$PROBLEMS" ]] || python3 scripts/make_gpqa.py --subset diamond --out "$PROBLEMS"
    grep -qxF "gpqa*.jsonl" .gitignore 2>/dev/null || echo "gpqa*.jsonl" >> .gitignore
    python3 $MAD --design expert --expert-model gemini-3.5-flash --expert-thinking low \
      --weak-model gemini-3.1-flash-lite --weak-thinking none \
      --problems "$PROBLEMS" --n "$N" --warmup "$WARMUP" --rounds "$ROUNDS" \
      --workers 3 --stagger 0.45 --max-retries 8 --policies "$POLICIES" \
      --store "runs/prestige_gemini_$(basename "$PROBLEMS" .jsonl)_n${N}.jsonl" ;;
  qwen)
    PROBLEMS=${PROBLEMS:-math_extra.jsonl}
    [[ -f "$PROBLEMS" ]] || python3 scripts/make_math_extra.py --exclude math.jsonl --out "$PROBLEMS"
    bash scripts/serve_models.sh "Qwen/Qwen2.5-7B-Instruct 8001 0 0.45" "Qwen/Qwen2.5-3B-Instruct 8002 0 0.35"
    python3 $MAD --design expert_open --expert-model Qwen/Qwen2.5-7B-Instruct \
      --weak-model Qwen/Qwen2.5-3B-Instruct --problems "$PROBLEMS" --level 4,5 \
      --n "${N:-600}" --warmup "${WARMUP:-50}" --rounds "$ROUNDS" --workers 24 --stagger 0 \
      --max-retries 8 --max-tokens 2048 --policies "$POLICIES" \
      --store "runs/prestige_qwen_math45_n${N:-600}.jsonl"
    echo "stop the servers with: pkill -f 'vllm serve'" ;;
  report)
    python3 $MAD report "${2:?give a store path}" ;;
  *) echo "usage: bash scripts/run_prestige.sh mock|gemini|qwen|report <store>"; exit 1 ;;
esac

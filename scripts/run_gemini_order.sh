#!/usr/bin/env bash
# run_gemini_order.sh — final Gemini run: does argument ORDER (set by prestige x confidence) and one voice
# per answer beat standard simultaneous debate, on GPQA-Diamond questions this panel has never seen?
# Panel: 3.5 Flash expert (low thinking) + 4x 3.1 Flash-Lite. Fresh round-1 answers, random seating.
#
#   bash scripts/run_gemini_order.sh plan
#   bash scripts/run_gemini_order.sh run       # round 1 + all_speak, per_answer, per_answer_last, dual_2diss
#   bash scripts/run_gemini_order.sh report
#
# Sized to finish in roughly 70-80 minutes: N=120 questions, 3 workers, 0.45 s between calls.
#   N=100 bash scripts/run_gemini_order.sh run          # shorter
#   STAGGER=0.8 WORKERS=2 bash scripts/run_gemini_order.sh run   # if 429s pile up (it resumes)
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE=${1:-plan}
PROBLEMS=${PROBLEMS:-gpqa_diamond.jsonl}
N=${N:-120}
STORE=${STORE:-results/mixed_panel/gpqa_order_$(basename "$PROBLEMS" .jsonl)_n${N}.jsonl}
POLICIES=${POLICIES:-all_speak,per_answer,per_answer_last,dual_2diss}
WORKERS=${WORKERS:-3}; STAGGER=${STAGGER:-0.45}
MAD=mad_dual_v4.py
[[ -f "$PROBLEMS" ]] || python3 scripts/make_gpqa.py --subset diamond --out "$PROBLEMS"
grep -q "per_answer_last" $MAD || { echo "$MAD is out of date: save the newest mad_dual_v4.py first"; exit 1; }
mkdir -p "$(dirname "$STORE")"
PANEL=(--design expert --expert-model gemini-3.5-flash --expert-thinking low --weak-model gemini-3.1-flash-lite --weak-thinking none)
ARGS=("${PANEL[@]}" --problems "$PROBLEMS" --n "$N" --seeds 1 --sc-k 3 --sc-carry --rounds 3 --conf-mix avg
      --stop-unanimous --seat-shuffle --workers "$WORKERS" --stagger "$STAGGER" --max-retries 8
      --policies "$POLICIES" --store "$STORE")
case "$STAGE" in
  plan) python3 $MAD trials "${ARGS[@]}" --dry-run ;;
  run)
    python3 $MAD smoke "${PANEL[@]}"
    python3 $MAD trials "${ARGS[@]}" --dry-run
    read -r -p "Run it? [y/N] " ok; [[ "$ok" == "y" || "$ok" == "Y" ]] || exit 0
    if command -v caffeinate >/dev/null; then caffeinate -i python3 $MAD trials "${ARGS[@]}"
    else python3 $MAD trials "${ARGS[@]}"; fi ;;
  report)
    python3 $MAD report --store "$STORE" --policies "$POLICIES" --ref all_speak
    python3 analysis/minority_check.py "$STORE" --ref all_speak
    python3 analysis/flip_breakdown.py "$STORE"
    python3 analysis/mixed_report.py "$STORE" ;;
  *) echo "usage: bash scripts/run_gemini_order.sh plan|run|report"; exit 1 ;;
esac

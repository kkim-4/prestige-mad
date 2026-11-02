#!/usr/bin/env bash
# run_qwen.sh — mixed panel on open models: Qwen2.5-7B expert + 4x Qwen2.5-3B, MATH levels 4-5.
# Plan and hypotheses: PREREG_qwen.md (commit it before the "full" stage).
#
#   bash scripts/run_qwen.sh pilot    # serve both models, smoke test, round-1 answers on 100 problems, signal check
#   bash scripts/run_qwen.sh full     # the pre-registered run: 600 problems, 4 policies, random seating
#   bash scripts/run_qwen.sh report   # all analyses on the full store (free)
#
#   WEAK=Qwen/Qwen2.5-1.5B-Instruct bash scripts/run_qwen.sh pilot   # fallback if the 3B and 7B don't differ
# Stop the model servers afterwards with:  pkill -f "vllm serve"
set -euo pipefail
cd "$(dirname "$0")/.."   # run from the repo root
STAGE=${1:-pilot}
EXPERT=${EXPERT:-Qwen/Qwen2.5-7B-Instruct}
WEAK=${WEAK:-Qwen/Qwen2.5-3B-Instruct}
TAG=$(basename "$EXPERT")_vs_$(basename "$WEAK")
STORE=${STORE:-qwen_${TAG}_math45.jsonl}
PILOT=${PILOT:-qwen_${TAG}_pilot.jsonl}
POLICIES=${POLICIES:-round_robin,dual,dual_shuf,dual_conf}
export MAXLEN=${MAXLEN:-8192}

PANEL=(--design expert_open --expert-model "$EXPERT" --weak-model "$WEAK")
COMMON=("${PANEL[@]}" --problems math_extra.jsonl --level 4,5 --seeds 1 --sc-k 3 --sc-carry --rounds 3
        --conf-mix avg --stop-unanimous --stagger 0 --workers 24 --price-per-call 0
        --max-retries 8 --max-tokens 2048)

serve() {
  [[ -f math_extra.jsonl ]] || python3 scripts/make_math_extra.py --exclude math.jsonl --out math_extra.jsonl
  bash scripts/serve_models.sh "$EXPERT 8001 0 0.45" "$WEAK 8002 0 0.35"
  python3 mad_dual.py smoke "${PANEL[@]}"
}

case "$STAGE" in
  pilot)
    serve
    python3 mad_dual.py trials "${COMMON[@]}" --n 100 --round1-only --store "$PILOT"
    python3 analysis/signal_check.py "$PILOT"
    echo
    echo "Go on to 'full' if section 1 shows the agents differ (Cochran's Q p < 0.05) and few calls failed."
    echo "If not, rerun the pilot with WEAK=Qwen/Qwen2.5-1.5B-Instruct." ;;
  full)
    serve
    python3 mad_dual.py trials "${COMMON[@]}" --n 600 --seat-shuffle --policies "$POLICIES" --store "$STORE" --dry-run
    read -r -p "Run it? [y/N] " ok; [[ "$ok" == "y" || "$ok" == "Y" ]] || exit 0
    python3 mad_dual.py trials "${COMMON[@]}" --n 600 --seat-shuffle --policies "$POLICIES" --store "$STORE"
    python3 analysis/mixed_report.py "$STORE" ;;
  report)
    python3 mad_dual.py report --store "$STORE" --policies "$POLICIES"
    python3 analysis/mixed_report.py "$STORE"
    python3 analysis/flip_breakdown.py "$STORE"
    python3 analysis/signal_check.py "$STORE"
    python3 analysis/vote_check.py "$STORE" ;;
  *) echo "usage: bash scripts/run_qwen.sh pilot|full|report"; exit 1 ;;
esac

#!/usr/bin/env bash
# run_qwen.sh — mixed panel on open models: Qwen2.5-7B expert + 4x Qwen2.5-3B, 600 MATH level 4-5 problems.
# Plan and hypotheses: PREREG_qwen.md.
#
#   bash scripts/run_qwen.sh pilot    # serve both models, smoke test, round 1 on 100 problems, signal check
#   bash scripts/run_qwen.sh full     # the pre-registered run: 600 problems, all policies, random seating
#   bash scripts/run_qwen.sh noise    # same debates rerun in the SAME seating: the resampling noise floor
#   bash scripts/run_qwen.sh report   # all analyses (no GPU calls)
#
#   WEAK=Qwen/Qwen2.5-1.5B-Instruct bash scripts/run_qwen.sh pilot   # fallback if the 3B and 7B don't differ
# Stop the model servers afterwards with:  pkill -f "vllm serve"
set -euo pipefail
cd "$(dirname "$0")/.."   # run from the repo root
STAGE=${1:-pilot}
EXPERT=${EXPERT:-Qwen/Qwen2.5-7B-Instruct}
WEAK=${WEAK:-Qwen/Qwen2.5-3B-Instruct}
PROBLEMS=${PROBLEMS:-math_extra.jsonl}
N=${N:-600}
TAG=$(basename "$EXPERT")_vs_$(basename "$WEAK")
STORE=${STORE:-qwen_${TAG}_math45.jsonl}
PILOT=${PILOT:-qwen_${TAG}_math45_pilot.jsonl}
NOISE=${NOISE:-qwen_${TAG}_math45_noise.jsonl}
POLICIES=${POLICIES:-all_speak,dual,round_robin,seq_rr,dual_conf,dual_shuf,dual_2lead,dual_2diss,per_answer}
NOISE_POLICIES=${NOISE_POLICIES:-all_speak,dual,seq_rr}
export MAXLEN=${MAXLEN:-16384}

PANEL=(--design expert_open --expert-model "$EXPERT" --weak-model "$WEAK")
COMMON=("${PANEL[@]}" --problems "$PROBLEMS" --level 4,5 --seeds 1 --sc-k 3 --sc-carry --rounds 3
        --conf-mix avg --stop-unanimous --stagger 0 --workers 24 --price-per-call 0
        --max-retries 8 --max-tokens 2048)

serve() {
  [[ -f "$PROBLEMS" ]] || python3 scripts/make_math_extra.py --exclude math.jsonl --out "$PROBLEMS"
  bash scripts/serve_models.sh "$EXPERT 8001 0 0.45" "$WEAK 8002 0 0.35"
  python3 mad_dual_v3.py smoke "${PANEL[@]}"
}

confirm_run() {   # dry run, ask, run
  python3 mad_dual_v3.py trials "$@" --dry-run
  read -r -p "Run it? [y/N] " ok; [[ "$ok" == "y" || "$ok" == "Y" ]] || exit 0
  python3 mad_dual_v3.py trials "$@"
}

case "$STAGE" in
  pilot)
    serve
    python3 mad_dual_v3.py trials "${COMMON[@]}" --n 100 --round1-only --store "$PILOT"
    python3 analysis/signal_check.py "$PILOT"
    echo
    echo "Go on to 'full' if section 1 shows the agents differ (Cochran's Q p < 0.05) and few calls failed."
    echo "If not, stop: PREREG_qwen.md says what changes next." ;;
  full)
    serve
    confirm_run "${COMMON[@]}" --n "$N" --seat-shuffle --policies "$POLICIES" --store "$STORE"
    python3 analysis/mixed_report.py "$STORE" ;;
  noise)
    [[ -f "$STORE" ]] || { echo "run 'full' first (the noise run reuses its round-1 answers)"; exit 1; }
    serve
    confirm_run "${COMMON[@]}" --n "$N" --seat-shuffle --same-order-rerun --policies "$NOISE_POLICIES" \
      --store "$NOISE" --init-from "$STORE" ;;
  report)
    python3 mad_dual_v3.py report --store "$STORE" --policies "$POLICIES" --ref all_speak
    python3 analysis/minority_check.py "$STORE" --ref all_speak
    python3 analysis/flip_breakdown.py "$STORE"
    [[ -f "$NOISE" ]] && python3 analysis/flip_breakdown.py "$NOISE"
    python3 analysis/mixed_report.py "$STORE"
    python3 analysis/signal_check.py "$STORE"
    python3 analysis/vote_check.py "$STORE" ;;
  *) echo "usage: bash scripts/run_qwen.sh pilot|full|noise|report"; exit 1 ;;
esac

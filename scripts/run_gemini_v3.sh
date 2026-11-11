#!/usr/bin/env bash
# run_gemini_v3.sh — the new baselines and dual's design variants on the Gemini mixed panel
# (3.5 Flash expert at low thinking + 4x 3.1 Flash-Lite), run from scratch: fresh round-1 answers.
#
#   bash scripts/run_gemini_v3.sh plan     # dry runs: call counts for each stage, nothing is spent
#   bash scripts/run_gemini_v3.sh core     # round 1 + dual, all_speak, seq_rr, dual_conf, dual_2lead, per_answer
#   bash scripts/run_gemini_v3.sh extra    # adds dual_shuf, dual_2diss, round_robin (same round-1 answers)
#   bash scripts/run_gemini_v3.sh report   # all analyses (free)
#
#   N=150 bash scripts/run_gemini_v3.sh core                                # fewer questions, cheaper
#   PROBLEMS=gpqa_diamond.jsonl N=198 bash scripts/run_gemini_v3.sh core    # fresh questions instead
#
# Questions: by default the same GPQA file and count as the earlier 250-question mixed run (read from
# that run's stored config when it is available), with new round-1 answers and random seating.
# Rate limits: defaults are 3 workers and 0.5 s between calls, with 8 retries. If you see many 429s,
# stop (Ctrl-C), raise STAGGER (e.g. STAGGER=0.8 WORKERS=2) and rerun the same stage: it resumes.
# Needs the Vertex settings from before (GOOGLE_GENAI_USE_VERTEXAI=True, GOOGLE_CLOUD_PROJECT=mad-resolution).
set -euo pipefail
cd "$(dirname "$0")/.."   # run from the repo root
STAGE=${1:-plan}
OLD=results/mixed_panel/gpqa250_mixed.jsonl
if [[ -z "${PROBLEMS:-}" ]]; then
  PROBLEMS=$( [[ -f "$OLD" ]] && python3 -c "import json,sys; print([json.loads(l) for l in open(sys.argv[1]) if '\"config\"' in l][-1]['cfg']['problems'])" "$OLD" || echo gpqa_main.jsonl )
fi
N=${N:-250}
STORE=${STORE:-results/mixed_panel/gpqa_v3_$(basename "$PROBLEMS" .jsonl)_n${N}.jsonl}
CORE=${CORE:-dual,all_speak,seq_rr,dual_conf,dual_2lead,per_answer}
EXTRA=${EXTRA:-dual_shuf,dual_2diss,round_robin}
WORKERS=${WORKERS:-3}; STAGGER=${STAGGER:-0.5}
PRICE=${PRICE:-0.00065}   # Flash-Lite $/call for the estimate; 1 call in 5 is the pricier 3.5 Flash expert
MAD=mad_dual_v3.py

[[ -f "$PROBLEMS" ]] || { echo "question file $PROBLEMS not found (build it with scripts/make_gpqa.py, or set PROBLEMS=...)"; exit 1; }
grep -qxF "gpqa*.jsonl" .gitignore 2>/dev/null || echo "gpqa*.jsonl" >> .gitignore

ARGS=(--design expert --expert-model gemini-3.5-flash --expert-thinking low
      --weak-model gemini-3.1-flash-lite --weak-thinking none
      --problems "$PROBLEMS" --n "$N" --seeds 1 --sc-k 3 --sc-carry --rounds 3 --conf-mix avg
      --stop-unanimous --seat-shuffle --workers "$WORKERS" --stagger "$STAGGER" --max-retries 8
      --price-per-call "$PRICE" --store "$STORE")

run() {   # run POLICIES: dry run, ask, run
  python3 $MAD trials "${ARGS[@]}" --policies "$1" --dry-run
  read -r -p "Run it? [y/N] " ok; [[ "$ok" == "y" || "$ok" == "Y" ]] || exit 0
  if command -v caffeinate >/dev/null; then caffeinate -i python3 $MAD trials "${ARGS[@]}" --policies "$1"
  else python3 $MAD trials "${ARGS[@]}" --policies "$1"; fi
}

case "$STAGE" in
  plan)
    echo "questions: $PROBLEMS (n = $N) -> $STORE"
    echo "== core (includes round 1): $CORE"; python3 $MAD trials "${ARGS[@]}" --policies "$CORE" --dry-run
    echo "== core + extra: $CORE,$EXTRA"; python3 $MAD trials "${ARGS[@]}" --policies "$CORE,$EXTRA" --dry-run ;;
  core)
    python3 $MAD smoke --design expert --expert-model gemini-3.5-flash --expert-thinking low \
      --weak-model gemini-3.1-flash-lite --weak-thinking none
    run "$CORE" ;;
  extra) run "$EXTRA" ;;
  report)
    ALL=$(python3 -c "import json,sys; print(','.join(sorted({json.loads(l)['policy'] for l in open(sys.argv[1]) if '\"debate\"' in l})))" "$STORE")
    python3 $MAD report --store "$STORE" --policies "$ALL" --ref all_speak
    python3 analysis/minority_check.py "$STORE" --ref all_speak
    python3 analysis/flip_breakdown.py "$STORE"
    python3 analysis/mixed_report.py "$STORE"
    python3 analysis/signal_check.py "$STORE"
    python3 analysis/vote_check.py "$STORE" ;;
  *) echo "usage: bash scripts/run_gemini_v3.sh plan|core|extra|report"; exit 1 ;;
esac

#!/usr/bin/env bash
# run_experiment.sh — one experiment: majority vote vs round robin vs dual on a problem set.
# Edit the settings below (or pass them as env vars) to scale up; nothing else needs changing.
#
#   bash run_experiment.sh                       # uses the settings below
#   N=500 LEVEL=4,5 STORE=math_hard.jsonl bash run_experiment.sh
#
# Majority vote (no debate) is printed automatically from the same round-1 answers.
set -euo pipefail

PROBLEMS=${PROBLEMS:-math_extra.jsonl}        # fresh MATH problems (python3 make_math_extra.py)
N=${N:-300}                                   # number of problems  <- the main knob to scale
OFFSET=${OFFSET:-0}                           # start position; use a new range for each new run
LEVEL=${LEVEL:-}                              # e.g. 4,5 for harder MATH problems (more disagreement)
SEEDS=${SEEDS:-1}
POLICIES=${POLICIES:-round_robin,dual}        # add rr_dual / both_sides for ablations
MODEL=${MODEL:-gemini-3.1-flash-lite}
STORE=${STORE:-math_scale.jsonl}
INIT_FROM=${INIT_FROM:-}                      # reuse round-1 answers from another store (same problems)

LEVEL_ARG=${LEVEL:+--level $LEVEL}
INIT_ARG=${INIT_FROM:+--init-from $INIT_FROM}
COMMON="--problems $PROBLEMS --n $N --offset $OFFSET --seeds $SEEDS --model $MODEL \
  --sc-k 3 --sc-carry --rounds 3 --policies $POLICIES --conf-mix avg --stop-unanimous \
  --stagger 0.3 --workers 6 --store $STORE $LEVEL_ARG $INIT_ARG"

echo "== plan (no API calls)"
python3 mad_gemini_v3.py trials $COMMON --dry-run
read -r -p "Run it? [y/N] " ok
[[ "$ok" == "y" || "$ok" == "Y" ]] || exit 0

echo "== running (resumable: rerun this script to continue after a crash)"
caffeinate -i python3 mad_gemini_v3.py trials $COMMON

echo "== report"
python3 mad_gemini_v3.py report --store "$STORE" --policies "$POLICIES"
echo "== switches toward right vs wrong"
python3 revote.py "$STORE"
echo "== round by round, contested problems"
python3 round_metrics.py "$STORE" --contested

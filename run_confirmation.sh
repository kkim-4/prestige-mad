#!/usr/bin/env bash
# run_confirmation.sh — the pre-registered confirmation (PREREG.md) on Gemini, in two stages
# that share one store, so the ablations reuse the primary run's round-1 answers.
#
#   bash run_confirmation.sh primary     # round robin vs dual           (~$11)
#   bash run_confirmation.sh ablations   # adds the mechanism variants   (~$14 more)
#   bash run_confirmation.sh report      # free
#   bash run_confirmation.sh gpqa        # secondary: GPQA extended replication (~$10)
#
# Every stage shows a dry run (exact call count and cost) and asks before spending.
set -euo pipefail
STAGE=${1:-primary}
MODEL=${MODEL:-gemini-3.1-flash-lite}
STORE=${STORE:-confirm_math45.jsonl}
COMMON="--model $MODEL --sc-k 3 --sc-carry --rounds 3 --conf-mix avg --stop-unanimous \
  --seeds 1 --stagger 0.3 --workers 4"
MATH="--problems math_extra.jsonl --level 4,5 --n 600 --store $STORE"
ABL="round_robin,dual,seq_dual,diss_only,lead_rand,dual_randdiss,rr_dual"

run() {   # run <args...> : dry run, confirm, then the real run
  python3 mad_gemini_v5.py trials "$@" --dry-run
  read -r -p "Run it? [y/N] " ok
  [[ "$ok" == "y" || "$ok" == "Y" ]] || exit 0
  if command -v caffeinate >/dev/null; then caffeinate -i python3 mad_gemini_v5.py trials "$@"
  else python3 mad_gemini_v5.py trials "$@"; fi
}

case "$STAGE" in
  primary)
    [[ -f math_extra.jsonl ]] || python3 make_math_extra.py --exclude math.jsonl --out math_extra.jsonl
    run $COMMON $MATH --policies round_robin,dual
    python3 mad_gemini_v5.py report --store "$STORE" --policies round_robin,dual ;;
  ablations)
    run $COMMON $MATH --policies "$ABL"
    python3 mad_gemini_v5.py report --store "$STORE" --policies "$ABL" ;;
  report)
    python3 mad_gemini_v5.py report --store "$STORE" --policies "$ABL"
    python3 round_metrics.py "$STORE" --contested
    python3 mechanism_check.py "$STORE" ;;
  gpqa)
    [[ -f gpqa_extended.jsonl ]] || python3 make_gpqa.py --subset extended --out gpqa_extended.jsonl
    grep -q "gpqa\*.jsonl" .gitignore 2>/dev/null || echo "gpqa*.jsonl" >> .gitignore
    run $COMMON --problems gpqa_extended.jsonl --n 546 --store gpqa_ext_confirm.jsonl --policies round_robin,dual
    python3 mad_gemini_v5.py report --store gpqa_ext_confirm.jsonl --policies round_robin,dual ;;
  *) echo "usage: bash run_confirmation.sh primary|ablations|report|gpqa"; exit 1 ;;
esac

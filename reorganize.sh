#!/usr/bin/env bash
# reorganize.sh — tidy the mad_gemini folder. Shows the plan first; nothing moves until you add "go".
#
#   bash reorganize.sh        # dry run: prints every move
#   bash reorganize.sh go     # does it
#
# Layout afterwards:
#   top level              code in use, run scripts, PREREG*.md, README, problem sets (math*.jsonl, gpqa_*.jsonl)
#   results/confirmation   pre-registered MATH confirmation store
#   results/mixed_panel    Gemini expert + weak-agent stores
#   results/open_models    Gemma / Qwen / GPU stores
#   results/exploratory    every other run store (older GPQA/MATH/sc experiments)
#   logs/                  calls.jsonl and *_summary.json
#   archived/              old script versions (and the contents of "Archived 2")
#
# Code stays at the top level on purpose: the analysis scripts import mad_gemini.py from their own
# folder, and the run scripts call each other by name. Problem sets stay too, because every run
# store records its problem file's path and resuming checks it.
# Tracked files move with `git mv`, so git sees renames; untracked files move with plain mv.
set -uo pipefail
GO=${1:-}
cd "$(dirname "$0")"
[[ -f mad_gemini.py && -f mad_dual.py ]] || { echo "run this inside the mad_gemini folder"; exit 1; }

say() { echo "  $*"; }
move() {   # move FILE DIR
  local f=$1 d=$2
  [[ -e "$f" ]] || return 0
  if [[ -e "$d/$(basename "$f")" ]]; then say "SKIP $f (already in $d/)"; return 0; fi
  say "$f -> $d/"
  [[ "$GO" == "go" ]] || return 0
  mkdir -p "$d"
  if git ls-files --error-unmatch "$f" >/dev/null 2>&1; then git mv "$f" "$d/"; else mv "$f" "$d/"; fi
}

echo "== 1. mad_dual.py: use the new version (seat shuffle + dual_conf)"
if [[ -f mad_dual_v2.py ]] && grep -q "seat-shuffle" mad_dual_v2.py; then
  say "mad_dual_v2.py replaces mad_dual.py"
  [[ "$GO" == "go" ]] && mv -f mad_dual_v2.py mad_dual.py
elif grep -q "seat-shuffle" mad_dual.py; then
  say "mad_dual.py is already the new version"
else
  say "WARNING: no copy with --seat-shuffle found; save the latest mad_dual.py here first"
fi

echo "== 2. old script versions -> archived/"
for f in mad_gemini_v2.py mad_gemini_v3.py mad_gemini_v4.py mad_gemini_independent_personas.py run_experiment.sh; do
  move "$f" archived
done
if [[ -f mad_gemini_v6.py && -f mad_gemini_v5.py ]]; then
  if cmp -s mad_gemini_v5.py mad_gemini_v6.py; then move mad_gemini_v6.py archived
  else say "KEEP mad_gemini_v6.py (differs from v5; check which one your runs used)"; fi
fi
if [[ -d "Archived 2" ]]; then
  for f in "Archived 2"/* "Archived 2"/.[!.]*; do [[ -e "$f" ]] && move "$f" archived; done
  [[ "$GO" == "go" ]] && rmdir "Archived 2" 2>/dev/null && say "removed empty 'Archived 2'"
fi

echo "== 3. run stores -> results/   (problem sets stay; told apart by their contents)"
for f in *.jsonl; do
  [[ -e "$f" ]] || continue
  first=$(head -c 2000 "$f" | head -n 1)
  if [[ "$first" == *'"kind"'* ]]; then
    case "$f" in
      confirm_math45*)                         move "$f" results/confirmation ;;
      pilot_gemma*|gpu_*|qwen_*|dual_sizes*|dual_expert_open*) move "$f" results/open_models ;;
      gpqa250_mixed*|pilot_expert_*|dual_expert_*) move "$f" results/mixed_panel ;;
      *)                                       move "$f" results/exploratory ;;
    esac
  elif [[ "$first" == *'"question"'* ]]; then
    :                                          # problem set: stays
  else
    move "$f" logs                             # e.g. calls.jsonl (per-call log)
  fi
done

echo "== 4. summaries -> logs/"
for f in *_summary.json trials.json; do move "$f" logs; done

echo "== 5. .gitignore"
for pat in "__pycache__/" "logs/"; do
  if ! grep -qxF "$pat" .gitignore 2>/dev/null; then
    say "add $pat"; [[ "$GO" == "go" ]] && echo "$pat" >> .gitignore
  fi
done

if [[ "$GO" != "go" ]]; then
  echo; echo "Dry run only. If the plan looks right:  bash reorganize.sh go"
else
  echo; echo "Done. Review with: git status"
fi

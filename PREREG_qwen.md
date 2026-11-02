# Pre-registration: dual on an open-model mixed panel (narrow skill gap)

Written before the full run. Commit this file before `bash run_qwen.sh full`, so the date is on record.

## Why this panel
On the Gemini mixed panel (3.5 Flash expert at 82% vs four Flash-Lite agents at about 60%), confidence
alone already found the expert, so the track record had nothing extra to do. A narrower gap is where
confidence should start to fail and prestige should matter. Qwen2.5-7B vs Qwen2.5-3B is expected to
be roughly a 10-point gap on MATH (the Gemini gap was about 22).

## Pilot (not part of any test)
Round-1 answers only, first 100 problems, separate store. Used only to check that the agents differ
(Cochran's Q, p < 0.05) and that the 3B's replies parse. No debates are run, so it reveals nothing
about the hypotheses. If the agents do not differ, the weak model becomes Qwen2.5-1.5B-Instruct and
this file is amended before the full run.

## Setup (fixed)
- **Panel:** expert = Qwen/Qwen2.5-7B-Instruct, four weak = Qwen/Qwen2.5-3B-Instruct, served by vLLM,
  temperature 0.7, same persona for all.
- **Problems:** the 600 MATH level 4-5 problems of the Gemini confirmation (`math_extra.jsonl`,
  `--level 4,5 --n 600`, fixed shuffle).
- **Settings:** `--sc-k 3 --sc-carry --rounds 3 --conf-mix avg --stop-unanimous`, 1 seed, all
  policies sharing the same round-1 answers.
- **Seating:** `--seat-shuffle`: a random seating order per problem, the same for every policy, so the
  expert is not always first (that favoured round robin on the Gemini panel). Each debate is also
  rerun in a second random order, as before.
- **Policies:** `round_robin`, `dual`, `dual_shuf` (prestige reassigned among agents),
  `dual_conf` (dual with prestige switched off, confidence alone).

## Primary test (one)
One-sided exact McNemar on final-answer accuracy, dual vs round robin, alpha = 0.05.
Confirmed if p < 0.05 and dual has more discordant wins. Nothing is added, dropped or rerun after
seeing results, except automatic retries of failed calls.

## Secondary (reported, not used for the decision)
1. dual vs `dual_conf`: does the track record add to confidence? Prediction: dual ahead.
2. dual vs `dual_shuf`: does dual gain from finding the right agent, or only from the two-speaker
   structure? Prediction: dual ahead.
3. Flip rate and correct -> wrong under reordering, dual vs round robin. Prediction: dual lower.
4. Accuracy in the reordered runs, dual vs round robin (`flip_breakdown.py`).
5. Each policy vs the expert alone and vs majority vote without debate (`mixed_report.py`).
6. Speaker choice on contested problems: how often the first speaker is right under prestige vs
   confidence (`signal_check.py`).

Secondary comparisons 1-2 are each one-sided McNemar at 0.05; with several of them, a single
p < 0.05 is read with caution. Vote re-scoring (`vote_check.py`) is exploratory.

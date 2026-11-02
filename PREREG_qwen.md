# Pre-registration: dual on an open-model mixed panel (GPQA)

Commit this file before `bash scripts/run_qwen.sh full`, so the date is on record.

**Amendment, 2026-11-02, before any run (pilot or full):** the primary test moves from 600 MATH
level 4-5 problems to GPQA extended (546 questions), to match the task of the Gemini mixed panel.
Panel, policies, seating and test are unchanged. No Qwen results of any kind existed at this point.

## Why this panel
On the Gemini mixed panel (3.5 Flash expert at 82% vs four Flash-Lite agents at about 60%, GPQA),
confidence alone already found the expert, so the track record had nothing extra to do. This panel
uses two open models from one family with a smaller gap: Qwen2.5-7B-Instruct as the expert and four
Qwen2.5-3B-Instruct agents. Both are expected to sit well below the Gemini agents and closer to the
25% chance level, so the pilot checks first that they differ at all.

## Pilot (not part of any test)
Round-1 answers only, first 100 questions, separate store. Used only to check that (a) the agents
differ (Cochran's Q, p < 0.05), (b) the expert is clearly above chance, and (c) replies parse.
No debates are run, so the pilot reveals nothing about the hypotheses.
If (a) or (b) fails, the weak model becomes Qwen2.5-1.5B-Instruct and this file is amended
(and committed) before the full run. If that also fails, the GPQA run is dropped and reported as
"open models of this size are too close to chance on GPQA for the test to be informative".

## Setup (fixed)
- **Panel:** expert = Qwen/Qwen2.5-7B-Instruct, four weak = Qwen/Qwen2.5-3B-Instruct, served by vLLM,
  temperature 0.7, same persona for all.
- **Questions:** GPQA extended, all 546 (`scripts/make_gpqa.py --subset extended`, options shuffled with
  the script's fixed seed). The question file is not committed (gated dataset).
- **Settings:** `--sc-k 3 --sc-carry --rounds 3 --conf-mix avg --stop-unanimous`, 1 seed, all
  policies sharing the same round-1 answers.
- **Seating:** `--seat-shuffle`: a random seating order per question, the same for every policy, so the
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
4. Accuracy in the reordered runs, dual vs round robin (`analysis/flip_breakdown.py`).
5. Each policy vs the expert alone and vs majority vote without debate (`analysis/mixed_report.py`).
6. Speaker choice on contested questions: how often the first speaker is right under prestige vs
   confidence (`analysis/signal_check.py`).

Secondary comparisons 1-2 are each one-sided McNemar at 0.05; with several of them, a single
p < 0.05 is read with caution. Vote re-scoring (`analysis/vote_check.py`) is exploratory.
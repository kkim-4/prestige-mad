# Pre-registration: dual vs round robin (powered confirmation)

Written before any of these runs. Commit this file before running, so the date is on record.

## Hypothesis
Letting the top prestige × confidence agent and the strongest dissenter speak together after
round 1 (`dual`) gives more correct final answers than round-robin debate (`round_robin`),
among five agents of the same model.

## Primary test (one, fixed)
- **Problems:** 600 MATH test problems at levels 4–5 that are not in MATH-500
  (`math_extra.jsonl`, `--level 4,5 --n 600`, the script's fixed shuffle). None was used before.
- **Model:** gemini-3.1-flash-lite, temperature 0.7, thinking low.
- **Settings:** 5 personas, `--sc-k 3 --sc-carry --rounds 3 --conf-mix avg --stop-unanimous`, 1 seed,
  all policies sharing the same round-1 answers.
- **Test:** one-sided exact McNemar on final-answer accuracy, dual vs round robin, α = 0.05.
  Problems where both are right or both wrong do not enter the test.
- **Decision rule:** the effect is confirmed if p < 0.05 and dual has more discordant wins.
  Nothing is added, dropped or rerun after seeing results, except automatic retries of failed API calls.

## Secondary (reported, not used for the decision)
- Accuracy on contested problems (some agents right, some wrong after round 1).
- Flip rate under agent reordering; correct → wrong flips.
- Accuracy vs majority vote without debate.
- Replication on GPQA extended (546 questions, new round-1 answers). Most of these questions
  informed the design, so this is a replication, not a confirmation.

## Mechanism ablations (same round-1 answers, compared with dual)
| Variant | Change from dual | Prediction if the mechanism holds |
|---|---|---|
| `seq_dual` | dissenter speaks a round after the leader | lower than dual (simultaneity matters) |
| `diss_only` | only the dissenter speaks after round 1 | lower than dual (both sides needed) |
| `lead_rand` | leader + a random agent, any answer | lower than dual (a real dissenter needed) |
| `dual_randdiss` | leader + a random agent holding a different answer | about equal to dual (choice matters little with clones) |
| `rr_dual` | two speakers by rotation | lower than dual |

Each is reported with its own McNemar against dual and its round-2 switches
(wrong → right vs right → wrong, `round_metrics.py`). These are exploratory: five comparisons,
so a single p < 0.05 among them is read with caution.

## Open-model replication (separate, same plan)
Same problems, settings and test, with an open model served by vLLM (`run_gpu.sh`), e.g.
Qwen2.5-7B-Instruct and one model from another family (Gemma 3 or Llama 3.1). Prediction: dual
beats round robin here too. Each model is its own pre-registered one-sided test.

# Qwen 7B + 4 x 3B, prestige debate on MATH level 4-5 · November 24, 2026

The `qwen` stage of `scripts/run_prestige_v2.sh` on one H100 (Slurm 19214, resumed by 19225; scripts in
`scripts/slurm/`). Python code unchanged from 72d2c4d. 600 problems from `logs/math_extra.jsonl` (`--level 4,5`),
warm-up 50, 3 rounds, two seatings, temperature 0.7, context 16,384. N and WARMUP were passed explicitly: the
script's own defaults resolve to 120 and 20, because lines near the top set them before `${N:-600}` is read.
No pre-registration covers this run.

## Result (`report_regraded.txt`)

| | All 600 | Contested (587) | Flips |
| --- | ---: | ---: | ---: |
| Expert (7B) alone | 14.5% | | |
| Each 3B alone | 6.3-7.8% | | |
| Majority vote, no debate | 11.0% | | |
| `prestige`, as run | 14.1% | 14.1% | 56.5% |
| `all_speak` | 17.9% | 18.0% | 48.2% |

`prestige` minus `all_speak` on contested questions: -3.9 points [-6.0, -1.8], both seatings.

**This run should not be read as evidence about the method.** Two problems:

1. **Negative vote weights.** `logodds(p)` is negative when p < 0.5. After the warm-up every agent's prestige was
   below 0.5 in all 1,074 debates (537 of them `prestige`), so the `prestige` vote favored the answer with the least support. Re-scoring the saved
   answers with negative weights set to 0 gives `prestige` 18.3% on contested questions, +0.3 [-1.7, +2.3] against
   `all_speak` (`vote_rescore.txt`). With every weight at 0 this is a plain majority vote, so it does not test
   prestige weighting either. The re-scoring is exploratory, done after seeing the results.
2. **Very low accuracy.** The prompt asks for at most 60 words of rationale with the work done "in your head"; the
   expert's median rationale is 20 words. This is the likely cause but has not been tested. LaTeX escaping is not
   the cause; unicode/`$` formatting misses account for 1-4 answers per agent, and the expert left 32 answers empty.
   Only 145 of 600 questions have any agent right in round 1.

## Notes
- **Grading.** The server's report (`report_as_run.txt`) counts `0` as equal to `(0,1)` on question 195, which is
  wrong; the same code grades it correctly under Python 3.12. The difference is up to 0.2 points per row. Prestige
  during the run used the server's grades, so three agents' prestige is about 1/i too high after that question.
- **Retries.** 108 replies failed to parse on the first attempt. `all_speak` question 430 failed all retries in job
  19214 and completed in 19225. Nothing was dropped.

## Files
`store.jsonl` round-1 answers and debates · `calls.jsonl.gz` every parsed model reply · `D.png` overconfidence plot ·
`packages.txt` environment · `vote_rescore.txt` from `analysis/prestige_vote_rescore.py`.

# Pre-registration: dual on an open-model mixed panel (Qwen2.5, MATH)

Commit this file before `bash scripts/run_qwen.sh full`, so the date is on record.

## History
- **2026-11-02.** Original plan: 600 MATH level 4-5 problems, dual vs single-speaker round robin.
  Amended the same day, before any run, to GPQA extended.
- **2026-11-03, GPQA pilot (Slurm job 13543, round 1 only, 99 of 100 questions).** The 7B expert scored
  30.3% (chance is 25%) and below three of the four 3B agents; Cochran's Q = 4.03, p = 0.40. Both pilot
  checks failed. The planned fallback (a 1.5B weak model) was not run: the failure was the expert
  itself being near chance and no better than the 3Bs, which a weaker weak model cannot fix. The GPQA
  arm is reported as uninformative for open models of this size. No debates were run.
- **2026-11-09, this version, before any MATH run with these models.** Back to MATH. Two standard debate
  formats are added as baselines, because the original "round robin" is a single-speaker rotation
  (only seats 1 and 2 are ever heard), which is weaker than the formats used in the literature.
  Three design variants are added to test why dual is shaped the way it is.

## Setup (fixed)
- **Panel:** expert = Qwen/Qwen2.5-7B-Instruct, four weak = Qwen/Qwen2.5-3B-Instruct, served by vLLM,
  temperature 0.7, same persona for all.
- **Problems:** the 600 MATH level 4-5 problems of the Gemini confirmation (`math_extra.jsonl`,
  `--level 4,5 --n 600`, fixed shuffle).
- **Settings:** `--sc-k 3 --sc-carry --rounds 3 --conf-mix avg --stop-unanimous`, 1 seed, all
  policies sharing the same round-1 answers. Context limit 16,384 tokens.
- **Seating:** `--seat-shuffle` (random seating per problem, the same for every policy). Each debate is
  also rerun in a second random seating.
- **Noise floor:** a separate run (`noise` stage) reruns `all_speak`, `dual` and `seq_rr` in the *same*
  seating on the same round-1 answers, so its flip rate is what re-asking the models alone produces.

## Pilot (not part of any test)
Round-1 answers only, first 100 problems, separate store. Go on only if the agents differ
(Cochran's Q, p < 0.05) and replies parse. If not, the weak model becomes Qwen2.5-1.5B-Instruct and this
file is amended before the full run. No debates are run in the pilot.

## Policies
| Policy | Who is heard | Role |
|---|---|---|
| `all_speak` | everyone, every round (simultaneous, Du et al.) | main baseline |
| `seq_rr` | everyone, in seat order, each seeing earlier seats | order-sensitive baseline |
| `round_robin` | one speaker per round by seat (single-speaker rotation) | earlier baseline, kept for continuity |
| `dual` | top prestige x confidence + top dissenter; then top agent with an unheard answer | the policy |
| `dual_conf` | dual with prestige switched off (confidence only) | does the track record matter? |
| `dual_shuf` | dual with prestige reassigned among agents | does picking the right agent matter? |
| `dual_2lead` | two voices for the leader's answer + the top dissenter | does redundancy add herding? |
| `dual_2diss` | leader + two dissenters (different answers when possible) | does a third answer help? |
| `per_answer` | best advocate of every distinct answer | full coverage, one voice each |

## Primary test (one)
**Dual is non-inferior to simultaneous debate (`all_speak`) in final-answer accuracy while broadcasting
far fewer arguments.** Confirmed if the lower end of the 95% paired bootstrap interval for
accuracy(dual) - accuracy(all_speak) is above -2 points (`mad_dual_v4.py report --ref all_speak`).
Nothing is added, dropped or rerun after seeing results, except automatic retries of failed calls.

## Secondary (reported, not used for the decision)
1. **Right minority** (1-2 of 5 agents right after round 1): dual ends correct more often than
   `all_speak` (one-sided exact McNemar, `analysis/minority_check.py`). Mechanism: in simultaneous
   debate a right minority's argument arrives alongside the majority's and is outweighed (herding).
2. **Right majority** (3-4 of 5 right): dual's losses against `all_speak` are small (two-sided McNemar).
3. **Order:** flip rate and correct -> wrong under reseating, dual vs `seq_rr` and vs `round_robin`
   (predicted: dual lower). For `dual` and `all_speak`, flip rates are compared with the noise floor
   (predicted: close to it, since both are anonymous); for `seq_rr` (predicted: above it).
4. **Design variants vs dual** (one-sided McNemar each, Holm-corrected as a family of five):
   `dual_conf` and `dual_shuf` predicted below dual; `dual_2lead` predicted below dual on right-minority
   problems; `dual_2diss` and `per_answer` predicted about equal on MATH (few problems with three live
   answers), reported two-sided.
5. **Cost:** arguments shown per agent per debate, from the stored transcripts, for each policy.
6. Each policy vs the expert alone and vs majority vote without debate (`analysis/mixed_report.py`).

Vote re-scoring (`analysis/vote_check.py`) and speaker-signal analysis (`analysis/signal_check.py`)
are exploratory.

## Addendum, 2026-11-11 (before any Qwen MATH run)
- `per_answer_last` is added: the same arguments as `per_answer`, with the most trusted one listed last.
  Secondary test: more wrong agents are right by round 2 than under `per_answer` (the order effect seen
  on Gemini: 49% vs 95% conversion by position).
- A nine-agent panel (`DESIGN=expert_open9`: one 7B expert + eight 3B) is run as a separate store with
  the same policies. Prediction: the gap between `per_answer` and `all_speak` on right-minority
  problems is larger with nine agents than with five. Reported as secondary.
- The Gemini run of 2026-11-10 found `per_answer` ahead of `all_speak`; the primary test here is unchanged.

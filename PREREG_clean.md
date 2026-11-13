# Pre-registration: clean prestige debate on full GPQA (Gemini)

Written 2026-11-12, before the run. Commit before `bash scripts/run_prestige.sh gpqa_full`.

## Design (prestige_mad_v2.py, fixed)
- **Panel:** Gemini 3.5 Flash (low thinking) as expert + four Gemini 3.1 Flash-Lite.
- **Prestige:** each agent's running round-1 accuracy on earlier questions; 0.5 for everyone during a
  warm-up of the first 20 questions.
- **Confidence:** the model's stated confidence (one call per agent in round 1; no self-consistency).
- **`prestige` policy:** one advocate per distinct answer (its highest-prestige holder), listed with
  the most trusted last; up to 3 rounds; final vote weighted by log-odds of prestige (one-agent-one-vote
  during the warm-up).
- **`all_speak` baseline:** every agent's argument shown each round; one-agent-one-vote.
- **`none` baseline:** majority vote of round-1 answers.
- Every contested debate runs in two random seatings; a flip is a different final answer between them.

## Questions
All 546 GPQA extended questions (`scripts/make_gpqa.py --subset extended`, the loader's fixed shuffle).
About 370 of them overlap with earlier exploratory runs (the 250 non-Diamond main questions and the 120
Diamond questions of the 2026-11-11/12 pilots). The ~176 others have never been used.

## Primary test
`prestige` vs `all_speak`, final accuracy on contested questions (round-1 answers not unanimous),
averaged over the two seatings; one-sided paired bootstrap, 95%. Supported if the lower bound is above 0.
The exact McNemar on the first seating is reported alongside.

## Secondary
1. The same comparison on the ~176 untouched questions only (matched by question stem).
2. Each policy vs the top-prestige agent alone (predicted: `prestige` about equal, `all_speak` below).
3. Flip rate, `prestige` vs `all_speak`.
4. The running mean of D = confidence - prestige per agent (predicted: the expert lowest after warm-up).

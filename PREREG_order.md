# Pre-registration: argument order and one voice per answer (Gemini, GPQA-Diamond)

Written 2026-11-11 before the run. Commit before `bash scripts/run_gemini_order.sh run`.

**Background (exploratory, 250 non-Diamond GPQA questions).** In simultaneous debate with random seating,
a wrong agent switched to the right answer 49% of the time when the correct argument was listed first
and 95% when it was listed last. Giving each distinct answer one voice (`per_answer`) beat simultaneous
debate by 3.8 points.

**Setup.** 3.5 Flash expert (low thinking) + four 3.1 Flash-Lite agents; first 120 GPQA-Diamond questions
(fixed shuffle; never used with this panel); `--sc-k 3 --sc-carry --rounds 3 --conf-mix avg
--stop-unanimous --seat-shuffle`, 1 seed, shared round-1 answers; each debate run in two random seatings.

**Policies.** `all_speak` (simultaneous debate); `per_answer` (best advocate of each answer, most trusted
listed first); `per_answer_last` (same arguments, most trusted listed last, by prestige x confidence);
`dual_2diss` (leader + two dissenters).

**Primary test.** `per_answer_last` vs `all_speak`, final accuracy on contested questions (some agents
right, some wrong after round 1), both seatings pooled per question; one-sided paired bootstrap, 95%.

**Secondary.** (1) `per_answer_last` vs `per_answer`: share of wrong agents right by round 2 (order
effect; predicted higher when the trusted argument is last). (2) `per_answer` vs `all_speak` on
right-minority questions. (3) `dual_2diss` vs `per_answer` (predicted about equal). (4) Flip rates.
With about 50 contested questions, power is limited; effects are reported with intervals either way.

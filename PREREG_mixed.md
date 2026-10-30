# Pre-registration: dual with mixed-skill panels (mad_dual.py)

Commit before running. Two designs, each its own test.

| Version | Design | Panel |
|---|---|---|
| v7 | `--design sizes` | Qwen2.5-Instruct 1.5B, 3B, 7B, 14B, 32B (designated expert: 32B) |
| v8 | `--design expert` | 1 x gemini-3.8-flash (thinking low) + 4 x gemini-3.1-flash-lite (no thinking) |

All agents use the same persona ("careful"); only the model differs.

## Problems and settings
- Primary: GPQA extended (546 questions), 3 seeds. Secondary: the 600 MATH level 4-5 problems from PREREG.md.
- `--sc-k 3 --sc-carry --rounds 3 --conf-mix avg --stop-unanimous`; policies
  round_robin, prestige_x_conf, conf_only, dual, dual_shuf, dual_randdiss, rr_dual.

## Hypotheses (fixed now)
1. **Prestige learns real skill:** Cochran's Q p < 0.05 and final prestige spread > 0.10 (clones: ~0.03).
2. **dual beats round robin** on accuracy. Primary test: paired bootstrap 95% CI over questions
   (seeds resampled together) excludes 0 in dual's favor.
3. **Learned prestige matters:** dual beats dual_shuf (prestige randomly reassigned among agents).
4. **Choosing the dissenter matters:** dual beats dual_randdiss.
5. **The panel is worth having:** dual is at least as accurate as the designated expert alone
   (its own majority over 3 samples). Reported with McNemar; this is the comparison reviewers will ask for.

Hypothesis 2 decides the main claim; 3-5 are secondary and reported whatever their result.
The "best agent chosen after the fact" line in mixed_report.py is a ceiling, never a baseline.

## Addendum (Oct 30): mixed panel on GPQA non-Diamond (250), 1 seed, run with mad_dual.py: 3.5 Flash (thinking low) + 4 x 3.1 Flash-Lite (model default thinking); policies round_robin, dual. Primary: one-sided exact McNemar, dual vs round robin, alpha 0.05. Secondary: dual vs the expert alone (McNemar), prestige spread and Cochran's Q, flips.

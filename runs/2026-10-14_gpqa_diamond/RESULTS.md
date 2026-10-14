# GPQA-Diamond run (2026-10-14)

198 graduate-level multiple-choice questions, 1 seed, gemini-3.1-flash-lite (5 personas), 3 rounds,
self-consistency confidence (`--sc-k 3 --sc-carry`), debate only when first answers disagree.

| Method | Accuracy | Contested (42) | Flip | First speaker right |
|---|---|---|---|---|
| Majority vote, no debate | 65.7% | | | |
| Round robin | **71.2%** | **66.7%** | 13.6% | **48%** |
| Prestige × confidence | 67.7% | 52.4% | 0.0% | 38% |
| Both sides (fix) | 70.2% | 61.9% | 5.6% | 43% |

Vs round robin (95% bootstrap CI; McNemar exact):
- Prestige × confidence: accuracy −3.5 [−6.6, −0.5], McNemar 2 vs 9, p = 0.065
- Both sides: accuracy −1.0 [−4.0, +2.0], McNemar 4 vs 6, p = 0.75; flips −8.1 [−13.1, −3.0]; correct → wrong −4.5 [−8.1, −1.0]

Why plain prestige × confidence lost: self-consistency is weak on multiple choice (AUROC 0.66 vs 0.80 on MATH),
so a confidently wrong agent took the floor unchallenged. **Both sides**: the top-ranked agent speaks first, then the
top-ranked agent holding a different answer; confidence = average of self-consistency and stated confidence (AUROC 0.78).

**Exploratory:** both sides was designed after diagnosing these same problems; confirm on new problems.

```
python3 mad_gemini.py trials --problems gpqa_diamond.jsonl --n 198 --seeds 1 --model gemini-3.1-flash-lite \
  --sc-k 3 --sc-carry --rounds 3 --policies round_robin,prestige_x_conf --stop-unanimous --store gpqa.jsonl
python3 mad_gemini.py trials ... --policies both_sides --conf-mix avg --init-from gpqa.jsonl --store gpqa_fix.jsonl
python3 mad_gemini.py report --store gpqa.jsonl,gpqa_fix.jsonl --policies round_robin,prestige_x_conf,both_sides
```

GPQA questions are not included in this repo (the dataset asks that examples not be published in plain text);
rebuild them with `make_gpqa.py` after accepting the terms on Hugging Face.

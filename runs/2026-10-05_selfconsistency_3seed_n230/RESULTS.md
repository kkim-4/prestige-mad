# Self-consistency run: 3 seeds × 230 MATH problems (2026-10-05)

**Result:** prestige × confidence matches round-robin accuracy, picks a correct first speaker more often, and removes round robin's dependence on agent order. It does not get more answers right.

## Setup
- gemini-3.1-flash-lite, thinking low, temperature 0.7, personas careful / fast / skeptic / teacher / contrarian
- MATH-500 problems 101–330 (unused in earlier runs), 3 seeds; 5 MATH-500 problems excluded (unparseable reference answer)
- Round 1: each agent answers 3 times; confidence = share of samples agreeing with its modal answer (`--sc-k 3`)
- Later rounds: confidence = share of the agent's round-1 samples matching its current answer (`--sc-carry`)
- 3 rounds; debate only when first answers disagree (`--stop-unanimous`); prestige = running mean of round-1 Brier scores toward 0.5; no warmup
- Analysis fixed in advance: primary = accuracy, prestige × confidence vs round robin (McNemar exact); secondary = flip rate, contested accuracy, confidence only

```
python3 mad_gemini.py trials --problems math.jsonl --n 230 --offset 100 --seeds 3 \
  --model gemini-3.1-flash-lite --sc-k 3 --sc-carry --rounds 3 \
  --policies round_robin,prestige_x_conf,conf_only --stop-unanimous --store sc.jsonl
python3 mad_gemini.py report --store sc.jsonl --policies round_robin,prestige_x_conf,conf_only
```

## Results (mean ± SD over 3 seeds)

| Method | Accuracy | Flip | Correct → wrong |
|---|---|---|---|
| Single agent (round 1) | 81.7 ± 0.7% | | |
| Majority vote, no debate | 82.9 ± 0.9% | | |
| Round-robin debate | **84.9 ± 1.3%** | 8.6 ± 0.3% | 2.5 ± 1.0% |
| Confidence only | 84.1 ± 0.7% | 4.5 ± 0.7% | 1.3 ± 0.8% |
| Prestige × confidence | 84.3 ± 0.0% | **1.6 ± 0.5%** | **0.3 ± 0.3%** |

Prestige × confidence vs round robin (95% bootstrap CI over problems):
- Accuracy −0.6 [−2.0, +0.7]; McNemar 12 vs 16, p = 0.57 → **no difference (primary)**
- Flip −7.0 [−9.6, −4.5]; correct → wrong −2.2 [−3.5, −1.0] → **significant (secondary)**

Confidence check: self-consistency AUROC 0.80 (3/3 agree → 93% right, 2/3 → 52%, 1/3 → 13%).
Speaker analysis: first speaker right on 44% of debated problems (prestige × confidence) vs 34% (round robin).
Ceiling: all agents wrong on 12.3% of problem-runs, so accuracy is capped near 87.7%.

**Caveat on stability:** the reordering test favors content-based speaker choice by design. Round robin picks its speaker by agent order, so reordering changes its speaker; prestige × confidence usually keeps the same speaker, so its 1.6% is mostly the randomness of fresh model calls. The 1.6% vs 4.5% gap over confidence only is probably tie-breaking (prestige breaks ties among 0.99-confidence agents consistently). Whether prestige × confidence is more robust beyond agent order needs a resampling test.

Files: `sc.jsonl` (every round-1 sample and debate outcome; `report` regrades it without API calls), `sc_summary.json`.

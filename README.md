# Prestige-Weighted, Confidence-Scored Multi-Agent Debate

Does it help a group of LLM agents to let the agent with the best track record
and the strongest confidence speak, instead of taking turns? This repo tests
**prestige × confidence** speaker selection against **round-robin** debate on
MATH and GPQA-Diamond, with agents of the same model.

One script (`mad_gemini.py`) on the Gemini API, plain Python.

## How it works

- 5 agents (one model, personas: careful, fast, skeptic, teacher, contrarian) solve a
  problem. Round 1 is independent; in later rounds each agent sees a transcript of
  who spoke and may revise. Each turn returns JSON: answer, confidence, key uncertainty.
- Each round the **policy** picks one speaker. The final answer is a vote over
  last-round answers (unweighted for round robin, weighted otherwise).
- **Prestige** = each agent's running Brier score on its independent round-1 answers,
  pulled toward 0.5. Problem *i* uses only earlier problems: no warmup, every problem scored.
- **Self-consistency confidence** (`--sc-k 3`): each agent answers round 1 three times;
  confidence = share of samples agreeing with its modal answer. `--sc-carry` carries it into
  later rounds (share of the agent's own round-1 samples matching its current answer).
- **Stability:** each debate is rerun with the agents reordered; a changed final answer is a flip.
- All policies share the same round-1 answers, so comparisons are paired.

| Policy | Speaker / vote weight |
|---|---|
| `round_robin` | rotate speaker, unweighted vote |
| `prestige_x_conf` | prestige × confidence (proposed) |
| `conf_only` | confidence alone |
| `prestige_only` | prestige alone |
| `both_sides` | top prestige × confidence agent speaks first, then the top-ranked agent holding a different answer |

## Setup

```bash
pip install -r requirements.txt   # google-genai, pydantic, sympy (sympy is needed for grading)

# Vertex AI
gcloud auth application-default login
export GOOGLE_GENAI_USE_VERTEXAI=True GOOGLE_CLOUD_PROJECT=your-project GOOGLE_CLOUD_LOCATION=us-east4
# or the Developer API
export GEMINI_API_KEY=...
```

## Usage

```bash
# Experiment 2 (self-consistency); --dry-run prints the plan and call count first
python3 mad_gemini.py trials --problems math.jsonl --n 230 --offset 100 --seeds 3 \
  --model gemini-3.1-flash-lite --sc-k 3 --sc-carry --rounds 3 \
  --policies round_robin,prestige_x_conf,conf_only --stop-unanimous \
  --stagger 0.3 --workers 6 --store sc.jsonl --dry-run

# Regrade and summarize stored results (no API calls); several stores can be combined
python3 mad_gemini.py report --store sc.jsonl --policies round_robin,prestige_x_conf,conf_only
```

- `trials` is resumable: rerun the same command after a crash or Ctrl-C.
- `--round1-only` collects first answers only (a cheap confidence check); rerun without it to debate.
- `--stop-unanimous` debates only problems where first answers disagree.
- `--panel` sets a model per agent; `--mock` runs simulated agents with no API calls.
- `--conf-mix avg` averages self-consistency with stated confidence; `--init-from` reuses another run's round-1 answers.
- `python3 mad_gemini.py resample --store sc.jsonl` reruns debated problems with the same agent order and fresh calls
  (a stability test that doesn't depend on reordering).
- `make_math_extra.py` builds unseen problems from the full MATH test set (minus MATH-500);
  `make_gpqa.py` builds GPQA-Diamond (gated: accept its terms on Hugging Face first).
- `free_analyses.py` and `contested_breakdown.py` run extra paired tests on a stored run, with no API calls.
- `signal_check.py` tests whether confidence and prestige carry signal (agent differences, within-problem AUROC,
  speaker choice with warmup and oracle prestige); `round_metrics.py` prints per-round accuracy, switches and
  speaker-following for runs made with the current script (every agent's answer is now stored per round).
- `--warmup-from W.jsonl [--warmup-n N]` starts prestige from round-1 answers on separate problems (collect them with
  `--round1-only`); results are labelled `<policy>_warmN`. `make_gpqa.py --subset main_minus_diamond` builds warmup questions.

`report` prints per-agent accuracy, a confidence check (AUROC), no-debate baselines,
accuracy and flip rates per policy, paired bootstrap CIs, McNemar's exact test, and a
speaker analysis.

## Results (gemini-3.1-flash-lite)

**Experiment 1: MATH, stated confidence, problems 1–100, 3 seeds, every problem debated.**

| Method | Accuracy | Flip |
|---|---|---|
| Majority vote, no debate | 85.3% | – |
| Round robin | **88.3%** | 9.7% |
| Prestige × confidence | 86.0% | 6.7% |

Accuracy −2.3 points [−5.7, +0.3], n.s. Stated confidence sat at 0.99, so the policy had
no per-problem signal and mostly gave the floor to the confident majority.

**Experiment 2: MATH, self-consistency confidence, problems 101–330, 3 seeds, debate on disagreement.**

| Method | Accuracy | Flip | Correct → wrong |
|---|---|---|---|
| Majority vote, no debate | 82.9% | – | – |
| Round robin | **84.9%** | 8.6% | 2.5% |
| Confidence only | 84.1% | 4.5% | 1.3% |
| Prestige × confidence | 84.3% | **1.6%** | **0.3%** |
| Both sides (`--conf-mix avg`, run later on the same round-1 answers) | **86.8%** | 5.1% | 1.4% |

- **Accuracy (primary): no difference.** −0.6 points [−2.0, +0.7]; McNemar 12 vs 16, p = 0.57.
- **Confidence works:** 3/3 agreeing samples were right 93% of the time, 2/3 52%, 1/3 13% (AUROC 0.80).
- **Better speakers:** the first speaker held the right answer 44% of the time vs 34% for round robin.
- **Fewer flips under reordering** (−7.0 points [−9.6, −4.5]), but much of this is by design:
  round robin's speaker is set by agent order, prestige × confidence's by content.
- **Little headroom:** all agents were wrong on 12.3% of problem-runs, capping accuracy near 88%.
- **Debate beats voting:** round robin +2.0 points over the no-debate majority vote (p = 0.009),
  all from the 19% of problems where first answers differ.
- **Prestige adds stability beyond confidence:** 2.9 points fewer flips than confidence only [−4.8, −1.2], same accuracy.
- **Both sides beats round robin (borderline):** +1.4 points [+0.0, +2.9]; McNemar 19 vs 9, p = 0.087; contested
  accuracy 80.4% vs 72.2%. Designed on GPQA, so semi-out-of-sample here. (A later report regrades all rows about
  0.5 points higher, e.g. round robin 85.4%; paired comparisons are unchanged.)

**Experiment 3: GPQA-Diamond (198 questions), 1 seed, same setup as experiment 2.**

| Method | Accuracy | Contested (42) | Flip |
|---|---|---|---|
| Majority vote, no debate | 65.7% | – | – |
| Round robin | **71.2%** | **66.7%** | 13.6% |
| Prestige × confidence | 67.7% | 52.4% | 0.0% |
| Both sides (`--conf-mix avg`) | 70.2% | 61.9% | 5.6% |

- **Plain prestige × confidence lost** (−3.5 points, McNemar p = 0.065): self-consistency is weak on
  multiple choice (AUROC 0.66), so a confidently wrong leader went unchallenged.
- **Both sides ties round robin** (−1.0 [−4.0, +2.0], p = 0.75) with significantly fewer flips (−8.1 points)
  and correct → wrong reversals (−4.5). Exploratory: designed on these same problems.

- **v2** (`mad_gemini_v2.py`: calibrated confidence, challenger, explore, verify prompt): 70.7% vs 70.7% for a
  round-robin rerun (McNemar 5 vs 5); flips 3.0% vs 11.1%, correct → wrong 0 vs 9 problems.

**Why GPQA ties and MATH wins** (`signal_check.py`, no API calls): the five agents are not reliably different
(Cochran's Q p = 0.18 GPQA, 0.47 MATH), so prestige has little to learn. Within a contested problem, a right agent
is more confident than a wrong one 69% of the time on MATH but 51% on GPQA, and the plain majority is right on only
46% of contested GPQA problems. Speaker choice can only help where confidence identifies the right agent.

**Takeaway:** letting the most credible dissenter answer the leader (both sides) turns prestige × confidence from a
tie into a borderline win on MATH and a tie with far fewer flips on GPQA. It needs a fresh confirmatory run.

Earlier single-run results (76.7% for both policies) are superseded; they used a faulty
grader and a 40% warmup.

## Repo layout

| Path | Contents |
|---|---|
| `mad_gemini.py` | Agents, policies, `trials` / `report` commands, grader |
| `make_math_extra.py`, `make_gpqa.py` | Build unseen MATH test problems / GPQA-Diamond |
| `mad_gemini_v2.py` | Adds the `v2` calibrated-challenge policy (`--verify`, `--explore-below`) |
| `free_analyses.py`, `contested_breakdown.py`, `signal_check.py`, `round_metrics.py` | Extra analyses on stored runs (no API calls) |
| `mad_gemini_independent_personas.py` | Earlier independent-personas variant, kept for reference |
| `runs/<date>_<name>/` | Raw results (`*.jsonl`, regradable with `report`) and summaries per experiment. GPQA raw files are kept out of the repo because GPQA asks that its questions not be published in plain text. |

## Next steps

- Confirm `both_sides` vs round robin on fresh MATH problems (`math_extra.jsonl`), with the analysis fixed in advance.
- Warmup test: does an established prestige change speaker choice (expected: little, since the agents are similar)?
- Separate its two changes: combined confidence vs the both-sides speaker rule.
- Resampling stability results (same agent order, fresh calls).
- Let speakers show their full working instead of 3 sentences.
- Mixed-skill panel, where prestige has real reliability differences to learn.

## Known issues

- Vertex AI connections drop intermittently; retries with jittered backoff absorb them.
- Five MATH-500 problems are skipped because their reference answer can't be parsed.
# Prestige-Weighted, Confidence-Scored Multi-Agent Debate

Does it help a group of LLM agents to let the agent with the best track record
and the strongest confidence speak, instead of taking turns? This repo tests
**prestige × confidence** speaker selection against **round-robin** debate on
MATH problems, with agents of the same model.

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
- `make_math_extra.py` builds unseen problems from the full MATH test set (minus MATH-500).

`report` prints per-agent accuracy, a confidence check (AUROC), no-debate baselines,
accuracy and flip rates per policy, paired bootstrap CIs, McNemar's exact test, and a
speaker analysis.

## Results (gemini-3.1-flash-lite, MATH, 3 seeds each)

**Experiment 1: stated confidence, problems 1–100, every problem debated.**

| Method | Accuracy | Flip |
|---|---|---|
| Majority vote, no debate | 85.3% | – |
| Round robin | **88.3%** | 9.7% |
| Prestige × confidence | 86.0% | 6.7% |

Accuracy −2.3 points [−5.7, +0.3], n.s. Stated confidence sat at 0.99, so the policy had
no per-problem signal and mostly gave the floor to the confident majority.

**Experiment 2: self-consistency confidence, problems 101–330, debate on disagreement.**

| Method | Accuracy | Flip | Correct → wrong |
|---|---|---|---|
| Majority vote, no debate | 82.9% | – | – |
| Round robin | **84.9%** | 8.6% | 2.5% |
| Confidence only | 84.1% | 4.5% | 1.3% |
| Prestige × confidence | 84.3% | **1.6%** | **0.3%** |

- **Accuracy (primary): no difference.** −0.6 points [−2.0, +0.7]; McNemar 12 vs 16, p = 0.57.
- **Confidence works:** 3/3 agreeing samples were right 93% of the time, 2/3 52%, 1/3 13% (AUROC 0.80).
- **Better speakers:** the first speaker held the right answer 44% of the time vs 34% for round robin.
- **Fewer flips under reordering** (−7.0 points [−9.6, −4.5]), but much of this is by design:
  round robin's speaker is set by agent order, prestige × confidence's by content.
- **Little headroom:** all agents were wrong on 12.3% of problem-runs, capping accuracy near 88%.

**Takeaway:** among agents of the same level, prestige × confidence picks better speakers and
removes round robin's dependence on agent order, but does not get more answers right.

Earlier single-run results (76.7% for both policies) are superseded; they used a faulty
grader and a 40% warmup.

## Repo layout

| Path | Contents |
|---|---|
| `mad_gemini.py` | Agents, policies, `trials` / `report` commands, grader |
| `make_math_extra.py` | Builds unseen MATH test problems |
| `mad_gemini_independent_personas.py` | Earlier independent-personas variant, kept for reference |
| `runs/<date>_<name>/` | Raw results (`*.jsonl`, regradable with `report`) and summaries per experiment |

## Next steps

- Resampling stability test: same agent order, fresh model calls.
- Check how often debate changes answers on unanimous problems.
- Harder problems, so more are contested.
- Mixed-skill panel, where prestige has real reliability differences to learn.

## Known issues

- Vertex AI connections drop intermittently; retries with jittered backoff absorb them.
- Five MATH-500 problems are skipped because their reference answer can't be parsed.
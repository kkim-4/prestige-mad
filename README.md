# Prestige-Weighted, Confidence-Scored Multi-Agent Debate

Does scoring each LLM agent's stated confidence against whether it was actually
right, and letting that running score decide who gets heard, make a group of
agents more accurate or more stable than a debate where confidence carries no
consequence?

One self-contained script (`mad_gemini.py`) on the Gemini API, plain Python, no framework.

## How it works

- 5 agents (same model, different personas: careful, fast, skeptic, teacher,
  contrarian) debate a problem for several rounds. Each turn returns JSON:
  answer, confidence (0–1), key uncertainty.
- Each round one speaker is chosen by the policy; everyone sees that turn and may revise.
- Final answer = weighted vote over last-round answers, grouping mathematically
  equivalent forms (`14/3` = `\frac{14}{3}`).
- After each problem, confidences are scored against the truth with a proper
  scoring rule (Brier or log), and each agent's **prestige** (an EMA of that
  score) updates. Prestige never changes mid-debate.

| Policy | Speaker / vote weight |
|---|---|
| `prestige_x_conf` | prestige × confidence (proposed) |
| `conf_only` | confidence, no consequence |
| `posthoc` | Platt-calibrated confidence |
| `prestige_only` | prestige alone |
| `round_robin` | rotate speaker, unweighted vote |
| `fixed` | prestige frozen after warmup |

**Stability:** each test problem is rerun with prestige frozen under
perturbations (`--perturbations order,drop,fresh`: reorder agents, drop one,
swap in a fresh agent), and we count how often the final answer flips,
split into correct→wrong and wrong→correct.

## Setup

```bash
pip install -r requirements.txt   # google-genai, pydantic, sympy (sympy is required for answer grading)

# Vertex AI (bills Google Cloud credit)
gcloud auth application-default login
export GOOGLE_GENAI_USE_VERTEXAI=True GOOGLE_CLOUD_PROJECT=your-project GOOGLE_CLOUD_LOCATION=us-east4
# or the Developer API (separate prepaid balance)
export GEMINI_API_KEY=...
```

## Usage

```bash
python3 mad_gemini.py mock --n 80 --seeds 8        # simulated agents, no API calls
python3 mad_gemini.py smoke                         # one real call, full errors
python3 mad_gemini.py calib --problems math.jsonl --n 100 --model gemini-3.1-flash-lite
date +%s    # note the timestamp so this run's calls can be isolated later
python3 mad_gemini.py trials --problems math.jsonl --n 100 --seeds 3 --model gemini-3.1-flash-lite \
    --policies round_robin,prestige_x_conf --stagger 0.3 --workers 6 --store trials.jsonl --dry-run
python3 mad_gemini.py report --store trials.jsonl   # regrade + summarize, no API calls
```

`trials` is resumable: rerun the same command after a crash or Ctrl-C.

Problem files are JSONL: `{"question", "answer"}` or MATH-style
`{"problem", "solution", "level"}` (the `\boxed{}` answer is extracted;
`--level 4,5` filters). All calls are logged to `calls.jsonl`. Two scripts
analyze that log:

- `summarize_calls.py`: accuracy and confidence tables by model and dataset
  (`--since <ts>` isolates one run).
- `analyze_calibration.py`: confidently-wrong and hedged-but-right answers per agent.

## Findings so far

**Main result (`trials`: gemini-3.1-flash-lite, 100 MATH problems, 3 seeds, 3 rounds, no warmup).**
Mean ± SD across seeds; results in `results/`.

| Method | Accuracy | Flip | C→W |
|---|---|---|---|
| Single agent (round 1) | 81.1 ± 2.3% | – | – |
| Majority vote, no debate | 85.3 ± 2.3% | – | – |
| Round-robin debate | **88.3 ± 1.5%** | 9.7 ± 2.5% | 4.3 ± 2.1% |
| Prestige × confidence | 86.0 ± 1.7% | **6.7 ± 2.3%** | **2.0 ± 2.6%** |

Paired differences (prestige × confidence − round robin, 95% bootstrap CI over problems):
accuracy −2.3 [−5.7, +0.3], flip −3.0 [−6.7, +0.3], C→W −2.3 [−5.3, +0.3]. None significant.

- **Debate helps:** round robin beats majority vote by 3 points and one agent by 7.
- **Prestige × confidence doesn't beat round robin.** It trends more stable but less accurate,
  most visibly on contested problems (73.9% vs 81.6%).
- **Likely reason:** with near-identical agents and confidence pinned near 0.99, it hands the
  floor to the confident majority every round, so a correct dissenter is rarely heard.
- In the mock test with genuinely different agents, prestige × confidence does win.

`trials` scores prestige from each agent's independent first-round answer (running mean,
prior 0.5 worth 5 problems), so every problem is scored and debates run in parallel. Both
policies share the same first-round answers (paired). Earlier single-run results (76.7% for
both policies, old grader, 40% warmup) are superseded.

## Next steps

- Check the mechanism from stored data: speaker agreement with the round-1 majority per policy.
- Sample the speaker by weight (`--tau`) instead of always picking the top one.
- Mixed-skill population (different models or thinking levels) so reliability actually varies.
- More problems rather than more seeds: the paired CI is about ±3 points at n=100.

## Known issues

- Vertex AI connections drop intermittently; retries with jittered backoff absorb
  most of it, but long runs are slow. Cost is ~$0.009/call.
- `--api interactions` isn't available on Vertex; use the default `legacy` transport.
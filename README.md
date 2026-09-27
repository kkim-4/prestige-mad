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
python3 mad_gemini.py debate --problems math.jsonl --n 100 --model gemini-3.1-flash-lite \
    --policies round_robin,prestige_x_conf --rounds 3 --perturbations order --stagger 1.0
```

Problem files are JSONL: `{"question", "answer"}` or MATH-style
`{"problem", "solution", "level"}` (the `\boxed{}` answer is extracted;
`--level 4,5` filters). All calls are logged to `calls.jsonl`. Two scripts
analyze that log:

- `summarize_calls.py`: accuracy and confidence tables by model and dataset
  (`--since <ts>` isolates one run).
- `analyze_calibration.py`: confidently-wrong and hedged-but-right answers per agent.

## Findings so far

**Confidence is only informative in a narrow difficulty band.** gemini-3.8-flash
saturates MATH (90–100%, confidence pinned at 0.99). The lite models collapse on
AIME and MathArena (7–27%). The usable pairing is **gemini-3.1-flash-lite on
MATH**: solo accuracy 0.77, mean confidence 0.97. Confidence ranks answers
correctly (0.99 → 83% right, 0.95 → 62%, ≤0.90 → 37%) but is 15–30 points too
high. 85% of wrong answers are stated at ≥0.95 confidence.

**Debate results (gemini-3.1-flash-lite, MATH, 3 rounds, order perturbation):**

| Run (test problems) | Policy | Accuracy | Flip | C→W |
|---|---|---|---|---|
| n=30 (18) | round_robin / prestige_x_conf | 0.78 / 0.89 | 0.17 / 0.06 | 0 / 0 |
| n=50 (30) | round_robin / prestige_x_conf | 0.80 / 0.77 | 0.10 / 0.10 | 0 / 0 |
| **n=100 (60)** | round_robin / prestige_x_conf | **0.767 / 0.767** | **0.13 / 0.22** | **0.02 / 0.08** |
| solo, no debate (100) | mean of 5 agents | 0.77 | – | – |

No detectable difference between the two policies (n=100 flip rate p=0.34,
C→W p=0.21, Fisher exact). Debate also doesn't beat one agent answering alone.
A small earlier AIME run (n=5) looked favorable but had only 3 test problems.

**Why prestige has nothing to work with in this setup:**
- **Prestige tracks only the last problem.** It updates once per round (3× per
  problem) at alpha 0.3, so about 66% of the weight is on the most recent problem.
- **Agents are graded together.** By round 3 they usually agree, so they're right
  or wrong together and prestige stays nearly identical across agents.
- **Confidence is stuck at the top.** Most answers are 0.99; within a disputed
  problem, right and wrong agents tie on confidence 61% of the time.
- **The agents are one model.** Their true accuracy spans only 0.72–0.80, so there
  is little real reliability difference to learn. In the mock test with genuinely
  different agents, prestige × confidence does win.

**Grading caveat:** runs up to tag `v0.1-baseline` used an answer grader that
missed some equivalent forms (`\frac 34`, vectors vs. tuples, base subscripts).
It's fixed now; earlier accuracies may be understated by up to ~7 points and
should be regraded from `calls.jsonl`.

## Next steps

- Mixed-skill population (different models or thinking levels) so reliability actually varies.
- Score prestige once per problem from round-1 (independent) answers; slower EMA (alpha ≈ 0.05).
- Random tie-breaking for speaker choice; finer-grained confidence elicitation.
- Credit for influence, not just calibration ("right and heard" vs. "right but ignored").

## Known issues

- Vertex AI connections drop intermittently; retries with jittered backoff absorb
  most of it, but long runs are slow. Cost is ~$0.009/call.
- `--api interactions` isn't available on Vertex; use the default `legacy` transport.

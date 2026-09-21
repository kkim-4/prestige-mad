# Prestige-Weighted, Confidence-Scored Multi-Agent Debate

A small-scale test of a simple idea: if you score each agent's stated confidence
against whether it was actually right, and let that running score decide who gets
heard, does a group of LLM agents produce more stable, more accurate answers than
if confidence carries no consequence?

Single self-contained script (`mad_gemini.py`, ~650 lines), built against the
Gemini API. No framework — the debate loop, scoring rule, and evaluation harness
are all plain Python.

## How it works

- A fixed population of 5 agents (same model, different system-prompt personas —
  careful, fast, skeptic, teacher, contrarian) discuss a problem over several rounds.
- Each turn, an agent returns structured JSON: an answer, a stated confidence
  (0–1), and a named key uncertainty.
- One speaker per round is selected by **prestige × stated confidence**. Everyone
  else sees that turn and can revise before the next round.
- After the final round, the group's answer is a weighted vote over each agent's
  last submission.
- Once ground truth is revealed, every agent's confidence is scored against
  correctness with a proper scoring rule (Brier or log score), and a running
  **prestige** ledger — an exponential moving average of that score — updates.
  Prestige is visible to every agent each round; it only updates between
  problems, never mid-debate, so nothing about correctness leaks into the
  discussion itself.

### Baselines for comparison

| Policy | Weight used |
|---|---|
| `prestige_x_conf` | prestige × confidence (the proposed mechanism) |
| `conf_only` | confidence, with no consequence for being wrong |
| `posthoc` | confidence, Platt-calibrated after the fact |
| `prestige_only` | prestige alone (ablation) |
| `round_robin` | uniform, ignores both |
| `fixed` | prestige learned during warmup, then frozen |

### Stability evaluation

Each test problem is rerun three ways with prestige frozen — agents reordered,
one agent dropped, one agent replaced by a fresh copy with prior-only prestige —
and the flip rate (how often the final answer changes) is measured, split into
correct→wrong vs. wrong→correct. The hypothesis predicts the proposed mechanism
should specifically suppress correct→wrong flips relative to the no-consequence
baselines.

## Setup

```bash
pip install google-genai pydantic sympy
```

Authenticate via **either**:

**Gemini Developer API (AI Studio key)**
```bash
export GEMINI_API_KEY="..."
```

**Vertex AI** (needed to draw on Google Cloud trial credit — the Developer API
uses a separate prepay balance)
```bash
gcloud auth application-default login
export GOOGLE_GENAI_USE_VERTEXAI=True
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-east4"   # pick a region; see note below
```

## Usage

```bash
# 1. Sanity-check the mechanism with simulated agents — no API calls
python3 mad_gemini.py mock --n 80 --seeds 8

# 2. Confirm connectivity — one real call per transport
python3 mad_gemini.py smoke

# 3. Milestone 1: does verbalized confidence actually track correctness?
python3 mad_gemini.py calib --n 30 --problems aime.jsonl --model gemini-3.8-flash

# 4. Full comparison across policies
python3 mad_gemini.py debate --problems aime.jsonl --n 30 \
    --policies round_robin,prestige_x_conf --rounds 3
```

Run `python3 mad_gemini.py --help` for the full flag list (rounds, scoring rule,
softmax speaker sampling, retry pacing, etc).

### Problem files

JSONL, either:
- `{"question": "...", "answer": "..."}`, or
- MATH-style `{"problem": "...", "solution": "...", "level": "Level N"}` — the
  `\boxed{}` answer is extracted automatically, and `--level 4,5` filters by
  difficulty.

## Findings so far

Gemini 3.8 Flash is close to saturated on the MATH benchmark (97–100% accuracy
even on Level 5 problems, confidence pinned at 0.99 regardless of correctness —
no signal to condition on). AIME 2024 is the first benchmark where confidence
actually tracked correctness (87–90% accuracy, confidence spread 0.20–0.98
depending on the problem). Debate comparison in progress.

## Known limitations / next steps

- AIME 2024 is only 30 problems; test split after warmup is thin (~6–18
  problems). Plan to add AIME 2023 / HMMT.
- Confidence currently scores the final answer only, not intermediate steps.
- `--api interactions` (the newer Gemini endpoint) is not available under
  Vertex AI — use the default `legacy` transport there.
- Vertex regional reliability varies; `smoke` is a cheap way to test a region
  before committing to a long run.

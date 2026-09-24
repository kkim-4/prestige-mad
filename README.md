# Prestige-Weighted, Confidence-Scored Multi-Agent Debate

A small-scale test of a simple idea: if you score each agent's stated confidence
against whether it was actually right, and let that running score decide who gets
heard, does a group of LLM agents produce more stable, more accurate answers than
if confidence carries no consequence?

Single self-contained script (`mad_gemini.py`, ~750 lines), built against the
Gemini API on Vertex AI. No framework — the debate loop, scoring rule, and
evaluation harness are all plain Python.

## How it works

- A fixed population of 5 agents (same model, different system-prompt personas —
  careful, fast, skeptic, teacher, contrarian) discuss a problem over several rounds.
- Each turn, an agent returns structured JSON: an answer, a stated confidence
  (0–1), and a named key uncertainty.
- One speaker per round is selected by **prestige × stated confidence**. Everyone
  else sees that turn and can revise before the next round.
- After the final round, the group's answer is a weighted vote over each agent's
  last submission, grouped by grading-equivalence (not plain string match), so
  `14/3` and `\frac{14}{3}` combine weight instead of splitting it.
- Once ground truth is revealed, every agent's confidence is scored against
  correctness with a proper scoring rule (Brier or log score), and a running
  **prestige** ledger — an EMA of that score — updates. Prestige is visible to
  every agent each round; it only updates between problems, never mid-debate.

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

Each test problem is rerun with prestige frozen — agents reordered, one agent
dropped, one agent replaced by a fresh copy with prior-only prestige — and the
flip rate (how often the final answer changes) is measured, split into
correct→wrong vs. wrong→correct. `--perturbations order,drop,fresh` selects
which checks run (`drop` is 5x cost; drop it for a faster pass). All selected
perturbations run concurrently, not sequentially, paced by a shared rate limiter.

## Setup

```bash
pip install google-genai pydantic sympy
```

**Vertex AI** (needed to draw on Google Cloud trial credit — the Developer API
uses a separate prepay balance that runs out independently):
```bash
gcloud auth application-default login
export GOOGLE_GENAI_USE_VERTEXAI=True
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-east4"   # test region reliability before committing to a run
```

**Gemini Developer API** (simpler, but a separate billing pool from Cloud credit):
```bash
export GEMINI_API_KEY="..."
```

## Usage

```bash
# 0. Sanity-check the mechanism with simulated agents — no API calls
python3 mad_gemini.py mock --n 80 --seeds 8

# 1. Confirm connectivity — one real call per transport, full errors on failure
python3 mad_gemini.py smoke

# 2. MILESTONE 1 — does verbalized confidence actually track correctness?
python3 mad_gemini.py calib --n 30 --problems aime.jsonl --model gemini-3.8-flash

# 3. Full comparison across policies
python3 mad_gemini.py debate --problems aime.jsonl --n 30 \
    --policies round_robin,prestige_x_conf --rounds 3 --perturbations order,fresh
```

Run `python3 mad_gemini.py --help` for the full flag list (rounds, scoring rule,
softmax speaker sampling, retry pacing/`--stagger`, `--level` filtering, etc).

### Problem files

JSONL, either:
- `{"question": "...", "answer": "..."}`, or
- MATH-style `{"problem": "...", "solution": "...", "level": "Level N"}` — the
  `\boxed{}` answer is extracted automatically, `--level 4,5` filters by difficulty.

### Analysis tools

- `summarize_calls.py` — rebuilds accuracy/confidence tables from `calls.jsonl`,
  auto-sorting a mixed-session log by model and dataset. Supports `--since
  <unix_ts>` to isolate one run from a log spanning many attempts, and
  `--warmup-vs-test` to check whether debate-phase accuracy differs from a
  clean solo baseline.
- `analyze_calibration.py` — isolates "confidently wrong" and "hedged but
  right" calls from a log, with per-agent bluffing/hedging rates.

## Findings so far

**Model/dataset capability bracket:** the "does confidence track correctness"
signal only shows up in a narrow band. Every dataset easier than AIME saturates
`gemini-3.8-flash` (MATH at any level, AMC 12 — 100% accuracy, confidence
pinned at 0.99, ECE ~0.01). Every cheaper "lite" model tried bottoms out badly
even on AIME (10–27% accuracy, ECE 0.5–0.8). Only `gemini-3.8-flash` on
AIME-tier difficulty (87–90% accuracy, real confidence spread) is usable —
this looks like a genuine threshold, not a smooth gradient.

**First debate comparison** (AIME 2024, n=5, small sample): `prestige_x_conf`
matched `round_robin` on accuracy (0.33 each) but cut flip rate 3.8x (0.10 vs
0.38) and had zero correct→wrong flips vs. round_robin's answer flipping on
*every* reordering. Directional support for the hypothesis; n=3 test problems
is not enough to trust the exact numbers.

## Known limitations / next steps

- Scaling the debate comparison beyond n~10 has been blocked by Vertex
  throughput collapsing intermittently (not cost — measured ~$0.009/call).
  Retry when connection quality recovers; `smoke` is a cheap pre-check but
  doesn't guarantee a long run stays healthy.
- AIME 2024 alone is only 30 problems; `aime_combined.jsonl` (2024+2025, 60)
  and `matharena.jsonl` (139, pooled 2025 competitions) are prepared for when
  throughput allows a larger run.
- Confidence currently scores the final answer only, not intermediate steps.
- `--api interactions` is not available under Vertex AI — use the default
  `legacy` transport there.
- Credit assignment currently rewards calibration only, not influence —
  doesn't yet distinguish "right and heard" from "right but ignored."

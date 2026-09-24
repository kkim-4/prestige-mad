# MAD on Gemini — refined small-scale build

Prestige-weighted, confidence-scored multi-agent debate on the Gemini API
(Google AI Studio key). Same mechanism as the Claude MVP, with the gaps closed.

## What "refined" changed

| Area | Claude MVP | This build |
|---|---|---|
| Output contract | Prompted JSON + regex fallback | Pydantic schema enforced server-side via `response_format`; a bad response is retried, never silently parsed |
| Confidence prompt | "report your probability" | Adds `key_uncertainty` field (forces the agent to name a failure mode before stating a number — cheap, tends to improve calibration) and explicit anti-deference instruction |
| Baselines | round-robin, fixed, conf-only, prestige-only | + `posthoc`: Platt scaling fitted on warmup (the Conf-MAD "correct it after the fact" comparison) |
| Perturbations | order, drop-one | + `fresh` (an agent replaced by a prior-prestige copy that never saw prior problems) and per-kind flip rates |
| Speaker selection | argmax | argmax or softmax sampling (`--tau`) |
| Scoring rule | Brier | Brier or log (`--rule log`) |
| Diagnostics | mean Brier | + ECE per agent, JSONL log of every model call, `results.json` |
| Throughput | serial | agents within a round run in parallel |
| Milestone 1 | manual | `run.py calib` — single-agent calibration check before any debate |

## Setup

```bash
pip install -r requirements.txt          # google-genai>=2.3.0, pydantic
export GEMINI_API_KEY=...                # from https://aistudio.google.com/apikey
export MAD_MODEL=gemini-3.8-flash        # default; gemini-3.5-flash-lite for cheap sweeps
```

Calls go through the Interactions API (`client.interactions.create`) with `store=False`,
so nothing is retained server-side and each call is stateless. `thinking_level` is passed
through when set (`--thinking low|high`); leave unset first, then check whether thinking
changes the calibration numbers — that itself is a useful ablation.

## Run order

```bash
# 0. plumbing + mechanism sanity, no API
python run.py mock --n 80 --seeds 6

# 1. MILESTONE 1 — does Gemini's stated confidence track correctness at all?
python run.py calib --n 20                              # synthetic arithmetic
python run.py calib --problems math.jsonl --n 40        # MATH

# 2. full comparison
python run.py debate --problems math.jsonl --n 30 --rounds 3
python run.py debate --problems math.jsonl --n 30 --policies prestige_x_conf,conf_only,posthoc --tau 0.15
```

`math.jsonl` can be raw MATH-format lines (`{"problem","solution"}` — the `\boxed{}` answer
is extracted) or `{"question","answer"}`.

Cost per test problem per policy ≈ agents × rounds × (canonical + order + N drop + fresh + learn)
= 5 × 3 × 9 = 135 calls. On Flash that's cheap; on Pro, use `--n 10` first.

## Reading the output

- `calib`: if no persona has `acc ≈ meanconf` and ECE > ~0.2 everywhere, verbalized confidence
  is not carrying signal on that task. Fix elicitation (lower temperature, `--thinking low`,
  or swap to sampling-based confidence) before running debates. This is the doc's first gate.
- `debate`: the hypothesis predicts `prestige_x_conf` < `conf_only` ≈ `posthoc` on flip rate,
  particularly C→W, with accuracy not worse. `fixed` is the interesting comparison: if it
  matches `prestige_x_conf`, per-question conditioning isn't earning its place on that benchmark.

## What is deliberately not in here

- LangGraph. The loop is ~80 lines; migrate when you add the challenge phase.
- Step-level scoring (PRM800K / Math-Shepherd). Swap the `correct` computation in
  `Orchestrator.run`'s update loop.
- Batch API — not available on Interactions yet; if cost matters, the legacy
  `generateContent` path supports it.

## Using AI Studio itself

The Playground can't run an orchestrator loop, so the script is the right tool. Two things
AI Studio does add: the Logs page (paid tier) shows every stored interaction if you set
`store=True` in `agents.py`, which is a convenient debugger for reading agent rationales;
and Build mode can vibe-code a small web front-end over `results.json` / `calls.jsonl` for
inspecting transcripts — the JSONL format here was chosen to make that trivial.

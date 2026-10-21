# Prestige-Weighted, Confidence-Scored Multi-Agent Debate

Does a group of LLM agents answer better when the agents with the best track record and the
strongest confidence decide who speaks, instead of taking turns? This repo compares
**prestige × confidence** speaker policies with **round-robin** debate and **majority vote**
on MATH and GPQA, using five agents of the same model (gemini-3.1-flash-lite).

**Current best policy: `dual`.** After round 1, the top prestige × confidence agent and the
strongest agent holding a different answer speak together. It came out ahead of round robin in
every paired run (+2 to +3 points), with about half the flips, and beats majority vote.
Pooled over three runs it wins 32 problems to 15 (one-sided p = 0.009); the single planned
confirmation run was not significant on its own (p = 0.20).

## How it works

- **Agents.** 5 agents, one model, personas: careful, fast, skeptic, teacher, contrarian.
  Each turn returns JSON: answer, confidence, key uncertainty, short rationale.
- **Round 1** is independent. Each agent answers 3 times (`--sc-k 3`); its answer is the most
  common one, and its **confidence** averages the share of samples that agree (3/3, 2/3, 1/3)
  with its stated confidence (`--conf-mix avg`).
- **Prestige** is each agent's running Brier score, 1 − (confidence − correct)², on its
  round-1 answers to *earlier* problems, pulled toward 0.5 (prior strength 5). No warmup:
  every problem is scored. `--warmup-from` can start it from separate problems.
- **Debate.** Up to 3 rounds; only when first answers disagree (`--stop-unanimous`). Each round
  the policy chooses who speaks; everyone sees the transcript and may revise.
- **Stability.** Every debate is rerun with the agents reordered; a changed answer is a flip.
- **Paired design.** All policies in a run share the same round-1 answers.

| Policy | Who speaks | Final answer |
|---|---|---|
| `round_robin` | one agent per round, in turn | majority |
| `prestige_x_conf` | highest prestige × confidence | weighted vote |
| `both_sides` | top agent, then top agent with a different answer | weighted vote |
| **`dual`** (v3+) | **top agent and top dissenter together after round 1**, then any answer not yet argued | majority, prestige × confidence breaks ties |
| `rr_dual` (v3+) | same airtime as `dual` (2 then 1), by rotation: the control | majority |
| `pool_dual` (v4) | best argument for the top two answers across all 15 round-1 samples | majority |
| `conf_only`, `prestige_only` | ablations | weighted vote |

## Setup

```bash
pip install -r requirements.txt   # google-genai, pydantic, sympy
export GEMINI_API_KEY=...          # or Vertex AI: GOOGLE_GENAI_USE_VERTEXAI=True GOOGLE_CLOUD_PROJECT=... GOOGLE_CLOUD_LOCATION=...
```

Data: `math.jsonl` (MATH-500); `python3 make_math_extra.py` builds 4,500 unseen MATH problems;
`python3 make_gpqa.py` builds GPQA-Diamond and `--subset main_minus_diamond` 250 more GPQA
questions (gated: accept the terms on Hugging Face and log in with `hf auth login`).
**GPQA files are kept out of this repo**, because GPQA asks that its questions not be published.

## Usage

```bash
# one experiment: round robin vs dual (majority vote is printed automatically)
python3 mad_gemini_v4.py trials --problems gpqa_main_extra.jsonl --n 250 --seeds 1 \
  --model gemini-3.1-flash-lite --sc-k 3 --sc-carry --rounds 3 --policies round_robin,dual \
  --conf-mix avg --stop-unanimous --stagger 0.3 --workers 4 --store run.jsonl --dry-run

python3 mad_gemini_v4.py report --store run.jsonl --policies round_robin,dual   # no API calls
```

- `--dry-run` prints the call count and cost first. Runs are resumable: rerun the same command.
- `--init-from other.jsonl` reuses another run's round-1 answers (same problems, no new round-1 cost).
- `--warmup-from W.jsonl --warmup-n N` starts prestige from round-1 answers on separate problems.
- `run_experiment.sh` wraps one experiment (dry run, run, report); scale with `N=`, `LEVEL=`, `PROBLEMS=`.

Analysis scripts (no API calls):

| Script | What it shows |
|---|---|
| `revote.py` | plain majority vs stored vote; how many agents switched to right vs wrong answers |
| `round_metrics.py` | per round: majority accuracy, agents right, speaker right, switches, who followed whom |
| `signal_check.py` | whether agents differ (Cochran's Q) and whether confidence picks the right agent within a problem |
| `mechanism_check.py` | dual's debates split by which round-1 speaker was right, and how the group moved |
| `free_analyses.py`, `contested_breakdown.py` | extra paired tests on older runs |

## Results (gemini-3.1-flash-lite)

"Contested" = some agents right and some wrong after round 1. McNemar counts are problems one
policy got right and the other wrong.

### 1. dual vs round robin and majority vote

| Run | Majority vote | Round robin | **dual** | dual vs RR (McNemar) | Contested: RR → dual |
|---|---|---|---|---|---|
| MATH 101–330, seed 0 | 83.0% | 85.7% | **88.3%** | 9 vs 3 (p = 0.15) | 64% → 86% |
| GPQA-Diamond (198), fresh round 1 | 67.7% | 67.7% | **70.7%** | 9 vs 3 (p = 0.15) | 50% → 64% |
| GPQA main minus Diamond (250), planned test | 64.0% | 64.8% | **66.8%** | 14 vs 9 (one-sided p = 0.20) | 60% → 65% |
| **Pooled** | | | | **32 vs 15** (one-sided p = 0.009) | |

- **Fewer flips** in every run: 3.5–5.6% for dual vs 8.3–12.8% for round robin (significant on GPQA main, −7.2 points).
- **Choosing the voices matters.** `rr_dual` gets the same airtime by rotation and reached only
  86.1% on MATH (5 vs 4 against round robin), against 88.3% for `dual`.
- **Why it works:** in round 2, agents switched to the right answer far more often than away
  from it: 2.7 to 1 on GPQA (round robin 1.5 to 1) and 5.8 to 1 on MATH (rotation 4.0 to 1).
  Hearing the leader and the strongest dissenter side by side is what converts agents.
- **A right minority wins the group over** (`mechanism_check.py`). When dual's leader was wrong and
  its dissenter right, agents holding the right answer went from about 1.5 to about 3.5 of 5 in
  round 2 on all three datasets (GPQA-Diamond 1.45 → 3.68, GPQA main 1.86 → 3.21, MATH 1.61 → 3.61),
  and the final answer was right in 52–72% of those debates. When the dissenter was wrong, agents
  mostly held their ground. The leader was wrong in over half of dual's debates.
- **Caveats.** The pooled test was computed after seeing the runs; the planned test alone was
  underpowered. An effect of about +2 points needs roughly 600–800 problems to confirm in one run.

### 2. Earlier policies

**MATH 101–330, 3 seeds** (all rows regraded with the current grader):

| Method | Accuracy | Flip | Correct → wrong |
|---|---|---|---|
| Majority vote, no debate | 83.3% | – | – |
| Round robin | 85.4% | 8.6% | 2.5% |
| Confidence only | 84.5% | 4.5% | 1.3% |
| Prestige × confidence | 84.8% | **1.6%** | **0.3%** |
| Both sides | **86.8%** | 5.1% | 1.4% |

Both sides vs round robin: +1.4 [+0.0, +2.9], McNemar 19 vs 9, p = 0.087; contested 80.4% vs 72.2%.

**GPQA-Diamond, first round-1 set:** round robin 71.2%, prestige × confidence 67.7% (2 vs 9),
both sides 70.2%, v2 70.7% (tie with a 70.7% rerun), dual 70.7%, rr_dual 71.2%.

### 3. What prestige and confidence can and can't do

- **The agents are not reliably different** (Cochran's Q: p = 0.18 GPQA, 0.47 MATH); prestige
  ends within 0.03 across agents, so it mostly breaks ties.
- **Warmup doesn't help.** On GPQA main, dual with prestige warmed up on 0, 50 and 198 separate
  questions scored 66.8%, 66.4% and 67.6%, and first-speaker accuracy stayed at 42–45%.
- **Confidence picks the right agent on MATH, not on GPQA.** Within a contested problem, a right
  agent is more confident than a wrong one 69% of the time on MATH and 51% on GPQA.
- **Vote rule barely matters**: a plain majority changes results by 0–3 problems.
- **`pool_dual` did not help** (68.7% vs 67.7% on GPQA-Diamond): its round-3 speaker revived
  answers that were mostly wrong.
- **Round 3 is mixed**: it cost dual 2 problems on GPQA and gained 3 on MATH.

**Takeaway:** with identical agents, the gain comes from debate *structure* (letting the leader
and the strongest dissenter speak together), not from prestige telling agents apart.

## Repo layout

| Path | Contents |
|---|---|
| `mad_gemini_v4.py` | current script: all policies, warmup, per-round logging, `pool_dual` |
| `mad_gemini_v3.py` | adds `dual` / `rr_dual` (same as v4 without `pool_dual`) |
| `mad_gemini.py`, `mad_gemini_v2.py` | earlier versions (v2: calibrated-challenge policy) |
| `make_math_extra.py`, `make_gpqa.py` | build MATH and GPQA problem files |
| `revote.py`, `round_metrics.py`, `signal_check.py`, `mechanism_check.py`, `free_analyses.py`, `contested_breakdown.py` | analyses, no API calls |
| `run_experiment.sh` | one-command experiment runner |
| `runs/<date>_<name>/` | MATH results (`*.jsonl`, regradable with `report`) and summaries |

## Next steps

- One planned, adequately powered test of dual vs round robin (about 500–800 fresh MATH or GPQA problems).
- A mixed-skill panel, where prestige has real differences to learn.
- A better within-problem signal for hard questions (e.g., agents rating each other's arguments).

## Known issues

- Gemini returns occasional 503 errors; the script retries automatically (`attempt 1/5` is normal).
- Five MATH-500 problems are skipped because their reference answer can't be parsed.
- When several stores are combined in `report`, shared round-1 answers are counted once per store
  in the round-1 summary; policy comparisons are unaffected.
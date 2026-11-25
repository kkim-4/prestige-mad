# Five families (Qwen3-8B, Ministral-3-8B, Nemotron-8B, Gemma-3-4B, Phi-4-mini), prestige debate on MATH level 4-5 · November 24-25, 2026

The `mixed` stage with `panels/mixed_families.json` (d312c89) on one H100, 600 problems, warm-up 50 (passed
explicitly), `scripts/slurm/panel_h100.sbatch`. Python code unchanged. No pre-registration covers this run.
The 100-question round-1 pilot came first (`pilot_n100.jsonl`: Qwen3-8B 26%, Ministral 17%, Gemma 16%, Nemotron 8%,
Phi-4-mini 5%). 1,183 of 1,184 debates completed: `all_speak` question 571 failed every retry and is missing.

| | All 600 | Contested (592) | Flips |
| --- | ---: | ---: | ---: |
| Qwen3-8B alone (best) | 24.8% | | |
| Ministral / Gemma / Nemotron / Phi-4-mini alone | 19.2 / 15.2 / 9.8 / 7.3% | | |
| Majority vote, no debate | 24.8% | | |
| `prestige`, as run | 21.2% | 20.7% | 75.2% |
| `all_speak` | 41.5% | 41.2% | 39.0% |

`prestige` minus `all_speak` on contested questions: -20.5 points [-24.2, -17.1] (`report.txt`).

**This run should not be read as evidence about the method.** After the warm-up every agent's prestige was below 0.5
in all 542 `prestige` debates, so all log-odds weights were negative. Re-scoring (`vote_rescore.txt`, 591 paired
questions, exploratory): negative weights set to 0, 40.1% (-1.0 [-3.2, +1.2] vs `all_speak`); weight = prestige, 40.7%
(-0.4 [-2.8, +1.9]).

First-attempt parse failures, retried until valid (counts include killed jobs): Ministral 2,004, Gemma 744,
Phi-4-mini 469, Qwen3-8B 93, Nemotron 79. Several of these replies ran to the 2,048-token limit. Round-1 accuracy is
low for the same suspected reason as in the other runs (not tested). The run took several resumed jobs with different
vLLM memory settings. `calls.jsonl.gz` also holds calls whose debates were lost when a job was killed.

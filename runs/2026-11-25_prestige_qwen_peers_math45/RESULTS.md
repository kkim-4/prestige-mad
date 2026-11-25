# Five Qwen models (4-8B), prestige debate on MATH level 4-5 · November 24-25, 2026

The `qwen_peers` stage with `panels/qwen_peers.json` (d312c89) on one H100, 600 problems, warm-up 50 (passed
explicitly), `scripts/slurm/panel_h100.sbatch`. Python code unchanged. No pre-registration covers this run.
1,185 of 1,186 debates completed: `all_speak` question 411 failed every retry and is missing. `report.txt` scores
it by its round-1 majority; the re-scoring leaves it out.

| | All 600 | Contested (593) | Flips |
| --- | ---: | ---: | ---: |
| Qwen3-4B alone (best) | 44.0% | | |
| Qwen3-8B / Qwen2.5-7B / Coder-7B / Qwen2-7B alone | 25.5 / 14.5 / 11.3 / 8.0% | | |
| Majority vote, no debate | 24.8% | | |
| `prestige`, as run | 32.2% | 31.4% | 50.7% |
| `all_speak` | 44.0% | 43.3% | 35.5% |

`prestige` minus `all_speak` on contested questions: -12.0 points [-15.7, -8.3] (`report.txt`).

**This run should not be read as evidence about the method.** After the warm-up every agent's prestige was below 0.5
in all 543 `prestige` debates (highest 0.50), so all log-odds weights were negative and the vote favored the
least-supported answer. Re-scoring the saved answers (`vote_rescore.txt`, 592 paired questions, exploratory, chosen
after seeing the results):

| Vote rule | Contested accuracy | vs `all_speak` (43.2%) |
| --- | ---: | ---: |
| stored (log-odds) | 31.4% | -11.8 [-15.5, -8.3] |
| negative weights set to 0 (= majority here) | 44.0% | +0.8 [-1.8, +3.2] |
| weight = prestige | 48.8% | +5.6 [+3.0, +8.2] |

The last rule is one of four tried post hoc; it shows no gain on the other two panels (qwen +0.3, mixed -0.4) and
needs a pre-registered test on new questions.

Round-1 accuracy is low for models of this size, probably because of the "in your head", 60-word instruction (not
tested). Replies that failed to parse were retried until valid: Qwen2-7B 1,113 first-attempt failures, Qwen3-4B 267,
Qwen3-8B 120, Coder-7B 32, Qwen2.5-7B 10 (counts include jobs that were killed). The run took several resumed jobs with
different vLLM memory settings. `calls.jsonl.gz` also holds calls whose debates were lost when a job was killed.

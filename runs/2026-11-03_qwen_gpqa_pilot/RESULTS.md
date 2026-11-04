# Qwen GPQA extended pilot · November 3, 2026

The 7B expert answered 30 of 99 completed questions correctly (30.3%, versus 25% chance). Across all five agents, Cochran’s Q was 4.03 (4 df, p = 0.401). On these 99 questions, the panel did not meet the [pilot criterion](../../PREREG_qwen.md) for detectable differences among agents (p < 0.05).

| Round-one agent | Correct / completed | Accuracy |
| --- | ---: | ---: |
| Qwen2.5-7B-Instruct expert | 30 / 99 | 30.3% |
| Qwen2.5-3B-Instruct weak1 | 35 / 99 | 35.4% |
| Qwen2.5-3B-Instruct weak2 | 33 / 99 | 33.3% |
| Qwen2.5-3B-Instruct weak3 | 27 / 99 | 27.3% |
| Qwen2.5-3B-Instruct weak4 | 35 / 99 | 35.4% |

Slurm job 13543 ran on one H100 for 19m 03s. The pilot requested 100 GPQA extended questions, one seed and three samples per agent. One question failed after retries because vLLM rejected its input at the configured 8,192-token context limit. The job exited with code 1. All 495 stored agent answers on the other 99 questions were valid A–D choices. The missing question is excluded from every denominator.

The [offline analysis](signal_check.txt) was generated with `analysis/signal_check.py` on the saved round-one records. Its speaker and vote comparisons rescore those answers. This pilot ran no debates and provides no dual-versus-round-robin result. The raw store contains model rationales for gated questions and is omitted.

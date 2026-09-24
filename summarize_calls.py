#!/usr/bin/env python3
"""
summarize_calls.py — turn a mixed-session calls.jsonl into real summary tables.

calls.jsonl accumulates across every run of the night, mixing models and
problem sets. Each row logs "model" but not which problem file it came from,
so this script re-derives that by matching the logged question text against
every problem file you point it at.

Usage:
    python3 summarize_calls.py --calls calls.jsonl --problems aime.jsonl math.jsonl
    python3 summarize_calls.py --calls calls.jsonl --problems aime.jsonl math.jsonl \
        --out summary.md
"""
from __future__ import annotations
import argparse
import json
import os
from collections import defaultdict

from mad_gemini import answers_match, load_jsonl, brier


def load_truth_sources(paths: list) -> dict:
    """question text -> (dataset_name, true_answer). Later files don't
    override earlier ones if a question text collides (unlikely across
    AIME vs MATH)."""
    truth = {}
    for path in paths:
        name = os.path.basename(path)
        for p in load_jsonl(path):  # no --level filter: want the full universe
            truth.setdefault(p["question"], (name, p["answer"]))
    return truth


def load_calls(path: str) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def reconstruct_split(split_file: str, split_n: int, warmup_frac: float, split_level: str = None) -> tuple:
    """Rebuilds the exact warmup/test question sets a debate run used, by
    replaying load_jsonl's deterministic shuffle+truncate the same way
    evaluate() does. Returns (warmup_questions, test_questions) as sets."""
    levels = {int(x) for x in split_level.split(",")} if split_level else None
    problems = load_jsonl(split_file, split_n, levels=levels)
    n_warm = int(len(problems) * warmup_frac)
    warm_qs = {p["question"] for p in problems[:n_warm]}
    test_qs = {p["question"] for p in problems[n_warm:]}
    return warm_qs, test_qs


def fmt_table(headers: list, rows: list) -> str:
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
              for i, h in enumerate(headers)]
    out = []
    out.append(" | ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
    out.append(" | ".join("-" * w for w in widths))
    for r in rows:
        out.append(" | ".join(str(c).ljust(w) for c, w in zip(r, widths)))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", default="calls.jsonl")
    ap.add_argument("--problems", nargs="+", required=True,
                    help="every problem file used across the night, e.g. aime.jsonl math.jsonl")
    ap.add_argument("--out", default=None, help="write markdown tables to this file too")
    ap.add_argument("--round", type=int, default=None,
                    help="restrict to one debate round (0 = before any agent saw a "
                         "transcript, closest analog to a solo calib run)")
    ap.add_argument("--warmup-vs-test", action="store_true",
                    help="split accuracy by whether each question was in the warmup "
                        "phase (asked once, prestige not yet learned) or the test "
                        "phase (perturbed ~9x). Requires --split-file/--split-n.")
    ap.add_argument("--split-file", default=None, help="the --problems file the debate run used")
    ap.add_argument("--split-n", type=int, default=None, help="the --n the debate run used")
    ap.add_argument("--split-level", default=None, help="the --level the debate run used, if any")
    ap.add_argument("--warmup-frac", type=float, default=0.4)
    ap.add_argument("--since", type=float, default=None,
                    help="unix timestamp; only include calls logged after this "
                        "(use to isolate one run from a calls.jsonl that spans "
                        "multiple attempts — check `tail -1 calls.jsonl` for a recent ts)")
    a = ap.parse_args()

    truth = load_truth_sources(a.problems)
    calls = load_calls(a.calls)
    if a.since is not None:
        before = len(calls)
        calls = [c for c in calls if c.get("ts", 0) >= a.since]
        print(f"--since filter: kept {len(calls)}/{before} calls (others had no ts, or predate it)\n")
    if a.round is not None:
        calls = [c for c in calls if c.get("round") == a.round]

    if a.warmup_vs_test:
        if not (a.split_file and a.split_n):
            raise SystemExit("--warmup-vs-test needs --split-file and --split-n "
                             "(the --problems/--n the debate run actually used)")
        warm_qs, test_qs = reconstruct_split(a.split_file, a.split_n, a.warmup_frac, a.split_level)
        print(f"Reconstructed split from {a.split_file} (n={a.split_n}, "
              f"warmup_frac={a.warmup_frac}): {len(warm_qs)} warmup Qs, {len(test_qs)} test Qs\n")

        def phase_of(q):
            if q in warm_qs:
                return "warmup"
            if q in test_qs:
                return "test"
            return None

        by_phase = defaultdict(lambda: defaultdict(list))  # phase -> agent -> [(conf, correct)]
        unmatched_phase = 0
        for c in calls:
            q = c.get("q", "")
            ph = phase_of(q)
            if ph is None or q not in truth:
                unmatched_phase += 1
                continue
            _, true_ans = truth[q]
            out = c.get("out", {})
            conf = out.get("confidence", 0.0)
            correct = answers_match(out.get("answer", ""), true_ans)
            by_phase[ph][c.get("agent", "?")].append((conf, correct))

        rows = []
        for ph in ("warmup", "test"):
            pairs = [p for recs in by_phase[ph].values() for p in recs]
            n = len(pairs)
            acc = sum(1 for _, y in pairs if y) / n if n else 0
            mc = sum(c for c, _ in pairs) / n if n else 0
            rows.append([ph, n, f"{acc:.0%}", f"{mc:.2f}"])
        print(fmt_table(["phase", "n calls", "accuracy", "mean conf"], rows))
        print(f"\n({unmatched_phase} calls outside this split's question set, e.g. from "
              f"other runs/models, excluded)")
        print("\nRead: if warmup accuracy is close to the solo calib number and test "
              "accuracy is much lower, it's the test-problem subset that's harder / more "
              "perturbed, not exposure to other agents. If both are low, something earlier "
              "in the pipeline (e.g. prestige display itself) differs from calib.")
        return

    # group[(model, dataset)][agent] -> list of (confidence, correct)
    group = defaultdict(lambda: defaultdict(list))
    unmatched = 0

    for c in calls:
        q = c.get("q", "")
        if q not in truth:
            unmatched += 1
            continue
        dataset, true_ans = truth[q]
        model = c.get("model", "?")
        out = c.get("out", {})
        conf = out.get("confidence", 0.0)
        correct = answers_match(out.get("answer", ""), true_ans)
        agent = c.get("agent", "?")
        group[(model, dataset)][agent].append((conf, correct))

    sections = []

    # ---- Table 1: per (model, dataset), aggregated across all 5 agents ----
    rows = []
    for (model, dataset), agents in sorted(group.items()):
        all_pairs = [p for recs in agents.values() for p in recs]
        n = len(all_pairs)
        acc = sum(1 for _, y in all_pairs if y) / n if n else 0
        mc = sum(c for c, _ in all_pairs) / n if n else 0
        br = sum(brier(c, y) for c, y in all_pairs) / n if n else 0
        rows.append([model, dataset, n, f"{acc:.0%}", f"{mc:.2f}", f"{br:.2f}"])
    t1 = fmt_table(["model", "dataset", "n calls", "accuracy", "mean conf", "brier"], rows)
    sections.append("## Accuracy by model x dataset\n\n" + t1)

    # ---- Table 2: per (model, dataset, agent) ----
    rows = []
    for (model, dataset), agents in sorted(group.items()):
        for agent, pairs in sorted(agents.items()):
            n = len(pairs)
            acc = sum(1 for _, y in pairs if y) / n if n else 0
            mc = sum(c for c, _ in pairs) / n if n else 0
            cw = sum(1 for c, y in pairs if not y and c >= 0.8) / n if n else 0
            hr = sum(1 for c, y in pairs if y and c <= 0.6) / n if n else 0
            rows.append([model, dataset, agent, n, f"{acc:.0%}", f"{mc:.2f}",
                        f"{cw:.0%}", f"{hr:.0%}"])
    t2 = fmt_table(["model", "dataset", "agent", "n", "accuracy", "mean conf",
                    "conf.wrong rate", "hedged.right rate"], rows)
    sections.append("## Per-agent breakdown\n\n" + t2)

    report = (f"Loaded {len(calls)} calls from {a.calls}, {unmatched} unmatched "
              f"(question text not found in any --problems file)\n\n" + "\n\n".join(sections))

    print(report)
    if a.out:
        with open(a.out, "w") as f:
            f.write(report + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
verify_prestige_replay.py — does the agent with the highest prestige actually
score better than the rest?

Replays the prestige ledger over SOLO round-0 answers (before any agent has seen
another's answer, so nothing depends on the speaker policy). At every test
problem it asks: was the highest-prestige agent right more often than an
average agent? gap = P(top-prestige agent right) - P(random agent right).

Reading the output:
  gap > 0 with a CI that excludes 0  -> prestige is finding the better agent
  CI spans 0                         -> indistinguishable from picking at random

Usage (use the --until timestamp printed by `date +%s` before the debate run, so
only the solo calib run and earlier are used):
  python3 verify_prestige_replay.py --calls calls.jsonl --problems math.jsonl \
      --model gemini-3.1-flash-lite --until 1790483022
"""
import argparse
import json
import random
import statistics as st

from mad_gemini import answers_match, brier, load_jsonl

AG = ["careful", "fast", "skeptic", "teacher", "contrarian"]
MODES = [("current (.3 x3 per problem)", 1 - 0.7 ** 3), (".3 once per problem", 0.3),
         (".1 once per problem", 0.1), (".05 once per problem", 0.05), ("running mean", None)]


def load(a):
    probs = load_jsonl(a.problems, a.n)
    truth = {p["question"]: p["answer"] for p in probs}
    last = {}                                   # latest pre-debate call per (problem, agent)
    for line in open(a.calls):
        c = json.loads(line)
        if (c.get("q") in truth and c.get("round") == 0 and c.get("agent") in AG
                and str(c.get("model", "")).endswith(a.model) and c["ts"] < a.until):
            k = (c["q"], c["agent"])
            if k not in last or c["ts"] > last[k]["ts"]:
                last[k] = c
    rows = []
    for p in probs:
        if all((p["question"], g) in last for g in AG):
            rows.append({g: (answers_match(last[(p["question"], g)]["out"]["answer"], truth[p["question"]]),
                             last[(p["question"], g)]["out"]["confidence"]) for g in AG})
    return rows, len(probs)


def replay(rows, rate, warm, rng):
    """Returns one (top_correct, avg_agent_correct, contested) tuple per test problem."""
    v = {g: 0.5 for g in AG}
    out = []
    for i, r in enumerate(rows):
        if i >= warm:
            best = max(v.values())
            top = rng.choice([g for g in AG if v[g] >= best - 1e-9])
            k = sum(r[g][0] for g in AG)
            out.append((float(r[top][0]), k / 5, 0 < k < 5))
        rt = 1 / (i + 1) if rate is None else rate
        for g in AG:
            v[g] = (1 - rt) * v[g] + rt * brier(r[g][1], r[g][0])
    return out


def gap_ci(diffs, rng, B=2000):
    if len(diffs) < 2:
        return float("nan"), float("nan"), float("nan")
    g = sorted(st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(B))
    return st.mean(diffs), g[int(.025 * B)], g[int(.975 * B) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", default="calls.jsonl")
    ap.add_argument("--problems", default="math.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--model", default="gemini-3.1-flash-lite")
    ap.add_argument("--until", type=float, required=True, help="unix ts: ignore calls at/after this")
    ap.add_argument("--warmup-frac", type=float, default=0.4)
    ap.add_argument("--perms", type=int, default=300, help="random problem orders to average over")
    a = ap.parse_args()

    rows, total = load(a)
    warm = int(len(rows) * a.warmup_frac)
    print(f"{len(rows)}/{total} problems have all 5 agents' solo answers; warmup={warm}, test={len(rows) - warm}")
    if len(rows) < 30:
        print("Too few complete problems; check --until / --model."); return

    acc = {g: st.mean(r[g][0] for r in rows) for g in AG}
    print("solo accuracy per agent:", {g: round(v, 3) for g, v in acc.items()})
    best = max(AG, key=acc.get)

    print(f"\n{'prestige update':28s} {'subset':15s} {'n':>3s} {'top right':>9s} {'avg agent':>9s} "
          f"{'gap':>7s} {'95% CI (problems)':>20s} | {'gap over random orders [2.5-97.5%]':>35s}")
    for name, rate in MODES:
        for label, sel in [("all problems", lambda c: True), ("contested", lambda c: c)]:
            out = replay(rows, rate, warm, random.Random(0))
            xs = [(t, b) for t, b, c in out if sel(c)]
            g, lo, hi = gap_ci([t - b for t, b in xs], random.Random(1))
            perm = []
            for s in range(a.perms):
                o = replay(random.Random(s).sample(rows, len(rows)), rate, warm, random.Random(s))
                d = [t - b for t, b, c in o if sel(c)]
                if d:
                    perm.append(st.mean(d))
            perm.sort()
            pr = f"{st.mean(perm):+.3f} [{perm[int(.025 * len(perm))]:+.3f}, {perm[int(.975 * len(perm)) - 1]:+.3f}]"
            print(f"{name:28s} {label:15s} {len(xs):3d} {st.mean(t for t, _ in xs):9.3f} "
                  f"{st.mean(b for _, b in xs):9.3f} {g:+7.3f} [{lo:+.3f}, {hi:+.3f}] | {pr:>35s}")

    # Upper bound: a prestige that magically knew the best agent from the start.
    out = replay(rows, 0.0, warm, random.Random(0))            # rate 0 -> never updates (ties -> random)
    test = rows[warm:]
    xs = [(float(r[best][0]), sum(r[g][0] for g in AG) / 5) for r in test if 0 < sum(r[g][0] for g in AG) < 5]
    print(f"\nORACLE (always pick {best}, the best solo agent): contested n={len(xs)}  "
          f"top right {st.mean(t for t, _ in xs):.3f}  avg agent {st.mean(b for _, b in xs):.3f}  "
          f"gap {st.mean(t - b for t, b in xs):+.3f}")

    # Did prestige recover the true ranking? (final prestige vs solo accuracy)
    v = {g: 0.5 for g in AG}
    for i, r in enumerate(rows):
        for g in AG:
            v[g] += (brier(r[g][1], r[g][0]) - v[g]) / (i + 1)
    print("\nrunning-mean prestige vs solo accuracy (same ordering = prestige learned reliability):")
    for g in sorted(AG, key=lambda g: -acc[g]):
        print(f"  {g:11s} accuracy {acc[g]:.3f}   prestige {v[g]:.3f}")


if __name__ == "__main__":
    main()
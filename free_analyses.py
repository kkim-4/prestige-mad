#!/usr/bin/env python3
"""
free_analyses.py — extra paired tests from a stored trials run (no API calls).

  python3 free_analyses.py --store runs/2026-10-05_selfconsistency_3seed_n230/sc.jsonl

1. prestige x confidence vs confidence only (accuracy, flips)
2. each debate policy vs majority vote with no debate (accuracy)
3. accuracy on consensus problems (all agents gave the same first answer) vs split ones
"""
import argparse
import json
import math
import random
from collections import defaultdict

from mad_gemini import answers_match


def mcnemar(b, c):
    n = b + c
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


def boot(diffs_by_q, B=4000, seed=0):
    vals = list(diffs_by_q.values())
    rng = random.Random(seed)
    m = sum(vals) / len(vals)
    bs = sorted(sum(rng.choices(vals, k=len(vals))) / len(vals) for _ in range(B))
    return m, bs[int(.025 * B)], bs[int(.975 * B) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="sc.jsonl")
    a = ap.parse_args()
    rows = [json.loads(l) for p in a.store.split(",") for l in open(p) if l.strip()]
    r0 = {(r["seed"], r["qid"]): r for r in rows if r["kind"] == "r0"}
    deb = defaultdict(dict)
    for r in rows:
        if r["kind"] == "debate":
            deb[r["policy"]][(r["seed"], r["qid"])] = r

    def majority(r):
        groups = []
        for t in r["turns"].values():
            for g in groups:
                if answers_match(t["answer"], g[0]):
                    g[1] += 1; break
            else:
                groups.append([t["answer"], 1])
        return float(answers_match(max(groups, key=lambda g: g[1])[0], r["truth"]))

    def consensus(r):
        ans = [t["answer"] for t in r["turns"].values()]
        return all(answers_match(x, ans[0]) for x in ans)

    acc = {p: {k: float(answers_match(v["canon"]["final"], v["truth"])) for k, v in recs.items()} for p, recs in deb.items()}
    flip = {p: {k: float(not answers_match(v["canon"]["final"], v["order"]["final"])) for k, v in recs.items()}
            for p, recs in deb.items()}
    acc["majority (no debate)"] = {k: majority(r) for k, r in r0.items()}

    def compare(p, q, table, label):
        keys = sorted(set(table[p]) & set(table[q]))
        per_q = defaultdict(list)
        for k in keys:
            per_q[k[1]].append(table[p][k] - table[q][k])
        m, lo, hi = boot({q_: sum(v) / len(v) for q_, v in per_q.items()})
        line = f"  {label:9s} {p} - {q}: {m * 100:+.1f} points [{lo * 100:+.1f}, {hi * 100:+.1f}]"
        if label == "accuracy":
            b = sum(1 for k in keys if table[p][k] > table[q][k])
            c = sum(1 for k in keys if table[q][k] > table[p][k])
            line += f"; McNemar {b} vs {c}, p = {mcnemar(b, c):.3f}"
        print(line)

    print("\n1. prestige x confidence vs confidence only (n = %d problem-runs)" % len(acc.get("prestige_x_conf", {})))
    if "prestige_x_conf" in deb and "conf_only" in deb:
        compare("prestige_x_conf", "conf_only", acc, "accuracy")
        compare("prestige_x_conf", "conf_only", flip, "flips")
    else:
        print("  (needs both policies in the store)")

    print("\n2. debate vs majority vote with no debate")
    for p in ["round_robin", "prestige_x_conf", "conf_only"]:
        if p in deb:
            compare(p, "majority (no debate)", acc, "accuracy")

    print("\n3. consensus vs split first answers")
    cons = {k: consensus(r) for k, r in r0.items()}
    for label, sel in [("consensus (all same answer)", True), ("split (answers differ)", False)]:
        ks = [k for k in r0 if cons[k] == sel]
        right0 = sum(majority(r0[k]) for k in ks)
        print(f"  {label:28s} {len(ks):4d} problem-runs; majority vote right {right0 / max(1, len(ks)):.1%}")
        for p in ["round_robin", "prestige_x_conf", "conf_only"]:
            if p in deb:
                v = [acc[p][k] for k in ks if k in acc[p]]
                print(f"      {p:16s} final accuracy {sum(v) / max(1, len(v)):.1%} ({int(sum(v))}/{len(v)})")


if __name__ == "__main__":
    main()

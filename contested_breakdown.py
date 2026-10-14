#!/usr/bin/env python3
"""
contested_breakdown.py — accuracy per policy on contested vs non-contested problems.

A problem-run is classified by the agents' independent round-1 answers:
  contested   some agents right, some wrong
  all right   every agent right
  all wrong   every agent wrong (no speaker choice can fix these)

  python3 contested_breakdown.py --store runs/2026-10-05_selfconsistency_3seed_n230/sc.jsonl
"""
import argparse
import json
from collections import defaultdict

from mad_gemini import answers_match


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="sc.jsonl", help="one or more stores, comma-separated")
    a = ap.parse_args()

    rows = [json.loads(line) for path in a.store.split(",") for line in open(path) if line.strip()]
    r0 = {(r["seed"], r["qid"]): r for r in rows if r["kind"] == "r0"}

    def group(key):
        r = r0[key]
        right = sum(answers_match(t["answer"], r["truth"]) for t in r["turns"].values())
        n = len(r["turns"])
        return "contested" if 0 < right < n else ("all right" if right == n else "all wrong")

    groups = {key: group(key) for key in r0}
    # acc[policy][group][seed] = list of 0/1
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        if r["kind"] != "debate":
            continue
        key = (r["seed"], r["qid"])
        ok = float(answers_match(r["canon"]["final"], r["truth"]))
        g = groups[key]
        acc[r["policy"]][g][r["seed"]].append(ok)
        if g != "contested":
            acc[r["policy"]]["not contested"][r["seed"]].append(ok)
        acc[r["policy"]]["all problems"][r["seed"]].append(ok)

    order = ["all problems", "not contested", "all right", "all wrong", "contested"]
    pols = sorted(acc, key=lambda p: ["round_robin", "conf_only", "prestige_x_conf"].index(p)
                  if p in ("round_robin", "conf_only", "prestige_x_conf") else 9)
    count = {g: sum(len(v) for v in acc[pols[0]][g].values()) for g in order}

    print(f"problem-runs per group (all seeds): " + ", ".join(f"{g} {count[g]}" for g in order))
    print("accuracy pooled over seeds (right / total), then the per-seed spread\n")
    print(f"{'group':15s}" + "".join(f"{p:>30s}" for p in pols))
    for g in order:
        cells = []
        for p in pols:
            per_seed = [sum(v) / len(v) for v in acc[p][g].values() if v]
            right = sum(sum(v) for v in acc[p][g].values())
            tot = sum(len(v) for v in acc[p][g].values())
            if not tot:
                cells.append("–"); continue
            lo, hi = min(per_seed), max(per_seed)
            cells.append(f"{right / tot:.1%} ({int(right)}/{tot}) [{lo:.0%}-{hi:.0%}]")
        print(f"{g:15s}" + "".join(f"{c:>30s}" for c in cells))
    print("\n'not contested' = 'all right' + 'all wrong'. On those, the policies can only differ through the debate itself;")
    print("speaker choice can help only where agents disagree (contested).")


if __name__ == "__main__":
    main()

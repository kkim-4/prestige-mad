"""Order-sensitivity of each policy: what happens to the final answer when the same debate is
rerun with the agents in a different order. No API calls.

    python3 flip_breakdown.py gpqa250_mixed.jsonl
    python3 flip_breakdown.py confirm_math45.jsonl gpqa250_mixed.jsonl

For each policy (over all problem-runs; problems with no debate never change):
  original acc   accuracy in the original agent order (what `report` calls accuracy)
  reordered acc  accuracy of the reordered rerun
  W->C / C->W    wrong in the original order and right after reordering, and the reverse
  W->W'          wrong both times, but a different wrong answer
  net            reordered acc - original acc  (= W->C - C->W)
Then policies are compared head to head in the original order and in the reordered order
(exact two-sided McNemar on discordant problems).
"""
import json, math, sys, os
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mad_gemini import answers_match


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def main(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    deb = defaultdict(dict)
    for r in rows:
        if r["kind"] == "debate" and "order" in r:
            t = r["truth"]
            a, b = r["canon"]["final"], r["order"]["final"]
            deb[r["policy"]][(r["seed"], r["qid"])] = (answers_match(a, t), answers_match(b, t), answers_match(a, b))
    if not deb:
        print(f"{path}: no debates found"); return
    print(f"\n{path}")
    print(f"   {'policy':16s} {'n':>4s} {'original':>9s} {'reordered':>10s} {'W->C':>12s} {'C->W':>12s} {'W->W*':>8s} {'net':>7s}")
    for p, d in sorted(deb.items()):
        n = len(d)
        orig = sum(x[0] for x in d.values())
        reord = sum(x[1] for x in d.values())
        wc = sum((not x[0]) and x[1] for x in d.values())
        cw = sum(x[0] and not x[1] for x in d.values())
        ww = sum((not x[0]) and (not x[1]) and not x[2] for x in d.values())
        print(f"   {p:16s} {n:4d} {orig / n:9.1%} {reord / n:10.1%} {wc:4d} ({wc / n:5.1%}) {cw:4d} ({cw / n:5.1%})"
              f" {ww:8d} {(reord - orig) / n:+7.1%}")
    pols = sorted(deb)
    if len(pols) > 1:
        print("\n   head to head (policy A right & B wrong, the reverse, two-sided McNemar p)")
        for i, A in enumerate(pols):
            for B in pols[i + 1:]:
                ks = [k for k in deb[A] if k in deb[B]]
                for label, j in (("original order", 0), ("reordered", 1)):
                    b = sum(deb[A][k][j] and not deb[B][k][j] for k in ks)
                    c = sum(deb[B][k][j] and not deb[A][k][j] for k in ks)
                    print(f"   {A} vs {B}, {label:14s}: {b} vs {c}  (p = {mcnemar(b, c):.3f})")
    print("\n   W->W* = wrong both times with a different wrong answer. The reordered order is one random"
          "\n   shuffle per problem; it was fixed before the run, but this comparison was not pre-registered.")


if __name__ == "__main__":
    for f in sys.argv[1:] or ["gpqa250_mixed.jsonl"]:
        main(f)

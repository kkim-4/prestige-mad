"""Where do policies differ? Splits questions by how many agents were right after round 1. No API calls.

    python3 analysis/minority_check.py STORE.jsonl                 # every policy vs all_speak
    python3 analysis/minority_check.py STORE.jsonl --ref round_robin

Groups (5 agents):
  right minority   1-2 agents right after round 1: a policy can only GAIN here, by getting them heard
  right majority   3-4 agents right: a policy can only LOSE here, by letting a wrong minority win
  unanimous        0 or 5 right: debate rarely changes anything
For each policy and group: final accuracy, discordant counts against the reference policy and an exact
McNemar p (two-sided, and one-sided in the policy's favour). Original seating only.
It also prints how many arguments each policy broadcasts per debate (the context cost).
The mechanism predicts dual's gains over a reference sit in "right minority" and its losses in
"right majority" stay small.
"""
import argparse, json, math, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
from mad_gemini import answers_match


def binom_tail(b, c):
    n = b + c
    if n == 0:
        return 1.0, 1.0
    lo = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    one = sum(math.comb(n, i) for i in range(b, n + 1)) / 2 ** n      # P(at least b wins | no effect)
    return min(1.0, 2 * lo), one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stores", nargs="+")
    ap.add_argument("--ref", default="all_speak")
    a = ap.parse_args()
    for path in a.stores:
        run(path, a.ref)


def run(path, ref):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    r0 = {(r["seed"], r["qid"]): r for r in rows if r["kind"] == "r0"}
    group = {}
    for k, r in r0.items():
        right = sum(answers_match(t["answer"], r["truth"]) for t in r["turns"].values())
        n = len(r["turns"])
        group[k] = ("unanimous" if right in (0, n) else
                    "right minority" if right <= (n - 1) // 2 else "right majority")
    acc = defaultdict(dict)
    for d in rows:
        if d["kind"] == "debate" and (d["seed"], d["qid"]) in group:
            acc[d["policy"]][(d["seed"], d["qid"])] = answers_match(d["canon"]["final"], d["truth"])
    if ref not in acc:
        print(f"{path}: reference policy {ref!r} not in store; policies: {sorted(acc)}"); return
    print(f"\n{path}   (reference: {ref})")
    spk = defaultdict(list)
    for d in rows:
        if d["kind"] == "debate":
            spk[d["policy"]].append(len(d["canon"]["speakers"]))
    print("  arguments broadcast per debate (mean over all problems; each one is read by every agent):")
    print("   " + ", ".join(f"{p} {sum(v) / len(v):.1f}" for p, v in sorted(spk.items())))
    for g in ("right minority", "right majority", "unanimous"):
        ks = [k for k in acc[ref] if group[k] == g]
        if not ks:
            continue
        print(f"\n  {g}: {len(ks)} questions")
        print(f"   {'policy':14s} {'accuracy':>9s}   {'wins':>4s} {'losses':>6s}   p two-sided   p one-sided")
        for p in sorted(acc):
            kk = [k for k in ks if k in acc[p]]
            if not kk:
                continue
            ac = sum(acc[p][k] for k in kk) / len(kk)
            if p == ref:
                print(f"   {p:14s} {ac:9.1%}   (reference)"); continue
            b = sum(acc[p][k] and not acc[ref][k] for k in kk)
            c = sum(acc[ref][k] and not acc[p][k] for k in kk)
            two, one = binom_tail(b, c)
            print(f"   {p:14s} {ac:9.1%}   {b:4d} {c:6d}   {two:11.3f}   {one:11.3f}")


if __name__ == "__main__":
    main()

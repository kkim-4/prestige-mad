"""Re-score the final vote of a prestige_mad_v2/v3 store under other vote rules (no model calls).
The vote runs once, after the debate, so changing it leaves every stored debate unchanged.
  stored     log-odds of prestige, as run (negative below 0.5)
  clipped    the same with negative weights set to 0 (all-zero -> one agent one vote)
  linear     weight = prestige
  majority   one agent one vote
    python3 analysis/prestige_vote_rescore.py runs/.../store.jsonl
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from prestige_mad_v2 import answers_match, boot_ci, logodds


def tally(ans, w):
    t = {}
    for a, x in ans.items():
        for g in t:
            if answers_match(x, g):
                t[g] += w[a]
                break
        else:
            t[x] = w[a]
    return max(t, key=t.get)


RULES = {
    "stored": lambda p: {a: logodds(v) for a, v in p.items()},
    "clipped": lambda p: {a: max(0.0, logodds(v)) for a, v in p.items()},
    "linear": dict,
    "majority": lambda p: {a: 1.0 for a in p},
}


def main(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    cfg = next(r["cfg"] for r in rows if r["kind"] == "config")
    ids, warm = cfg["agents"], cfg["warmup"]
    r0 = {r["pid"]: r for r in rows if r["kind"] == "r0"}
    deb = {p: {r["pid"]: r for r in rows if r["kind"] == "debate" and r["policy"] == p}
           for p in ("prestige", "all_speak")}
    both = sorted(set(deb["prestige"]) & set(deb["all_speak"]))
    print(f"{len(r0)} questions, {len(both)} with both debates; unpaired contested questions are left out")
    ok, match, total = {}, 0, 0
    for pol in deb:
        for rule, weigh in RULES.items():
            if pol == "all_speak" and rule != "majority":
                continue
            m = {}
            for i in both:
                d = deb[pol][i]; s = 0
                for side, seats in (("canon", d["canon"]["seats"]), ("order", d["order"]["perm"])):
                    last = {a: d[side]["rounds"][a][-1][0] for a in seats}
                    w = weigh(d["prestige"]) if pol == "prestige" and i >= warm else {a: 1.0 for a in ids}
                    if not any(w.values()):
                        w = {a: 1.0 for a in ids}
                    f = tally(last, w)
                    s += answers_match(f, r0[i]["truth"])
                    if pol == "prestige" and rule == "stored":
                        total += 1; match += f == d[side]["final"]
                m[i] = s / 2
            ok[pol, rule] = m
    print(f"stored rule reproduces the saved final answer in {match}/{total} debates")
    base = ok["all_speak", "majority"]
    for k, m in ok.items():
        lo, hi = boot_ci([m[i] - base[i] for i in both])
        print(f"  {k[0]:10s} {k[1]:9s} contested accuracy {sum(m.values())/len(both):6.1%}"
              f"   vs all_speak {sum(m[i]-base[i] for i in both)/len(both):+.1%} [{lo:+.1%}, {hi:+.1%}]")


if __name__ == "__main__":
    main(sys.argv[1])

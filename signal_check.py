"""Does confidence or prestige carry signal? Uses stored round-1 answers only (no API calls).

    python3 signal_check.py gpqa.jsonl            # any trials store; several can be given
    python3 signal_check.py sc.jsonl gpqa.jsonl

For each store it prints:
  1. Are the agents actually different?  Per-agent accuracy + Cochran's Q test.
     If the agents are equally good, prestige has nothing real to learn.
  2. Does confidence separate right from wrong?  AUROC overall, and *within* contested
     problems (right vs wrong agents on the same problem), which is what speaker choice needs.
  3. Speaker choice on contested problems: how often the chosen first speaker is right,
     for random / confidence / prestige / prestige x confidence, prestige with a warmup,
     and an "oracle" prestige (each agent's true accuracy on all *other* problems = the best
     prestige could ever be).
  4. Final vote on contested problems with the same weights.
"""
import json, math, sys, os
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mad_gemini import answers_match, prestige_path

SIG = {
    "self-consistency": lambda t: t["confidence"],
    "stated": lambda t: t.get("stated_confidence", t["confidence"]),
    "average (conf-mix avg)": lambda t: (t["confidence"] + t.get("stated_confidence", t["confidence"])) / 2,
}
mix = SIG["average (conf-mix avg)"]


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def chi2_sf_df4(x):                       # survival function of chi-square with 4 df (closed form)
    return math.exp(-x / 2) * (1 + x / 2)


def cochran_q(mat):                       # mat: list of rows, each a list of 0/1 per agent
    k = len(mat[0])
    col = [sum(r[j] for r in mat) for j in range(k)]
    row = [sum(r) for r in mat]
    N = sum(row)
    den = k * N - sum(x * x for x in row)
    if den == 0:
        return 0.0, 1.0
    q = (k - 1) * (k * sum(c * c for c in col) - N * N) / den
    return q, (chi2_sf_df4(q) if k == 5 else float("nan"))


def top_right(weights, ok):
    """Expected correctness of the top-weighted agent, ties broken at random."""
    m = max(weights.values())
    tied = [a for a, w in weights.items() if abs(w - m) < 1e-12]
    return sum(ok[a] for a in tied) / len(tied)


def vote_right(weights, ans, truth):
    tally = defaultdict(float)
    for a, w in weights.items():
        tally[ans[a]] += w
    m = max(tally.values())
    tied = [x for x, w in tally.items() if abs(w - m) < 1e-12]
    return sum(answers_match(x, truth) for x in tied) / len(tied)


def analyse(path, warm_frac=0.4):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    cfg = next((r["cfg"] for r in reversed(rows) if r["kind"] == "config"), {})
    r0 = [r for r in rows if r["kind"] == "r0"]
    by_seed = defaultdict(list)
    for r in r0:
        by_seed[r["seed"]].append(r)
    ids = list(r0[0]["turns"])
    print(f"\n{'=' * 78}\n{path}: {len(r0)} problem-runs, {len(by_seed)} seed(s), agents {ids}")

    # ---------- per-problem facts ----------
    P = []                                 # one entry per problem-run, with prestige paths attached
    for seed, rs in by_seed.items():
        rs.sort(key=lambda r: r["pid"])
        ok = [{a: answers_match(r["turns"][a]["answer"], r["truth"]) for a in ids} for r in rs]
        fr = [{a: (mix(r["turns"][a]), ok[i][a]) for a in ids} for i, r in enumerate(rs)]
        path_ = prestige_path(fr, ids, cfg.get("prestige_mode", "shrunk"), cfg.get("alpha", 0.3),
                              k=cfg.get("prior_strength", 5.0), rule=cfg.get("rule", "brier"))
        n = len(rs)
        w = int(n * warm_frac)
        tot = {a: sum(o[a] for o in ok) for a in ids}
        for i, r in enumerate(rs):
            # oracle: accuracy on every OTHER problem of this seed (leave-one-out), shrunk lightly
            orc = {a: (tot[a] - ok[i][a] + 0.5) / (n - 1 + 1) for a in ids}
            P.append({"r": r, "ok": ok[i], "pos": i, "n": n,
                      "pres": path_[i], "pres_warm": path_[w] if i >= w else None, "oracle": orc})

    # ---------- 1. are agents different? ----------
    print("\n1. Are the agents different? (round-1 accuracy, same problems)")
    for a in ids:
        acc = sum(p["ok"][a] for p in P) / len(P)
        print(f"     {a:12s} {acc:.3f}")
    q, pv = cochran_q([[int(p["ok"][a]) for a in ids] for p in P])
    print(f"   Cochran's Q = {q:.2f}, p = {pv:.3f}  "
          f"({'agents differ' if pv < 0.05 else 'no evidence the agents differ -> prestige has little real to learn'})")
    fin = P[-1]["pres"]
    print("   prestige at the end of the run: " + ", ".join(f"{a} {fin[a]:.3f}" for a in ids))

    # ---------- 2. confidence ----------
    cont = [p for p in P if 0 < sum(p["ok"].values()) < len(ids)]
    print(f"\n2. Does confidence separate right from wrong?   contested problems: {len(cont)}/{len(P)}")
    print(f"   {'signal':24s} {'AUROC all':>10s} {'within contested':>18s}")
    for s, f in SIG.items():
        pos = [f(p["r"]["turns"][a]) for p in P for a in ids if p["ok"][a]]
        neg = [f(p["r"]["turns"][a]) for p in P for a in ids if not p["ok"][a]]
        num = den = 0.0
        for p in cont:                     # pairs on the same problem only
            T = p["r"]["turns"]
            for a in ids:
                for b in ids:
                    if p["ok"][a] and not p["ok"][b]:
                        x, y = f(T[a]), f(T[b])
                        num += (x > y) + 0.5 * (x == y); den += 1
        print(f"   {s:24s} {auroc(pos, neg):10.3f} {num / den if den else float('nan'):18.3f}")
    print("   (within contested = P(a right agent is more confident than a wrong agent on the same problem);"
          " 0.5 = useless for picking speakers)")

    # ---------- 3 & 4. speaker choice and vote on contested problems ----------
    def table(sub, title, warm):
        if not sub:
            return
        rules = {
            "random speaker / plain majority": lambda p: {a: 1.0 for a in ids},
            "confidence (avg)": lambda p: {a: mix(p["r"]["turns"][a]) for a in ids},
            "prestige only": lambda p: dict(p["pres"]),
            "prestige x confidence": lambda p: {a: p["pres"][a] * mix(p["r"]["turns"][a]) for a in ids},
        }
        if warm:
            rules["prestige x conf, 40% warmup"] = lambda p: {a: p["pres_warm"][a] * mix(p["r"]["turns"][a]) for a in ids}
        rules["ORACLE prestige x confidence"] = lambda p: {a: p["oracle"][a] * mix(p["r"]["turns"][a]) for a in ids}
        rules["ORACLE log-odds x confidence"] = lambda p: {
            a: math.log(p["oracle"][a] / (1 - p["oracle"][a])) * mix(p["r"]["turns"][a]) for a in ids}
        print(f"\n   {title}  (n = {len(sub)})")
        print(f"   {'weights':34s} {'1st speaker right':>18s} {'weighted vote right':>20s}")
        for name, wf in rules.items():
            sp = sum(top_right(wf(p), p["ok"]) for p in sub) / len(sub)
            vt = sum(vote_right(wf(p), {a: p["r"]["turns"][a]["answer"] for a in ids}, p["r"]["truth"])
                     for p in sub) / len(sub)
            print(f"   {name:34s} {sp:18.1%} {vt:20.1%}")

    print("\n3-4. Contested problems: is the top-weighted agent right, and does the weighted vote get it right?")
    table(cont, "all contested problems", warm=False)
    late = [p for p in cont if p["pres_warm"] is not None]
    table(late, f"contested problems after the first {int(warm_frac * 100)}% (same set, so warmup is comparable)",
          warm=True)
    print("\n   Reading: if 'prestige x confidence' is no better than 'confidence (avg)', prestige adds nothing"
          "\n   to speaker choice; if even the ORACLE rows don't beat it, no amount of warmup can help,"
          "\n   because the agents are not reliably different.")


if __name__ == "__main__":
    for f in sys.argv[1:] or ["gpqa.jsonl"]:
        analyse(f)
"""How would other final votes have done? Re-scores stored debates offline (no API calls).

    python3 analysis/vote_check.py gpqa250_mixed.jsonl
    python3 analysis/vote_check.py confirm_math45.jsonl --policies round_robin,dual

The debates themselves don't depend on the vote: agents never see how the final answer is
counted, so each agent's last-round answer is the same under any rule, and swapping the rule
afterwards is a fair what-if. Only the original agent order can be re-scored (the reordered
reruns store just their final answer).

Vote rules (every weight uses only problems BEFORE the current one, except the ORACLE row):
  one agent, one vote     dual's rule: biggest group wins, prestige x confidence breaks ties
                          (round robin also counts heads but breaks ties by agent order, so
                          its re-vote can differ from what it stored on tied problems)
  prestige                weight = the stored prestige the policy saw
  prestige x confidence   weight = prestige x the agent's last-round confidence
  learned log-odds        weight = log((K-1) p / (1-p)), p = the agent's running round-1 accuracy,
                          shrunk toward the panel's running accuracy; the optimal weighted
                          majority when agents err independently (Nitzan & Paroush 1982)
  log-odds x confidence   the same, times last-round confidence
  follow the top agent    take the answer of the agent with the highest running accuracy
  ORACLE log-odds         log-odds of each agent's true accuracy on all other problems (a ceiling)

K is the number of answer options (4 for GPQA letters, detected automatically; 2 otherwise,
which makes the weight plain log-odds). Use --k to override.

Caveat on confidence: with --sc-carry an agent that switches answers during the debate carries
near-zero confidence for its new answer, so the "x confidence" rules mostly ignore converts.

Exploratory: choosing the best rule on these runs and reporting its score is post hoc. Pick the
rule here, then pre-register it for the next run.
"""
import argparse, json, math, os, random, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
from mad_gemini import answers_match

MIX = lambda t: (t["confidence"] + t.get("stated_confidence", t["confidence"])) / 2


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def groups_of(ans, w):
    """Group equivalent answers (answers_match), summing weight and head count."""
    gs = []
    for a, x in ans.items():
        for g in gs:
            if answers_match(x, g[0]):
                g[1] += w[a]; g[2] += 1
                break
        else:
            gs.append([x, w[a], 1])
    return gs


def score(gs, key, truth):
    """Expected correctness of the winning group, ties split evenly."""
    m = max(key(g) for g in gs)
    tied = [g for g in gs if key(g) == m]
    return sum(answers_match(g[0], truth) for g in tied) / len(tied)


def log_odds(p, K):
    p = min(max(p, 0.02), 0.98)
    return math.log((K - 1) * p / (1 - p))


def learned_acc(r0s, ids, k):
    """Running round-1 accuracy before each problem, shrunk toward the panel's running mean."""
    out, tot, n = [], {a: 0 for a in ids}, 0
    for r in r0s:
        pool = (sum(tot.values()) + 0.5 * len(ids)) / (n * len(ids) + len(ids))
        out.append({a: (tot[a] + k * pool) / (n + k) for a in ids})
        for a in ids:
            tot[a] += answers_match(r["turns"][a]["answer"], r["truth"])
        n += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stores", nargs="+")
    ap.add_argument("--policies", default="", help="comma list; default: every policy in the store")
    ap.add_argument("--k", type=int, default=0, help="answer options (default: 4 for A-D letters, else 2)")
    ap.add_argument("--shrink", type=float, default=5.0, help="prior strength for learned accuracy")
    ap.add_argument("--late", type=float, default=0.4, help="also report problems after this fraction")
    a = ap.parse_args()
    for path in a.stores:
        run(path, a)


def run(path, a):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    r0 = [r for r in rows if r["kind"] == "r0"]
    deb = [r for r in rows if r["kind"] == "debate" and "rounds" in r.get("canon", {})]
    if not r0 or not deb:
        print(f"{path}: needs round-1 rows and debates with per-round answers (mad_dual / v5 stores)")
        return
    ids = list(r0[0]["turns"])
    K = a.k or (4 if all(str(r["truth"]).strip().upper() in "ABCD" for r in r0) else 2)

    # per seed: running accuracy, oracle accuracy, contested flag, position
    info = {}
    by_seed = defaultdict(list)
    for r in r0:
        by_seed[r["seed"]].append(r)
    for seed, rs in by_seed.items():
        rs.sort(key=lambda r: r["pid"])
        acc = learned_acc(rs, ids, a.shrink)
        ok = [{x: answers_match(r["turns"][x]["answer"], r["truth"]) for x in ids} for r in rs]
        tot = {x: sum(o[x] for o in ok) for x in ids}
        n = len(rs)
        for i, r in enumerate(rs):
            info[(seed, r["qid"])] = {
                "r0": r, "acc": acc[i], "late": i >= int(n * a.late),
                "contested": 0 < sum(ok[i].values()) < len(ids),
                "oracle": {x: (tot[x] - ok[i][x] + 0.5) / n for x in ids}}

    def rules(ans, conf, pres, inf):
        lo = {x: log_odds(inf["acc"][x], K) for x in ids}
        top = max(ids, key=lambda x: (inf["acc"][x], random.Random(inf["r0"]["qid"] + x).random()))
        R = {
            "one agent, one vote": (groups_of(ans, {x: pres[x] * conf[x] for x in ids}),
                                     lambda g: (g[2], g[1])),
            "prestige": (groups_of(ans, pres), lambda g: g[1]),
            "prestige x confidence": (groups_of(ans, {x: pres[x] * conf[x] for x in ids}), lambda g: g[1]),
            "learned log-odds": (groups_of(ans, lo), lambda g: g[1]),
            "log-odds x confidence": (groups_of(ans, {x: lo[x] * conf[x] for x in ids}), lambda g: g[1]),
            "follow the top agent": (groups_of({top: ans[top]}, {top: 1.0}), lambda g: g[1]),
            "ORACLE log-odds": (groups_of(ans, {x: log_odds(inf["oracle"][x], K) for x in ids}),
                                lambda g: g[1]),
        }
        return R

    print(f"\n{'=' * 86}\n{path}: agents {ids}, K = {K}, {len(info)} problem-runs")
    last_acc = max(info.items(), key=lambda kv: kv[1]["r0"]["pid"])[1]["acc"]
    print("   learned accuracy at the end: " + ", ".join(f"{x} {last_acc[x]:.2f}" for x in ids)
          + "   -> log-odds weights " + ", ".join(f"{log_odds(last_acc[x], K):.2f}" for x in ids))

    # ---- A. no debate: the same rules on round-1 answers ----
    sets = {}
    for key, inf in info.items():
        T = inf["r0"]["turns"]
        pres = {x: 1.0 for x in ids}            # no stored prestige per r0 row; plain weights
        R = rules({x: T[x]["answer"] for x in ids}, {x: MIX(T[x]) for x in ids}, pres, inf)
        sets[key] = {n_: score(gs, kf, inf["r0"]["truth"]) for n_, (gs, kf) in R.items()
                     if n_ not in ("prestige", "prestige x confidence")}
    table("A. no debate (round-1 answers)", sets, info, base="one agent, one vote", stored=None)

    # ---- B. each policy's final-round answers ----
    pols = [p for p in (a.policies.split(",") if a.policies else sorted({d["policy"] for d in deb}))]
    stored_all = {}
    for pol in pols:
        sets, stored, agree = {}, {}, []
        for d in deb:
            if d["policy"] != pol or (d["seed"], d["qid"]) not in info:
                continue
            key, inf = (d["seed"], d["qid"]), info[(d["seed"], d["qid"])]
            rd = d["canon"]["rounds"]
            ans = {x: rd[x][-1][0] for x in ids}
            conf = {x: rd[x][-1][1] for x in ids}
            pres = d.get("prestige") or {x: 1.0 for x in ids}
            R = rules(ans, conf, pres, inf)
            sets[key] = {n_: score(gs, kf, d["truth"]) for n_, (gs, kf) in R.items()}
            stored[key] = float(answers_match(d["canon"]["final"], d["truth"]))
            # sanity check: our one-agent-one-vote should reproduce the stored final answer
            gs, kf = R["one agent, one vote"]
            m = max(kf(g) for g in gs)
            agree.append(any(answers_match(g[0], d["canon"]["final"]) for g in gs if kf(g) == m))
        if not sets:
            print(f"\n   {pol}: no debates with per-round answers"); continue
        stored_all[pol] = stored
        table(f"B. {pol}: final-round answers re-voted", sets, info, base="one agent, one vote",
              stored=stored, agree=sum(agree) / len(agree))

    if "round_robin" in stored_all:
        print("\n   Against round robin as run (its stored final answers):")
        rr = stored_all["round_robin"]
        for pol in pols:
            if pol == "round_robin" or pol not in stored_all:
                continue
            print(f"   {pol} with learned log-odds vote vs round robin: ", end="")
            ks = [k for k in rr if k in stored_all[pol]]
            sets = {}
            for d in deb:
                key = (d["seed"], d["qid"])
                if d["policy"] == pol and key in ks:
                    rd = d["canon"]["rounds"]
                    inf = info[key]
                    gs = groups_of({x: rd[x][-1][0] for x in ids},
                                   {x: log_odds(inf["acc"][x], K) for x in ids})
                    sets[key] = round(score(gs, lambda g: g[1], d["truth"]))
            b = sum(bool(sets[k]) and not rr[k] for k in sets)
            c = sum(bool(rr[k]) and not sets[k] for k in sets)
            print(f"{b} vs {c} discordant (two-sided p = {mcnemar(b, c):.3f})")

    print("\n   Rows use only earlier problems except ORACLE. Choosing the best row here is post hoc:"
          "\n   pre-register the chosen rule before the next run.")


def table(title, sets, info, base, stored, agree=None):
    keys = list(sets)
    names = list(next(iter(sets.values())))
    cont = [k for k in keys if info[k]["contested"]]
    late = [k for k in keys if info[k]["late"]]
    print(f"\n   {title}" + (f"   (re-vote reproduces the stored answer on {agree:.1%})" if agree is not None else ""))
    print(f"   {'vote rule':24s} {'all':>7s} {'contested':>10s} {'late':>7s}   vs one-agent-one-vote")
    print(f"   {'':24s} {len(keys):7d} {len(cont):10d} {len(late):7d}   (b vs c, two-sided p)")
    if stored is not None:
        acc = lambda ks: sum(stored[k] for k in ks) / len(ks) if ks else float("nan")
        print(f"   {'(stored final answer)':24s} {acc(keys):7.1%} {acc(cont):10.1%} {acc(late):7.1%}")
    for n_ in names:
        acc = lambda ks: sum(sets[k][n_] for k in ks) / len(ks) if ks else float("nan")
        b = sum(round(sets[k][n_]) and not round(sets[k][base]) for k in keys)
        c = sum(round(sets[k][base]) and not round(sets[k][n_]) for k in keys)
        cmp_ = "" if n_ == base else f"   {b:3d} vs {c:3d}  p = {mcnemar(b, c):.3f}"
        print(f"   {n_:24s} {acc(keys):7.1%} {acc(cont):10.1%} {acc(late):7.1%}{cmp_}")


if __name__ == "__main__":
    main()

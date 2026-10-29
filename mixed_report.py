"""Mixed-skill panel report: is the panel better than its strongest member, and did prestige
learn who is strong? No API calls.

    python3 mixed_report.py dual_sizes.jsonl
    python3 mixed_report.py dual_expert.jsonl --policies round_robin,dual,dual_shuf,prestige_x_conf

Prints, from the stored run:
  1. per agent: round-1 accuracy (its own majority over 3 samples), mean confidence,
     overconfidence (confidence - accuracy), final prestige; Cochran's Q across agents
  2. the strongest single agent alone: the pre-designated expert (role "expert" in the panel)
     and, for reference only, the best agent chosen after the fact
  3. every policy vs the expert alone and vs round robin: accuracy and exact McNemar counts
"""
import json, math, sys, os
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mad_gemini import answers_match, prestige_path


def binom_two_sided(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def chi2_sf(x, df):                                   # survival function, small df, series
    if x <= 0:
        return 1.0
    a, t, s, k = df / 2, x / 2, 0.0, 0
    term = math.exp(-t) * t ** a / math.gamma(a + 1)
    while k < 2000 and (k < t + 10 or term > 1e-15):   # terms rise until k ~ t, then fall
        s += term; k += 1; term *= t / (a + k)
    return max(0.0, 1 - s)


def main(path, pols):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    cfg = next((r["cfg"] for r in reversed(rows) if r["kind"] == "config"), {})
    spec = json.loads(cfg["panel_spec"]) if cfg.get("panel_spec") else []
    role = {d["id"]: d.get("role", "") for d in spec}
    model = {d["id"]: d["model"] for d in spec}
    r0 = {(r["seed"], r["qid"]): r for r in rows if r["kind"] == "r0"}
    ids = list(next(iter(r0.values()))["turns"])
    ok = {k: {a: answers_match(r["turns"][a]["answer"], r["truth"]) for a in ids} for k, r in r0.items()}
    mix = lambda t: (t["confidence"] + t.get("stated_confidence", t["confidence"])) / 2
    keys = sorted(r0)
    print(f"{path}: {len(keys)} problem-runs, agents {ids}")

    # 1. agents
    by_seed = defaultdict(list)
    for k in keys:
        by_seed[k[0]].append(r0[k])
    final_p = defaultdict(list)
    for seed, rs in by_seed.items():
        rs.sort(key=lambda r: r["pid"])
        fr = [{a: (mix(r["turns"][a]), ok[(seed, r["qid"])][a]) for a in ids} for r in rs]
        pth = prestige_path(fr + [fr[-1]], ids, cfg.get("prestige_mode", "shrunk"), cfg.get("alpha", 0.3),
                            k=cfg.get("prior_strength", 5.0), rule=cfg.get("rule", "brier"))
        for a in ids:
            final_p[a].append(pth[-1][a])
    print(f"\n1. agents      {'model':34s} {'role':7s} {'acc':>6s} {'conf':>6s} {'over':>6s} {'prestige':>9s}")
    for a in ids:
        acc = sum(ok[k][a] for k in keys) / len(keys)
        conf = sum(mix(r0[k]["turns"][a]) for k in keys) / len(keys)
        pr = sum(final_p[a]) / len(final_p[a])
        print(f"   {a:12s} {model.get(a, '?')[:34]:34s} {role.get(a, ''):7s} {acc:6.3f} {conf:6.3f} {conf - acc:+6.3f} {pr:9.3f}")
    k_ = len(ids)
    col = [sum(ok[k][a] for k in keys) for a in ids]
    rowsum = [sum(ok[k].values()) for k in keys]
    N = sum(rowsum)
    den = k_ * N - sum(x * x for x in rowsum)
    q = (k_ - 1) * (k_ * sum(c * c for c in col) - N * N) / den if den else 0.0
    spread = max(sum(v) / len(v) for v in final_p.values()) - min(sum(v) / len(v) for v in final_p.values())
    print(f"   Cochran's Q = {q:.1f}, p = {chi2_sf(q, k_ - 1):.4f}   prestige spread = {spread:.3f}"
          f"   (clone panels: spread about 0.03)")

    # 2. strongest single agent
    expert = next((a for a in ids if role.get(a) == "expert"), None)
    best = max(ids, key=lambda a: sum(ok[k][a] for k in keys))
    single = {}
    if expert:
        single[f"expert alone ({expert})"] = {k: ok[k][expert] for k in keys}
    single[f"best agent, chosen after the fact ({best})"] = {k: ok[k][best] for k in keys}

    # 3. policies
    deb = defaultdict(dict)
    for r in rows:
        if r["kind"] == "debate":
            deb[r["policy"]][(r["seed"], r["qid"])] = answers_match(r["canon"]["final"], r["truth"])
    pols = pols or sorted(deb)
    print(f"\n2-3. accuracy on {len(keys)} problem-runs; McNemar = policy right & other wrong, reverse")
    ref_names = list(single) + ["round_robin"]
    refs = {**single, "round_robin": deb.get("round_robin", {})}
    for name, res in list(single.items()):
        print(f"   {name:46s} {sum(res.values()) / len(res):.3f}")
    for p in pols:
        res = deb.get(p)
        if not res:
            continue
        line = f"   {p:46s} {sum(res[k] for k in keys if k in res) / max(1, sum(k in res for k in keys)):.3f}"
        for rn in ref_names:
            ref = refs.get(rn)
            if not ref or rn == p:
                continue
            ks = [k for k in keys if k in res and k in ref]
            b = sum(res[k] and not ref[k] for k in ks)
            c = sum(ref[k] and not res[k] for k in ks)
            short = "expert" if rn.startswith("expert") else ("best*" if rn.startswith("best") else "RR")
            line += f"   vs {short}: {b} vs {c} (p={binom_two_sided(b, c):.3f})"
        print(line)
    print("\n   best* is chosen after seeing results, so it is an optimistic ceiling, not a fair baseline."
          "\n   Seeds reuse problems: for several seeds, prefer the bootstrap CIs from `report`.")


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    pols = None
    if "--policies" in sys.argv:
        pols = sys.argv[sys.argv.index("--policies") + 1].split(",")
        args = [x for x in args if x != sys.argv[sys.argv.index("--policies") + 1]]
    for f in args:
        main(f, pols)

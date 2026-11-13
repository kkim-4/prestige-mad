"""prestige_mad_v2.py — clean redesign of prestige-weighted debate, v2: every debate runs in two
random seatings (flip rates), the report adds paired significance tests, and --fresh-against
reports the questions earlier runs never used.

Two signals, each one thing:
  * prestige p_i(t)  = agent i's running ACCURACY on round-1 answers before question t
                       (uniform 0.5 during a warm-up of the first W questions)
  * confidence c_i   = the agent's STATED confidence this question (no self-consistency term)

We track D_i = c_i - p_i, the overconfidence gap, per agent per question, and plot it over time.

Debate (policy 'prestige'):
  1. Every agent answers once (round 1).
  2. If they disagree: each distinct answer gets ONE advocate (its highest-prestige holder),
     and the advocates are laid out most-trusted LAST, since agents weight the last argument
     they read most. Everyone then revises. Repeat until agreement or --rounds is reached.
  3. Final answer: a vote weighted by log-odds of prestige (a chronically-wrong agent's vote
     shrinks toward zero). During the warm-up the vote is one-agent-one-vote.

Baselines: 'all_speak' (standard simultaneous debate) and 'none' (majority vote, no debate).

    python3 prestige_mad_v2.py --mock --n 60 --policies prestige,all_speak,none
    python3 prestige_mad_v2.py --design expert --n 250 --problems gpqa_main_extra.jsonl --policies prestige,all_speak
    python3 prestige_mad_v2.py report --store runs/foo.jsonl      # accuracy + D-over-time plot

No API calls in --mock or in report.
"""
import argparse, hashlib, json, math, os, random, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mad_dual import (AgentTurn, GeminiAgent, OpenAIAgent, MockAgent, answers_match,
                      load_jsonl, synthetic, panel_spec, _spec_agent, _qid, _levels)

PRIOR = 0.5


# ---------- signals ----------
def prestige_path(round1, ids, warmup):
    """round1: list per question (in order) of {agent: correct(bool)}.
    Returns p[i] before each question = running accuracy, or 0.5 during warm-up."""
    path, hit, seen = [], {a: 0 for a in ids}, 0
    for r in round1:
        if seen < warmup:
            path.append({a: PRIOR for a in ids})
        else:
            path.append({a: hit[a] / seen for a in ids})
        for a in ids:
            hit[a] += int(r[a])
        seen += 1
    return path


def stem_key(question):
    """Hash of the question text without its answer options, so the same GPQA question
    matches across files whose options were shuffled differently."""
    stem = question.split("\n\n(A)")[0].strip()
    return hashlib.md5(stem.encode()).hexdigest()[:12]


def logodds(p):
    p = min(max(p, 0.02), 0.98)
    return math.log(p / (1 - p))


# ---------- one debate ----------
def advocates(turns, prestige, conf):
    """One advocate per distinct answer; most-trusted (prestige, then confidence) LAST."""
    best = {}
    for a, ans in turns.items():
        for g in best:
            if answers_match(ans, g):
                if (prestige[a], conf[a]) > (prestige[best[g]], conf[best[g]]):
                    best[g] = a
                break
        else:
            best[ans] = a
    return sorted(best.values(), key=lambda a: (prestige[a], conf[a]))   # ascending -> top last


def vote(turns, prestige, conf, weighted):
    tally = defaultdict(float)
    for a, ans in turns.items():
        w = logodds(prestige[a]) if weighted else 1.0
        for g in list(tally):
            if answers_match(ans, g):
                tally[g] += w
                break
        else:
            tally[ans] += w
    return max(tally, key=tally.get)


def run_debate(agents, problem, prestige, policy, rounds, first, warm):
    ids = [a.agent_id for a in agents]
    ans = {a.agent_id: first[a.agent_id].answer for a in agents}
    conf = {a.agent_id: first[a.agent_id].confidence for a in agents}
    rat = {a.agent_id: first[a.agent_id].rationale for a in agents}
    transcript, rnds = [], {aid: [(ans[aid], round(conf[aid], 3))] for aid in ids}
    for rnd in range(rounds - 1):
        if len(set(ans.values())) == 1:
            break
        if policy == "all_speak":
            speakers = ids
        else:
            speakers = advocates(ans, prestige, conf)
        board = [{"round": rnd, "speaker": s, "speaker_prestige": prestige[s],
                  "confidence": conf[s], "answer": ans[s], "rationale": rat[s]} for s in speakers]
        transcript += board
        outs = {}
        with ThreadPoolExecutor(max_workers=len(agents)) as ex:
            res = list(ex.map(lambda a: (a.agent_id, a.propose(problem, transcript, prestige)), agents))
        for aid, t in res:
            t = t.clipped()
            ans[aid], conf[aid], rat[aid] = t.answer, t.confidence, t.rationale
            rnds[aid].append((t.answer, round(t.confidence, 3)))
    final = vote(ans, prestige, conf, weighted=(policy == "prestige" and not warm))
    return {"final": final, "speakers": [b["speaker"] for b in transcript], "rounds": rnds}


# ---------- trial loop ----------
def build_panel(a):
    if a.mock:
        accs = [0.85, 0.62, 0.60, 0.58, 0.58]
        return [MockAgent(f"{'expert' if i == 0 else 'weak%d' % i}", accuracy=accs[i], conf_bias=0.25 if i else 0.1)
                for i in range(5)]
    spec = panel_spec(a)
    if not spec:
        sys.exit("give --design or --panel-file (or --mock)")
    return [_spec_agent(d, a) for d in spec]


def cmd_run(a):
    os.makedirs(os.path.dirname(a.store) or ".", exist_ok=True)
    store = open(a.store, "a")
    def emit(r): store.write(json.dumps(r) + "\n"); store.flush()
    probs = (load_jsonl(a.problems, a.n, levels=_levels(a)) if a.problems else synthetic(a.n, 0))[:a.n]
    agents = build_panel(a)
    ids = [ag.agent_id for ag in agents]
    emit({"kind": "config", "cfg": {"n": a.n, "problems": a.problems, "warmup": a.warmup,
          "rounds": a.rounds, "policies": a.policies, "agents": ids}})

    # phase 1: one independent answer per agent per question
    print(f"round 1: {len(probs)} questions x {len(ids)} agents")
    r1 = [None] * len(probs)
    def ask(i):
        p = probs[i]
        with ThreadPoolExecutor(max_workers=len(agents)) as ex:
            outs = list(ex.map(lambda ag: (ag.agent_id, ag.propose(p, [], None).clipped()), agents))
        return i, {aid: t for aid, t in outs}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, turns in ex.map(ask, range(len(probs))):
            r1[i] = turns
            emit({"kind": "r0", "pid": i, "qid": _qid(probs[i]), "stem": stem_key(probs[i]["question"]),
                  "truth": probs[i]["answer"],
                  "turns": {aid: {"answer": t.answer, "confidence": t.confidence} for aid, t in turns.items()}})

    correct1 = [{aid: answers_match(t.answer, probs[i]["answer"]) for aid, t in r1[i].items()} for i in range(len(probs))]
    path = prestige_path(correct1, ids, a.warmup)

    # D = c - p per agent per question (round-1 stated confidence vs prestige-before)
    for i in range(len(probs)):
        D = {aid: r1[i][aid].confidence - path[i][aid] for aid in ids}
        emit({"kind": "D", "pid": i, "D": D, "p": path[i],
              "c": {aid: r1[i][aid].confidence for aid in ids}})

    # phase 2: debates (only contested questions change anything)
    for pol in a.policies.split(","):
        if pol == "none":
            continue
        todo = [i for i in range(len(probs)) if len({r1[i][aid].answer for aid in ids}) > 1]
        print(f"policy {pol}: {len(todo)} contested debates")
        def do(i, pol=pol):
            q = _qid(probs[i])
            seats = agents[:]; random.Random(f"seat:{q}").shuffle(seats)     # random seating per question
            perm = agents[:]; random.Random(f"order:{q}").shuffle(perm)      # a second seating, for flips
            with ThreadPoolExecutor(max_workers=2) as ex2:
                f1 = ex2.submit(run_debate, seats, probs[i], path[i], pol, a.rounds, r1[i], i < a.warmup)
                f2 = ex2.submit(run_debate, perm, probs[i], path[i], pol, a.rounds, r1[i], i < a.warmup)
                return i, f1.result(), f2.result(), [x.agent_id for x in seats], [x.agent_id for x in perm]
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for i, res, res2, seats, perm in ex.map(do, todo):
                emit({"kind": "debate", "policy": pol, "pid": i, "qid": _qid(probs[i]),
                      "truth": probs[i]["answer"], "prestige": path[i],
                      "canon": {**res, "seats": seats}, "order": {**res2, "perm": perm}})
    store.close()
    print(f"-> {a.store}")
    report(a.store, a.fresh_against)


# ---------- report ----------
def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def boot_ci(diff, B=5000, seed=0):
    rng = random.Random(seed); n = len(diff)
    if n == 0:
        return float("nan"), float("nan")
    xs = sorted(sum(diff[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return xs[int(.025 * B)], xs[int(.975 * B)]


def seen_stems(spec):
    """spec: 'file[:n],file[:n]' -> stems of the problems those runs used (same loader and shuffle)."""
    out = set()
    for part in filter(None, (spec or "").split(",")):
        path, _, n = part.partition(":")
        if not os.path.exists(path):
            print(f"(fresh subset: {path} not found, skipped)"); continue
        for p in load_jsonl(path, int(n) if n else None):
            out.add(stem_key(p["question"]))
    return out


def report(path, fresh_against=None):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    r1 = {r["pid"]: r for r in rows if r["kind"] == "r0"}
    cfg = next(r["cfg"] for r in rows if r["kind"] == "config")
    ids, pols = cfg["agents"], [p for p in cfg["policies"].split(",") if p != "none"]
    deb = defaultdict(dict)
    for r in rows:
        if r["kind"] == "debate":
            deb[r["policy"]][r["pid"]] = r
    n = len(r1)
    print(f"\n{path}: {n} questions, agents {ids}")
    acc1 = {a: sum(answers_match(r1[i]["turns"][a]["answer"], r1[i]["truth"]) for i in r1) / n for a in ids}
    print("round-1 accuracy (= final prestige): " + ", ".join(f"{a} {acc1[a]:.3f}" for a in ids))
    top = max(ids, key=lambda a: acc1[a])

    def final_ok(pol, i, which):
        if i in deb[pol]:
            return answers_match(deb[pol][i][which]["final"], deb[pol][i]["truth"])
        return _maj(r1[i], ids) == 1                                   # unanimous: no debate ran
    def flipped(pol, i):
        d = deb[pol].get(i)
        return bool(d) and "order" in d and not answers_match(d["canon"]["final"], d["order"]["final"])

    def table(keys, label):
        if not keys:
            return
        cont = [i for i in keys if len({r1[i]["turns"][a]["answer"] for a in ids}) > 1]
        maj = sum(_maj(r1[i], ids) == 1 for i in keys) / len(keys)
        exp = sum(answers_match(r1[i]["turns"][top]["answer"], r1[i]["truth"]) for i in keys) / len(keys)
        print(f"\n== {label}: {len(keys)} questions, {len(cont)} contested")
        print(f"   majority vote (no debate) {maj:.1%}   top-prestige agent alone ({top}) {exp:.1%}")
        print(f"   {'policy':12s} {'all (2 seatings avg)':>21s} {'contested':>10s} {'flips':>7s}")
        mean = {}
        for pol in pols:
            m = {i: (final_ok(pol, i, "canon") + final_ok(pol, i, "order" if i in deb[pol] and "order" in deb[pol][i] else "canon")) / 2 for i in keys}
            mean[pol] = m
            fl = sum(flipped(pol, i) for i in keys) / len(keys)
            print(f"   {pol:12s} {sum(m.values())/len(keys):21.1%} {sum(m[i] for i in cont)/max(1,len(cont)):10.1%} {fl:7.1%}")
        if "prestige" in mean and "all_speak" in mean and cont:
            b = sum(final_ok("prestige", i, "canon") and not final_ok("all_speak", i, "canon") for i in cont)
            c = sum(final_ok("all_speak", i, "canon") and not final_ok("prestige", i, "canon") for i in cont)
            diff = [mean["prestige"][i] - mean["all_speak"][i] for i in cont]
            lo, hi = boot_ci(diff)
            print(f"   prestige vs all_speak, contested: {sum(diff)/len(diff):+.1%} (both seatings) 95% CI [{lo:+.1%}, {hi:+.1%}]")
            print(f"     exact McNemar, first seating: prestige right & all_speak wrong {b}, reverse {c}, p = {mcnemar(b, c):.3f}")
            e = {i: answers_match(r1[i]["turns"][top]["answer"], r1[i]["truth"]) for i in keys}
            for pol in ("prestige", "all_speak"):
                b2 = sum(final_ok(pol, i, "canon") and not e[i] for i in keys)
                c2 = sum(e[i] and not final_ok(pol, i, "canon") for i in keys)
                print(f"     {pol} vs top agent alone: {b2} vs {c2}, p = {mcnemar(b2, c2):.3f}")

    table(list(r1), "all questions")
    if fresh_against:
        seen = seen_stems(fresh_against)
        fresh = [i for i in r1 if r1[i].get("stem") and r1[i]["stem"] not in seen]
        table(fresh, f"untouched questions (not in {fresh_against})")

    # D-over-time plot
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        Ds = sorted([r for r in rows if r["kind"] == "D"], key=lambda r: r["pid"])
        if Ds:
            run = {a: [] for a in ids}; tot = {a: 0.0 for a in ids}
            for t, r in enumerate(Ds, 1):
                for a in ids:
                    tot[a] += r["D"][a]; run[a].append(tot[a] / t)
            plt.figure(figsize=(7, 4.3))
            for a in ids:
                plt.plot(range(1, len(Ds) + 1), run[a], lw=2.2 if a == top else 1,
                         color="#185FA5" if a == top else "#993C1D", alpha=1 if a == top else .6)
            plt.axhline(0, color="#999", lw=.6); plt.xlabel("questions seen (t)")
            plt.ylabel("running mean D = c - p")
            plt.title("Overconfidence gap over time (blue = top-prestige agent)")
            out = os.path.splitext(path)[0] + "_D.png"; plt.tight_layout(); plt.savefig(out, dpi=130)
            print(f"\nwrote {out}")
    except ImportError:
        print("(matplotlib not available; skipped D plot)")


def _maj(r, ids):
    from collections import Counter
    c = Counter()
    for a in ids:
        placed = False
        for g in list(c):
            if answers_match(r["turns"][a]["answer"], g):
                c[g] += 1; placed = True; break
        if not placed:
            c[r["turns"][a]["answer"]] += 1
    win = max(c, key=c.get)
    return int(answers_match(win, r["truth"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "report"])
    ap.add_argument("store_arg", nargs="?", help="report: path to a store")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--design", default=None); ap.add_argument("--panel-file", default=None)
    ap.add_argument("--expert-model", default=None); ap.add_argument("--weak-model", default=None)
    ap.add_argument("--expert-thinking", default="low"); ap.add_argument("--weak-thinking", default="none")
    ap.add_argument("--problems", default=None); ap.add_argument("--level", default=None)
    ap.add_argument("--n", type=int, default=60); ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=3); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7); ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-retries", type=int, default=8); ap.add_argument("--schema-mode", default="json_schema")
    ap.add_argument("--openai-extra", default=None); ap.add_argument("--provider", default=None); ap.add_argument("--api", default=None)
    ap.add_argument("--log", default="calls.jsonl"); ap.add_argument("--stagger", type=float, default=0.0)
    ap.add_argument("--policies", default="prestige,all_speak,none")
    ap.add_argument("--store", default="runs/prestige_run.jsonl")
    ap.add_argument("--fresh-against", default=None,
                    help="report: 'file[:n],...' problem files earlier runs used; also report the questions not in them")
    a = ap.parse_args()
    if a.cmd == "report":
        report(a.store_arg or a.store, a.fresh_against)
    else:
        from mad_dual import _RATE_LIMITER
        _RATE_LIMITER.min_interval = a.stagger
        cmd_run(a)


if __name__ == "__main__":
    main()

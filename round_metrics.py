"""Per-round metrics for debates stored by `mad_gemini.py trials` (needs runs made after
per-round logging was added: debate rows with canon.rounds). No API calls.

    python3 round_metrics.py gpqa_rounds.jsonl
    python3 round_metrics.py gpqa_rounds.jsonl --contested    # only problems with some right, some wrong

For each policy, on problems where the debate ran, round by round:
  vote right   plain majority vote over the agents' answers in that round
  agents right mean number of agents (out of 5) holding the right answer
  speaker right share of rounds where the chosen speaker held the right answer
  switches     agents that changed answer this round: wrong->right, right->wrong, wrong->other wrong
  followed     share of switches that moved to the answer the previous speaker argued
"""
import json, sys, os
from collections import defaultdict, Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mad_gemini import answers_match


def majority_right(answers, truth):
    groups = []                                     # group equivalent answers
    for x in answers:
        for g in groups:
            if answers_match(g[0], x):
                g.append(x); break
        else:
            groups.append([x])
    m = max(len(g) for g in groups)
    tied = [g for g in groups if len(g) == m]
    return sum(answers_match(g[0], truth) for g in tied) / len(tied)


def main(path, contested_only=False):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    deb = [r for r in rows if r["kind"] == "debate" and r["canon"].get("rounds") and r["canon"]["speakers"]]
    if not deb:
        sys.exit("no debates with per-round answers in this store (run trials with the updated mad_gemini.py)")
    by_pol = defaultdict(list)
    for r in deb:
        R = r["canon"]["rounds"]
        if contested_only:
            n_right = sum(answers_match(v[0][0], r["truth"]) for v in R.values())
            if not 0 < n_right < len(R):
                continue
        by_pol[r["policy"]].append(r)

    for pol, rs in by_pol.items():
        nr = max(len(v) for r in rs for v in r["canon"]["rounds"].values())
        print(f"\n== {pol}: {len(rs)} debated problem-runs" + (" (contested only)" if contested_only else ""))
        print(f"   {'round':>5s} {'vote right':>11s} {'agents right':>13s} {'speaker right':>14s}"
              f" {'W->R':>6s} {'R->W':>6s} {'W->W':>6s} {'followed':>9s}")
        for k in range(nr):
            vote = agents = 0.0
            spk_ok = spk_n = 0
            sw = Counter(); fol = fol_n = 0
            for r in rs:
                R, truth, spk = r["canon"]["rounds"], r["truth"], r["canon"]["speakers"]
                cur = {a: v[min(k, len(v) - 1)][0] for a, v in R.items()}   # stopped early -> carry last
                vote += majority_right(list(cur.values()), truth)
                agents += sum(answers_match(x, truth) for x in cur.values())
                if k < len(spk):                                            # speaker chosen in round k
                    s = spk[k]; spk_n += 1
                    spk_ok += answers_match(R[s][min(k, len(R[s]) - 1)][0], truth)
                if k > 0:
                    prev = {a: v[min(k - 1, len(v) - 1)][0] for a, v in R.items()}
                    said = None
                    if k - 1 < len(spk):
                        s = spk[k - 1]; said = R[s][min(k - 1, len(R[s]) - 1)][0]
                    for a in R:
                        if answers_match(prev[a], cur[a]):
                            continue
                        was, now = answers_match(prev[a], truth), answers_match(cur[a], truth)
                        sw["W->R" if now else ("R->W" if was else "W->W")] += 1
                        if said is not None:
                            fol_n += 1; fol += answers_match(cur[a], said)
            n = len(rs)
            sp = f"{spk_ok / spk_n:.1%}" if spk_n else "-"
            fl = f"{fol / fol_n:.0%}" if fol_n else "-"
            print(f"   {k + 1:>5d} {vote / n:11.1%} {agents / n:13.2f} {sp:>14s}"
                  f" {sw['W->R']:6d} {sw['R->W']:6d} {sw['W->W']:6d} {fl:>9s}")
    print("\nRound 1 = independent answers (identical across policies); later rounds show what each"
          "\npolicy's speakers did to the group.")


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    main(args[0] if args else "gpqa_rounds.jsonl", "--contested" in sys.argv)
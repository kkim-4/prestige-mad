"""Why does dual help? Splits dual's debates by which of its two round-1 speakers was right,
and shows how the group moved in round 2 and where it ended. No API calls.

    python3 mechanism_check.py gpqa_v4.jsonl gpqa_main_v4.jsonl math_dual.jsonl
"""
import json, sys, os
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mad_gemini import answers_match


def main(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    deb = [r for r in rows if r["kind"] == "debate" and r["policy"] == "dual"
           and r["canon"].get("rounds") and r["canon"].get("speaker_rounds")]
    if not deb:
        print(f"\n{path}: no dual debates with per-round data"); return
    groups = defaultdict(list)
    for r in deb:
        R, truth, spk, srd = r["canon"]["rounds"], r["truth"], r["canon"]["speakers"], r["canon"]["speaker_rounds"]
        first = [s for s, q in zip(spk, srd) if q == 0]
        ok = lambda a, k: answers_match(R[a][min(k, len(R[a]) - 1)][0], truth)
        lead = ok(first[0], 0)
        diss = ok(first[1], 0) if len(first) > 1 else None
        key = ("leader right" if lead else "leader wrong") + (
            "" if diss is None else (", dissenter right" if diss else ", dissenter wrong"))
        r1 = sum(ok(a, 0) for a in R)
        r2 = sum(ok(a, 1) for a in R)
        groups[key].append((r1, r2, answers_match(r["canon"]["final"], truth)))
    print(f"\n{path}: {len(deb)} dual debates")
    print(f"   {'round-1 speakers':36s} {'n':>4s} {'agents right r1 -> r2':>22s} {'final right':>12s}")
    for key in sorted(groups):
        g = groups[key]
        n = len(g)
        print(f"   {key:36s} {n:4d} {sum(x[0] for x in g)/n:9.2f} -> {sum(x[1] for x in g)/n:4.2f}"
              f"      {sum(x[2] for x in g)/n:9.0%}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
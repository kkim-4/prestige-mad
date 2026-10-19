"""Would a plain majority over the agents' final answers beat the stored weighted vote?
No API calls.   python3 revote.py sc.jsonl math_fix.jsonl gpqa.jsonl gpqa_fix.jsonl
"""
import json, sys, os
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mad_gemini import answers_match
def maj(ans, truth):
    g=[]
    for x in ans:
        for gg in g:
            if answers_match(gg[0],x): gg.append(x); break
        else: g.append([x])
    m=max(len(x) for x in g); t=[x for x in g if len(x)==m]
    return sum(answers_match(x[0],truth) for x in t)/len(t)
for f in sys.argv[1:]:
    rows=[json.loads(l) for l in open(f)]
    r0={(r["seed"],r["qid"]):r for r in rows if r["kind"]=="r0"}
    by=defaultdict(list)
    for r in rows:
        if r["kind"]=="debate" and r["canon"]["speakers"]: by[r["policy"]].append(r)
    print("\n"+f)
    for p,rs in sorted(by.items()):
        n=len(rs)
        stored=sum(answers_match(r["canon"]["final"],r["truth"]) for r in rs)
        unw=sum(maj(list(r["canon"]["last"].values()),r["truth"]) for r in rs)
        # agents who switched from round-1 answer
        conv_right=sum(1 for r in rs for a,x in r["canon"]["last"].items()
                       if not answers_match(r0[(r["seed"],r["qid"])]["turns"][a]["answer"],x) and answers_match(x,r["truth"]))
        conv_wrong=sum(1 for r in rs for a,x in r["canon"]["last"].items()
                       if not answers_match(r0[(r["seed"],r["qid"])]["turns"][a]["answer"],x) and not answers_match(x,r["truth"]))
        print(f"  {p:16s} debated {n:3d}  stored weighted vote right {stored:5.1f}  plain majority of final answers {unw:5.1f}"
              f"   agents switched: to right {conv_right}, to wrong {conv_wrong}")
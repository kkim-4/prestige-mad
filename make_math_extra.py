#!/usr/bin/env python3
"""
make_math_extra.py — builds math_extra.jsonl: the MATH test set (5,000 problems,
EleutherAI/hendrycks_math on Hugging Face, MIT license) minus every problem
already in math.jsonl (MATH-500), so the new problems are guaranteed unseen.

  pip install datasets
  python3 make_math_extra.py --exclude math.jsonl --out math_extra.jsonl
"""
import argparse
import json
import re

SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus"]


def key(text):
    return re.sub(r"\s+", " ", text).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", default="math.jsonl")
    ap.add_argument("--out", default="math_extra.jsonl")
    a = ap.parse_args()
    from datasets import load_dataset

    seen = set()
    with open(a.exclude) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                seen.add(key(d.get("problem") or d.get("question", "").split("\n\nGive only")[0]))

    kept = dropped = 0
    with open(a.out, "w") as out:
        for subj in SUBJECTS:
            for d in load_dataset("EleutherAI/hendrycks_math", subj, split="test"):
                if key(d["problem"]) in seen:
                    dropped += 1
                    continue
                out.write(json.dumps({"problem": d["problem"], "solution": d["solution"],
                                      "level": d["level"], "type": d["type"]}) + "\n")
                kept += 1
    print(f"wrote {kept} problems to {a.out}; skipped {dropped} already in {a.exclude}")
    if dropped < 450:
        print(f"WARNING: expected about 500 overlaps with MATH-500, found {dropped}. "
              f"Check that {a.exclude} is MATH-500 before trusting that the new problems are unseen.")


if __name__ == "__main__":
    main()
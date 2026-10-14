#!/usr/bin/env python3
"""
make_gpqa.py — builds gpqa_diamond.jsonl (198 graduate-level multiple-choice
science questions) in the {"question","answer"} format mad_gemini.py reads.

GPQA is gated: accept the terms at huggingface.co/datasets/Idavidrein/gpqa,
then `huggingface-cli login`.

  pip install datasets huggingface_hub
  python3 make_gpqa.py --out gpqa_diamond.jsonl

Choices are shuffled with a fixed seed, so the correct letter is spread over A-D
and the file is identical on every machine.
"""
import argparse
import json
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gpqa_diamond.jsonl")
    a = ap.parse_args()
    from datasets import load_dataset
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    rng = random.Random(0)
    letters = "ABCD"
    with open(a.out, "w") as f:
        for d in ds:
            opts = [d["Correct Answer"], d["Incorrect Answer 1"], d["Incorrect Answer 2"], d["Incorrect Answer 3"]]
            opts = [str(o).strip() for o in opts]
            order = list(range(4))
            rng.shuffle(order)
            body = "\n".join(f"({letters[j]}) {opts[i]}" for j, i in enumerate(order))
            q = (d["Question"].strip() + "\n\n" + body +
                 "\n\nAnswer with only the letter of the correct option (A, B, C or D).")
            f.write(json.dumps({"question": q, "answer": letters[order.index(0)],
                                "level": d.get("High-level domain", "")}) + "\n")
    print(f"wrote {len(ds)} questions to {a.out}")


if __name__ == "__main__":
    main()

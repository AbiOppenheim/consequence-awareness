#!/usr/bin/env python
"""Phase 2 prep (CPU): build data/contrast/refusal.jsonl from Arditi's train splits.

Why this file exists
--------------------
The r_hat gate is a machinery check: does OUR extraction reproduce Arditi's refusal direction?
To make that a clean test we run both sides on the SAME prompts. So we copy Arditi's exact
harmful/harmless instructions (AdvBench-derived vs Alpaca) into our own schema. Our caching
stage reads this file to produce OUR r_hat; the Arditi runner feeds these SAME prompts to
Arditi's own extractor. Identical inputs, different extraction code -> the cosine tests the
machinery and nothing else.

We take a balanced n per class (default 128, mirroring Arditi's n_train). We do NOT apply
Arditi's optional model-based refusal-score filtering (that needs the GPU model and only
trims a few borderline prompts); difference-in-means is a mean, so it is robust to that.

Reads:  external/refusal_direction/dataset/splits/{harmful,harmless}_train.json
Writes: data/contrast/refusal.jsonl
        rows: {id, text, label, split, source}   label 1 = harmful, 0 = harmless
"""

import argparse
import json
import random

import _bootstrap  # noqa: F401
from consequence.config import REPO_ROOT

SPLITS = REPO_ROOT / "external" / "refusal_direction" / "dataset" / "splits"


def load_instructions(name: str) -> list[str]:
    p = SPLITS / f"{name}.json"
    if not p.exists():
        raise SystemExit(
            f"missing {p}\n"
            "clone andyrdt/refusal_direction into external/ first (see external/README.md)."
        )
    return [r["instruction"] for r in json.loads(p.read_text())]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128, help="prompts per class (mirrors Arditi n_train)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/contrast/refusal.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    harmful = load_instructions("harmful_train")
    harmless = load_instructions("harmless_train")
    harmful = rng.sample(harmful, min(args.n, len(harmful)))
    harmless = rng.sample(harmless, min(args.n, len(harmless)))

    rows = []
    for i, t in enumerate(harmful):
        rows.append({"id": f"advbench_train_{i}", "text": t, "label": 1,
                     "split": "train", "source": "advbench"})
    for i, t in enumerate(harmless):
        rows.append({"id": f"alpaca_train_{i}", "text": t, "label": 0,
                     "split": "train", "source": "alpaca"})

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"[build] {len(rows)} rows ({len(harmful)} harmful + {len(harmless)} harmless) -> {out}")


if __name__ == "__main__":
    main()

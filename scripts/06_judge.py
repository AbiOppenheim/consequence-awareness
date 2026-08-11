#!/usr/bin/env python
"""Stage 06 (API, CPU): three-way judging of generations.

    python scripts/06_judge.py --generations artifacts/generations/fiction_jailbreaks_L14.jsonl

Adds label (refusal/bypass/degenerate) + strongreject + coherent to every row, writes the
scored rows to artifacts/scores/, and prints per-condition summaries. Never StrongREJECT alone.

Requires ANTHROPIC_API_KEY. The API call itself is stubbed in consequence.judge.score_batch;
this wrapper fixes the I/O flow so nothing else changes once the call is wired.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
from consequence import data as D
from consequence import judge as J
from consequence.config import load_config, resolve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen.yaml")
    ap.add_argument("--generations", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    rows = D.load_jsonl(args.generations)
    scored = J.score_batch(rows, model=cfg["judge"]["model"],
                           use_batch_api=cfg["judge"]["use_batch_api"])

    out = resolve(cfg["paths"]["scores"]) / (Path(args.generations).stem + "_scored.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in scored:
            f.write(json.dumps(r) + "\n")

    by_cond = defaultdict(list)
    for r in scored:
        by_cond[(r["condition"], r.get("alpha"))].append(r)
    summary = {f"{c}@alpha={a}": J.summarize(rs) for (c, a), rs in sorted(by_cond.items())}

    with open(out.with_suffix(".summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    for k, v in summary.items():
        print(f"[{k}] refusal={v['refusal_rate']:.2f} bypass={v['bypass_rate']:.2f} "
              f"degen={v['degenerate_rate']:.2f} SR={v['mean_strongreject']:.2f}")


if __name__ == "__main__":
    main()

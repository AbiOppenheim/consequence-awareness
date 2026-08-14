#!/usr/bin/env python
"""Stage 06 (API, CPU): three-way judging of generations.

    python scripts/06_judge.py --generations artifacts/generations/fiction_jailbreaks_L14.jsonl

Adds label (refusal/bypass/degenerate) + strongreject + coherent to every row, writes the
scored rows to artifacts/scores/, and prints per-condition summaries. Never StrongREJECT alone.

Needs the API key for whichever provider `judge.provider` names in the config: OPENAI_API_KEY
for 'openai' (the default), ANTHROPIC_API_KEY for 'anthropic'. The judge prompt, schema and
parsing are shared, so the provider changes the bill and nothing about what is measured.
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
    ap.add_argument("--limit", type=int, default=None,
                    help="judge only the first N rows. Use it once against the live API before "
                         "spending on the full file: it proves the key, the model name and the "
                         "structured-output request are all accepted, for a fraction of a cent. "
                         "Writes to a _limitN_scored.jsonl so it cannot be mistaken for a "
                         "complete scoring.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rows = D.load_jsonl(args.generations)
    if args.limit:
        rows = rows[:args.limit]
        print(f"[judge] --limit {args.limit}: TRIAL RUN, not a complete scoring")
    jcfg = cfg["judge"]
    scored = J.score_batch(rows, model=jcfg["model"],
                           provider=jcfg.get("provider", "openai"),
                           use_batch_api=jcfg.get("use_batch_api", False),
                           max_workers=jcfg.get("max_workers", 8))

    suffix = f"_limit{args.limit}_scored.jsonl" if args.limit else "_scored.jsonl"
    out = resolve(cfg["paths"]["scores"]) / (Path(args.generations).stem + suffix)
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

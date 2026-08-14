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


def key(row: dict) -> tuple:
    """Identifies one generation across the generations file and the scored file."""
    return (row["condition"], row.get("alpha"), row.get("idx"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen.yaml")
    ap.add_argument("--generations", required=True)
    ap.add_argument("--resume", action="store_true",
                    help="re-judge ONLY the rows that failed last time, keeping the successful "
                         "verdicts from the existing scored file. A rate-limited run leaves "
                         "most rows judged; this pays for the remainder instead of the lot.")
    ap.add_argument("--max-failed", type=float, default=0.02,
                    help="fail the stage if more than this FRACTION of rows went unjudged. "
                         "Missing judgments are not neutral: they land unevenly across "
                         "conditions, and a summary built on them looks authoritative while "
                         "resting on a biased subsample.")
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
    suffix = f"_limit{args.limit}_scored.jsonl" if args.limit else "_scored.jsonl"
    out = resolve(cfg["paths"]["scores"]) / (Path(args.generations).stem + suffix)

    kept = []
    if args.resume:
        if not out.exists():
            raise SystemExit(f"--resume needs an existing {out.name} to resume from")
        # Match on (condition, alpha, idx), never on row count. The generations file GROWS when
        # a condition is appended — `05_generate.py --extra-direction` does exactly that — so a
        # length check would reject the most common reason to resume after the first one.
        # The generations file is the source of truth: a verdict is kept only if its row is
        # still there, and every row without a successful verdict gets judged.
        prior = {key(r): r for r in D.load_jsonl(out)}
        kept, todo = [], []
        for r in rows:
            got = prior.get(key(r))
            (kept if got and got.get("label") in J.LABELS else todo).append(got or r)
        new = sum(1 for r in todo if key(r) not in prior)
        rows = todo
        print(f"[resume] {len(kept)} verdicts kept | {len(rows) - new} to re-judge, "
              f"{new} never judged")
        if not rows:
            print("[resume] nothing left to judge")

    judged = J.score_batch(rows, model=jcfg["model"],
                           provider=jcfg.get("provider", "openai"),
                           use_batch_api=jcfg.get("use_batch_api", False),
                           max_workers=jcfg.get("max_workers", 4),
                           max_retries=jcfg.get("max_retries", 8)) if rows else []
    scored = kept + judged
    # Keep the generation order, so the scored file lines up with the generations file.
    order = {key(r): i for i, r in enumerate(D.load_jsonl(args.generations))}
    scored.sort(key=lambda r: order.get(key(r), 0))

    n_failed = sum(r.get("label") not in J.LABELS for r in scored)
    frac = n_failed / max(len(scored), 1)

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
        flag = "  <- INCOMPLETE" if v["n_failed"] else ""
        print(f"[{k}] refusal={v['refusal_rate']:.2f} bypass={v['bypass_rate']:.2f} "
              f"degen={v['degenerate_rate']:.2f} SR={v['mean_strongreject']:.2f} "
              f"n={v['n_judged']}/{v['n_total']}{flag}")

    if frac > args.max_failed:
        raise SystemExit(
            f"\n{n_failed}/{len(scored)} rows ({frac:.0%}) went unjudged — above the "
            f"{args.max_failed:.0%} threshold.\n"
            f"  The verdicts that DID land are saved in {out.name}, so nothing is lost.\n"
            "  Missing judgments are not spread evenly: they accumulate in whichever conditions\n"
            "  were being judged when the budget ran out, so the surviving rows are a biased\n"
            "  subsample and any dose-response read off them may be a rate-limit curve.\n"
            "  Re-run with --resume to judge only the failures (and lower judge.max_workers if\n"
            "  they were 429s)."
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Stage 14 (CPU, no API): draw the human-agreement sample from the judge's verdicts.

    python scripts/14_sample_verdicts.py

Picks ~50 scored generations, STRATIFIED BY JUDGE LABEL, and writes them to
artifacts/human_check/sample.jsonl for stage 15 to tag by hand. Every number in this project
rests on gpt-4.1-mini, and that instrument has already produced two wrong conclusions this
project caught (the XSTest rubric; StrongREJECT disagreeing with its own label). This stage is
the first half of checking it.

Why stratified and not a simple random sample. At the population rates, 50 random rows from the
L18 attack set would contain about 5 bypasses -- and 'bypass' is the label the causal claim is
made of. Equal-ish allocation buys per-label reliability, which is what we actually want to
report. The cost is that raw agreement over the sample is NOT the population agreement, so each
row carries a sampling `weight` (= stratum size / rows drawn from it) and stage 15 reports both
the unweighted per-label figures and the weight-corrected population estimate.

Selection is by hash of (seed, row key), not by a shuffled index. So the same rows are chosen
however the files are ordered, and appending a condition to a scored file later leaves the
already-tagged rows in the sample instead of redrawing the lot (Rule 4).
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
from consequence import data as D
from consequence import judge as J
from consequence.config import load_config, resolve
from consequence.io import git_sha

# Rows per (file, judge label). The rare labels carry the claims -- 'bypass' is the attack
# succeeding, 'refused' is the over-refusal cost -- so they get as many rows as the common
# ones. Fixed here, before any agreement number is seen.
DEFAULT_ALLOC = {
    "fiction_jailbreaks_L18_scored.jsonl": {"refusal": 12, "bypass": 10, "degenerate": 8},
    "xstest_L18_scored.jsonl": {"answered": 8, "refused": 8, "degenerate": 4},
}


def row_key(stem: str, row: dict) -> str:
    """Stable identity of one verdict across the scored file, the sample, and the tags."""
    return f"{stem}|{row['condition']}|{row.get('alpha')}|{row.get('idx')}"


def hash_rank(seed: int, key: str) -> str:
    """Deterministic per-row sort key. Uniform over rows, independent of file order."""
    return hashlib.sha256(f"{seed}|{key}".encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen.yaml")
    ap.add_argument("--scores", nargs="*", default=None,
                    help="scored .jsonl files to sample from. Default: the two files the "
                         "headline numbers come from (fiction_jailbreaks_L18, xstest_L18).")
    ap.add_argument("--seed", type=int, default=None, help="default: the config seed")
    ap.add_argument("--out", default="artifacts/human_check/sample.jsonl")
    ap.add_argument("--alloc", default=None,
                    help="JSON overriding rows-per-label, e.g. '{\"bypass\": 20}' applied to "
                         "every file. Use to TOP UP a finished sample, never to redraw one "
                         "after seeing the agreement number.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    scores_dir = resolve(cfg["paths"]["scores"])
    files = [Path(p) for p in args.scores] if args.scores else \
            [scores_dir / name for name in DEFAULT_ALLOC]
    override = json.loads(args.alloc) if args.alloc else {}

    sample, design = [], []
    for path in files:
        if not path.exists():
            raise SystemExit(f"no such scored file: {path}")
        stem = path.name
        rows = D.load_jsonl(path)
        rubrics = {r.get("rubric", "harmful") for r in rows}
        if len(rubrics) != 1:
            raise SystemExit(f"{stem} mixes rubrics {rubrics}; judge it under one and re-run")
        rubric = rubrics.pop()

        by_label = defaultdict(list)
        for r in rows:
            by_label[r["label"]].append(r)

        alloc = dict(DEFAULT_ALLOC.get(stem, {}))
        alloc.update({k: v for k, v in override.items() if k in J.RUBRICS[rubric].labels})
        if not alloc:
            raise SystemExit(
                f"no allocation for {stem}. Add one to DEFAULT_ALLOC or pass --alloc.")

        for label, want in alloc.items():
            pool = by_label.get(label, [])
            take = min(want, len(pool))
            if take < want:
                print(f"[sample] {stem}: only {len(pool)} rows labelled {label!r}, "
                      f"taking all of them (asked for {want})")
            chosen = sorted(pool, key=lambda r: hash_rank(seed, row_key(stem, r)))[:take]
            # Inverse sampling probability. A stratum we sampled heavily must not speak for the
            # population as loudly as its row count in this file suggests.
            weight = len(pool) / take if take else 0.0
            for r in chosen:
                sample.append({
                    "key": row_key(stem, r),
                    "source": stem,
                    "rubric": rubric,
                    "condition": r["condition"],
                    "alpha": r.get("alpha"),
                    "idx": r.get("idx"),
                    "prompt": r["prompt"],
                    "completion": r["completion"],
                    # Kept in the sample file so stage 15 can stay blind by simply not printing
                    # them, and so the report needs no second pass over the scored files.
                    "judge_label": r["label"],
                    "judge_coherent": r.get("coherent"),
                    "judge_strongreject": r.get("strongreject"),
                    "weight": weight,
                })
            design.append({"source": stem, "rubric": rubric, "label": label,
                           "population": len(pool), "drawn": take, "weight": weight})

    # Present the rows in hash order across files too, so the tagger does not walk the attack
    # set and then the control set -- an ordering the hand can learn.
    sample.sort(key=lambda r: hash_rank(seed, r["key"]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in sample:
            f.write(json.dumps(r) + "\n")

    # Sidecar (committed -- it carries no completions, only the design that produced them).
    meta = {
        "n": len(sample),
        "seed": seed,
        "sources": [str(p) for p in files],
        "design": design,
        "git_sha": git_sha(),
        "note": "Strata are judge labels; `weight` = population/drawn for reweighting.",
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"\n[sample] {len(sample)} verdicts -> {out}")
    for d in design:
        print(f"  {d['source']:<40} {d['label']:<11} {d['drawn']:>3} of {d['population']:>5}"
              f"   weight {d['weight']:.1f}")
    print(f"\nNext:  python scripts/15_tag_verdicts.py")


if __name__ == "__main__":
    main()

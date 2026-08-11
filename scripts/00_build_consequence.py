#!/usr/bin/env python
"""Assemble data/contrast/consequence.jsonl from two normalized source files.

This is a BUILD step, not a data file you edit. Edit the sources, then rerun:
    data/contrast/benign_tasks.jsonl   one row per task    (the [TASK] content)
    data/contrast/framings.jsonl       one row per framing (the real/hypo wrapper, {task} slot)
        -> consequence.jsonl           GENERATED

Two modes:
  --mode sample  (default)  draw a fixed number of CONTRAST PAIRS (below). With many framings the
                            full cross product is larger than we need.
  --mode full               the exhaustive cross product (framings x tasks), the old behaviour.

What a "pair" is, and why we sample this way
--------------------------------------------
A pair is the project's unit: ONE task rendered under ONE real framing and ONE hypo framing ->
two rows (label 1 and 0). Sampling pairs (rather than assigning a random task to each framing
independently) is what preserves the science:

  - Both rows of a pair share the SAME task, so the task multiset is IDENTICAL between the real
    and hypo classes. Task content therefore cancels in mean(real)-mean(hypo) -> the direction
    isolates framing, not topic. (CLAUDE.md 2: matched task content across every contrast pair.)
  - Framings are cycled in reshuffled passes -> every framing is used about equally (max diff 1),
    which is the whole point of expanding to 140 framings: maximise framing diversity.
  - Within a pair's framing-couple, the task is chosen least-used-first -> all tasks are used
    about equally, so no task is over-represented (no topic bias).
  - Held-out stays a framing-level split; pairs are drawn WITHIN a split, and the pair budget is
    shared between train and held-out in proportion to their framing counts, so every framing
    (train or held-out) gets ~equal usage.

Scientific checks baked in as before: every framing has a {task} slot; tasks never leak framing
vocabulary; held-out routes are reported against train routes; and (sample mode) the real/hypo
task multisets are asserted identical.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TASKS = REPO / "data" / "contrast" / "benign_tasks.jsonl"
FRAMINGS = REPO / "data" / "contrast" / "framings.jsonl"
OUT = REPO / "data" / "contrast" / "consequence.jsonl"

# framing vocabulary that must NOT leak into a task (would let the probe cheat).
FRAMING_TOKENS = {
    "real", "live", "production", "customer", "executed", "execute", "simulation", "simulate",
    "dream", "fiction", "fictional", "imagine", "character", "novel", "story", "play", "exercise",
    "operation", "operations", "role", "scenario", "hypothetical", "infrastructure", "verbatim",
    "counterfactual", "sandbox", "sandboxed", "assistant", "make-believe",
}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def validate(tasks, framings):
    """The build-time invariants. Returns a per-polarity route-disjointness report."""
    for fr in framings:
        if "{task}" not in fr["text"]:
            raise SystemExit(f"framing {fr['framing_id']} has no {{task}} slot")
        if "route" not in fr:
            raise SystemExit(f"framing {fr['framing_id']} has no 'route' (its semantic mechanism)")
    leaks = [
        (t["task_id"], w)
        for t in tasks
        for w in t["text"].lower().replace(",", " ").replace(".", " ").split()
        if w in FRAMING_TOKENS
    ]
    if leaks:
        raise SystemExit(f"tasks leak framing vocabulary (would let the probe cheat): {leaks}")
    if len({f["framing_id"] for f in framings}) != len(framings):
        raise SystemExit("duplicate framing_id")
    if len({t["task_id"] for t in tasks}) != len(tasks):
        raise SystemExit("duplicate task_id")

    report = []
    for pol in ("real", "hypo"):
        tr = {f["route"] for f in framings if f["polarity"] == pol and f["split"] == "train"}
        ho = {f["route"] for f in framings if f["polarity"] == pol and f["split"] == "heldout"}
        overlap = tr & ho
        if overlap:
            report.append(f"  WARNING [{pol}] held-out reuses train route(s) {sorted(overlap)} "
                          f"— held-out is a weaker test of the concept")
        else:
            report.append(f"  [{pol}] routes disjoint  train={sorted(tr)}  held-out={sorted(ho)}")
    return report


def emit_row(fr, task, pair_id):
    return {
        "id": f"{fr['framing_id']}__{task['task_id']}",
        "framing": fr["polarity"],          # -> label: real=1, hypo=0
        "split": fr["split"],               # held-out is by framing (template)
        "template_id": fr["framing_id"],    # group key for the probe split
        "route": fr["route"],               # semantic mechanism (for per-route analysis)
        "task_id": task["task_id"],
        "task_source": "authored_benign",
        "pair_id": pair_id,                 # links the real and hypo row of one contrast pair
        "text": fr["text"].replace("{task}", task["text"]),
    }


def _reshuffling_cycle(items, rng):
    """Yield items forever, reshuffling each full pass -> near-uniform usage (max diff 1)."""
    pool = []
    while True:
        if not pool:
            pool = items[:]
            rng.shuffle(pool)
        yield pool.pop()


def build_full(tasks, framings):
    return [emit_row(fr, t, f"full_{fr['framing_id']}__{t['task_id']}")
            for fr in framings for t in tasks]


def build_sample(tasks, framings, n_pairs, seed):
    rng = random.Random(seed)
    task_ids = [t["task_id"] for t in tasks]
    task_by_id = {t["task_id"]: t for t in tasks}

    real = [f for f in framings if f["polarity"] == "real"]
    n_real = len(real)

    # Budget per split, proportional to its share of real framings -> every framing (train or
    # held-out) gets ~equal usage. Held-out gets its proportional share; train gets the rest.
    n_real_heldout = sum(f["split"] == "heldout" for f in real)
    budget = {"heldout": round(n_pairs * n_real_heldout / n_real)}
    budget["train"] = n_pairs - budget["heldout"]

    rows, pid = [], 0
    for split in ("train", "heldout"):
        R = [f for f in real if f["split"] == split]
        H = [f for f in framings if f["polarity"] == "hypo" and f["split"] == split]
        if not R or not H or budget[split] == 0:
            continue
        # pair real & hypo framings by index (arbitrary coupling; only serves task-matching).
        # if the two sides differ in count, cycle the shorter to cover the longer.
        m = max(len(R), len(H))
        couples = [(R[i % len(R)], H[i % len(H)]) for i in range(m)]
        n = budget[split]

        couple_cycle = _reshuffling_cycle(couples, rng)
        counts = Counter()                    # global task usage within this split
        used = defaultdict(set)               # tasks already given to a specific couple
        for _ in range(n):
            real_fr, hypo_fr = next(couple_cycle)
            key = (real_fr["framing_id"], hypo_fr["framing_id"])
            cand = [tid for tid in task_ids if tid not in used[key]]
            if not cand:                      # couple exhausted all tasks -> allow reuse
                used[key].clear()
                cand = task_ids
            tid = min(cand, key=lambda x: (counts[x], rng.random()))
            counts[tid] += 1
            used[key].add(tid)
            task = task_by_id[tid]
            rows.append(emit_row(real_fr, task, f"{split}_{pid}"))
            rows.append(emit_row(hypo_fr, task, f"{split}_{pid}"))
            pid += 1
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sample", "full"], default="sample")
    ap.add_argument("--pairs", type=int, default=1000, help="contrast pairs (sample mode); 2 rows each")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tasks = load_jsonl(TASKS)
    framings = load_jsonl(FRAMINGS)
    report = validate(tasks, framings)

    if args.mode == "full":
        rows = build_full(tasks, framings)
    else:
        rows = build_sample(tasks, framings, args.pairs, args.seed)

    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # --- diagnostics -------------------------------------------------------
    real_rows = [r for r in rows if r["framing"] == "real"]
    hypo_rows = [r for r in rows if r["framing"] == "hypo"]
    real_tasks = Counter(r["task_id"] for r in real_rows)
    hypo_tasks = Counter(r["task_id"] for r in hypo_rows)
    if args.mode == "sample":
        # the invariant: real and hypo classes must see the same task multiset.
        assert real_tasks == hypo_tasks, "real/hypo task multiset mismatch — invariant broken"

    fr_used = Counter(r["template_id"] for r in rows)
    print(f"mode={args.mode}  tasks={len(tasks)}  framings={len(framings)}  ->  {len(rows)} rows "
          f"({len(real_rows)} real / {len(hypo_rows)} hypo)")
    print(f"  held-out rows = {sum(r['split']=='heldout' for r in rows)}  "
          f"train rows = {sum(r['split']=='train' for r in rows)}")
    print(f"  framing coverage = {len(fr_used)}/{len(framings)} used  "
          f"(per-framing rows: min {min(fr_used.values())}, max {max(fr_used.values())})")
    if real_tasks:
        print(f"  task usage per class: min {min(real_tasks.values())}, "
              f"max {max(real_tasks.values())}, all {len(task_ids_used(real_tasks, tasks))}/{len(tasks)} used")
        print(f"  real/hypo task multiset identical: {real_tasks == hypo_tasks}")
    print("\n".join(report))
    print(f"  wrote {OUT.relative_to(REPO)}")


def task_ids_used(counter, tasks):
    return [t["task_id"] for t in tasks if counter.get(t["task_id"], 0) > 0]


if __name__ == "__main__":
    main()

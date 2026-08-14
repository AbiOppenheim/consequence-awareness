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


def resolve_rubric(value: str, generations: str, cfg: dict) -> J.Rubric:
    """Which question to ask the judge about this file: 'harmful' or 'benign'.

    Auto-detection reads the eval NAME from the .meta.json sidecar stage 05 wrote, not the
    prompts or the filename, and looks it up in judge.benign_evals. Guessing from the prompt
    text is what a judge would do, and the whole point of this split is that a benign prompt
    dressed in sensitive-sounding words ("kill a process") is exactly what fools it.

    No sidecar means no way to tell an over-refusal control from an attack set, and the wrong
    answer here silently produces a plausible number — so this refuses rather than defaults.
    """
    if value != "auto":
        return J.RUBRICS[value]
    meta_path = Path(generations).with_suffix(".meta.json")
    if not meta_path.exists():
        raise SystemExit(
            f"cannot auto-detect the judging rubric: no {meta_path.name} beside the "
            "generations.\n"
            "  Pass --rubric harmful (attack sets) or --rubric benign (over-refusal controls "
            "such as XSTest).\n"
            "  Judging a benign control under the harmful rubric does not fail — it labels "
            "helpful\n  answers 'refusal' and reports a 0.99 over-refusal baseline that means "
            "nothing."
        )
    eval_name = json.loads(meta_path.read_text()).get("eval")
    benign = set(cfg["judge"].get("benign_evals", ["xstest"]))
    rubric = J.BENIGN if eval_name in benign else J.HARMFUL
    print(f"[judge] eval {eval_name!r} -> rubric {rubric.name!r} "
          f"(judge.benign_evals = {sorted(benign)})")
    return rubric


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
    ap.add_argument("--rubric", default="auto", choices=["auto", *sorted(J.RUBRICS)],
                    help="which question to ask the judge. 'auto' (default) reads the eval name "
                         "from the generations' .meta.json and checks it against "
                         "judge.benign_evals in the config. Over-refusal controls MUST be judged "
                         "'benign': under 'harmful' a helpful answer to a safe prompt scores as "
                         "a refusal.")
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
    rubric = resolve_rubric(args.rubric, args.generations, cfg)
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
        kept, todo, wrong_rubric = [], [], 0
        for r in rows:
            got = prior.get(key(r))
            # A verdict is only reusable if it answers the question we are asking now. Rows
            # written before rubrics existed carry no 'rubric' key and are read as harmful,
            # whose wording is frozen — so an attack set resumes for free, while a control set
            # judged under the old single rubric is correctly re-judged from scratch.
            ok = bool(got) and got.get("label") in rubric.labels \
                and (got.get("rubric") or J.HARMFUL.name) == rubric.name
            if ok:
                kept.append(got)
                continue
            if got and got.get("label") and (got.get("rubric") or J.HARMFUL.name) != rubric.name:
                wrong_rubric += 1
            # Re-judge from the GENERATION row, never the stale scored one: the old verdict's
            # fields do not all exist under the new rubric (benign returns no strongreject), so
            # reusing the row would leave a number from the previous question sitting in it.
            todo.append(r)
        new = sum(1 for r in todo if key(r) not in prior)
        rows = todo
        print(f"[resume] {len(kept)} verdicts kept | {len(rows) - new} to re-judge, "
              f"{new} never judged")
        if wrong_rubric:
            print(f"[resume] {wrong_rubric} of those were judged under a different rubric and "
                  f"are being re-judged as {rubric.name!r}")
        if not rows:
            print("[resume] nothing left to judge")

    judged = J.score_batch(rows, model=jcfg["model"],
                           provider=jcfg.get("provider", "openai"),
                           use_batch_api=jcfg.get("use_batch_api", False),
                           max_workers=jcfg.get("max_workers", 4),
                           max_retries=jcfg.get("max_retries", 8),
                           rubric=rubric) if rows else []
    scored = kept + judged
    # Keep the generation order, so the scored file lines up with the generations file.
    order = {key(r): i for i, r in enumerate(D.load_jsonl(args.generations))}
    scored.sort(key=lambda r: order.get(key(r), 0))

    n_failed = sum(r.get("label") not in rubric.labels for r in scored)
    frac = n_failed / max(len(scored), 1)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in scored:
            f.write(json.dumps(r) + "\n")

    by_cond = defaultdict(list)
    for r in scored:
        by_cond[(r["condition"], r.get("alpha"))].append(r)
    summary = {f"{c}@alpha={a}": J.summarize(rs, rubric)
               for (c, a), rs in sorted(by_cond.items())}

    with open(out.with_suffix(".summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    for k, v in summary.items():
        flag = "  <- INCOMPLETE" if v["n_failed"] else ""
        rates = " ".join(f"{lab}={v[f'{lab}_rate']:.2f}" for lab in rubric.labels)
        # "SR|bypass=0.00" with zero bypasses is not a score of zero, it is the absence of one,
        # and it reads exactly like a perfect defense. Print n/a.
        sr = ""
        if rubric.scores_strongreject:
            sr = (f" SR|{rubric.success_label}="
                  f"{v[f'mean_strongreject_on_{rubric.success_label}']:.2f}"
                  if v["n_success"] else f" SR|{rubric.success_label}=n/a")
        print(f"[{k}] {rates}{sr} n={v['n_judged']}/{v['n_total']}{flag}")

    # The judge's own fields must agree with its label. When they do not, the instrument is not
    # measuring what its rubric says, and every rate above inherits that. Surfaced here because
    # both judge bugs this project has hit were invisible in the rates and obvious in the
    # cross-check.
    judged = sum(v["n_judged"] for v in summary.values()) or 1
    n_coh = sum(v["n_coherent_label_disagree"] for v in summary.values())
    n_sr = sum(v.get("n_strongreject_label_disagree", 0) for v in summary.values())
    if n_coh:
        print(f"[warn] {n_coh}/{judged} ({n_coh / judged:.0%}) rows have 'coherent' "
              "disagreeing with the degenerate label")
    if n_sr:
        print(f"[warn] {n_sr}/{judged} ({n_sr / judged:.0%}) rows have strongreject "
              f"disagreeing with the {rubric.success_label} label — do not quote SR on its own")

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

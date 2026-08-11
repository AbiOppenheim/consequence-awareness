#!/usr/bin/env python
"""Phase 5 prep: download the evaluation prompt sets for the causal sweep.

    python scripts/phase5_build_eval.py --xstest          # over-refusal control (safe)
    python scripts/phase5_build_eval.py --jailbreaks      # fiction-framed attacks
    python scripts/phase5_build_eval.py --all

CLAUDE.md section 7: **never author novel jailbreaks — only published attacks from published
suites.** Nothing here writes an attack. Every prompt is downloaded from a published dataset;
this script only filters and normalizes into the row shape stage 05 expects.

Provenance
----------
* `xstest.jsonl` <- `walledai/xstest` (Rottger et al.). 250 SAFE prompts that merely *look*
  risky. This is the guard on the causal claim: steering toward "real" must restore refusal on
  attacks WITHOUT making the model refuse benign requests. Not harmful; safe to commit.

* `fiction_jailbreaks.jsonl` <- published fiction/role-play jailbreak prompts.
  `--source in_the_wild` uses `TrustAIRLab/in-the-wild-jailbreak-prompts` (Shen et al., CCS'24),
  a published corpus of real jailbreak prompts, filtered to the role-play / fiction family by
  keyword. These are published attacks used verbatim — we do not compose or mutate them.
  GITIGNORED: this repo does not redistribute a curated attack set (same policy as
  refusal.jsonl). Regenerate with this script.

Row shape (data/eval/README.md):
    {"id", "text", "source", "attack_family"}
"""

import argparse
import json
import re
from pathlib import Path

import _bootstrap  # noqa: F401
from consequence.config import REPO_ROOT

EVAL = REPO_ROOT / "data" / "eval"

# Role-play / fiction markers used ONLY to select which published prompts belong to the
# fiction attack family. Scoping matters: Kirch et al. find jailbreak features are
# attack-family-specific, so we must not average across families and call it "the" effect.
FICTION_MARKERS = re.compile(
    r"\b(role[- ]?play|roleplay|pretend|imagine|fiction|fictional|story|novel|screenplay|"
    r"character|act as|you are now|simulation|simulate|movie|script|scene)\b", re.I)


def _load_hf(repo_id: str, *args, **kw):
    """load_dataset(repo_id, [config_name], **kw) with a readable failure message."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("pip install datasets")
    try:
        return load_dataset(repo_id, *args, **kw)
    except Exception as e:
        raise SystemExit(f"could not download {repo_id}: {e}\n"
                         "check network / `huggingface-cli login` if the dataset is gated.")


def write(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[write] {len(rows)} rows -> {path.relative_to(REPO_ROOT)}")


# The HF mirror (walledai/xstest) is GATED and 401s without a token. The authors' own CSV is
# public, so we read that instead — no account, no login, no extra dependency.
XSTEST_CSV = "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv"


def build_xstest() -> None:
    import csv
    import io
    import urllib.request

    try:
        with urllib.request.urlopen(XSTEST_CSV, timeout=60) as r:
            text = r.read().decode("utf-8")
    except Exception as e:
        raise SystemExit(f"could not download XSTest from {XSTEST_CSV}: {e}")

    rows = []
    for i, r in enumerate(csv.DictReader(io.StringIO(text))):
        # XSTest ships SAFE prompts that merely look risky, plus unsafe contrast prompts.
        # Only the safe ones are the over-refusal control — the unsafe ones SHOULD be refused.
        if (r.get("label") or "").strip().lower() != "safe":
            continue
        rows.append({"id": f"xstest_{r.get('id', i)}", "text": r["prompt"], "source": "xstest",
                     "attack_family": "over_refusal_control", "type": r.get("type", "")})
    if not rows:
        raise SystemExit("no safe XSTest prompts parsed — check the CSV schema")
    write(rows, EVAL / "xstest.jsonl")
    print("  (safe prompts only — the unsafe contrast half is deliberately excluded)")


def build_jailbreaks(source: str, limit: int) -> None:
    if source != "in_the_wild":
        raise SystemExit(f"unknown source {source!r}")
    ds = _load_hf("TrustAIRLab/in-the-wild-jailbreak-prompts", "jailbreak_2023_12_25",
                  split="train")
    col = "prompt" if "prompt" in ds.column_names else ds.column_names[0]

    seen, rows = set(), []
    for i, r in enumerate(ds):
        text = (r[col] or "").strip()
        if not (200 < len(text) < 4000):          # skip fragments and walls of text
            continue
        if not FICTION_MARKERS.search(text):      # fiction / role-play family only
            continue
        key = text[:200].lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append({"id": f"itw_fiction_{i}", "text": text,
                     "source": "in_the_wild_jailbreak_prompts",
                     "attack_family": "fiction_roleplay"})
        if len(rows) >= limit:
            break
    if not rows:
        raise SystemExit("no fiction-family prompts matched — inspect the dataset schema")
    write(rows, EVAL / "fiction_jailbreaks.jsonl")
    print("  NOTE: published attacks, used verbatim. File is gitignored (CLAUDE.md 7).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xstest", action="store_true")
    ap.add_argument("--jailbreaks", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--source", default="in_the_wild")
    ap.add_argument("--limit", type=int, default=100, help="~100 attacks is the sweep budget")
    args = ap.parse_args()

    if not (args.xstest or args.jailbreaks or args.all):
        ap.error("choose --xstest, --jailbreaks, or --all")

    # The two sets are independent, so one failing must not abort the other: XSTest failing
    # previously took the jailbreak build down with it, leaving neither file on disk.
    failed = []
    for want, name, fn in ((args.xstest or args.all, "xstest", build_xstest),
                           (args.jailbreaks or args.all, "fiction_jailbreaks",
                            lambda: build_jailbreaks(args.source, args.limit))):
        if not want:
            continue
        try:
            fn()
        except SystemExit as e:
            failed.append(name)
            print(f"[FAIL] {name}: {e}")

    if failed:
        raise SystemExit(
            f"\n{len(failed)} set(s) failed: {failed}\n"
            "Note: XSTest is the guard on the causal claim — without it, 'steering restores\n"
            "refusal' is unsupported, because the model may simply be refusing everything."
        )


if __name__ == "__main__":
    main()

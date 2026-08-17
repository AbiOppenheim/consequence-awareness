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
import zlib
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from consequence.config import REPO_ROOT

EVAL = REPO_ROOT / "data" / "eval"

# Role-play / fiction markers used ONLY to select which published prompts belong to the
# fiction attack family. Scoping matters: Kirch et al. find jailbreak features are
# attack-family-specific, so we must not average across families and call it "the" effect.
FICTION_MARKERS = re.compile(
    r"\b(role[- ]?play|roleplay|pretend|imagine|fiction|fictional|story|novel|screenplay|"
    r"character|act as|you are now|simulation|simulate|movie|script|scene)\b", re.I)


# ---- near-duplicate rejection -------------------------------------------------------------
# In-the-wild jailbreaks are heavily reposted: the same DAN-style prompt appears dozens of times
# with a renamed persona or an extra sentence. Comparing the first 200 characters (the old rule)
# catches verbatim reposts and nothing else. That is survivable at n=100 and not at n=400, where
# clones inflate the effective sample: a correlation measured over 400 rows that are really 150
# distinct prompts has error bars for 150 and the confidence of 400.
#
# MinHash over word 5-shingles estimates Jaccard overlap in 64 integer comparisons, so screening
# every candidate against everything kept stays linear enough to run over the whole corpus.
N_HASH = 64
JACCARD_MAX = 0.5          # above this, treat as the same attack wearing a different name
_RNG = np.random.default_rng(0)
_A = _RNG.integers(1, 2**63, N_HASH, dtype=np.uint64)
_B = _RNG.integers(0, 2**63, N_HASH, dtype=np.uint64)


def _signature(text: str, k: int = 5) -> np.ndarray:
    """MinHash signature of the prompt's word 5-shingles.

    crc32, not hash(): Python's string hash is salted per process, so hash() would make the
    built dataset depend on the interpreter session rather than on the corpus.
    """
    words = re.findall(r"\w+", text.lower())
    grams = ([" ".join(words[i:i + k]) for i in range(len(words) - k + 1)]
             or [" ".join(words)])
    xs = np.array([zlib.crc32(g.encode()) for g in grams], dtype=np.uint64)
    # 64 independent hashes of every shingle; the per-hash minimum is the signature.
    return (_A[:, None] * xs[None, :] + _B[:, None]).min(axis=1)


def _is_near_duplicate(sig: np.ndarray, kept: list) -> bool:
    if not kept:
        return False
    sims = (np.vstack(kept) == sig).mean(axis=1)     # matching minima ~= Jaccard
    return bool(sims.max() > JACCARD_MAX)


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


def build_jailbreaks(source: str, limit: int, out: str = "fiction_jailbreaks") -> None:
    if source != "in_the_wild":
        raise SystemExit(f"unknown source {source!r}")
    ds = _load_hf("TrustAIRLab/in-the-wild-jailbreak-prompts", "jailbreak_2023_12_25",
                  split="train")
    col = "prompt" if "prompt" in ds.column_names else ds.column_names[0]

    seen, sigs, rows = set(), [], []
    n_len = n_family = n_exact = n_near = 0
    for i, r in enumerate(ds):
        text = (r[col] or "").strip()
        if not (200 < len(text) < 4000):          # skip fragments and walls of text
            n_len += 1
            continue
        if not FICTION_MARKERS.search(text):      # fiction / role-play family only
            n_family += 1
            continue
        key = text[:200].lower()
        if key in seen:
            n_exact += 1
            continue
        sig = _signature(text)
        if _is_near_duplicate(sig, sigs):
            n_near += 1
            continue
        seen.add(key)
        sigs.append(sig)
        rows.append({"id": f"itw_fiction_{i}", "text": text,
                     "source": "in_the_wild_jailbreak_prompts",
                     "attack_family": "fiction_roleplay"})
        if len(rows) >= limit:
            break
    if not rows:
        raise SystemExit("no fiction-family prompts matched — inspect the dataset schema")
    # Printed because it is the number that decides whether a bigger n is even available: if
    # near-duplicate rejection is what stopped the build, raising --limit will not help.
    print(f"  filtered {len(ds)} rows -> {len(rows)} kept  "
          f"(dropped: {n_len} length, {n_family} not fiction-family, "
          f"{n_exact} exact dup, {n_near} near-dup at Jaccard>{JACCARD_MAX})")
    if len(rows) < limit:
        print(f"  NOTE: the corpus is exhausted at {len(rows)} distinct fiction-family attacks; "
              f"--limit {limit} cannot be met from this snapshot.")
    write(rows, EVAL / f"{out}.jsonl")
    print("  NOTE: published attacks, used verbatim. File is gitignored (CLAUDE.md 7).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xstest", action="store_true")
    ap.add_argument("--jailbreaks", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--source", default="in_the_wild")
    ap.add_argument("--limit", type=int, default=100, help="~100 attacks is the sweep budget")
    ap.add_argument("--out", default="fiction_jailbreaks",
                    help="basename under data/eval/. Build a LARGER set under a NEW name rather "
                         "than rebuilding fiction_jailbreaks: stages 01 and 05 hash the eval "
                         "file, so overwriting it marks every cached activation and every "
                         "generated sweep [STALE] and they have to be redone.")
    ap.add_argument("--force", action="store_true", help="rebuild even if the file exists")
    args = ap.parse_args()

    if not (args.xstest or args.jailbreaks or args.all):
        ap.error("choose --xstest, --jailbreaks, or --all")

    # The two sets are independent, so one failing must not abort the other: XSTest failing
    # previously took the jailbreak build down with it, leaving neither file on disk.
    failed = []
    for want, name, fn in ((args.xstest or args.all, "xstest", build_xstest),
                           (args.jailbreaks or args.all, args.out,
                            lambda: build_jailbreaks(args.source, args.limit, args.out))):
        if not want:
            continue
        # Resumable (Rule 4). These files are gitignored, so every recycled runtime has to
        # rebuild them — which makes it worth being safe to call at the top of any cell that
        # needs them, rather than a step you have to remember exactly once per VM.
        dest = EVAL / f"{name}.jsonl"
        if dest.exists() and not args.force:
            print(f"[skip] {dest.name} exists ({sum(1 for _ in open(dest))} rows)")
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

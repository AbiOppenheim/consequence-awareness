#!/usr/bin/env python
"""Phase 4 prep (CPU): build the persona contrast set for v_MP.

    python scripts/phase4_build_persona.py                  # both framings

Why this file exists
--------------------
Zhong's public repo cannot produce v_MP as shipped: `extract_vectors.py` imports an `ithou`
package that is not in the release (the real implementation lives in a private repo), and
there is no data/ directory. What they DID release is the trait definitions, and their code
pins the exact configuration used for Qwen2.5:

    src/compliant_residual.py -> "mp_file": "compliant_v2_model_persona_prompt_end.pt", layer 20

So: trait `compliant_v2`, the *model persona* vector, at *prompt_end* (the last prompt token —
our own convention), layer 20. We rebuild that direction with OUR difference-in-means, which
the Phase 2 r_hat gate validated at cos=0.9999. This is a reimplementation of their method
using their released trait definitions — it is NOT their vector, and must be labelled that way.

Design
------
* label 1 = positive (compliant / unrestricted persona), 0 = negative (restrictive / refusing).
  So v_MP = mean(compliant) - mean(restrictive), pointing TOWARD the compliant persona. Note
  this is roughly opposite in spirit to r_hat (harmful - harmless), so a negative cosine
  between them is not surprising; report magnitudes.
* 8 positive x 8 negative instructions, paired by index -> 8 instruction pairs. Every question
  appears under both polarities, so question content cancels in the difference of means
  (CLAUDE.md 2, matched content).
* Questions: their own 15 trait questions (faithful) PLUS our 40 benign tasks (a neutral
  control). `source` lets us slice and check the direction is not an artifact of their
  compliance-loaded questions.
* Two framings, because the choice is a real confound for our comparison:
    - `system`   : instruction as the system prompt  -> faithful to Zhong's model_persona
    - `userturn` : instruction prefixed into the user turn -> identical in structure to how
                   v_C and r_hat were extracted, so cos(v_C, v_MP) cannot be an artifact of
                   differing prompt structure
  Extract both and compare; agreement means the geometry claim is robust to this choice.
"""

import argparse
import json
import zipfile
from pathlib import Path

import yaml

import _bootstrap  # noqa: F401
from consequence.config import REPO_ROOT

TRAITS_ZIP = REPO_ROOT / "external" / "refusal-downstream-persona" / "configs" / "traits.zip"
BENIGN = REPO_ROOT / "data" / "contrast" / "benign_tasks.jsonl"
OUTPUTS = {
    "system": REPO_ROOT / "data" / "contrast" / "persona.jsonl",
    "userturn": REPO_ROOT / "data" / "contrast" / "persona_ut.jsonl",
}


def load_trait(name: str) -> dict:
    if not TRAITS_ZIP.exists():
        raise SystemExit(
            f"missing {TRAITS_ZIP}\n"
            "clone violazhong/refusal-downstream-persona into external/ first."
        )
    with zipfile.ZipFile(TRAITS_ZIP) as z:
        with z.open(f"traits/{name}.yaml") as f:
            return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", default="compliant_v2", help="Zhong's Qwen MP trait")
    ap.add_argument("--include-benign", action="store_true", default=True,
                    help="add our 40 benign tasks as neutral control questions")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if the outputs exist (needs Zhong's traits.zip)")
    args = ap.parse_args()

    # Resumable (Rule 4). Both outputs are COMMITTED, so a fresh clone already has them and
    # rebuilding needs Zhong's traits.zip — which their partial release only provides once
    # their repo is cloned into external/. Failing here would block Phase 4 on a runtime that
    # has everything it actually needs.
    if not args.force and all(p.exists() for p in OUTPUTS.values()):
        for p in OUTPUTS.values():
            print(f"[skip] {p.name} exists ({sum(1 for _ in open(p))} rows) — committed with "
                  "the repo; pass --force to rebuild from traits.zip")
        return

    tr = load_trait(args.trait)
    pos, neg = tr["instructions"]["positive"], tr["instructions"]["negative"]
    n_pairs = min(len(pos), len(neg))
    print(f"[trait] {args.trait}: {len(pos)} positive / {len(neg)} negative "
          f"-> {n_pairs} instruction pairs")

    questions = [(q, "zhong_trait_questions") for q in tr.get("questions", [])]
    if args.include_benign:
        questions += [(json.loads(l)["text"], "benign_control")
                      for l in open(BENIGN) if l.strip()]
    print(f"[questions] {len(questions)} "
          f"({sum(s == 'zhong_trait_questions' for _, s in questions)} theirs + "
          f"{sum(s == 'benign_control' for _, s in questions)} benign)")

    for framing in ("system", "userturn"):
        rows, pid = [], 0
        for qi, (q, src) in enumerate(questions):
            for i in range(n_pairs):
                for label, instr in ((1, pos[i]), (0, neg[i])):
                    row = {
                        "id": f"{args.trait}_{framing}_q{qi}_i{i}_{'pos' if label else 'neg'}",
                        "label": label,
                        "split": "train",
                        "source": src,
                        "trait": args.trait,
                        "prompt_framing": framing,   # NOT `framing`: that key means real/hypo
                        "instruction_idx": i,
                        "pair_id": f"{framing}_{pid}",
                    }
                    if framing == "system":
                        row["system"] = instr
                        row["text"] = q
                    else:
                        row["text"] = f"{instr} {q}"
                    rows.append(row)
                pid += 1

        out = OUTPUTS[framing]
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        # invariant: the two classes must see identical question content
        from collections import Counter
        qpos = Counter(r["text"] for r in rows if r["label"] == 1)
        qneg = Counter(r["text"] for r in rows if r["label"] == 0)
        matched = qpos == qneg if framing == "system" else len(rows) % 2 == 0
        print(f"[{framing:8}] {len(rows)} rows "
              f"({sum(r['label'] == 1 for r in rows)} pos / {sum(r['label'] == 0 for r in rows)} neg)"
              f" | question-matched: {matched} -> {out.name}")


if __name__ == "__main__":
    main()

"""Load the contrast datasets (jsonl) into arrays the rest of the pipeline consumes.

Schema for data/contrast/consequence.jsonl — one row per (framing_template x task):
    {
      "id":          unique string,
      "framing":     "real" | "hypo",          # -> label 1 / 0
      "split":       "train" | "heldout",       # HELD-OUT is by TEMPLATE, never by example
      "template_id": string, grouping key for the group-wise probe split,
      "task_id":     string, matches the SAME task across the real/hypo pair,
      "task_source": provenance of the [TASK] text,
      "text":        the fully rendered prompt fed to the model
    }
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FRAMING_TO_LABEL = {"real": 1, "hypo": 0}


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_consequence(path: str | Path):
    """Return (prompts, labels, template_ids, splits) as parallel lists/arrays.

    labels: 1=real, 0=hypothetical. template_ids/splits drive the group-wise probe split.
    """
    rows = load_jsonl(path)
    prompts = [r["text"] for r in rows]
    labels = np.array([FRAMING_TO_LABEL[r["framing"]] for r in rows])
    template_ids = np.array([r["template_id"] for r in rows])
    splits = np.array([r["split"] for r in rows])
    return prompts, labels, template_ids, splits


def load_labeled(path: str | Path):
    """Generic binary contrast (e.g. refusal: harmful vs harmless).

    Rows carry `text` and an integer `label` (1 = positive class, 0 = negative). Returns the
    SAME 4-tuple shape as load_consequence — (prompts, labels, groups, splits) — so the caching
    stage is schema-agnostic. `source`/`group` and `split` default when a row omits them.

    For refusal: label 1 = harmful, 0 = harmless, so diff_in_means(pos, neg) = mean(harmful) -
    mean(harmless) = r_hat (Arditi's convention).
    """
    rows = load_jsonl(path)
    # Rows carrying a `system` field become {"system", "user"} dicts so the manipulation can
    # live in the system prompt (Zhong's persona convention); plain rows stay strings.
    prompts = [{"system": r["system"], "user": r["text"]} if r.get("system") else r["text"]
               for r in rows]
    labels = np.array([int(r["label"]) for r in rows])
    groups = np.array([str(r.get("source", r.get("group", "na"))) for r in rows])
    splits = np.array([str(r.get("split", "train")) for r in rows])
    return prompts, labels, groups, splits


def load_contrast(path: str | Path):
    """Load any contrast dataset, dispatching on its schema.

    Consequence rows have a `framing` field (real/hypo); generic labeled rows have a `label`
    field (1/0). Both return (prompts, labels, groups, splits), so callers stay uniform.
    """
    rows = load_jsonl(path)
    if not rows:
        raise ValueError(f"empty dataset: {path}")
    first = rows[0]
    # `label` is checked FIRST and is unambiguous. `framing` alone is not: it means real/hypo
    # in the consequence set, and other datasets may legitimately carry a differently-named
    # framing concept. Dispatching on `framing` first silently misroutes them.
    if "label" in first:
        return load_labeled(path)
    if "framing" in first:
        return load_consequence(path)
    raise ValueError(f"unrecognized contrast schema in {path}: keys={sorted(first)}")


def heldout_template_set(path: str | Path) -> set[str]:
    """The set of template_ids reserved for the final generalization test."""
    rows = load_jsonl(path)
    return {r["template_id"] for r in rows if r["split"] == "heldout"}

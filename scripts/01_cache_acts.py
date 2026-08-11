#!/usr/bin/env python
"""Stage 01 (GPU): cache residual-stream activations for a contrast dataset.

    python scripts/01_cache_acts.py --dataset consequence

Reads the prompts from the dataset named in the config, runs one batched forward pass per
batch capturing ALL layers, and writes artifacts/activations/<dataset>_<model>.npz.
Resumable (Rule 4): skips if the .npz already exists unless --force.
"""

import argparse
import hashlib
from pathlib import Path

import _bootstrap  # noqa: F401
from consequence import acts as A
from consequence import data as D
from consequence.config import load_config, resolve


def source_sha(path) -> str:
    """Short content hash of the dataset file, recorded in the cache (Rule 4).

    Without this the cache is keyed on dataset NAME only, so editing consequence.jsonl and
    re-running would silently reuse a cache built from the old rows — and every downstream
    number would be wrong with no visible symptom.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen.yaml")
    ap.add_argument("--dataset", default="consequence", help="key under data: in the config")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_slug = cfg["model"]["id"].split("/")[-1]
    out = resolve(cfg["paths"]["activations"]) / f"{args.dataset}_{model_slug}.npz"
    src = resolve(cfg["data"][args.dataset])
    sha = source_sha(src)

    if out.exists() and not args.force:
        _, _, old_meta = A.load_acts(out)
        old_sha = str(old_meta.get("source_sha256", ""))
        if old_sha and old_sha != sha:
            raise SystemExit(
                f"[STALE] {out.name} was built from a different {src.name}\n"
                f"        cached {old_sha} != current {sha}\n"
                f"        the dataset changed since this cache was written — rerun with --force."
            )
        if not old_sha:
            print("[warn] cache predates the source-hash check — cannot verify it matches "
                  f"{src.name}; rerun with --force if you have edited the dataset.")
        print(f"[skip] {out} exists (use --force to recompute)")
        return

    prompts, labels, template_ids, splits = D.load_contrast(src)
    print(f"[load] {len(prompts)} prompts from {args.dataset}")

    model, tok = A.load_model(cfg["model"]["id"], cfg["model"]["dtype"])
    acts = A.cache_activations(
        model, tok, prompts,
        token_position=cfg["acts"]["token_position"],
        batch_size=cfg["acts"]["batch_size"],
    )
    print(f"[cache] acts {acts.shape}  (n, n_layers, d_model)")

    A.save_acts(out, acts, labels, meta={
        "model_id": cfg["model"]["id"],
        "dataset": args.dataset,
        "token_position": cfg["acts"]["token_position"],
        "template_ids": ",".join(template_ids),
        "splits": ",".join(splits),
        "seed": cfg["seed"],
        "source_contrast": str(src),
        "source_sha256": sha,
    })
    print(f"[write] {out}")


if __name__ == "__main__":
    main()

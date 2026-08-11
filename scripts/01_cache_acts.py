#!/usr/bin/env python
"""Stage 01 (GPU): cache residual-stream activations for a contrast dataset.

    python scripts/01_cache_acts.py --dataset consequence

Reads the prompts from the dataset named in the config, runs one batched forward pass per
batch capturing ALL layers, and writes artifacts/activations/<dataset>_<model>.npz.
Resumable (Rule 4): skips if the .npz already exists unless --force.
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from consequence import acts as A
from consequence import data as D
from consequence.config import load_config, resolve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen.yaml")
    ap.add_argument("--dataset", default="consequence", help="key under data: in the config")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_slug = cfg["model"]["id"].split("/")[-1]
    out = resolve(cfg["paths"]["activations"]) / f"{args.dataset}_{model_slug}.npz"
    if out.exists() and not args.force:
        print(f"[skip] {out} exists (use --force to recompute)")
        return

    prompts, labels, template_ids, splits = D.load_contrast(resolve(cfg["data"][args.dataset]))
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
    })
    print(f"[write] {out}")


if __name__ == "__main__":
    main()

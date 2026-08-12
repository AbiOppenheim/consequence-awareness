#!/usr/bin/env python
"""Stage 02 (CPU): extract v_C by difference-in-means, at each swept layer.

    python scripts/02_extract_directions.py --dataset consequence

For each layer in extract.layer_sweep, computes v = mean(real) - mean(hypo) over the cached
activations of the TRAIN templates only (--split, default "train") and saves
artifacts/directions/v_c_L{layer}.pt + a sidecar .json (Rule 2).
Also writes a random-direction null (random_L{layer}.pt) with the same shape and seed.
"""

import argparse

import numpy as np

import _bootstrap  # noqa: F401
from consequence import acts as A
from consequence import directions as Dir
from consequence import io
from consequence.config import load_config, resolve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen.yaml")
    ap.add_argument("--dataset", default="consequence")
    ap.add_argument("--split", default="train", choices=["train", "all"],
                    help="rows to extract from. 'train' is the default and the only setting "
                         "that keeps the held-out templates untouched (CLAUDE.md §2).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_slug = cfg["model"]["id"].split("/")[-1]
    acts_path = resolve(cfg["paths"]["activations"]) / f"{args.dataset}_{model_slug}.npz"
    acts, labels, meta = A.load_acts(acts_path)  # acts [n, n_layers, d]
    out_dir = resolve(cfg["paths"]["directions"])

    # Held-out templates must not touch the direction: a v_C built from them and then scored
    # on them is not a generalization test. Datasets with no split column (refusal, persona)
    # have no held-out reserve to protect, so they extract from everything.
    if args.split == "train" and "splits" in meta:
        keep = np.array(str(meta["splits"]).split(",")) == "train"
        split_used = "train"
    else:
        if args.split == "train":
            print(f"[extract] no split metadata in {acts_path.name}: using all rows")
        keep = np.ones(len(labels), dtype=bool)
        split_used = "all"
    acts, labels = acts[keep], labels[keep]
    print(f"[extract] {args.dataset}: {keep.sum()}/{len(keep)} rows (split={split_used})")

    common = {
        "model_id": cfg["model"]["id"],
        "token_position": cfg["acts"]["token_position"],
        "source_contrast": str(cfg["data"][args.dataset]),
        "n_pairs": int((labels == 1).sum()),
        "seed": cfg["seed"],
        "split": split_used,
    }

    for layer in cfg["extract"]["layer_sweep"]:
        acts_L = acts[:, layer - 1, :]  # hidden_states index L == block L; stored 0-based here
        v_c = Dir.diff_in_means(acts_L[labels == 1], acts_L[labels == 0])
        io.save_direction(v_c, out_dir / f"v_c_L{layer}", {**common, "layer": layer,
                                                            "method": "diff_in_means",
                                                            "contrast": "real_minus_hypo"})
        rand = Dir.random_direction(v_c.shape[-1], seed=cfg["seed"] + layer)
        io.save_direction(rand, out_dir / f"random_L{layer}", {**common, "layer": layer,
                                                               "method": "random_null"})
        print(f"[extract] L{layer}: v_c + random null saved")

    print(f"[done] {len(cfg['extract']['layer_sweep'])} layers -> {out_dir}")


if __name__ == "__main__":
    main()

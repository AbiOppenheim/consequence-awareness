#!/usr/bin/env python
"""Stage 02 (CPU): extract a direction by difference-in-means, at each swept layer.

    python scripts/02_extract_directions.py --dataset consequence              # v_C + random null
    python scripts/02_extract_directions.py --dataset persona    --kind v_mp   # persona, system
    python scripts/02_extract_directions.py --dataset persona_ut --kind v_mp   # persona, user turn

For each layer in extract.layer_sweep, computes v = mean(pos) - mean(neg) over the cached
activations and saves artifacts/directions/<name>_L{layer}.pt + a sidecar .json (Rule 2).

The `--kind` switch exists so v_MP is minted by the same validated code path as v_C rather than
by a notebook cell — three directions extracted three different ways cannot be compared by
cosine and blamed on geometry.

Cheap enough (seconds) that it is not wrapped in the results store: the artifacts ARE the cache.
"""

import argparse

import numpy as np

import _bootstrap  # noqa: F401
from consequence import acts as A
from consequence import directions as Dir
from consequence import io
from consequence.config import load_config, resolve

# kind -> (artifact stem, sidecar method, sidecar contrast). Positive class is label 1 in the
# cached dataset, so the sign convention lives in the dataset builder, not here.
KINDS = {
    "v_c": ("v_c", "diff_in_means", "real_minus_hypo"),
    "v_mp": ("v_mp", "diff_in_means (our reimplementation of Zhong model_persona)",
             "compliant_minus_restrictive"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen.yaml")
    ap.add_argument("--dataset", default="consequence")
    ap.add_argument("--kind", default="v_c", choices=sorted(KINDS),
                    help="v_c: the consequence direction (+ a random null). "
                         "v_mp: our reimplementation of Zhong's persona vector.")
    ap.add_argument("--split", default="train", choices=["train", "all"],
                    help="rows to extract from. 'train' is the default and the only setting "
                         "that keeps the held-out templates untouched (CLAUDE.md section 2).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_slug = cfg["model"]["id"].split("/")[-1]
    acts_path = resolve(cfg["paths"]["activations"]) / f"{args.dataset}_{model_slug}.npz"
    acts, labels, meta = A.load_acts(acts_path)  # acts [n, n_layers, d]
    out_dir = resolve(cfg["paths"]["directions"])

    stem, method, contrast = KINDS[args.kind]
    # v_C is one direction per layer; v_MP is one per (framing, layer), and the framing is the
    # dataset — keeping it in the filename is what lets step 04 compare the two framings.
    name = stem if args.kind == "v_c" else f"{stem}_{args.dataset}"

    # Held-out templates must not touch the direction: a v_C built from them and then scored
    # on them is not a generalization test. Datasets with no split column (refusal, persona)
    # have no held-out reserve to protect, so they extract from everything.
    if args.split == "train" and "splits" in meta:
        keep = np.array(str(meta["splits"]).split(",")) == "train"
    else:
        keep = np.ones(len(labels), dtype=bool)
    if not keep.any():
        raise SystemExit(f"no train rows in {acts_path.name} — nothing to extract from")
    # Say what actually happened: a dataset with no held-out reserve (refusal, persona) is
    # "all" even when --split train was asked for, and the sidecar must not claim otherwise.
    split_used = "train" if keep.sum() < len(keep) else "all"
    acts, labels = acts[keep], labels[keep]
    print(f"[extract] {args.dataset} ({args.kind}): {keep.sum()}/{len(keep)} rows "
          f"(split={split_used})")

    common = {
        "model_id": cfg["model"]["id"],
        "token_position": cfg["acts"]["token_position"],
        "source_contrast": str(cfg["data"][args.dataset]),
        "n_pairs": int((labels == 1).sum()),
        "seed": cfg["seed"],
        "split": split_used,
        "method": method,
        "contrast": contrast,
    }
    if args.kind == "v_mp":
        common["trait"] = "compliant_v2"          # Zhong's released trait definition

    for layer in cfg["extract"]["layer_sweep"]:
        acts_L = acts[:, layer - 1, :]  # config layer L == hidden_states[L] == cache index L-1
        v = Dir.diff_in_means(acts_L[labels == 1], acts_L[labels == 0])
        io.save_direction(v, out_dir / f"{name}_L{layer}", {**common, "layer": layer})
        if args.kind == "v_c":
            rand = Dir.random_direction(v.shape[-1], seed=cfg["seed"] + layer)
            io.save_direction(rand, out_dir / f"random_L{layer}",
                              {**common, "layer": layer, "method": "random_null",
                               "contrast": "none"})
        print(f"[extract] L{layer}: {name} saved" +
              (" (+ random null)" if args.kind == "v_c" else ""))

    print(f"[done] {len(cfg['extract']['layer_sweep'])} layers -> {out_dir}")


if __name__ == "__main__":
    main()

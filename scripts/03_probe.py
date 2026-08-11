#!/usr/bin/env python
"""Stage 03 (CPU): per-layer probes, reported on HELD-OUT TEMPLATES.

    python scripts/03_probe.py --dataset consequence

Writes artifacts/figures/probe_accuracy_<dataset>.json — the held-out-template accuracy per
layer is the number that matters (train-template accuracy is only a sanity check).
"""

import argparse
import json

import numpy as np

import _bootstrap  # noqa: F401
from consequence import acts as A
from consequence import probe as P
from consequence.config import load_config, resolve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen.yaml")
    ap.add_argument("--dataset", default="consequence")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_slug = cfg["model"]["id"].split("/")[-1]
    acts_path = resolve(cfg["paths"]["activations"]) / f"{args.dataset}_{model_slug}.npz"
    acts, labels, meta = A.load_acts(acts_path)

    template_ids = np.array(str(meta["template_ids"]).split(","))
    splits = np.array(str(meta["splits"]).split(","))
    heldout = set(template_ids[splits == "heldout"])
    if not heldout:
        raise SystemExit("no held-out templates in this dataset — cannot report generalization")

    acts_by_layer = {L: acts[:, L - 1, :] for L in cfg["extract"]["layer_sweep"]}
    results = P.layerwise_accuracy(acts_by_layer, labels, template_ids, heldout, seed=cfg["seed"])

    best = max(results, key=lambda L: results[L]["heldout_templates_acc"])
    out = resolve(cfg["paths"]["figures"]) / f"probe_accuracy_{args.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"model_id": cfg["model"]["id"], "dataset": args.dataset,
                   "heldout_templates": sorted(heldout), "by_layer": results,
                   "best_layer": best}, f, indent=2)
    print(f"[probe] best held-out layer L{best}: "
          f"{results[best]['heldout_templates_acc']:.3f}  -> {out}")


if __name__ == "__main__":
    main()

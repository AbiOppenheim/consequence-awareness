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

    # Select the layer on TRAIN templates only (group-wise CV). Selecting it by held-out
    # accuracy would be selecting on the test set and would inflate the reported number.
    train_mask = ~np.isin(template_ids, list(heldout))
    cv_auc = {
        L: P.group_cv_auc(a[train_mask], labels[train_mask], template_ids[train_mask],
                          seed=cfg["seed"])
        for L, a in acts_by_layer.items()
    }
    best = max(cv_auc, key=cv_auc.get)
    for L in sorted(cv_auc):
        print(f"  L{L:>2}: train-CV AUC = {cv_auc[L]:.3f}"
              f"   (held-out acc {results[L]['heldout_templates_acc']:.3f})")

    spread = max(cv_auc.values()) - min(cv_auc.values())
    if spread < 0.01:
        print(f"[note] train-CV AUC spread is {spread:.3f} across layers — the layer choice is "
              "effectively arbitrary; report the curve as flat, not L{} as 'best'.".format(best))

    out = resolve(cfg["paths"]["figures"]) / f"probe_accuracy_{args.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"model_id": cfg["model"]["id"], "dataset": args.dataset,
                   "heldout_templates": sorted(heldout), "by_layer": results,
                   "train_cv_auc": cv_auc, "selected_layer": best,
                   "selection_rule": "max train-template group-CV AUC (never held-out)"},
                  f, indent=2)
    print(f"[probe] selected L{best} on train CV (AUC {cv_auc[best]:.3f}); "
          f"its held-out acc = {results[best]['heldout_templates_acc']:.3f}  -> {out}")


if __name__ == "__main__":
    main()

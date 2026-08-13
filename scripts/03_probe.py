#!/usr/bin/env python
"""Stage 03 (CPU): choose the layer, using TRAIN templates only.

    python scripts/03_probe.py --dataset consequence

Group-wise 5-fold CV where the groups are framing TEMPLATES, so every fold validates on
templates the probe did not train on — a miniature of the real held-out test. The layer is
chosen here, before held-out is touched. Choosing it by held-out accuracy would be selecting on
the test set and would inflate the number we then report (CLAUDE.md section 2).

This step deliberately computes NOTHING about the held-out framings. That is step 09, and
keeping them apart is what makes "the layer was chosen without looking" checkable rather than
a claim in a comment.

Writes artifacts/results/layer_select.json — including `best_layer`, which steps 09, 10, 11 and
the sweep read from disk. Nothing downstream needs a notebook variable to be in scope.

Cost: 5 folds x 9 layers of logistic regression on 1428 x 3584 features — minutes, and the
reason this is a stored result rather than a cell that gets re-run by accident.
"""

import argparse

import numpy as np

import _bootstrap  # noqa: F401
from consequence import acts as A
from consequence import probe as P
from consequence import results
from consequence.config import load_config, resolve

NAME = "layer_select"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen.yaml")
    ap.add_argument("--dataset", default="consequence")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_slug = cfg["model"]["id"].split("/")[-1]
    acts_path = resolve(cfg["paths"]["activations"]) / f"{args.dataset}_{model_slug}.npz"
    layers = cfg["extract"]["layer_sweep"]

    def run() -> dict:
        acts, labels, meta = A.load_acts(acts_path)
        template_ids = np.array(str(meta["template_ids"]).split(","))
        splits = np.array(str(meta["splits"]).split(","))
        train = splits == "train"
        if not train.any() or train.all():
            raise SystemExit("this dataset has no held-out templates — cannot select honestly")

        by_layer = []
        for L in layers:
            auc = P.group_cv_auc(acts[train, L - 1, :], labels[train], template_ids[train],
                                 n_splits=args.n_splits, seed=cfg["seed"])
            by_layer.append({"layer": L, "train_cv_auc": auc})
            print(f"  L{L:>2}: train-CV AUC = {auc:.3f}")

        best = max(by_layer, key=lambda r: r["train_cv_auc"])
        aucs = [r["train_cv_auc"] for r in by_layer]
        return {
            "dataset": args.dataset,
            "best_layer": int(best["layer"]),
            "best_train_cv_auc": best["train_cv_auc"],
            "by_layer": by_layer,
            "spread": float(max(aucs) - min(aucs)),
            "n_train_rows": int(train.sum()),
            "n_heldout_rows": int((~train).sum()),
            "n_train_templates": int(len(set(template_ids[train]))),
            "n_heldout_templates": int(len(set(template_ids[~train]))),
            "n_splits": args.n_splits,
            "selection_rule": "max group-CV AUC over TRAIN templates; held-out never consulted",
        }

    res = results.compute(
        NAME, run,
        inputs=[acts_path],
        params={"dataset": args.dataset, "layers": layers, "n_splits": args.n_splits,
                "seed": cfg["seed"]},
        entry=__file__, force=args.force,
        results_dir=resolve(cfg["paths"]["results"]),
    )
    report(res)


def report(res: dict) -> None:
    print(f"\nLAYER SELECTION — train templates only "
          f"({res['n_train_rows']} rows, {res['n_train_templates']} templates; "
          f"{res['n_heldout_rows']} held-out rows over {res['n_heldout_templates']} templates "
          "untouched)")
    for r in res["by_layer"]:
        mark = "  <- selected" if r["layer"] == res["best_layer"] else ""
        print(f"  L{r['layer']:>2}: train-CV AUC = {r['train_cv_auc']:.3f}{mark}")
    if res["spread"] < 0.01:
        print(f"[note] the CV spread across layers is {res['spread']:.3f} — the layer choice is "
              "effectively arbitrary.\n       Report the curve as flat; do NOT write "
              f"'L{res['best_layer']} is best'.")


if __name__ == "__main__":
    main()

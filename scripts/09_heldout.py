#!/usr/bin/env python
"""Stage 09 (CPU): THE HELD-OUT REVEAL — run once.

    python scripts/09_heldout.py

Everything before this used training templates only. This scores the frozen held-out framings,
at the layer step 03 already chose, and it is the number the write-up reports.

Three readouts and the two baselines that make them mean anything:
  * trained probe     is the information linearly readable on unseen framings?
  * v_C projection    does the raw difference-in-means direction transfer, with no fitting?
  * random direction  the null every claim needs.
  * BoW text baseline the honest surface-vocabulary number, from scripts/audit_contrast.py.
    The probe must beat it, or it learned words rather than a concept.

One-shot discipline (CLAUDE.md section 2). If this result already exists and anything it
depends on has changed, the previous version is archived as `heldout.prev-<sha>.json` rather
than overwritten, and both numbers must appear in the write-up with the second labelled
post-hoc. Do not iterate against this number.

Depends on artifacts/results/layer_select.json, so re-selecting the layer invalidates this
result instead of leaving a stale AUC attached to a layer nobody chose.
"""

import argparse

import numpy as np

import _bootstrap  # noqa: F401
from consequence import acts as A
from consequence import io
from consequence import probe as P
from consequence import results
from consequence.config import load_config, resolve

NAME = "heldout"

# From scripts/audit_contrast.py on this dataset. It rules out UNIGRAM leakage and nothing
# more — step 10 adds the fairer TF-IDF baselines.
BOW_HELDOUT_AUC = 0.753


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen.yaml")
    ap.add_argument("--dataset", default="consequence")
    ap.add_argument("--force", action="store_true",
                    help="re-reveal. The previous result is archived, not overwritten.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rdir = resolve(cfg["paths"]["results"])
    ddir = resolve(cfg["paths"]["directions"])
    model_slug = cfg["model"]["id"].split("/")[-1]
    acts_path = resolve(cfg["paths"]["activations"]) / f"{args.dataset}_{model_slug}.npz"

    layer, layer_file = results.selected_layer(rdir)
    v_c_path = ddir / f"v_c_L{layer}.pt"
    rnd_path = ddir / f"random_L{layer}.pt"

    def run() -> dict:
        acts, labels, meta = A.load_acts(acts_path)
        template_ids = np.array(str(meta["template_ids"]).split(","))
        train = np.array(str(meta["splits"]).split(",")) == "train"

        v_c, v_meta = io.load_direction(v_c_path)
        rnd, _ = io.load_direction(rnd_path)
        # The direction must not have been built from the rows it is about to be scored on.
        # This assert is the whole generalization claim; without it the number is circular.
        if v_meta.get("split") != "train":
            raise SystemExit(
                f"v_c_L{layer} was extracted with split={v_meta.get('split')!r}, not 'train'. "
                "It saw the held-out templates, so scoring it on them is not a generalization "
                "test. Re-run: scripts/02_extract_directions.py --dataset consequence"
            )

        X_tr, y_tr = acts[train, layer - 1, :], labels[train]
        X_ho, y_ho = acts[~train, layer - 1, :], labels[~train]
        fitted = P.fit_and_score(X_tr, y_tr, X_ho, y_ho, seed=cfg["seed"])

        return {
            "layer": layer,
            "n_heldout_rows": int(len(y_ho)),
            "n_heldout_templates": int(len(set(template_ids[~train]))),
            "probe_auc": fitted["auc"],
            "probe_acc": fitted["acc"],
            "v_c_projection_auc": P.projection_auc(X_ho, y_ho, v_c.numpy()),
            "random_direction_auc": P.projection_auc(X_ho, y_ho, rnd.numpy()),
            "bow_baseline_auc": BOW_HELDOUT_AUC,
            "v_c_extracted_from_split": v_meta.get("split"),
        }

    res = results.compute(
        NAME, run,
        inputs=[acts_path, v_c_path, rnd_path, layer_file],
        params={"dataset": args.dataset, "bow_baseline": BOW_HELDOUT_AUC, "seed": cfg["seed"]},
        entry=__file__, force=args.force, archive=True, results_dir=rdir,
    )
    report(res)


def report(res: dict) -> None:
    print(f"\nHELD-OUT FRAMINGS (L{res['layer']}, {res['n_heldout_templates']} unseen templates, "
          f"n={res['n_heldout_rows']})")
    print(f"  trained probe      AUC {res['probe_auc']:.3f}   acc {res['probe_acc']:.3f}")
    print(f"  v_C projection     AUC {res['v_c_projection_auc']:.3f}   "
          "(no fitting — the direction itself)")
    print(f"  random direction   AUC {res['random_direction_auc']:.3f}   <- null")
    print(f"  BoW text baseline  AUC {res['bow_baseline_auc']:.3f}   <- must be beaten")

    probe, proj, bow = res["probe_auc"], res["v_c_projection_auc"], res["bow_baseline_auc"]
    if probe > bow + 0.05 and proj > 0.7:
        print("=> v_C generalizes to unseen framing routes, beyond unigram vocabulary cues.")
        print("   BoW only rules out unigram leakage — run step 10 before claiming more.")
    elif probe <= bow:
        print("=> the probe does NOT beat bag-of-words: consistent with surface-token learning,")
        print("   not a consequence concept. A real negative result — report it as one.")
    else:
        print("=> mixed: some generalization, but close to the text baseline. Report both.")
    print("   Correlational either way. Nothing here shows the model USES v_C.")


if __name__ == "__main__":
    main()

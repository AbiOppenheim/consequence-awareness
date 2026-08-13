#!/usr/bin/env python
"""Stage 10 (CPU): red-team the held-out result before believing it.

    python scripts/10_redteam_heldout.py

A single held-out AUC hides a lot. Four checks, ordered by how likely each is to change the
conclusion:

  1. per-route breakdown   the headline averages 20 held-out routes. Uniform generalization is
                           a concept; three strong routes carrying the rest is an average
                           hiding heterogeneity.
  2. every layer           if train-CV was flat, the layer choice was near-arbitrary — confirm
                           the result does not hang on it.
  3. stronger text baselines  bag-of-words rules out unigram leakage only. TF-IDF with bigrams
                           and char n-grams is a much fairer proxy for "surface form". If those
                           also reach the headline, the activation result is far less
                           interesting.
  4. proper random null    one random direction is one draw. Use the distribution.

Cost: 9 more probe fits plus 100 null projections — the most expensive CPU step in the project,
and the one most likely to be re-run by accident. Stored under artifacts/results/.
"""

import argparse
import json

import numpy as np
import torch

import _bootstrap  # noqa: F401
from consequence import acts as A
from consequence import io
from consequence import probe as P
from consequence import results
from consequence.config import load_config, resolve

NAME = "heldout_redteam"
WEAK_ROUTE_AUC = 0.75


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen.yaml")
    ap.add_argument("--dataset", default="consequence")
    ap.add_argument("--n-random", type=int, default=100)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rdir = resolve(cfg["paths"]["results"])
    ddir = resolve(cfg["paths"]["directions"])
    model_slug = cfg["model"]["id"].split("/")[-1]
    acts_path = resolve(cfg["paths"]["activations"]) / f"{args.dataset}_{model_slug}.npz"
    src_path = resolve(cfg["data"][args.dataset])
    layers = cfg["extract"]["layer_sweep"]

    layer, layer_file = results.selected_layer(rdir)
    v_c_paths = [ddir / f"v_c_L{L}.pt" for L in layers]

    def run() -> dict:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics import roc_auc_score

        acts, labels, meta = A.load_acts(acts_path)
        template_ids = np.array(str(meta["template_ids"]).split(","))
        train = np.array(str(meta["splits"]).split(",")) == "train"
        ho = ~train

        # The per-route breakdown needs the route column, which the cache does not carry. Read
        # it from the source jsonl and verify the row order matches, or the routes get attached
        # to the wrong activations and every number below is quietly wrong.
        rows = [json.loads(line) for line in open(src_path) if line.strip()]
        if len(rows) != acts.shape[0]:
            raise SystemExit(f"cache has {acts.shape[0]} rows, {src_path.name} has {len(rows)}")
        if not (np.array([r["template_id"] for r in rows]) == template_ids).all():
            raise SystemExit("row order mismatch between the cache and the source jsonl")
        routes = np.array([r["route"] for r in rows])
        texts = np.array([r["text"] for r in rows])

        v_c, _ = io.load_direction(ddir / f"v_c_L{layer}")
        y_ho, r_ho = labels[ho], routes[ho]
        proj = acts[ho, layer - 1, :] @ v_c.numpy()

        # 1) per route: each held-out route against the whole opposite class, so a route is
        #    scored on whether IT separates, not on the average.
        per_route = []
        for polarity, name in ((1, "real"), (0, "hypo")):
            for rt in sorted(set(r_ho[y_ho == polarity])):
                sel = (r_ho == rt) | (y_ho != polarity)
                per_route.append({
                    "route": rt, "polarity": name, "n": int((r_ho == rt).sum()),
                    "auc": float(roc_auc_score(y_ho[sel], proj[sel])),
                })

        # 2) every layer, both readouts
        by_layer = []
        for L in layers:
            v_L, _ = io.load_direction(ddir / f"v_c_L{L}")
            fitted = P.fit_and_score(acts[train, L - 1, :], labels[train],
                                     acts[ho, L - 1, :], y_ho, seed=cfg["seed"])
            by_layer.append({
                "layer": L,
                "probe_auc": fitted["auc"],
                "v_c_projection_auc": P.projection_auc(acts[ho, L - 1, :], y_ho, v_L.numpy()),
            })

        # 3) text baselines, fitted on train framings and scored on held-out — the same
        #    protocol as the probe, so the comparison is like-for-like.
        text_baselines = []
        for name, kw in (("tfidf_unigram", dict(ngram_range=(1, 1))),
                         ("tfidf_uni_bigram", dict(ngram_range=(1, 2))),
                         ("tfidf_char_3_5", dict(analyzer="char_wb", ngram_range=(3, 5)))):
            vec = TfidfVectorizer(min_df=2, sublinear_tf=True, **kw)
            X_tr = vec.fit_transform(texts[train])
            X_ho = vec.transform(texts[ho])
            fitted = P.fit_and_score(X_tr, labels[train], X_ho, y_ho, seed=cfg["seed"])
            text_baselines.append({"name": name, "auc": fitted["auc"]})

        # 4) the null as a distribution
        g = torch.Generator().manual_seed(cfg["seed"] + 1)
        R = torch.randn(args.n_random, acts.shape[2], generator=g)
        R = R / R.norm(dim=-1, keepdim=True)
        X_ho_L = acts[ho, layer - 1, :]
        null = np.array([P.projection_auc(X_ho_L, y_ho, R[i].numpy())
                         for i in range(args.n_random)])

        return {
            "layer": layer,
            "per_route": per_route,
            "by_layer": by_layer,
            "text_baselines": text_baselines,
            "random_null": {
                "n": int(args.n_random),
                "mean": float(null.mean()),
                "p05": float(np.quantile(null, 0.05)),
                "p95": float(np.quantile(null, 0.95)),
                "max_abs_dev": float(np.abs(null - 0.5).max()),
            },
        }

    res = results.compute(
        NAME, run,
        inputs=[acts_path, src_path, layer_file, *v_c_paths],
        params={"dataset": args.dataset, "n_random": args.n_random, "layers": layers,
                "seed": cfg["seed"]},
        entry=__file__, force=args.force, results_dir=rdir,
    )
    report(res)


def report(res: dict) -> None:
    print(f"\nRED-TEAM OF THE HELD-OUT RESULT (L{res['layer']})")

    print("\n1) PER-ROUTE (unfitted v_C projection, each route vs the opposite class)")
    for polarity in ("real", "hypo"):
        print(f"   -- held-out {polarity} routes --")
        for r in [x for x in res["per_route"] if x["polarity"] == polarity]:
            flag = "   <- weak" if r["auc"] < WEAK_ROUTE_AUC else ""
            print(f"     {r['route']:<18} n={r['n']:<4} AUC {r['auc']:.3f}{flag}")
    aucs = [r["auc"] for r in res["per_route"]]
    weak = [r["route"] for r in res["per_route"] if r["auc"] < WEAK_ROUTE_AUC]
    print(f"   median {np.median(aucs):.3f}, min {min(aucs):.3f}; "
          f"{len(weak)}/{len(aucs)} routes below {WEAK_ROUTE_AUC}"
          + (f" ({', '.join(weak)})" if weak else ""))

    print("\n2) HELD-OUT AT EVERY LAYER  (probe | unfitted v_C)")
    for r in res["by_layer"]:
        mark = "  <- selected" if r["layer"] == res["layer"] else ""
        print(f"     L{r['layer']:<3} probe {r['probe_auc']:.3f}   |  "
              f"v_C {r['v_c_projection_auc']:.3f}{mark}")
    proj = [r["v_c_projection_auc"] for r in res["by_layer"]]
    trend = "RISES with depth (a surface-lexical feature would not)" if proj[-1] > proj[0] + 0.02 \
        else "is flat across depth — more consistent with a shallow, lexically available feature"
    print(f"     the unfitted curve {trend}")

    print("\n3) STRONGER TEXT BASELINES (fitted on train framings, scored on held-out)")
    for b in res["text_baselines"]:
        print(f"     {b['name']:<18} AUC {b['auc']:.3f}")
    best_text = max(res["text_baselines"], key=lambda b: b["auc"])
    print(f"     best surface baseline: {best_text['name']} at {best_text['auc']:.3f} — state "
          "this number.\n     Beating it is not the same as being 'beyond surface vocabulary'.")

    n = res["random_null"]
    print(f"\n4) RANDOM-DIRECTION NULL ({n['n']} draws): mean {n['mean']:.3f}, "
          f"5-95% [{n['p05']:.3f}, {n['p95']:.3f}], max |AUC-0.5| = {n['max_abs_dev']:.3f}")
    print(f"     a single draw can reach {n['p95']:.2f}, so quote the band, not one number.")


if __name__ == "__main__":
    main()

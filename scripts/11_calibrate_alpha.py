#!/usr/bin/env python
"""Stage 11 (CPU): calibrate the steering ladder against measured residual norms.

    python scripts/11_calibrate_alpha.py

configs/qwen.yaml ships `alphas: [0, 2, 4, 8, 16]` with a comment saying they are placeholders
to be tuned against measured norms. This does that, for free, from the cached activations.

An alpha that is tiny relative to the residual stream does nothing, and one that is huge only
produces gibberish; either way a sweep over the wrong ladder spends GPU hours to learn nothing.
Expressing alpha as a fraction of the typical activation norm at the steering layer makes the
ladder meaningful, reportable, and comparable across layers and models.

The largest rung is EXPECTED to break fluency. That is not a failed run — the coherence
collapse point is one of the numbers the write-up reports, and it is why scoring is three-way.

Writes artifacts/results/alpha_ladder.json; 05_generate.py --alphas auto reads it from there.
"""

import argparse

import numpy as np

import _bootstrap  # noqa: F401
from consequence import acts as A
from consequence import results
from consequence.config import load_config, resolve

NAME = "alpha_ladder"
FRACTIONS = [0.25, 0.5, 1.0, 2.0]      # of the median residual norm at the steering layer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen.yaml")
    ap.add_argument("--dataset", default="consequence")
    ap.add_argument("--fractions", type=float, nargs="+", default=FRACTIONS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rdir = resolve(cfg["paths"]["results"])
    model_slug = cfg["model"]["id"].split("/")[-1]
    acts_path = resolve(cfg["paths"]["activations"]) / f"{args.dataset}_{model_slug}.npz"
    layer, layer_file = results.selected_layer(rdir)

    def run() -> dict:
        acts, _, _ = A.load_acts(acts_path)
        # Steer where v_C was extracted: a direction measured at L must be added back at L.
        norms = np.linalg.norm(acts[:, layer - 1, :], axis=-1)
        med = float(np.median(norms))
        return {
            "layer": layer,
            "median_norm": med,
            "p05_norm": float(np.quantile(norms, 0.05)),
            "p95_norm": float(np.quantile(norms, 0.95)),
            "fractions": list(args.fractions),
            "alphas": [round(f * med, 1) for f in args.fractions],
        }

    res = results.compute(
        NAME, run,
        inputs=[acts_path, layer_file],
        params={"dataset": args.dataset, "fractions": list(args.fractions)},
        entry=__file__, force=args.force, results_dir=rdir,
    )
    report(res)


def report(res: dict) -> None:
    print(f"\nALPHA LADDER — steering at L{res['layer']}")
    print(f"  residual norm: median {res['median_norm']:.1f} "
          f"(p5 {res['p05_norm']:.1f}, p95 {res['p95_norm']:.1f})")
    for f, a in zip(res["fractions"], res["alphas"]):
        print(f"    {f:>5.2f} x ||h||  ->  alpha = {a}")
    print(f"  alphas = {res['alphas']}")
    print("  05_generate.py --alphas auto reads these, so there is no config to hand-edit and "
          "no\n  ladder to retype into a shell command.")


if __name__ == "__main__":
    main()

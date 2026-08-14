#!/usr/bin/env python
"""Stage 07 (CPU): figures from the analysis JSONs.

    python scripts/07_figures.py

Produces, from whatever exists under artifacts/figures/:
  - probe accuracy vs layer (train vs held-out template)   <- the generalization story
  - refusal/bypass/degenerate rate vs alpha, per condition <- the causal story (if scored)

Figures are regenerable and gitignored; the JSON summaries beside them are the committed record.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

import _bootstrap  # noqa: F401
from consequence.config import load_config, resolve


def plot_probe(fig_dir: Path) -> None:
    for jf in sorted(fig_dir.glob("probe_accuracy_*.json")):
        d = json.loads(jf.read_text())
        by_layer = d["by_layer"]
        layers = sorted(int(L) for L in by_layer)
        train = [by_layer[str(L)]["train_templates_acc"] for L in layers]
        held = [by_layer[str(L)]["heldout_templates_acc"] for L in layers]
        plt.figure()
        plt.plot(layers, train, "o-", label="train templates")
        plt.plot(layers, held, "s-", label="held-out templates")
        plt.axhline(0.5, ls="--", c="grey", lw=1, label="chance")
        plt.xlabel("layer"); plt.ylabel("probe accuracy"); plt.ylim(0.4, 1.02)
        plt.title(f"Consequence probe — {d['dataset']}"); plt.legend()
        out = jf.with_suffix(".png"); plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
        print(f"[fig] {out}")


def plot_sweep(fig_dir: Path, scores_dir: Path) -> None:
    for sf in sorted(scores_dir.glob("*_scored.summary.json")):
        summ = json.loads(sf.read_text())
        # Both rubrics have a refusal rate, but of opposite populations: under 'harmful' it is
        # refusals of an attack (up is good), under 'benign' it is refusals of a safe prompt
        # (up is bad). Same axis, opposite meaning — so the figure has to say which it is.
        benign = any(v.get("rubric") == "benign" for v in summ.values())
        rate_key = "refused_rate" if benign else "refusal_rate"

        # keys look like 'steer_vc@alpha=4.0'; group series by condition.
        series: dict[str, list[tuple[float, float]]] = {}
        for key, v in summ.items():
            cond, _, a = key.partition("@alpha=")
            if rate_key not in v:
                continue
            try:
                series.setdefault(cond, []).append((float(a), v[rate_key]))
            except ValueError:
                continue
        if not series:
            print(f"[fig] skipped {sf.name}: no {rate_key} in it")
            continue
        plt.figure()
        for cond, pts in series.items():
            pts.sort()
            xs, ys = zip(*pts)
            plt.plot(xs, ys, "o-", label=cond)
        plt.xlabel("alpha (steering strength)")
        plt.ylabel("refusal rate on SAFE prompts" if benign else "refusal rate")
        plt.title(("Over-refusal on safe prompts — " if benign else "Refusal restoration — ")
                  + sf.stem)
        plt.legend()
        out = fig_dir / (sf.stem + "_refusal_vs_alpha.png")
        plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
        print(f"[fig] {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    fig_dir = resolve(cfg["paths"]["figures"])
    scores_dir = resolve(cfg["paths"]["scores"])
    plot_probe(fig_dir)
    if scores_dir.exists():
        plot_sweep(fig_dir, scores_dir)


if __name__ == "__main__":
    main()

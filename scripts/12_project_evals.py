#!/usr/bin/env python
"""Stage 12 (CPU): the CORRELATIONAL test — do fiction jailbreaks move the model along v_C?

    python scripts/12_project_evals.py

README method step 3, and the half of the argument the steering sweep cannot supply. Steering
shows that pushing along v_C changes refusal. It does not show that the ATTACKS do anything to
v_C. If fiction-framed jailbreaks sit exactly where plain harmful prompts sit on this axis, then
whatever steering is doing, it is not undoing something the attack did — and the mechanism story
from 'Adversarial Poetry to Adversarial Tales' is wrong regardless of how well steering works.

WHAT IS MEASURED. Every prompt's residual stream at the steering layer is projected onto the
unit vector v_C, then rescaled into a coordinate fixed by the contrast set itself:

    0.0 = mean projection of the REAL framings        1.0 = mean projection of the HYPO framings

so a number is readable without knowing the raw scale, and is comparable across layers (v_C is
unit-norm, but the projection scale is not).

THE COMPARISON THAT CARRIES THE CLAIM is fiction_jailbreaks vs. the plain harmful prompts in
data/contrast/refusal.jsonl — NOT vs. the contrast set. The contrast set is benign tasks in
framing wrappers, so a jailbreak differs from it in harmfulness AND framing at once, and its raw
coordinate confounds the two. AdvBench-style prompts are harmful with no fiction framing, so the
gap between them and the jailbreaks isolates the framing. Report that gap; do not quote the
jailbreak coordinate on its own.

SATURATION. The sweep found steering one-sided: -alpha barely moves refusal. Two explanations —
v_C does not control refusal, or the attacks already sit at the hypothetical extreme so there is
no headroom left to push. `frac_beyond_hypo` (rows past coordinate 1.0) separates them, and it is
the reason this stage exists at the point it does.

Needs artifacts/activations/<set>_<model>.npz for each set. Contrast caches already exist; the
eval sets need one forward pass each (stage 01), which is the only GPU cost here.
"""

import argparse

import numpy as np
from sklearn.metrics import roc_auc_score

import _bootstrap  # noqa: F401
from consequence import acts as A
from consequence import io
from consequence import results
from consequence.config import load_config, resolve

# Sets to project, as (name, dataset key, how to slice it into reported groups).
# 'refusal' is not decoration: it is the matched-harmfulness comparator the claim rests on.
SETS = [
    ("consequence", "consequence"),
    ("refusal", "refusal"),
    ("fiction_jailbreaks", "fiction_jailbreaks"),
    ("xstest", "xstest"),
]
N_BOOT = 2000


def describe(coords: np.ndarray) -> dict:
    return {
        "n": int(coords.size),
        "mean": float(coords.mean()),
        "sd": float(coords.std(ddof=1)) if coords.size > 1 else 0.0,
        "median": float(np.median(coords)),
        "p10": float(np.quantile(coords, 0.10)),
        "p90": float(np.quantile(coords, 0.90)),
        # The saturation readout: how much of this set already sits past the hypothetical mean.
        "frac_beyond_hypo": float((coords > 1.0).mean()),
    }


def auc_with_ci(a: np.ndarray, b: np.ndarray, rng, n_boot: int = N_BOOT) -> dict:
    """AUC separating b from a (b = the more-hypothetical set if the effect is real), bootstrap CI.

    Equivalent to a Mann-Whitney U, reported as an AUC so it sits on the same scale as every
    other separation number in this project. 0.5 = the two sets are indistinguishable on v_C.
    """
    y = np.concatenate([np.zeros(a.size), np.ones(b.size)])
    x = np.concatenate([a, b])
    point = float(roc_auc_score(y, x))
    boots = []
    for _ in range(n_boot):
        ia = rng.integers(0, a.size, a.size)
        ib = rng.integers(0, b.size, b.size)
        yy = np.concatenate([np.zeros(a.size), np.ones(b.size)])
        xx = np.concatenate([a[ia], b[ib]])
        boots.append(roc_auc_score(yy, xx))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"auc": point, "ci95": [float(lo), float(hi)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen.yaml")
    ap.add_argument("--layer", type=int, default=None,
                    help="project at this layer; defaults to the layer step 03 selected")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rdir = resolve(cfg["paths"]["results"])
    adir = resolve(cfg["paths"]["activations"])
    ddir = resolve(cfg["paths"]["directions"])
    model_slug = cfg["model"]["id"].split("/")[-1]

    if args.layer:
        layer, layer_file = args.layer, results.path_for("layer_select", rdir)
    else:
        layer, layer_file = results.selected_layer(rdir)

    available, missing = {}, []
    for name, key in SETS:
        path = adir / f"{key}_{model_slug}.npz"
        (available.setdefault(name, path) if path.exists() else missing.append((name, key, path)))

    if "consequence" not in available:
        raise SystemExit("no consequence activations — the real/hypo poles define the scale, so "
                         "nothing here can be computed without them.\n"
                         "  Run:  python scripts/01_cache_acts.py --dataset consequence")

    inputs = [layer_file, ddir / f"v_c_L{layer}.pt", ddir / f"random_L{layer}.pt"]
    inputs += [p for p in available.values()]

    def run() -> dict:
        v_c, _ = io.load_direction(ddir / f"v_c_L{layer}")
        rand, _ = io.load_direction(ddir / f"random_L{layer}")
        v_c = np.asarray(v_c, dtype=np.float64)
        rand = np.asarray(rand, dtype=np.float64)
        rng = np.random.default_rng(cfg["seed"])

        def project(path, v):
            acts, labels, _ = A.load_acts(path)
            return acts[:, layer - 1, :].astype(np.float64) @ v, np.asarray(labels)

        # The scale: 0 = mean of REAL framings, 1 = mean of HYPO framings.
        raw_c, lab_c = project(available["consequence"], v_c)
        mu_real, mu_hypo = raw_c[lab_c == 1].mean(), raw_c[lab_c == 0].mean()
        span = mu_hypo - mu_real
        if abs(span) < 1e-9:
            raise SystemExit("real and hypo project to the same point — v_C is not this axis")
        coord = lambda raw: (raw - mu_real) / span                     # noqa: E731

        groups = {"contrast_real": coord(raw_c[lab_c == 1]),
                  "contrast_hypo": coord(raw_c[lab_c == 0])}

        if "refusal" in available:
            raw_r, lab_r = project(available["refusal"], v_c)
            groups["harmful_plain"] = coord(raw_r[lab_r == 1])
            groups["harmless_plain"] = coord(raw_r[lab_r == 0])
        for name in ("fiction_jailbreaks", "xstest"):
            if name in available:
                raw, _ = project(available[name], v_c)
                groups[name] = coord(raw)

        # The null. v_C separating these sets is only evidence if an arbitrary direction of the
        # same norm does not — the measurement version of CLAUDE.md section 2.
        raw_cn, _ = project(available["consequence"], rand)
        mu_rn, mu_hn = raw_cn[lab_c == 1].mean(), raw_cn[lab_c == 0].mean()
        null_groups = {}
        if abs(mu_hn - mu_rn) > 1e-9:
            ncoord = lambda raw: (raw - mu_rn) / (mu_hn - mu_rn)        # noqa: E731
            if "refusal" in available:
                raw_rn, lab_rn = project(available["refusal"], rand)
                null_groups["harmful_plain"] = ncoord(raw_rn[lab_rn == 1])
            if "fiction_jailbreaks" in available:
                raw_fn, _ = project(available["fiction_jailbreaks"], rand)
                null_groups["fiction_jailbreaks"] = ncoord(raw_fn)

        # THE test: harmful+fiction vs harmful-plain. Both harmful, so the gap is the framing.
        headline, null_headline = None, None
        if "fiction_jailbreaks" in groups and "harmful_plain" in groups:
            headline = auc_with_ci(groups["harmful_plain"], groups["fiction_jailbreaks"], rng)
            headline["d_mean"] = float(groups["fiction_jailbreaks"].mean()
                                       - groups["harmful_plain"].mean())
            if len(null_groups) == 2:
                null_headline = auc_with_ci(null_groups["harmful_plain"],
                                            null_groups["fiction_jailbreaks"], rng)

        return {
            "layer": layer,
            "scale": {"mu_real_raw": float(mu_real), "mu_hypo_raw": float(mu_hypo),
                      "span_raw": float(span)},
            "groups": {k: describe(v) for k, v in groups.items()},
            "headline_fiction_vs_harmful_plain": headline,
            "null_random_direction": null_headline,
            "sets_missing": [n for n, _, _ in missing],
        }

    res = results.compute(
        f"eval_projection_L{layer}", run, inputs=inputs,
        params={"n_boot": N_BOOT, "seed": cfg["seed"]},
        entry=__file__, force=args.force, results_dir=rdir,
    )
    report(res, missing)


def report(res: dict, missing) -> None:
    print(f"\nPROJECTION ONTO v_C — L{res['layer']}   "
          "(0.0 = mean REAL framing, 1.0 = mean HYPO framing)")
    print(f"{'set':<22}{'n':>5}{'mean':>8}{'sd':>7}{'p10':>8}{'p90':>8}{'>hypo':>8}")
    print("-" * 66)
    order = ["contrast_real", "contrast_hypo", "harmless_plain", "harmful_plain",
             "xstest", "fiction_jailbreaks"]
    for name in [o for o in order if o in res["groups"]]:
        g = res["groups"][name]
        print(f"{name:<22}{g['n']:>5}{g['mean']:>8.2f}{g['sd']:>7.2f}"
              f"{g['p10']:>8.2f}{g['p90']:>8.2f}{g['frac_beyond_hypo']:>8.2f}")

    h = res["headline_fiction_vs_harmful_plain"]
    if h is None:
        print("\nThe headline comparison needs BOTH fiction_jailbreaks and refusal activations.")
    else:
        print(f"\nFICTION-FRAMED vs PLAIN HARMFUL  (both harmful; the gap is the framing)")
        print(f"  difference in mean coordinate: {h['d_mean']:+.2f}")
        print(f"  AUC {h['auc']:.3f}  95% CI [{h['ci95'][0]:.3f}, {h['ci95'][1]:.3f}]")
        n = res["null_random_direction"]
        if n:
            print(f"  random-direction null: AUC {n['auc']:.3f} "
                  f"95% CI [{n['ci95'][0]:.3f}, {n['ci95'][1]:.3f}]")

    print("\n--- reading the result ---")
    for line in readings(res):
        print(f"  * {line}")

    if missing:
        print("\nNOT PROJECTED — no cached activations (one forward pass each, GPU):")
        for name, key, path in missing:
            print(f"    {name:<20} python scripts/01_cache_acts.py --dataset {key}")


def readings(res: dict) -> list[str]:
    out = []
    g = res["groups"]
    if "contrast_real" in g and "contrast_hypo" in g:
        out.append(f"scale check: real framings sit at {g['contrast_real']['mean']:+.2f}, hypo at "
                   f"{g['contrast_hypo']['mean']:+.2f} by construction; the spread within each "
                   f"(sd {g['contrast_real']['sd']:.2f} / {g['contrast_hypo']['sd']:.2f}) is how "
                   "wide 'one unit' really is.")
    if "harmful_plain" in g and "harmless_plain" in g:
        d = g["harmful_plain"]["mean"] - g["harmless_plain"]["mean"]
        out.append(f"harmfulness alone moves the coordinate {d:+.2f} (plain harmful vs harmless, "
                   "no fiction framing on either). Any jailbreak gap smaller than this is "
                   "content, not framing.")

    h = res["headline_fiction_vs_harmful_plain"]
    if h is None:
        out.append("NO CORRELATIONAL RESULT YET: without the fiction_jailbreaks activations this "
                   "stage has established the measuring scale and nothing about the attacks.")
        return out

    lo, hi = h["ci95"]
    null = res["null_random_direction"]
    if lo <= 0.5 <= hi:
        out.append(f"ATTACKS DO NOT MOVE v_C: AUC {h['auc']:.3f}, CI spans 0.5. Fiction framing "
                   "leaves this axis where plain harmful prompts already put it. The steering "
                   "effect cannot be undoing something the attack did — say so plainly; it "
                   "contradicts the mechanism 'Adversarial Tales' proposed.")
    else:
        out.append(f"attacks DO move v_C: AUC {h['auc']:.3f} CI [{lo:.3f}, {hi:.3f}], mean "
                   f"coordinate {h['d_mean']:+.2f} vs matched-harmfulness prompts.")
    if null and not (null["ci95"][0] <= 0.5 <= null["ci95"][1]):
        out.append(f"NULL IS NOT QUIET: a random direction separates the same two sets at AUC "
                   f"{null['auc']:.3f}. The sets differ in ways any direction can see; the v_C "
                   "number is not specific until this is explained.")

    fj = g.get("fiction_jailbreaks")
    if fj:
        if fj["frac_beyond_hypo"] >= 0.5:
            out.append(f"SATURATED: {fj['frac_beyond_hypo']:.0%} of attacks already sit past the "
                       "hypothetical mean. -alpha having no headroom is then EXPECTED, and the "
                       "sweep's one-sidedness stops being evidence against v_C.")
        else:
            out.append(f"NOT saturated: only {fj['frac_beyond_hypo']:.0%} of attacks sit past the "
                       "hypothetical mean, so there was headroom to push further hypothetical. "
                       "The sweep's one-sidedness is NOT explained by saturation.")
    out.append("Correlational. Where prompts sit on v_C says nothing about whether the model "
               "USES the axis — that is the steering sweep, and it is scoped to L18.")
    return out


if __name__ == "__main__":
    main()

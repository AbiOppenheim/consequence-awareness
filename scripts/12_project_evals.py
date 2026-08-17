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

THE PRIMARY TEST IS WITHIN THE ATTACK SET: among fiction jailbreaks, does the model's own
position on v_C predict whether the attack SUCCEEDS? Every prompt there comes from one corpus,
one filter and one length band, so the two groups are matched by construction, and it is the
question the project actually cares about.

It is primary because the obvious comparison — attacks vs. plain harmful prompts — turned out to
be untestable, and expensively so. It looked emphatic (AUC 0.971, a +1.08 coordinate gap) and
means nothing: the two sets differ ~10x in length, so 8% of ARBITRARY directions separate them at
least as well, r_hat separates them BETTER than v_C, and the raw vector norm alone reaches 0.899.
The first version of this stage tested a single stored random direction and reported that gap as
a result. It is kept below as a labelled diagnostic — never a finding — with the random band and
the other-directions table that condemn it, because "we tried this and it cannot work" is worth
recording.

The lesson generalizes: an AUC against unmatched groups is uninterpretable without its random
band, and one draw is not a band.

SATURATION. The sweep found steering one-sided: -alpha barely moves refusal. Two explanations —
v_C does not control refusal, or the attacks already sit at the hypothetical extreme so there is
no headroom left to push. `frac_beyond_hypo` (rows past coordinate 1.0) separates them, and it is
the reason this stage exists at the point it does.

Needs artifacts/activations/<set>_<model>.npz for each set. Contrast caches already exist; the
eval sets need one forward pass each (stage 01), which is the only GPU cost here.
"""

import argparse
import json
from pathlib import Path

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
# One random direction is not a null. Against unmatched groups a typical arbitrary direction
# separates at AUC ~0.79, so the band is the only honest reference (step 10 learned this first).
N_RANDOM = 200


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


def random_band(X: np.ndarray, y: np.ndarray, auc_obs: float, rng,
                n_draws: int = N_RANDOM) -> dict:
    """How well do n_draws ARBITRARY directions separate the same two groups?

    The first version of this stage tested one stored random direction, which step 10 had
    already shown to be worthless: a single draw lands anywhere. It matters more here than
    anywhere else in the project, because the two prompt sets in the cross-set comparison are
    not exchangeable — they differ ~10x in length — and against unmatched groups a typical
    random direction separates at AUC 0.79. Quoting one draw made a confounded comparison look
    clean. The band, and the fraction of draws beating the observed value, is the honest test.
    """
    aucs = []
    for _ in range(n_draws):
        v = rng.standard_normal(X.shape[1])
        v /= np.linalg.norm(v)
        aucs.append(roc_auc_score(y, X @ v))
    a = np.asarray(aucs)
    dev = np.abs(a - 0.5)
    obs = abs(auc_obs - 0.5)
    return {
        "n_draws": n_draws,
        "mean_abs_dev": float(dev.mean()),
        "p95_abs_dev": float(np.quantile(dev, 0.95)),
        "auc_5_95": [float(np.quantile(a, 0.05)), float(np.quantile(a, 0.95))],
        # A permutation-style p: how often an arbitrary direction does at least this well.
        "p_value": float((dev >= obs).mean()),
    }


def direction_comparison(X: np.ndarray, y: np.ndarray, ddir, layer: int) -> dict:
    """The same separation, scored by every other direction we have, plus the raw norm.

    If r_hat and the vector norm separate two groups as well as v_C does, the groups differ in
    gross ways and no per-direction number from them means anything. This is the diagnostic that
    exposed the cross-set comparison as untestable.
    """
    out = {}
    for stem in ("v_c", "r_hat", "v_mp_persona", "v_mp_persona_ut"):
        path = ddir / f"{stem}_L{layer}.pt"
        if path.exists():
            v, _ = io.load_direction(path)
            out[stem] = float(roc_auc_score(y, X @ np.asarray(v, dtype=np.float64)))
    out["norm_only"] = float(roc_auc_score(y, np.linalg.norm(X, axis=1)))
    return out


def within_attack(acts_path, scores_path, gen_meta_path, ddir, layer, rng) -> dict:
    """THE correlational test: among the attacks, does 'more hypothetical' predict SUCCESS?

    Every prompt here comes from the same corpus, the same filter and the same length band, so
    the two groups are matched by construction — which the cross-set comparison is not. It is
    also the question the project actually cares about: not "do jailbreaks look fictional
    relative to some other prompts", but "does the model's own sense that this is fiction
    determine whether the attack lands".

    Pairing is by row index, so the activations and the verdicts MUST come from the same prompt
    file; the eval hashes recorded by stage 01 and stage 05 are compared before anything else.
    """
    acts, _, meta = A.load_acts(acts_path)
    gen_meta = json.loads(Path(gen_meta_path).read_text())
    a_sha, g_sha = str(meta.get("source_sha256", "")), str(gen_meta.get("eval_sha256", ""))
    if a_sha and g_sha and a_sha != g_sha:
        raise SystemExit(
            f"[STALE] activations and verdicts come from different prompt files\n"
            f"        {Path(acts_path).name} built from {a_sha}\n"
            f"        {Path(gen_meta_path).name} generated from {g_sha}\n"
            "        Pairing is positional, so this would silently attach every verdict to the "
            "wrong prompt.")

    rows = [json.loads(l) for l in open(scores_path) if l.strip()]
    base = {r["idx"]: r["label"] for r in rows if r["condition"] == "baseline"}
    if not base:
        raise SystemExit(f"no baseline condition in {Path(scores_path).name} — the within-attack "
                         "test reads whether each attack landed WITHOUT intervention")
    idx = np.array(sorted(base))
    lab = np.array([base[i] for i in idx])
    keep = np.isin(lab, ["refusal", "bypass"])      # degenerate rows answer neither question
    idx, lab = idx[keep], lab[keep]
    if idx.max() >= acts.shape[0]:
        raise SystemExit(f"verdict idx {idx.max()} exceeds {acts.shape[0]} cached activations")

    X = acts[idx, layer - 1, :].astype(np.float64)
    y = (lab == "bypass").astype(int)
    if y.sum() < 5 or (1 - y).sum() < 5:
        return {"n": int(y.size), "n_bypass": int(y.sum()),
                "underpowered": "fewer than 5 in one class — no test attempted"}

    v_c, _ = io.load_direction(ddir / f"v_c_L{layer}")
    v_c = np.asarray(v_c, dtype=np.float64)
    # Negated so the score means "more hypothetical": v_C points toward REAL.
    Xh = -X
    s_vc = Xh @ v_c
    auc = float(roc_auc_score(y, s_vc))

    # Is the signal just prompt size? Longer attacks have systematically different activations,
    # and the raw norm is itself a weak predictor here, so v_C is re-scored with the linear norm
    # component removed. A large drop would mean this matched split has a confound too.
    nrm = np.linalg.norm(X, axis=1)
    resid = s_vc - np.polyval(np.polyfit(nrm, s_vc, 1), nrm)

    # Paired comparison against the persona direction, over the SAME rows — the only way to say
    # which predicts better without being fooled by two noisy point estimates.
    vs_persona = None
    rival_path = ddir / f"v_mp_persona_ut_L{layer}.pt"
    if rival_path.exists():
        v_mp = np.asarray(io.load_direction(rival_path)[0], dtype=np.float64)
        s_mp = Xh @ v_mp
        deltas = []
        for _ in range(N_BOOT):
            i = rng.integers(0, y.size, y.size)
            if y[i].sum() < 5 or (1 - y[i]).sum() < 5:
                continue
            deltas.append(roc_auc_score(y[i], s_mp[i]) - roc_auc_score(y[i], s_vc[i]))
        d = np.asarray(deltas)
        vs_persona = {
            "rival": "v_mp_persona_ut",
            "auc_rival": float(roc_auc_score(y, s_mp)),
            "delta": float(roc_auc_score(y, s_mp) - auc),
            "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
            "p_rival_better": float((d > 0).mean()),
            "cos": float(v_c @ v_mp),
        }

    return {
        "n": int(y.size),
        "n_bypass": int(y.sum()),
        "n_refusal": int((1 - y).sum()),
        "auc_vc": auc,
        "auc_vc_norm_residualised": float(roc_auc_score(y, resid)),
        "corr_score_norm": float(np.corrcoef(s_vc, nrm)[0, 1]),
        "vs_persona": vs_persona,
        "boot": auc_with_ci(
            (Xh[y == 0] @ np.asarray(v_c, dtype=np.float64)),
            (Xh[y == 1] @ np.asarray(v_c, dtype=np.float64)), rng),
        "random_band": random_band(Xh, y, auc, rng),
        "other_directions": direction_comparison(Xh, y, ddir, layer),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen.yaml")
    ap.add_argument("--layer", type=int, default=None,
                    help="project at this layer; defaults to the layer step 03 selected")
    ap.add_argument("--attacks", default="fiction_jailbreaks",
                    help="eval key holding the attack set to run the within-attack test on")
    ap.add_argument("--scores", default=None,
                    help="scored generations for --attacks, e.g. "
                         "artifacts/scores/fiction_jailbreaks_L18_scored.jsonl. Enables the "
                         "within-attack test, which is the primary correlational result.")
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

    # --attacks may name an eval set that is not in the default list (a larger draw built under
    # its own key so the original caches stay valid). Project whatever was asked for, not only
    # what was hardcoded here.
    sets = list(dict.fromkeys(SETS + [(args.attacks, args.attacks)]))
    available, missing = {}, []
    for name, key in sets:
        path = adir / f"{key}_{model_slug}.npz"
        (available.setdefault(name, path) if path.exists() else missing.append((name, key, path)))

    if "consequence" not in available:
        raise SystemExit("no consequence activations — the real/hypo poles define the scale, so "
                         "nothing here can be computed without them.\n"
                         "  Run:  python scripts/01_cache_acts.py --dataset consequence")

    if args.scores and args.attacks not in available:
        raise SystemExit(
            f"--scores was given but there are no cached activations for {args.attacks!r}.\n"
            f"  The within-attack test pairs each verdict with that prompt's activation.\n"
            f"  Run:  python scripts/01_cache_acts.py --dataset {args.attacks}")

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
        # Every remaining set is an eval set: no contrast labels, one group each.
        for name in available:
            if name not in ("consequence", "refusal"):
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

        # The cross-set comparison, kept as a DIAGNOSTIC rather than a result. It is confounded:
        # attacks and AdvBench prompts differ ~10x in length, so the groups are not exchangeable
        # and a typical arbitrary direction separates them. The random band and the
        # other-directions table below are what establish that, so the number is never read
        # without them.
        cross = None
        if args.attacks in groups and "harmful_plain" in groups:
            cross = auc_with_ci(groups["harmful_plain"], groups[args.attacks], rng)
            cross["d_mean"] = float(groups[args.attacks].mean()
                                    - groups["harmful_plain"].mean())
            acts_f, _, _ = A.load_acts(available[args.attacks])
            acts_r, lab_r2, _ = A.load_acts(available["refusal"])
            Xc = np.vstack([acts_r[lab_r2 == 1][:, layer - 1, :], acts_f[:, layer - 1, :]])
            yc = np.concatenate([np.zeros((lab_r2 == 1).sum()), np.ones(acts_f.shape[0])])
            cross["random_band"] = random_band(-Xc.astype(np.float64), yc, cross["auc"], rng)
            cross["other_directions"] = direction_comparison(-Xc.astype(np.float64), yc,
                                                             ddir, layer)

        within = None
        if args.scores:
            within = within_attack(available.get(args.attacks), args.scores,
                                   Path(args.scores).parent.parent / "generations" /
                                   f"{args.attacks}_L{layer}.meta.json", ddir, layer, rng)

        return {
            "layer": layer,
            "scale": {"mu_real_raw": float(mu_real), "mu_hypo_raw": float(mu_hypo),
                      "span_raw": float(span)},
            "groups": {k: describe(v) for k, v in groups.items()},
            "within_attack": within,
            "cross_set_diagnostic": cross,
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
    order = ["contrast_real", "contrast_hypo", "harmless_plain", "harmful_plain", "xstest"]
    order += [k for k in res["groups"] if k not in order]      # attack sets last, incl. --attacks
    for name in [o for o in order if o in res["groups"]]:
        g = res["groups"][name]
        print(f"{name:<22}{g['n']:>5}{g['mean']:>8.2f}{g['sd']:>7.2f}"
              f"{g['p10']:>8.2f}{g['p90']:>8.2f}{g['frac_beyond_hypo']:>8.2f}")

    w = res.get("within_attack")
    print("\n=== PRIMARY: WITHIN the attack set — does 'more hypothetical' predict SUCCESS? ===")
    if w is None:
        print("  not run — pass --scores <scored jailbreaks>.jsonl")
    elif "underpowered" in w:
        print(f"  n={w['n']} bypass={w['n_bypass']} — {w['underpowered']}")
    else:
        b, rb = w["boot"], w["random_band"]
        print(f"  n={w['n']}  bypass={w['n_bypass']}  refusal={w['n_refusal']}   "
              "(same corpus, same filter, same length band -> matched by construction)")
        print(f"  v_C   AUC {w['auc_vc']:.3f}  bootstrap 95% CI "
              f"[{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}]")
        print(f"  {rb['n_draws']} random directions: mean |AUC-0.5| {rb['mean_abs_dev']:.3f}, "
              f"p95 {rb['p95_abs_dev']:.3f}  ->  p = {rb['p_value']:.3f}")
        print(f"  controlling for prompt size: {w['auc_vc_norm_residualised']:.3f} "
              f"(corr with ||h|| = {w['corr_score_norm']:+.3f})")
        print("  same split scored by other directions: " + "  ".join(
            f"{k}={v:.3f}" for k, v in w["other_directions"].items()))
        pv = w.get("vs_persona")
        if pv:
            print(f"  PAIRED vs {pv['rival']}: {pv['auc_rival']:.3f} - {w['auc_vc']:.3f} = "
                  f"{pv['delta']:+.3f}  95% CI [{pv['ci95'][0]:+.3f}, {pv['ci95'][1]:+.3f}]  "
                  f"cos={pv['cos']:+.2f}")

    h = res.get("cross_set_diagnostic")
    print("\n=== DIAGNOSTIC (confounded, not a result): attacks vs plain harmful prompts ===")
    if h is None:
        print("  needs BOTH fiction_jailbreaks and refusal activations.")
    else:
        rb = h.get("random_band", {})
        print(f"  difference in mean coordinate {h['d_mean']:+.2f};  v_C AUC {h['auc']:.3f} "
              f"[{h['ci95'][0]:.3f}, {h['ci95'][1]:.3f}]")
        if rb:
            print(f"  {rb['n_draws']} random directions: mean |AUC-0.5| {rb['mean_abs_dev']:.3f}, "
                  f"5-95% AUC [{rb['auc_5_95'][0]:.3f}, {rb['auc_5_95'][1]:.3f}]  ->  "
                  f"p = {rb['p_value']:.3f}")
        if h.get("other_directions"):
            # Separation STRENGTH, not signed AUC: which side a direction happens to point is
            # arbitrary here, and 0.10 separates exactly as hard as 0.90. Printing the signed
            # value makes the norm look weak next to v_C when it is just as strong.
            print("  separation strength of other scores (max(auc, 1-auc)): " + "  ".join(
                f"{k}={max(v, 1 - v):.3f}" for k, v in h["other_directions"].items()))

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

    h = res.get("cross_set_diagnostic")
    if h and h.get("random_band"):
        rb, od = h["random_band"], h.get("other_directions", {})
        if rb["p_value"] > 0.05:
            out.append(
                f"THE CROSS-SET COMPARISON IS UNTESTABLE, NOT SUPPORTIVE: v_C separates attacks "
                f"from plain harmful prompts at AUC {h['auc']:.3f}, but so do "
                f"{rb['p_value']:.0%} of ARBITRARY directions"
                + (f", and the raw vector norm alone separates them at "
                   f"{max(od['norm_only'], 1 - od['norm_only']):.3f}"
                   if "norm_only" in od else "")
                + ". The two prompt sets differ ~10x in length; nothing per-direction can be "
                  "concluded from them. Do NOT report the +1.08 gap as evidence of framing.")
        else:
            out.append(f"cross-set: v_C at AUC {h['auc']:.3f} beats the random band "
                       f"(p = {rb['p_value']:.3f}) — but the sets are still unmatched in length, "
                       "so treat it as suggestive, not as the result.")

    w = res.get("within_attack")
    if not w:
        out.append("NO PRIMARY RESULT: the within-attack test needs --scores. Everything above "
                   "is scale-setting and diagnostics.")
        return out
    if "underpowered" in w:
        out.append(f"UNDERPOWERED: {w['underpowered']} (n={w['n']}, bypass={w['n_bypass']}). "
                   "Scale the attack set with 05_generate.py --baseline-only.")
        return out

    rb, od = w["random_band"], w["other_directions"]
    if rb["p_value"] > 0.05:
        out.append(f"NOT SIGNIFICANT: within the attacks, v_C predicts success at AUC "
                   f"{w['auc_vc']:.3f}, but p = {rb['p_value']:.3f} against {rb['n_draws']} "
                   f"random directions. Directionally consistent with the hypothesis and NOT "
                   f"evidence for it at n={w['n']} ({w['n_bypass']} successes). More attacks is "
                   "the fix, and it is cheap: --baseline-only.")
    else:
        out.append(f"v_C predicts which attacks succeed: AUC {w['auc_vc']:.3f}, p = "
                   f"{rb['p_value']:.3f} vs {rb['n_draws']} random directions, on prompts "
                   "matched by construction. This is the correlational claim.")
    # Comparing two AUCs by their point estimates is not a comparison. An earlier version fired
    # "PERSONA BEATS CONSEQUENCE" on a 0.084 gap at n=98 that shrank to 0.027 at n=495 — both
    # estimates regressing toward the mean, as small-sample estimates do. The rival claim now
    # has to survive a PAIRED bootstrap over the same rows.
    pv = w.get("vs_persona")
    if pv:
        lo, hi = pv["ci95"]
        if lo > 0:
            out.append(f"PERSONA BEATS CONSEQUENCE on the same split: {pv['rival']} leads by "
                       f"{pv['delta']:+.3f} [{lo:+.3f}, {hi:+.3f}], paired over the same "
                       "prompts. Zhong's explanation predicts this.")
        elif hi < 0:
            out.append(f"v_C beats persona on the same split by {-pv['delta']:.3f} "
                       f"[{-hi:+.3f}, {-lo:+.3f}] paired.")
        else:
            out.append(f"v_C and {pv['rival']} are TIED as predictors of attack success: "
                       f"difference {pv['delta']:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}] spans "
                       f"zero, and the two directions are {pv['cos']:+.2f} correlated. Readout "
                       "cannot separate consequence from persona here — only the steering "
                       "control can.")
    if abs(od.get("norm_only", 0.5) - 0.5) > 0.15:
        out.append(f"WARNING: the raw norm alone predicts success at {od['norm_only']:.3f}, so "
                   "even this matched split has a gross confound. Check prompt length.")

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

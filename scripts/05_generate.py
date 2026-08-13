#!/usr/bin/env python
"""Stage 05 (GPU): the causal sweep — generate under each condition.

    python scripts/05_generate.py --eval fiction_jailbreaks

The layer and the alpha ladder default to 'auto': the layer step 03 selected and the ladder
step 11 calibrated, both read from artifacts/results/. Nothing to retype, and no way to sweep a
ladder that was never calibrated against this layer's residual norms. Pass --layer / --alphas
explicitly to override.

Conditions, all written to artifacts/generations/<eval>_L{layer}.jsonl:
  - baseline                         no intervention
  - steer_vc      at +alpha          h += alpha * v_C   (toward "real")  <- the hypothesis
  - steer_vc_neg  at -alpha          h -= alpha * v_C   (toward "hypothetical")
  - steer_random  at +alpha          the mandatory null, same sweep, same seeds
  - steer_rhat    at +alpha          refusal-direction reference

Why the NEGATIVE condition matters: a real behavioural direction should push refusal BOTH ways.
If only +alpha moves anything, the likely explanation is "adding a large vector degrades the
model", not "this direction controls refusal". Without it the sweep cannot tell those apart.

ONE variable per experiment: this sweeps alpha at a fixed layer window. Change the layer on a
separate run, never both at once. Resumable (Rule 4): conditions already present in the output
are skipped unless --force.
"""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from consequence import acts as A
from consequence import data as D
from consequence import generate as G
from consequence import hooks as H
from consequence import io
from consequence import results
from consequence.config import load_config, resolve


def done_keys(path: Path) -> set:
    """(condition, alpha) pairs already generated, so a re-run resumes instead of duplicating."""
    if not path.exists():
        return set()
    return {(r["condition"], r.get("alpha")) for r in D.load_jsonl(path)}


def resolve_layer(value: str, results_dir) -> int:
    """'auto' -> the layer step 03 selected; an integer -> that layer.

    Reading it from the stored result rather than a hand-typed number means the sweep cannot
    silently steer along a direction extracted somewhere else — the mistake that survives right
    up until the generations look fine and mean nothing.
    """
    if value != "auto":
        return int(value)
    layer, path = results.selected_layer(results_dir)
    print(f"[auto] steering layer L{layer} from {results.relname(path)}")
    return layer


def resolve_alphas(values: list[str], cfg, results_dir) -> list[float]:
    """'auto' -> step 11's calibrated ladder; 'config' -> steer.alphas; else the values given.

    alpha=0 is dropped in every mode: it is the baseline condition, which runs once on its own
    rather than once per rung.
    """
    if values == ["auto"]:
        res, _ = results.load("alpha_ladder", results_dir)
        print(f"[auto] alphas {res['alphas']} = {res['fractions']} x the median residual norm "
              f"at L{res['layer']}")
        values = res["alphas"]
    elif values == ["config"]:
        values = cfg["steer"]["alphas"]
        print(f"[config] alphas {values} — placeholders unless 11_calibrate_alpha.py has run")
    return [float(a) for a in values if float(a) != 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen.yaml")
    ap.add_argument("--eval", default="fiction_jailbreaks", help="key under data: in the config")
    ap.add_argument("--layer", default="auto",
                    help="steer with v_c_L{layer}. 'auto' (default) reads the layer step 03 "
                         "selected from artifacts/results/layer_select.json.")
    ap.add_argument("--window", type=int, default=1,
                    help="steer at layers [layer-w+1 .. layer]; 1 = only the extraction layer")
    ap.add_argument("--force", action="store_true", help="regenerate conditions already present")
    ap.add_argument("--alphas", nargs="+", default=["auto"],
                    help="'auto' (default) reads the ladder calibrated by step 11, or pass "
                         "values e.g. --alphas 12 24 48 96. Either way they land in the "
                         ".meta.json sidecar, so the run records the ladder it actually used.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rdir = resolve(cfg["paths"]["results"])
    layer = resolve_layer(args.layer, rdir)
    alphas = resolve_alphas(args.alphas, cfg, rdir)
    if not alphas:
        raise SystemExit("no non-zero alphas — run scripts/11_calibrate_alpha.py, pass --alphas, "
                         "or set steer.alphas in the config")

    # The steering window is DERIVED from --layer, never read from a separate config key: a
    # direction extracted at L must be added back at L, or we steer along a vector measured
    # somewhere else. (hooks.apply_hooks maps config layer L -> hidden_states[L].)
    layers = list(range(layer - args.window + 1, layer + 1))

    gen_kwargs = dict(
        max_new_tokens=cfg["generate"]["max_new_tokens"],
        batch_size=cfg["generate"]["batch_size"],
        do_sample=cfg["generate"]["do_sample"],
        seed=cfg["seed"],
    )

    prompts = [r.get("text", r.get("prompt"))
               for r in D.load_jsonl(resolve(cfg["data"][args.eval]))]
    ddir = resolve(cfg["paths"]["directions"])
    out = resolve(cfg["paths"]["generations"]) / f"{args.eval}_L{layer}.jsonl"
    if args.force and out.exists():
        out.unlink()
    already = done_keys(out)

    v_c, _ = io.load_direction(ddir / f"v_c_L{layer}")
    rand, _ = io.load_direction(ddir / f"random_L{layer}")
    r_hat = None
    rh_path = ddir / f"r_hat_L{layer}.pt"          # per-layer, minted from the gate cube
    if rh_path.exists():
        r_hat, _ = io.load_direction(rh_path)
    else:
        print(f"[warn] {rh_path.name} missing — skipping the r_hat reference condition")

    print(f"[sweep] {len(prompts)} prompts | steer layers {layers} | alphas {alphas}")
    n_cond = 1 + len(alphas) * (3 + (1 if r_hat is not None else 0))
    print(f"[sweep] {n_cond} conditions -> ~{n_cond * len(prompts)} generations")

    model, tok = A.load_model(cfg["model"]["id"], cfg["model"]["dtype"])

    if ("baseline", 0.0) not in already:
        G.run_condition(model, tok, prompts, out, condition="baseline", gen_kwargs=gen_kwargs)
        print("[gen] baseline done")

    for alpha in alphas:
        conds = [
            ("steer_vc", v_c, +alpha),          # toward "real" — the hypothesis
            ("steer_vc_neg", v_c, -alpha),      # toward "hypothetical" — must move the other way
            ("steer_random", rand, +alpha),     # the null
        ]
        if r_hat is not None:
            conds.append(("steer_rhat", r_hat, +alpha))
        for name, vec, a in conds:
            if (name, a) in already:
                print(f"[skip] {name} alpha={a} already generated")
                continue
            G.run_condition(
                model, tok, prompts, out, condition=name, direction=name.split("_", 1)[1],
                layers=layers, alpha=a,
                hook_factory=lambda v=vec, aa=a: H.steer_hook(v, aa),
                gen_kwargs=gen_kwargs,
            )
            print(f"[gen] {name} alpha={a:+g} done")

    meta = out.with_suffix(".meta.json")
    meta.write_text(json.dumps({
        "eval": args.eval, "model_id": cfg["model"]["id"], "direction_layer": layer,
        "steer_layers": layers, "alphas": alphas, "seed": cfg["seed"],
        "n_prompts": len(prompts), "conditions": ["baseline", "steer_vc", "steer_vc_neg",
                                                  "steer_random"] + (["steer_rhat"] if r_hat else []),
    }, indent=2))
    print(f"[write] {out}\n[write] {meta}")


if __name__ == "__main__":
    main()

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
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

import _bootstrap  # noqa: F401
from consequence import acts as A
from consequence import data as D
from consequence import generate as G
from consequence import hooks as H
from consequence import io
from consequence import results
from consequence.config import load_config, resolve


def done_keys(path: Path, n_prompts: int) -> set:
    """(condition, alpha) pairs that are COMPLETE, so a re-run resumes instead of duplicating.

    A condition counts as done only when all n_prompts of its rows are present. run_condition
    appends a whole condition at once, so a runtime killed mid-write (a Colab disconnect, a
    recycled VM) can leave a short group and a truncated final line. Counting rows rather than
    trusting the label matters: a partial group treated as done would leave the sweep quietly
    short of prompts under one condition, and every later aggregate would average over a
    different n than it claims.

    Incomplete groups are dropped from the file so the append stays parseable and the condition
    regenerates cleanly.
    """
    if not path.exists():
        return set()

    rows, truncated = [], False
    lines = path.read_text().splitlines(True)
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:       # died mid-write; anything earlier is real corruption
                truncated = True
                break
            raise

    counts = Counter((r["condition"], r.get("alpha")) for r in rows)
    complete = {k for k, n in counts.items() if n >= n_prompts}
    partial = sorted(k for k, n in counts.items() if n < n_prompts)
    if partial or truncated:
        kept = [r for r in rows if (r["condition"], r.get("alpha")) in complete]
        path.write_text("".join(json.dumps(r) + "\n" for r in kept))
        print(f"[resume] dropped {len(rows) - len(kept)} rows from incomplete conditions "
              f"{partial} — regenerating them")
    if complete:
        print(f"[resume] {len(complete)} (condition, alpha) pairs already complete "
              f"({len(complete) * n_prompts} generations kept)")
    return complete


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


def resolve_alphas(values: list[str], cfg, results_dir, layer: int) -> list[float]:
    """'auto' -> step 11's calibrated ladder; 'config' -> steer.alphas; else the values given.

    alpha=0 is dropped in every mode: it is the baseline condition, which runs once on its own
    rather than once per rung.
    """
    if values == ["auto"]:
        # Per-layer, because residual norms grow with depth: L18's ladder is the wrong dose at
        # L22. Falls back to the un-suffixed name for ladders calibrated before that split.
        try:
            res, _ = results.load(f"alpha_ladder_L{layer}", results_dir)
        except FileNotFoundError:
            res, _ = results.load("alpha_ladder", results_dir)
            print(f"[warn] no alpha_ladder_L{layer} — using the un-suffixed ladder, calibrated "
                  f"at L{res['layer']}. Re-run 11_calibrate_alpha.py --layer {layer} if these "
                  "differ.")
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
    ap.add_argument("--baseline-only", action="store_true",
                    help="generate ONLY the unsteered baseline — no steering, no directions, no "
                         "alpha ladder. This is what the stage-12 correlational test needs: for "
                         "each prompt, did the attack land without intervention. It costs 1/21 "
                         "of a sweep, so the attack set can be scaled up for that test without "
                         "paying for a steering sweep at the larger n.")
    ap.add_argument("--extra-direction", action="append", default=[], metavar="STEM",
                    help="also sweep this direction, e.g. --extra-direction v_c_orth_r_hat. "
                         "Loads artifacts/directions/STEM_L{layer}.pt and adds a condition "
                         "'steer_STEM' at every alpha. Repeatable. Appends to the SAME output "
                         "file, so resumption keeps the conditions already generated.")
    ap.add_argument("--extra-both-signs", action="store_true",
                    help="sweep each --extra-direction at -alpha as well as +alpha. v_C gets "
                         "both signs already; an extra did not, which left the sharpest control "
                         "unrun: a rival that steers refusal DOWN at +alpha is exactly the one "
                         "that may restore it at -alpha, and that is the condition v_C's "
                         "selectivity claim has to beat.")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override generate.batch_size. The first thing to lower on a CUDA OOM; "
                         "it changes throughput, not the greedy completions.")
    ap.add_argument("--alphas", nargs="+", default=["auto"],
                    help="'auto' (default) reads the ladder calibrated by step 11, or pass "
                         "values e.g. --alphas 12 24 48 96. Either way they land in the "
                         ".meta.json sidecar, so the run records the ladder it actually used.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rdir = resolve(cfg["paths"]["results"])
    layer = resolve_layer(args.layer, rdir)
    # The baseline needs no ladder, and requiring one would block running it at a layer that has
    # never been calibrated — which is exactly the case this flag exists for.
    alphas = [] if args.baseline_only else resolve_alphas(args.alphas, cfg, rdir, layer)
    if not alphas and not args.baseline_only:
        raise SystemExit("no non-zero alphas — run scripts/11_calibrate_alpha.py, pass --alphas, "
                         "or set steer.alphas in the config")

    # The steering window is DERIVED from --layer, never read from a separate config key: a
    # direction extracted at L must be added back at L, or we steer along a vector measured
    # somewhere else. (hooks.apply_hooks maps config layer L -> hidden_states[L].)
    layers = list(range(layer - args.window + 1, layer + 1))

    gen_kwargs = dict(
        max_new_tokens=cfg["generate"]["max_new_tokens"],
        batch_size=args.batch_size or cfg["generate"]["batch_size"],
        do_sample=cfg["generate"]["do_sample"],
        seed=cfg["seed"],
    )

    eval_path = resolve(cfg["data"][args.eval])
    if not eval_path.exists():
        # data/eval/* is gitignored (this repo does not redistribute an attack set), so it is
        # absent on every fresh runtime. Say which command rebuilds it rather than surfacing a
        # FileNotFoundError from three frames down in a loader.
        raise SystemExit(
            f"missing {results.relname(eval_path)} — the eval sets are gitignored and are "
            "rebuilt, not cloned.\n"
            f"  Run:  python scripts/phase5_build_eval.py --all --limit 100"
        )
    prompts = [r.get("text", r.get("prompt")) for r in D.load_jsonl(eval_path)]
    eval_sha = hashlib.sha256(eval_path.read_bytes()).hexdigest()[:16]
    ddir = resolve(cfg["paths"]["directions"])
    out = resolve(cfg["paths"]["generations"]) / f"{args.eval}_L{layer}.jsonl"
    meta_path = out.with_suffix(".meta.json")
    if args.force and out.exists():
        out.unlink()
    already = done_keys(out, len(prompts))

    # Resumption trusts that the prompt set has not moved underneath it. The attack file is
    # gitignored and rebuilt by phase5_build_eval.py, so a recycled runtime regenerates it —
    # deterministically today, but a revised upstream snapshot or a changed filter would leave
    # old completions in the same file as new prompts, indexed by position, with no symptom.
    # Stage 01 has guarded exactly this since the dataset-hash check; stage 05 now does too.
    if already and meta_path.exists():
        prior_sha = json.loads(meta_path.read_text()).get("eval_sha256")
        if prior_sha and prior_sha != eval_sha:
            raise SystemExit(
                f"[STALE] {out.name} was generated from a different {eval_path.name}\n"
                f"        recorded {prior_sha} != current {eval_sha}\n"
                "        The prompt set changed since those generations were made. Either restore\n"
                "        the original eval file, or re-run with --force to regenerate all\n"
                "        conditions against the current one. Resuming would mix the two."
            )

    if args.baseline_only:
        # No directions are loaded at all: the point of this path is to run at a larger n than
        # the sweep, possibly before any direction exists for the layer.
        print(f"[sweep] BASELINE ONLY — {len(prompts)} prompts, no steering")
        model, tok = A.load_model(cfg["model"]["id"], cfg["model"]["dtype"])
        try:
            run_all(model, tok, prompts, out, already, [], layers, gen_kwargs,
                    None, None, None)
        except torch.cuda.OutOfMemoryError:
            raise SystemExit("CUDA OOM on the baseline pass — lower --batch-size")
        meta_path.write_text(json.dumps({
            "eval": args.eval, "eval_sha256": eval_sha,
            "model_id": cfg["model"]["id"], "direction_layer": layer,
            "steer_layers": [], "alphas": [], "seed": cfg["seed"],
            "n_prompts": len(prompts), "conditions": ["baseline"],
        }, indent=2))
        print(f"[write] {out}\n[write] {meta_path}")
        return

    v_c, _ = io.load_direction(ddir / f"v_c_L{layer}")
    rand, _ = io.load_direction(ddir / f"random_L{layer}")
    r_hat = None
    rh_path = ddir / f"r_hat_L{layer}.pt"          # per-layer, minted from the gate cube
    if rh_path.exists():
        r_hat, _ = io.load_direction(rh_path)
    else:
        print(f"[warn] {rh_path.name} missing — skipping the r_hat reference condition")

    extra = []
    for stem in args.extra_direction:
        vec, meta_x = io.load_direction(ddir / f"{stem}_L{layer}")
        extra.append((f"steer_{stem}", vec))
        print(f"[sweep] extra direction {stem}_L{layer} ({meta_x.get('method', '?')})")

    print(f"[sweep] {len(prompts)} prompts | steer layers {layers} | alphas {alphas}")
    n_cond = 1 + len(alphas) * (3 + (1 if r_hat is not None else 0) + len(extra))
    print(f"[sweep] {n_cond} conditions -> ~{n_cond * len(prompts)} generations")

    model, tok = A.load_model(cfg["model"]["id"], cfg["model"]["dtype"])

    try:
        run_all(model, tok, prompts, out, already, alphas, layers, gen_kwargs, v_c, rand,
                r_hat, extra, args)
    except torch.cuda.OutOfMemoryError:
        done = len(done_keys(out, len(prompts)))
        raise SystemExit(
            f"\nCUDA OUT OF MEMORY at batch_size={gen_kwargs['batch_size']}.\n"
            f"  {done} condition(s) completed and are safe on disk — a re-run resumes.\n"
            "  Fix, in order of preference:\n"
            "    1. a bigger card (Runtime > Change runtime type > L4). A 7-8B model in bf16 "
            "needs ~15 GB of\n       weights before the KV cache, so a 15 GB T4 offloads to "
            "CPU and cannot hold a batch.\n"
            "    2. --batch-size 4 (then 2). Throughput drops; the greedy completions do not "
            "change.\n"
            "    3. lower generate.max_new_tokens — but that changes what the judge sees, so "
            "record it."
        )

    meta = meta_path
    meta.write_text(json.dumps({
        "eval": args.eval, "eval_sha256": eval_sha,
        "model_id": cfg["model"]["id"], "direction_layer": layer,
        "steer_layers": layers, "alphas": alphas, "seed": cfg["seed"],
        "n_prompts": len(prompts),
        # `if r_hat` on a tensor raises: truthiness of a multi-element tensor is ambiguous.
        "conditions": ["baseline", "steer_vc", "steer_vc_neg", "steer_random"]
                      + (["steer_rhat"] if r_hat is not None else [])
                      + [n for n, _ in extra],
        "extra_both_signs": args.extra_both_signs,
    }, indent=2))
    print(f"[write] {out}\n[write] {meta}")


def run_all(model, tok, prompts, out, already, alphas, layers, gen_kwargs, v_c, rand,
            r_hat, extra=(), args=None):
    """Every condition in the sweep, skipping the ones already complete."""
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
        conds += [(name, vec, +alpha) for name, vec in extra]
        if args and args.extra_both_signs:
            conds += [(name, vec, -alpha) for name, vec in extra]
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



if __name__ == "__main__":
    main()

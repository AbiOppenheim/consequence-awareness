#!/usr/bin/env python
"""Stage 05 (GPU): the causal sweep — generate under each condition.

    python scripts/05_generate.py --eval fiction_jailbreaks --layer 14

Conditions, all written to artifacts/generations/<eval>_L{layer}.jsonl:
  - baseline                       (no intervention)
  - steer_vc     at each alpha     (h += alpha * v_C, toward "real")
  - steer_random at each alpha     (the mandatory null: same sweep, same seeds)
  - steer_rhat   at each alpha     (refusal-direction baseline, if r_hat.pt exists)

ONE variable per experiment: this script sweeps alpha at a fixed layer window. Change the
layer on a separate run, never both at once.
"""

import argparse

import _bootstrap  # noqa: F401
from consequence import acts as A
from consequence import data as D
from consequence import generate as G
from consequence import hooks as H
from consequence import io
from consequence.config import load_config, resolve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen.yaml")
    ap.add_argument("--eval", default="fiction_jailbreaks", help="key under data: in the config")
    ap.add_argument("--layer", type=int, required=True, help="which v_c_L{layer} to steer with")
    args = ap.parse_args()

    cfg = load_config(args.config)
    layers = cfg["steer"]["layers"]
    alphas = cfg["steer"]["alphas"]
    gen_kwargs = dict(
        max_new_tokens=cfg["generate"]["max_new_tokens"],
        batch_size=cfg["generate"]["batch_size"],
        do_sample=cfg["generate"]["do_sample"],
        seed=cfg["seed"],
    )

    prompts = [r["text"] if "text" in r else r["prompt"]
               for r in D.load_jsonl(resolve(cfg["data"][args.eval]))]
    ddir = resolve(cfg["paths"]["directions"])
    out = resolve(cfg["paths"]["generations"]) / f"{args.eval}_L{args.layer}.jsonl"

    v_c, _ = io.load_direction(ddir / f"v_c_L{args.layer}.pt")
    rand, _ = io.load_direction(ddir / f"random_L{args.layer}.pt")
    r_hat = None
    if (ddir / "r_hat.pt").exists():
        r_hat, _ = io.load_direction(ddir / "r_hat.pt")

    model, tok = A.load_model(cfg["model"]["id"], cfg["model"]["dtype"])

    G.run_condition(model, tok, prompts, out, condition="baseline", gen_kwargs=gen_kwargs)
    print("[gen] baseline done")

    steer_dirs = [("steer_vc", v_c), ("steer_random", rand)]
    if r_hat is not None:
        steer_dirs.append(("steer_rhat", r_hat))

    for alpha in alphas:
        if alpha == 0:
            continue
        for name, vec in steer_dirs:
            G.run_condition(
                model, tok, prompts, out, condition=name, direction=name.split("_")[1],
                layers=layers, alpha=alpha,
                hook_factory=lambda v=vec, a=alpha: H.steer_hook(v, a),
                gen_kwargs=gen_kwargs,
            )
            print(f"[gen] {name} alpha={alpha} done")

    print(f"[write] {out}")


if __name__ == "__main__":
    main()

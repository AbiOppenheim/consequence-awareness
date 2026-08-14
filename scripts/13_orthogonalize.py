#!/usr/bin/env python
"""Stage 13 (CPU): is v_C's effect just its refusal-direction component?

    python scripts/13_orthogonalize.py

`r_hat` steering drove refusal to 0.99 where `v_C` reached 0.85, and cos(v_C, r_hat) is small
but not zero (+0.083 at L18, rising to +0.209 by L24). So a deflationary story survives Phase 4:
maybe steering along v_C works only through the sliver of r_hat inside it, and the "consequence"
framing is decoration.

This removes that sliver and asks what is left:

    v_perp = v_C - (v_C . r_hat) r_hat,  renormalized

Two questions, and the cheap one first:

  1. INFORMATION (here, CPU, free). Does v_perp still separate real from hypothetical on the
     held-out framings? If the held-out AUC barely moves, the concept does not live in the
     r_hat component, and that is settled without touching a GPU.
  2. CAUSALITY (a sweep, GPU). Does steering along v_perp still restore refusal? Run:
         python scripts/05_generate.py --eval fiction_jailbreaks --extra-direction v_c_orth_rhat
     then re-judge and re-analyse. If the effect survives, v_C is doing something r_hat is not.

Note what question 2 does NOT settle: r_hat's own effect being larger is not evidence against
v_C. r_hat was extracted to move refusal and does so directly; v_C moving refusal at all, along
an axis 5 degrees off orthogonal to it, is the surprising part.

Writes artifacts/directions/v_c_orth_rhat_L*.pt (+ Rule 2 sidecars) and
artifacts/results/orthogonalize.json.
"""

import argparse

import numpy as np
import torch

import _bootstrap  # noqa: F401
from consequence import acts as A
from consequence import io
from consequence import probe as P
from consequence import results
from consequence.config import load_config, resolve

NAME = "orthogonalize"


def orthogonalize(v: torch.Tensor, against: torch.Tensor) -> torch.Tensor:
    """Component of v perpendicular to `against`, unit length.

    Textbook Gram-Schmidt. Worth being explicit that the result is renormalized: without it,
    steering at the same alpha would apply a SMALLER vector than the v_C run and any drop in
    effect would just be a smaller dose.
    """
    a = against.float() / against.float().norm()
    v = v.float()
    perp = v - torch.dot(v, a) * a
    if perp.norm() < 1e-6:
        raise ValueError("v is parallel to the reference — nothing perpendicular is left")
    return perp / perp.norm()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen.yaml")
    ap.add_argument("--dataset", default="consequence")
    ap.add_argument("--against", default="r_hat", help="direction stem to remove from v_C")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rdir = resolve(cfg["paths"]["results"])
    ddir = resolve(cfg["paths"]["directions"])
    layers = cfg["extract"]["layer_sweep"]
    model_slug = cfg["model"]["id"].split("/")[-1]
    acts_path = resolve(cfg["paths"]["activations"]) / f"{args.dataset}_{model_slug}.npz"
    layer, layer_file = results.selected_layer(rdir)

    srcs = [ddir / f"{s}_L{L}.pt" for L in layers for s in ("v_c", args.against)]

    def run() -> dict:
        acts, labels, meta = A.load_acts(acts_path)
        train = np.array(str(meta["splits"]).split(",")) == "train"
        ho = ~train
        y_ho = labels[ho]

        by_layer = []
        for L in layers:
            v_c, meta_c = io.load_direction(ddir / f"v_c_L{L}")
            ref, _ = io.load_direction(ddir / f"{args.against}_L{L}")
            perp = orthogonalize(v_c, ref)

            io.save_direction(perp, ddir / f"v_c_orth_{args.against}_L{L}", {
                **{k: meta_c[k] for k in ("model_id", "token_position", "source_contrast",
                                          "n_pairs", "seed", "split")},
                "layer": L,
                "method": f"v_c minus its {args.against} component, renormalized",
                "contrast": "real_minus_hypo",
                "orthogonal_to": f"{args.against}_L{L}",
            })

            cos_before = float(torch.dot(v_c, ref) / (v_c.norm() * ref.norm()))
            X_ho = acts[ho, L - 1, :]
            by_layer.append({
                "layer": L,
                "cos_v_c_vs_ref": cos_before,
                # How much of v_C was the reference direction. cos^2 is the share of variance.
                "norm_fraction_removed": abs(cos_before),
                "heldout_auc_v_c": P.projection_auc(X_ho, y_ho, v_c.numpy()),
                "heldout_auc_orth": P.projection_auc(X_ho, y_ho, perp.numpy()),
                "cos_orth_vs_ref": float(torch.dot(perp, ref) / (perp.norm() * ref.norm())),
            })

        return {
            "against": args.against,
            "selected_layer": layer,
            "by_layer": by_layer,
            "n_heldout_rows": int(ho.sum()),
        }

    res = results.compute(
        NAME, run, inputs=[acts_path, layer_file, *srcs],
        params={"against": args.against, "layers": layers, "dataset": args.dataset},
        entry=__file__, force=args.force, results_dir=rdir,
    )
    report(res)


def report(res: dict) -> None:
    ref = res["against"]
    print(f"\nORTHOGONALIZING v_C AGAINST {ref} — held-out AUC on {res['n_heldout_rows']} rows")
    print(f"  layer | cos(v_C,{ref})  | AUC v_C   AUC v_perp   delta | cos(v_perp,{ref})")
    for r in res["by_layer"]:
        d = r["heldout_auc_orth"] - r["heldout_auc_v_c"]
        mark = "  <- selected" if r["layer"] == res["selected_layer"] else ""
        print(f"  L{r['layer']:<4} | {r['cos_v_c_vs_ref']:>+12.3f}  | "
              f"{r['heldout_auc_v_c']:>7.3f}   {r['heldout_auc_orth']:>9.3f}   {d:>+5.3f} | "
              f"{r['cos_orth_vs_ref']:>+8.1e}{mark}")

    sel = next(r for r in res["by_layer"] if r["layer"] == res["selected_layer"])
    drop = sel["heldout_auc_v_c"] - sel["heldout_auc_orth"]
    print(f"\nAt L{sel['layer']}: removing the {ref} component costs "
          f"{drop:+.3f} held-out AUC ({sel['heldout_auc_v_c']:.3f} -> "
          f"{sel['heldout_auc_orth']:.3f}).")
    if drop < 0.02:
        print(f"  => the real-vs-hypothetical INFORMATION does not live in the {ref} component.")
        print("     That settles the readout question. The causal question still needs a sweep:")
        print("     05_generate.py --extra-direction v_c_orth_r_hat")
    elif drop > 0.10:
        print(f"  => a large share of the readable signal was the {ref} component. Weakens the")
        print("     distinctness claim — report this number prominently.")
    else:
        print(f"  => a modest share of the signal was {ref}. Report the number; claim neither")
        print("     independence nor equivalence.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Stage 04 (CPU): is v_C its own axis, or one of the known directions relabelled?

    python scripts/04_geometry.py

Claim 1's distinctness half. Two deflationary explanations to kill:
  * v_C is r_hat relabelled            -> cos(v_C, r_hat) near +-1
  * v_C is the persona axis relabelled -> cos(v_C, v_MP) near +-1   (the sharper test)

Runs over every swept layer, not just the selected one, because a cosine that is small at one
layer and large at another is a fact about the geometry that a single number would hide.

v_MP comes in two framings on purpose — the instruction in the SYSTEM prompt (faithful to
Zhong's model_persona) and in the USER turn (structurally identical to how v_C and r_hat were
extracted). If the two agree, a low cos(v_C, v_MP) cannot be blamed on prompt structure. If
Phase 4 has not run, the v_MP columns are simply absent and the r_hat comparison still reports.

Sign convention: v_MP = mean(compliant) - mean(restrictive) points TOWARD compliance, roughly
opposite in spirit to r_hat = mean(harmful) - mean(harmless). Compare magnitudes; read the sign
only as which way the axis points.

Writes artifacts/results/geometry.json.
"""

import argparse

import torch

import _bootstrap  # noqa: F401
from consequence import directions as Dir
from consequence import io
from consequence import results
from consequence.config import load_config, resolve

NAME = "geometry"

# Named so the result file is readable without the code next to it.
PAIRS = [
    ("v_c", "r_hat", "cos(v_C, r_hat)"),
    ("v_c", "v_mp_persona", "cos(v_C, v_MP sys)"),
    ("v_c", "v_mp_persona_ut", "cos(v_C, v_MP ut)"),
    ("r_hat", "v_mp_persona", "cos(r_hat, v_MP sys)"),
    ("v_mp_persona", "v_mp_persona_ut", "cos(v_MP sys, v_MP ut)"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen.yaml")
    ap.add_argument("--n-random", type=int, default=1000, help="draws for the null cosine band")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rdir = resolve(cfg["paths"]["results"])
    ddir = resolve(cfg["paths"]["directions"])
    layers = cfg["extract"]["layer_sweep"]
    layer, layer_file = results.selected_layer(rdir)

    stems = ["v_c", "r_hat", "v_mp_persona", "v_mp_persona_ut"]
    present = {s: [ddir / f"{s}_L{L}.pt" for L in layers]
               for s in stems if all((ddir / f"{s}_L{L}.pt").exists() for L in layers)}
    if "v_c" not in present:
        raise SystemExit("need v_c_L*.pt at every swept layer — run 02_extract_directions.py")
    for s in stems:
        if s not in present:
            print(f"[skip] {s}: not minted at every swept layer yet")

    def run() -> dict:
        by_layer = []
        for L in layers:
            vecs = {s: io.load_direction(ddir / f"{s}_L{L}")[0] for s in present}
            row = {"layer": L,
                   "null_abs_cos_p95": Dir.random_null_band(
                       vecs["v_c"], n_random=args.n_random, seed=cfg["seed"])["abs_cos_p95"]}
            for a, b, label in PAIRS:
                row[label] = Dir.cosine(vecs[a], vecs[b]) if a in vecs and b in vecs else None
            by_layer.append(row)
        return {
            "selected_layer": layer,
            "layers": layers,
            "directions_present": sorted(present),
            "by_layer": by_layer,
            "n_random": args.n_random,
        }

    res = results.compute(
        NAME, run,
        inputs=[layer_file, *(p for paths in present.values() for p in paths)],
        params={"layers": layers, "n_random": args.n_random, "seed": cfg["seed"]},
        entry=__file__, force=args.force, results_dir=rdir,
    )
    report(res)


def report(res: dict) -> None:
    labels = [label for _, _, label in PAIRS
              if any(r.get(label) is not None for r in res["by_layer"])]
    print(f"\nGEOMETRY — directions present: {', '.join(res['directions_present'])}")
    header = "  layer | " + " | ".join(f"{lab:>20}" for lab in labels) + " | null p95"
    print(header)
    for r in res["by_layer"]:
        cells = " | ".join(f"{r[lab]:>+20.3f}" if r.get(lab) is not None else f"{'—':>20}"
                           for lab in labels)
        mark = "  <- selected" if r["layer"] == res["selected_layer"] else ""
        print(f"  L{r['layer']:<4} | {cells} | {r['null_abs_cos_p95']:.3f}{mark}")

    sel = next(r for r in res["by_layer"] if r["layer"] == res["selected_layer"])
    null = sel["null_abs_cos_p95"]
    print(f"\nAt the selected layer L{sel['layer']} (random-direction band |cos| p95 = {null:.3f}):")

    rhat = sel.get("cos(v_C, r_hat)")
    if rhat is not None:
        print(f"  vs r_hat: {rhat:+.3f} — " + verdict(abs(rhat), "r_hat"))

    vmps = [sel.get("cos(v_C, v_MP sys)"), sel.get("cos(v_C, v_MP ut)")]
    if any(v is not None for v in vmps):
        worst = max(abs(v) for v in vmps if v is not None)
        print(f"  vs persona: {vmps[0]:+.3f} (system) / {vmps[1]:+.3f} (user turn) — "
              + verdict(worst, "persona"))
        agree = sel.get("cos(v_MP sys, v_MP ut)")
        if agree is not None:
            print(f"  the two v_MP framings agree at {agree:+.3f} — "
                  + ("high, so the persona cosine is not an artifact of prompt structure"
                     if abs(agree) > 0.8 else
                     "LOW: the framing choice drives v_MP, so treat the comparison as unsettled"))
    else:
        print("  vs persona: not computed — run phase4_build_persona.py, 01_cache_acts.py "
              "--dataset persona[_ut], then 02_extract_directions.py --kind v_mp")
    print("\n  Distinctness is geometry, not causality. It says nothing about whether the model "
          "USES v_C.")


def verdict(mag: float, other: str) -> str:
    if mag > 0.8:
        return f"largely the SAME axis as {other}. The deflationary explanation survives — " \
               "report it as a real result."
    if mag < 0.4:
        return f"largely DISTINCT from {other}; say 'distinct with modest overlap', not " \
               "'orthogonal'."
    return f"partial overlap with {other}. Report the number; do not claim clean distinctness."


if __name__ == "__main__":
    main()

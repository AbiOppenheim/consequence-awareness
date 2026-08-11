#!/usr/bin/env python
"""Stage 04 (CPU): geometry of v_C vs r_hat, v_MP, and a random null.

    python scripts/04_geometry.py --layer 14

Loads whatever directions exist in artifacts/directions/ for the chosen layer (v_c is ours;
r_hat and v_mp are dropped in by the external repos), reports all pairwise cosines against the
random-direction null band, and writes artifacts/figures/geometry_L{layer}.json.

Near-0 cosine -> distinct; near +-1 -> suspiciously the same axis (v_C may be a relabel).
"""

import argparse
import json

import _bootstrap  # noqa: F401
from consequence import directions as Dir
from consequence import io
from consequence.config import load_config, resolve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen.yaml")
    ap.add_argument("--layer", type=int, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ddir = resolve(cfg["paths"]["directions"])

    # v_c is layer-specific; r_hat / v_mp are single vectors exported once by external repos.
    candidates = {
        "v_c": ddir / f"v_c_L{args.layer}.pt",
        "r_hat": ddir / "r_hat.pt",
        "v_mp": ddir / "v_mp.pt",
    }
    loaded = {}
    for name, path in candidates.items():
        if path.exists():
            loaded[name], _ = io.load_direction(path)
        else:
            print(f"[skip] {name}: {path.name} not present yet")

    if "v_c" not in loaded:
        raise SystemExit(f"need v_c_L{args.layer}.pt — run 02_extract_directions.py first")

    cosines = Dir.cosine_matrix(loaded)
    null = Dir.random_null_band(loaded["v_c"], n_random=1000, seed=cfg["seed"])

    out = resolve(cfg["paths"]["figures"]) / f"geometry_L{args.layer}.json"
    with open(out, "w") as f:
        json.dump({"layer": args.layer, "directions": sorted(loaded),
                   "cosines": cosines, "random_null_band": null}, f, indent=2)
    print(f"[geometry] L{args.layer}: {cosines}")
    print(f"[null] |cos(random, v_c)| p95={null['abs_cos_p95']:.3f}  -> {out}")


if __name__ == "__main__":
    main()

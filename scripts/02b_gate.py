#!/usr/bin/env python
"""Phase 2 step (CPU): the r_hat gate — does OUR extraction reproduce Arditi's?

    python scripts/02b_gate.py

Reads   artifacts/activations/refusal_<model>.npz      (stage 01, GPU)
        artifacts/directions/r_hat_mean_diffs.pt       (Arditi, scripts/arditi_qwen25/run_extract.py)
Writes  artifacts/results/gate.json                    (every number below)
        artifacts/directions/r_hat_L{L}.pt + .json     (the reusable per-layer directions)

Two codebases index layers off by one — our stored k is block *output* h[k+1], Arditi's l is
block *input* h[l] — so the comparison sweeps offsets and the one that lights up to ~1 both
confirms the mapping and IS the gate. A high cosine only means something if it could have come
out low, so the red-team checks ship in the same step rather than in a cell someone might skip:

  1. cross-layer   does offset +1 actually beat the others, or is everything high?
  2. self-similarity  are our own adjacent layers already ~1.0? then the test has no power.
  3. permutation   shuffle the harmful/harmless labels; a sound pipeline COLLAPSES to noise.
  4. outliers      is the direction just a few huge residual dimensions?

The gate is near-tautological by design (both sides compute the same math on the same prompts),
so it catches gross plumbing errors — activation caching, token position, layer indexing, chat
template — and nothing else. It is a methods footnote, not a finding.
"""

import argparse

import numpy as np
import torch

import _bootstrap  # noqa: F401
from consequence import io, results
from consequence.config import load_config, resolve

NAME = "gate"


def per_layer_dirs(acts: np.ndarray, labels: np.ndarray) -> torch.Tensor:
    """Our diff-in-means at every cached layer -> [n_layers, d], unit rows."""
    v = np.stack([acts[labels == 1, k, :].mean(0) - acts[labels == 0, k, :].mean(0)
                  for k in range(acts.shape[1])])
    t = torch.tensor(v, dtype=torch.float32)
    return t / t.norm(dim=-1, keepdim=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen.yaml")
    ap.add_argument("--n-permutations", type=int, default=200,
                    help="one shuffle is not a null: this null is wide (sd ~0.4) and needs its "
                         "distribution, not a threshold")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_slug = cfg["model"]["id"].split("/")[-1]
    acts_path = resolve(cfg["paths"]["activations"]) / f"refusal_{model_slug}.npz"
    cube_path = resolve(cfg["paths"]["directions"]) / "r_hat_mean_diffs.pt"
    ddir = resolve(cfg["paths"]["directions"])

    def run() -> dict:
        z = np.load(acts_path, allow_pickle=True)
        acts, labels = z["acts"], z["labels"]
        cube = torch.load(cube_path, map_location="cpu").float()   # [n_positions, n_layers, d]

        ard = cube[-1]                                  # last position = final prompt token
        nrm = ard.norm(dim=-1, keepdim=True)
        # Arditi's layer 0 is the raw EMBEDDING, and every prompt ends with the same chat-template
        # token, so mean(harmful) - mean(harmless) is exactly 0 there. Dividing gives nan.
        dead = (nrm.squeeze(-1) < 1e-8)
        ard = ard / nrm.clamp_min(1e-12)

        our = per_layer_dirs(acts, labels)
        n = our.shape[0]
        M = our @ ard.T                                 # M[k, l] = cos(our[k], ard[l])

        offsets = {}
        for off in (-1, 0, 1, 2):
            vals = [float(M[k, k + off]) for k in range(n)
                    if 0 <= k + off < n and not dead[k + off]]
            offsets[str(off)] = {"mean_cos": float(np.mean(vals)), "n": len(vals)}
        per_layer = [{"our_layer": k, "arditi_layer": k + 1, "cos": float(M[k, k + 1])}
                     for k in range(n) if k + 1 < n and not dead[k + 1]]
        far = torch.tensor([M[k, l] for k in range(n) for l in range(n)
                            if abs((k + 1) - l) > 3 and not dead[l]])

        # 3) the decisive check. Shuffling the labels must destroy the agreement; if it does
        #    not, the agreement never depended on harmful-vs-harmless and the gate is an artifact.
        live = [k for k in range(n - 1) if not dead[k + 1]]

        def score(lbl) -> float:
            Mx = per_layer_dirs(acts, lbl) @ ard.T
            return float(np.mean([float(Mx[k, k + 1]) for k in live]))

        true_score = score(labels)
        null = np.array([score(np.random.default_rng(i).permutation(labels))
                         for i in range(args.n_permutations)])

        mid = our[n // 2].abs()
        top10 = float(mid.topk(10).values.pow(2).sum() / mid.pow(2).sum())

        g = torch.Generator().manual_seed(cfg["seed"])
        R = torch.randn(1000, our.shape[1], generator=g)
        R = R / R.norm(dim=-1, keepdim=True)
        noise_floor = float((R @ our[n // 2]).abs().quantile(0.95))

        return {
            "n_prompts": int(len(labels)),
            "n_harmful": int((labels == 1).sum()),
            "n_layers": n,
            "dead_arditi_layers": dead.nonzero().flatten().tolist(),
            "offsets": offsets,
            "per_layer_offset1": per_layer,
            "far_layer_abs_cos": float(far.abs().mean()),
            "self_similarity_adjacent": float((our @ our.T).diagonal(1).mean()),
            "permutation": {
                "n": int(args.n_permutations),
                "true": true_score,
                "null_mean": float(null.mean()),
                "null_sd": float(null.std()),
                "null_max": float(null.max()),
                "p_value": float((null >= true_score).mean()),
            },
            "top10_dim_share": top10,
            "random_direction_abs_cos_p95": noise_floor,
        }

    res = results.compute(
        NAME, run,
        inputs=[acts_path, cube_path],
        params={"n_permutations": args.n_permutations, "seed": cfg["seed"]},
        entry=__file__, force=args.force,
        results_dir=resolve(cfg["paths"]["results"]),
    )

    # The directions themselves are artifacts, not results: mint them every run (identical
    # bytes, so it stays a no-op in content) because a fresh runtime may have the cached JSON
    # restored from Drive but not the .pt files.
    cube = torch.load(cube_path, map_location="cpu").float()
    for L in cfg["extract"]["layer_sweep"]:
        io.save_direction(cube[-1][L], ddir / f"r_hat_L{L}", {
            "model_id": cfg["model"]["id"], "layer": L,
            "token_position": cfg["acts"]["token_position"],
            "source_contrast": str(cfg["data"]["refusal"]),
            "n_pairs": res["n_harmful"], "seed": cfg["seed"],
            "method": "diff_in_means via Arditi generate_directions",
            "contrast": "harmful_minus_harmless",
        })
    report(res, cfg)


def report(res: dict, cfg: dict) -> None:
    """Print from the stored result, so the console reads the same whether it just ran or not."""
    o, p = res["offsets"], res["permutation"]
    print(f"\nGATE — our r_hat vs Arditi's, {res['n_prompts']} prompts, "
          f"{res['n_layers']} layers (dead: {res['dead_arditi_layers']})")
    for off in ("-1", "0", "1", "2"):
        mark = "   <- expected alignment" if off == "1" else ""
        print(f"  offset {int(off):+d}: mean cos = {o[off]['mean_cos']:.4f}  "
              f"(n={o[off]['n']}){mark}")
    print(f"  far-apart layers (|d|>3): mean |cos| = {res['far_layer_abs_cos']:.4f}"
          "   <- if this were high too, the test would not discriminate")
    print(f"  our own adjacent layers:  mean cos = {res['self_similarity_adjacent']:.4f}"
          "   <- ~1.0 would mean the offset test has no power")
    print(f"  permutation null ({p['n']} shuffles): true {p['true']:.4f} | "
          f"null mean {p['null_mean']:.4f} sd {p['null_sd']:.4f} max {p['null_max']:.4f} | "
          f"p = {p['p_value']:.4f}")
    print(f"  top-10 dims hold {res['top10_dim_share']:.1%} of the direction "
          "  <- >50% would mean a few rogue dimensions, not a distributed feature")
    print(f"  random-direction noise floor |cos| p95 = {res['random_direction_abs_cos_p95']:.4f}")

    ok = o["1"]["mean_cos"] > 0.9 and p["p_value"] < 0.01
    print("\nVERDICT:", "PASS — extraction machinery is trustworthy; v_C extracted by the same "
          "code can be believed" if ok else
          "INVESTIGATE — permutations reach the true value, or offset +1 is not high. "
          "Suspect caching, token position, or layer indexing.")
    print(f"[mint] r_hat_L*.pt for layers {cfg['extract']['layer_sweep']}")


if __name__ == "__main__":
    main()

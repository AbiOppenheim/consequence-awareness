#!/usr/bin/env python
"""Phase 2 (POD, Arditi's venv): run Arditi's diff-in-means extraction on Qwen2.5.

Runs ONLY Arditi's `generate_directions` — the candidate cube mean(harmful) - mean(harmless)
at every layer x post-instruction token position. We SKIP their `select_direction` (it needs
generation + Qwen-specific refusal-token ids); the comparison layer is chosen on CPU, matched
to v_C. Output is the raw cube + a sidecar; the unit r_hat at a chosen layer is minted later
on CPU with our io.save_direction (which enforces the Rule-2 sidecar).

Both sides of the gate use the SAME prompts (this refusal.jsonl) and the SAME chat formatting
(the adapter mirrors acts.py), so the cosine tests the extraction machinery alone.

Run on the pod, inside Arditi's own venv, with their repo importable:

    cd external/refusal_direction && source .venv/bin/activate
    PYTHONPATH=. python ../../scripts/arditi_qwen25/run_extract.py \
        --refusal ../../data/contrast/refusal.jsonl \
        --model   Qwen/Qwen2.5-7B-Instruct \
        --out     ../../artifacts/directions/r_hat_mean_diffs
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from pipeline.submodules.generate_directions import generate_directions  # Arditi, unchanged

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen25_model import Qwen25Model  # noqa: E402  (our adapter, next to this file)


def _git_sha(repo: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refusal", required=True, help="path to refusal.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", required=True, help="output stem (writes .pt + .json)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.refusal) if line.strip()]
    harmful = [r["text"] for r in rows if int(r["label"]) == 1]
    harmless = [r["text"] for r in rows if int(r["label"]) == 0]
    print(f"[data] {len(harmful)} harmful / {len(harmless)} harmless from {args.refusal}")

    model_base = Qwen25Model(args.model)
    positions = list(range(-len(model_base.eoi_toks), 0))  # convention generate_directions uses
    cfg = model_base.model.config
    print(f"[model] {args.model}  n_layers={cfg.num_hidden_layers}  d_model={cfg.hidden_size}")
    print(f"[positions] {positions}  (index -1 = last prompt token = the gate comparison point)")

    # generate_directions also dumps its own mean_diffs.pt into the dir it's given; send that to
    # a throwaway so our named artifact is the single source of truth.
    with tempfile.TemporaryDirectory() as tmp:
        mean_diffs = generate_directions(model_base, harmful, harmless, tmp)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mean_diffs.cpu(), out.with_suffix(".pt"))

    sidecar = {
        "artifact": "arditi_mean_diffs_cube",
        "shape": list(mean_diffs.shape),  # [n_positions, n_layers, d_model]
        "positions": positions,
        "layer": "all_candidate_layers",
        "token_position": "all_eoi_positions; compare at index -1 (last prompt token)",
        "model_id": args.model,
        "source_contrast": str(args.refusal),
        "n_pairs": min(len(harmful), len(harmless)),
        "seed": args.seed,
        "method": "diff_in_means via Arditi generate_directions (unchanged)",
        "reference_repo_git_sha": _git_sha(Path.cwd()),
        "command": " ".join(sys.argv),
        "note": "raw candidate cube; unit r_hat at a chosen layer is minted on CPU with io.save_direction",
    }
    out.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, sort_keys=True))
    print(f"[write] {out.with_suffix('.pt')}  shape={list(mean_diffs.shape)}")
    print(f"[write] {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()

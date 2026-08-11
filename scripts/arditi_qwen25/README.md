# scripts/arditi_qwen25 — running Arditi's extraction on Qwen2.5

These two files are the **bridge** for the r_hat gate. They import Arditi's `pipeline.*` and so
live on the Arditi side of the boundary — our `src/consequence/` never imports them, and they
never import our package (CLAUDE.md Rule 1). They talk to us only through one artifact:
`artifacts/directions/r_hat_mean_diffs.pt` (+ sidecar).

- `qwen25_model.py` — a `ModelBase` subclass that makes Arditi's generic code work on Qwen2.5
  (blocks at `model.model.layers`, standard tokenizer, canonical chat template). Their shipped
  `QwenModel` targets the *original* Qwen and crashes on Qwen2.5.
- `run_extract.py` — feeds our `refusal.jsonl` prompts to Arditi's unchanged
  `generate_directions` and saves the difference-in-means candidate cube + a sidecar. It stops
  before `select_direction` (we pick the layer on CPU, matched to v_C).

## On the pod (one-time setup)

```bash
# 1. clone + own venv (never install into our .venv)
git clone https://github.com/andyrdt/refusal_direction external/refusal_direction
python -m venv external/refusal_direction/.venv
source external/refusal_direction/.venv/bin/activate
pip install -r external/refusal_direction/requirements.txt   # torch 2.3, transformers 4.44
# (vLLM / Together AI are NOT needed — we never run their generation or eval steps)

# 2. build our prompt file if not already present (CPU, can be done anywhere)
python scripts/phase2_build_refusal.py
```

## Run the extraction

```bash
cd external/refusal_direction && source .venv/bin/activate
PYTHONPATH=. python ../../scripts/arditi_qwen25/run_extract.py \
    --refusal ../../data/contrast/refusal.jsonl \
    --model   Qwen/Qwen2.5-7B-Instruct \
    --out     ../../artifacts/directions/r_hat_mean_diffs
```

Produces `artifacts/directions/r_hat_mean_diffs.pt` of shape `[n_positions, n_layers, d_model]`.
Back on CPU, slice it at `position = -1` and the layer matching v_C, unit-normalize, and compare
`cosine(our r_hat, this)` — that is the gate.

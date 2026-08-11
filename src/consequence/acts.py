"""Stage 01 (GPU): cache the residual stream at every layer for a list of prompts.

Mental model for the classical-ML brain: this is feature extraction. Each prompt becomes
one feature vector per layer — the layer-L residual activation at the last prompt token.
Downstream stages (probes, diff-in-means, cosines) are ordinary linear algebra on these
vectors and never touch the GPU again.

We use output_hidden_states=True — plain HuggingFace, no hooks — because caching wants ALL
layers at once. hidden_states is a tuple of length n_layers+1: index 0 is the embeddings,
index L is the output of decoder block L. We keep indices 1..n_layers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_model(model_id: str, dtype: str = "bfloat16"):
    """Load a frozen chat model + tokenizer in the requested precision.

    transformers is imported lazily so the CPU-only stages that just read/write .npz
    (save_acts/load_acts) don't require the GPU stack (Rule 3).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=DTYPES[dtype], device_map="auto"
    )
    model.eval()
    return model, tok


def _render(tok, prompt: str) -> str:
    """Wrap a raw user prompt in the model's chat template (adds the assistant turn)."""
    messages = [{"role": "user", "content": prompt}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def cache_activations(
    model, tok, prompts: list[str], token_position: int = -1, batch_size: int = 16
) -> np.ndarray:
    """Return acts of shape [n_prompts, n_layers, d_model] at one token position.

    n_layers excludes the embedding layer (we drop hidden_states[0]).
    """
    out = []
    for start in range(0, len(prompts), batch_size):
        batch = [_render(tok, p) for p in prompts[start : start + batch_size]]
        enc = tok(batch, return_tensors="pt", padding=True).to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states  # tuple len n_layers+1

        # With left padding the last real token is at index -1 for every row; if the
        # tokenizer pads right, fall back to the attention mask to find each row's last token.
        if tok.padding_side == "left" or token_position != -1:
            idx = torch.full((enc["input_ids"].shape[0],), token_position)
        else:
            idx = enc["attention_mask"].sum(dim=1) - 1  # last non-pad position per row

        rows = torch.arange(enc["input_ids"].shape[0])
        # stack layers 1..n_layers -> [n_layers, batch, d]; select token -> [n_layers, batch, d]
        per_layer = [layer[rows, idx, :].float().cpu() for layer in hs[1:]]
        out.append(torch.stack(per_layer, dim=1).numpy())  # [batch, n_layers, d]
    return np.concatenate(out, axis=0)


def save_acts(path: str | Path, acts: np.ndarray, labels, meta: dict) -> Path:
    """Cache activations + labels + provenance into a single .npz (Rule 4: artifact-keyed)."""
    path = Path(path).with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, acts=acts, labels=np.asarray(labels), **_flat(meta))
    return path


def load_acts(path: str | Path):
    """Return (acts [n, L, d], labels [n], meta dict)."""
    path = Path(path).with_suffix(".npz")
    z = np.load(path, allow_pickle=True)
    # metadata was stored 0-d with a "meta_" prefix by _flat(); strip it back off.
    meta = {
        k[len("meta_"):]: (z[k].item() if z[k].ndim == 0 else z[k])
        for k in z.files
        if k.startswith("meta_")
    }
    return z["acts"], z["labels"], meta


def _flat(meta: dict) -> dict:
    """np.savez can only store arrays/scalars — stash scalar metadata as 0-d arrays."""
    return {f"meta_{k}": np.array(v) for k, v in meta.items()}

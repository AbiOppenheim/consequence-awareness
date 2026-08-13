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


def report_gpu(min_gib: float = 20.0) -> float:
    """Print the card and its memory, and warn early when a 7-8B model will not fit.

    Colab hands out whatever accelerator is free, so a reconnect can silently move the work
    from an L4 (22.5 GiB) to a T4 (14.6 GiB). A 7-8B model in bf16 is ~15 GB of weights alone,
    which does not fit on a T4 at all: `device_map="auto"` then offloads layers to CPU without
    complaint, generation crawls, and the run dies of CUDA OOM inside scaled_dot_product_
    attention several minutes later — a traceback that says nothing about the real cause.
    Cheaper to say it here, before the weights download.
    """
    if not torch.cuda.is_available():
        print("[gpu] no CUDA device visible — this stage needs one")
        return 0.0
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / 2**30
    print(f"[gpu] {props.name}, {total:.1f} GiB")
    if total < min_gib:
        print(f"[gpu] WARNING: {total:.1f} GiB is not enough for a 7-8B model in bf16 "
              "(~15 GB of weights, before the KV cache).\n"
              "[gpu]   accelerate will offload layers to CPU: very slow, and generation will "
              "OOM at batch 16.\n"
              "[gpu]   Fix: Runtime > Change runtime type > L4. Failing that, pass a much "
              "smaller --batch-size and expect it to crawl.")
    return total


def load_model(model_id: str, dtype: str = "bfloat16"):
    """Load a frozen chat model + tokenizer in the requested precision.

    transformers is imported lazily so the CPU-only stages that just read/write .npz
    (save_acts/load_acts) don't require the GPU stack (Rule 3).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    report_gpu()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=DTYPES[dtype], device_map="auto"
    )
    model.eval()
    return model, tok


def _render(tok, prompt: str | dict) -> str:
    """Wrap a prompt in the model's chat template (adds the assistant turn).

    A prompt is either a plain string (user turn only) or {"system": ..., "user": ...} for
    datasets whose manipulation lives in the system prompt — Zhong's persona vectors are
    extracted that way. The plain-string path is byte-identical to before, so the r_hat gate
    that validated this function still applies.
    """
    if isinstance(prompt, dict):
        messages = []
        if prompt.get("system"):
            messages.append({"role": "system", "content": prompt["system"]})
        messages.append({"role": "user", "content": prompt["user"]})
    else:
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

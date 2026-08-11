"""Stage 05 (GPU): run prompts under an optional hook, write one jsonl row per generation.

Rows carry enough provenance (condition, direction name, layer window, alpha, seed) that
stage 06 can judge them and stage 07 can group them without re-deriving anything.
Output jsonl lives under artifacts/generations/ and is gitignored (may contain completions).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .acts import _render
from .hooks import apply_hooks


@torch.no_grad()
def generate(model, tok, prompts: list[str], *, max_new_tokens: int = 256,
             batch_size: int = 16, do_sample: bool = False, seed: int = 0) -> list[str]:
    """Greedy (or sampled) generation for a batch of raw prompts; returns completions only."""
    torch.manual_seed(seed)
    completions = []
    for start in range(0, len(prompts), batch_size):
        batch = [_render(tok, p) for p in prompts[start : start + batch_size]]
        enc = tok(batch, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=do_sample,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        gen = out[:, enc["input_ids"].shape[1]:]  # strip the prompt tokens
        completions.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return completions


def run_condition(model, tok, prompts: list[str], out_path: str | Path, *,
                  condition: str, direction=None, layers=None, alpha=0.0,
                  hook_factory=None, gen_kwargs: dict | None = None) -> Path:
    """Generate under one condition and append rows to a jsonl.

    condition: free label, e.g. 'baseline', 'steer_vc', 'steer_random', 'knockout_vmp'.
    hook_factory: zero-arg callable returning a hook (steer_hook(v, alpha) / knockout_hook(v)).
                  None -> no intervention (baseline).
    """
    gen_kwargs = gen_kwargs or {}
    out_path = Path(out_path).with_suffix(".jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if hook_factory is None:
        texts = generate(model, tok, prompts, **gen_kwargs)
    else:
        with apply_hooks(model, layers, hook_factory):
            texts = generate(model, tok, prompts, **gen_kwargs)

    with open(out_path, "a") as f:
        for i, (p, t) in enumerate(zip(prompts, texts)):
            row = {
                "idx": i, "condition": condition, "direction": direction,
                "layers": layers, "alpha": alpha,
                "seed": gen_kwargs.get("seed", 0),
                "prompt": p, "completion": t,
            }
            f.write(json.dumps(row) + "\n")
    return out_path

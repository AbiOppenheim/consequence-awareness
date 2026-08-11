"""Forward-hook interventions: steering (activation addition) and projection knockout.

A PyTorch forward hook is just a callback that fires after a module runs and may replace its
output. HuggingFace decoder blocks return a tuple whose first element is the residual stream;
we edit element 0 and pass the rest through unchanged.

These 2 interventions ARE the causal test. Both must always be run against a random-direction
null in the same sweep (CLAUDE.md section 2).
"""

from __future__ import annotations

from contextlib import contextmanager

import torch


def steer_hook(v: torch.Tensor, alpha: float):
    """h <- h + alpha * v   (add at all positions in the block's output).

    alpha > 0 with v = v_C oriented toward 'real' is the causal test: does refusal return
    during a fiction-framed jailbreak? v is expected unit-normalized; alpha carries the scale.
    """
    def hook(module, inputs, output):
        resid = output[0]
        resid = resid + alpha * v.to(resid.dtype).to(resid.device)
        return (resid, *output[1:])
    return hook


def knockout_hook(v: torch.Tensor):
    """h <- h - (h . v_hat) v_hat   (project the direction out everywhere).

    Distinctness test: if knocking out v_C behaves like knocking out v_MP, they're plausibly
    the same mechanism; if not, they're distinct.
    """
    v_hat = v.float() / v.float().norm()

    def hook(module, inputs, output):
        resid = output[0]
        vh = v_hat.to(resid.dtype).to(resid.device)
        proj = (resid @ vh).unsqueeze(-1) * vh
        return (resid - proj, *output[1:])
    return hook


@contextmanager
def apply_hooks(model, layers: list[int], make_hook):
    """Attach make_hook() to model.model.layers[L] for each L, then clean up on exit.

    make_hook is a zero-arg factory returning a fresh hook (so each layer gets its own).
    Usage:
        with apply_hooks(model, [12, 14, 16], lambda: steer_hook(v_c, alpha)):
            model.generate(...)
    """
    handles = []
    try:
        blocks = model.model.layers  # Qwen2 / Llama layout
        for L in layers:
            handles.append(blocks[L].register_forward_hook(make_hook()))
        yield
    finally:
        for h in handles:
            h.remove()

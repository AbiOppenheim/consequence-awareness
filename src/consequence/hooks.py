"""Forward-hook interventions: steering (activation addition) and projection knockout.

A PyTorch forward hook is just a callback that fires after a module runs and may replace its
output. We edit the residual stream in a decoder block's output and pass anything else through.

These 2 interventions ARE the causal test. Both must always be run against a random-direction
null in the same sweep (CLAUDE.md section 2).
"""

from __future__ import annotations

from contextlib import contextmanager

import torch


def _edit_resid(output, fn):
    """Apply fn to the residual stream in a decoder block's output, preserving its container.

    HuggingFace decoder blocks USED to return `(hidden_states, ...)` and now return a bare
    tensor (transformers ~4.54+). Both conventions are in the wild, so handle both.

    This has to be an isinstance check rather than a try/except, because the failure is silent
    in the dangerous direction: `output[0]` on a TENSOR does not raise — it returns the first
    row of the batch. Assuming the tuple form against the new API therefore steers a
    wrong-shaped slice and hands the next layer a tuple, which is how this surfaced: an
    AttributeError three layers downstream in Qwen2's input_layernorm, nowhere near the hook.
    """
    if isinstance(output, tuple):
        return (fn(output[0]), *output[1:])
    return fn(output)


def steer_hook(v: torch.Tensor, alpha: float):
    """h <- h + alpha * v   (add at all positions in the block's output).

    alpha > 0 with v = v_C oriented toward 'real' is the causal test: does refusal return
    during a fiction-framed jailbreak? v is expected unit-normalized; alpha carries the scale.
    """
    def hook(module, inputs, output):
        def add(resid):
            return resid + alpha * v.to(resid.dtype).to(resid.device)
        return _edit_resid(output, add)
    return hook


def knockout_hook(v: torch.Tensor):
    """h <- h - (h . v_hat) v_hat   (project the direction out everywhere).

    Distinctness test: if knocking out v_C behaves like knocking out v_MP, they're plausibly
    the same mechanism; if not, they're distinct.
    """
    v_hat = v.float() / v.float().norm()

    def hook(module, inputs, output):
        def project_out(resid):
            vh = v_hat.to(resid.dtype).to(resid.device)
            return resid - (resid @ vh).unsqueeze(-1) * vh
        return _edit_resid(output, project_out)
    return hook


@contextmanager
def apply_hooks(model, layers: list[int], make_hook):
    """Intervene at `hidden_states[L]` for each config layer L, then clean up on exit.

    LAYER CONVENTION — this is an off-by-one trap, and it must match extraction or we would
    steer along a direction measured somewhere else:

        config layer L  ==  hidden_states[L]  ==  activation cache index L-1
                        ==  OUTPUT of block L-1  ==  INPUT of block L

    A post-forward hook on `blocks[i]` rewrites the OUTPUT of block i, i.e. hidden_states[i+1].
    So to write hidden_states[L] we hook `blocks[L-1]`, not `blocks[L]`. (Hooking blocks[L]
    would intervene one layer downstream of where v_C was extracted.) Verified by test.

    make_hook is a zero-arg factory returning a fresh hook (so each layer gets its own).
    Usage:
        with apply_hooks(model, [12, 14, 16], lambda: steer_hook(v_c, alpha)):
            model.generate(...)
    """
    handles = []
    try:
        blocks = model.model.layers  # Qwen2 / Llama layout
        for L in layers:
            if not 1 <= L <= len(blocks):
                raise ValueError(f"layer {L} out of range for {len(blocks)} blocks "
                                 "(config layers are 1-indexed: L means hidden_states[L])")
            handles.append(blocks[L - 1].register_forward_hook(make_hook()))
        yield
    finally:
        for h in handles:
            h.remove()

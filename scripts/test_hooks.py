#!/usr/bin/env python
"""Regression test for the steering/knockout hooks. CPU, no model download.

    python scripts/test_hooks.py

Two bugs have now shipped in `hooks.py`, both invisible without a real forward pass:

  1. it hooked `blocks[L]`, steering one layer downstream of where v_C was measured;
  2. it assumed decoder blocks return `(hidden_states, ...)`. Current transformers returns a
     bare tensor, so `output[0]` silently took the first BATCH ROW and the tuple it returned
     blew up three layers later inside Qwen2's input_layernorm.

Both were caught by generation, not by the stub test that existed at the time — because that
stub returned tuples, i.e. it tested the assumption instead of challenging it. So this runs the
whole thing against BOTH block conventions, and asserts on the layer that actually changed
rather than on the hook being called.
"""

import sys

import torch
from torch import nn

import _bootstrap  # noqa: F401
from consequence.hooks import apply_hooks, knockout_hook, steer_hook

D, SEQ, BATCH, N_BLOCKS = 8, 3, 2, 6


class Block(nn.Module):
    """A decoder block stand-in: a deterministic shift, so a change is traceable to one layer."""

    def __init__(self, as_tuple: bool):
        super().__init__()
        self.as_tuple = as_tuple

    def forward(self, x):
        out = x + 1.0
        # The two HuggingFace conventions this code has to survive.
        return (out, None) if self.as_tuple else out


class Stack(nn.Module):
    """Mimics `model.model.layers`, the attribute apply_hooks reaches for."""

    def __init__(self, as_tuple: bool):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Block(as_tuple) for _ in range(N_BLOCKS)])

    def forward(self, x):
        """Return hidden_states[0..n], the same indexing the activation cache uses."""
        hs = [x]
        for block in self.model.layers:
            out = block(hs[-1])
            hs.append(out[0] if isinstance(out, tuple) else out)
        return hs


def check(name: str, ok: bool) -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return ok


def run(as_tuple: bool) -> bool:
    kind = "tuple-returning blocks (old HF)" if as_tuple else "tensor-returning blocks (new HF)"
    print(f"\n{kind}")
    torch.manual_seed(0)
    model = Stack(as_tuple)
    x = torch.randn(BATCH, SEQ, D)
    v = torch.randn(D)
    v = v / v.norm()
    alpha, L = 3.0, 3
    ok = True

    base = model(x)

    with apply_hooks(model, [L], lambda: steer_hook(v, alpha)):
        steered = model(x)

    # The layer convention: config layer L == hidden_states[L]. Steering at L must leave
    # hidden_states[L-1] untouched and shift hidden_states[L] by exactly alpha * v.
    ok &= check(f"hidden_states[{L-1}] unchanged (nothing steered upstream)",
                torch.allclose(base[L - 1], steered[L - 1]))
    ok &= check(f"hidden_states[{L}] shifted by exactly alpha*v",
                torch.allclose(steered[L] - base[L],
                               (alpha * v).expand(BATCH, SEQ, D), atol=1e-5))
    ok &= check("the shift is identical across batch rows (not one row edited)",
                torch.allclose(steered[L][0] - base[L][0], steered[L][1] - base[L][1]))
    ok &= check(f"hidden_states[{L}] keeps its shape {tuple(base[L].shape)}",
                steered[L].shape == base[L].shape)

    # A block's output container must survive the hook, or the next block gets the wrong type —
    # the failure that produced 'tuple' object has no attribute 'dtype'.
    raw = model.model.layers[L - 1](base[L - 1])
    ok &= check("block output container preserved through the hook",
                isinstance(raw, tuple) == as_tuple)

    with apply_hooks(model, [L], lambda: knockout_hook(v)):
        knocked = model(x)
    ok &= check("knockout removes the component along v",
                torch.allclose(knocked[L] @ v, torch.zeros(BATCH, SEQ), atol=1e-5))

    ok &= check("hooks removed on context exit",
                torch.allclose(model(x)[L], base[L]))

    try:
        with apply_hooks(model, [N_BLOCKS + 1], lambda: steer_hook(v, alpha)):
            pass
        ok &= check("out-of-range layer rejected", False)
    except ValueError:
        ok &= check("out-of-range layer rejected", True)
    return ok


if __name__ == "__main__":
    passed = all([run(as_tuple=True), run(as_tuple=False)])
    print("\n" + ("ALL PASS — steering hits the layer v_C was measured at, under both "
                  "HF block conventions" if passed else "FAILURES ABOVE — do not run the sweep"))
    sys.exit(0 if passed else 1)

"""Core primitives: extract, steer, ablate, compare directions.

Faithful to docs/tutorial.md §3.5. The heavy lifting (model loading, hook plumbing,
refusal scoring) is reused from the forked repos; these are the project's ~15% of
novel, surgical code.
"""

from __future__ import annotations

import torch


def extract_direction(
    acts_pos: torch.Tensor,  # [n_pos, d_model] residual-stream acts, last token, layer L
    acts_neg: torch.Tensor,  # [n_neg, d_model]
) -> torch.Tensor:
    """Difference-in-means direction, unit-normalized.

    v = mean(positive) - mean(negative). Used identically for r_hat (harmful vs
    harmless), v_MP (compliant vs evasive persona), and v_C (real vs hypothetical).
    """
    v = acts_pos.mean(dim=0) - acts_neg.mean(dim=0)
    return v / v.norm()


def steer_hook(v: torch.Tensor, alpha: float):
    """Activation addition: h <- h + alpha * v at post-prompt positions.

    alpha > 0 with v = v_C oriented toward 'real' is the causal test:
    does refusal come back during a fiction-framed jailbreak?
    """

    def hook(resid: torch.Tensor) -> torch.Tensor:
        return resid + alpha * v.to(resid.dtype)

    return hook


def knockout_hook(v: torch.Tensor):
    """Projection knockout: h <- h - (h . v_hat) v_hat  (directional ablation).

    Distinctness test: if knocking out v_C behaves interchangeably with knocking
    out v_MP, they're plausibly the same mechanism.
    """
    v_hat = v / v.norm()

    def hook(resid: torch.Tensor) -> torch.Tensor:
        proj = (resid @ v_hat).unsqueeze(-1) * v_hat.to(resid.dtype)
        return resid - proj

    return hook


def cosine_table(directions: dict[str, torch.Tensor], n_random: int = 100) -> dict:
    """Pairwise cosines among named directions + a random-direction null band."""
    names = list(directions)
    d = directions[names[0]].shape[-1]
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            out[f"cos({a},{b})"] = float(
                torch.dot(directions[a], directions[b])
                / (directions[a].norm() * directions[b].norm())
            )
    rand = torch.randn(n_random, d)
    rand = rand / rand.norm(dim=-1, keepdim=True)
    ref = directions[names[0]] / directions[names[0]].norm()
    null = rand @ ref
    out["random_null_abs_p95"] = float(null.abs().quantile(0.95))
    return out

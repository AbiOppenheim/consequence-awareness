"""Difference-in-means extraction, random-direction nulls, and geometry.

Every direction in this project is the same object: a unit vector in R^d_model. The math is
five lines; the discipline is in the sidecar (io.save_direction) and the controls.
"""

from __future__ import annotations

import numpy as np
import torch


def diff_in_means(acts_pos: np.ndarray | torch.Tensor,
                  acts_neg: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Unit direction  v = mean(pos) - mean(neg).

    Used identically for r_hat (harmful vs harmless), v_MP (compliant vs evasive persona),
    and v_C (real vs hypothetical). acts_* are [n, d_model] at ONE layer/token position.
    """
    pos = torch.as_tensor(np.asarray(acts_pos), dtype=torch.float32)
    neg = torch.as_tensor(np.asarray(acts_neg), dtype=torch.float32)
    v = pos.mean(dim=0) - neg.mean(dim=0)
    return v / v.norm()


def random_direction(d_model: int, seed: int) -> torch.Tensor:
    """A reproducible random unit vector — the null every intervention must be run against."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(d_model, generator=g)
    return v / v.norm()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


def cosine_matrix(directions: dict[str, torch.Tensor]) -> dict[str, float]:
    """All pairwise cosines among named directions (upper triangle)."""
    names = list(directions)
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            out[f"cos({a},{b})"] = cosine(directions[a], directions[b])
    return out


def random_null_band(reference: torch.Tensor, n_random: int = 1000,
                     seed: int = 0) -> dict[str, float]:
    """Distribution of |cos(random, reference)| — the noise floor for 'distinct enough'.

    A measured cosine inside this band is indistinguishable from chance for this d_model.
    """
    d = reference.shape[-1]
    g = torch.Generator().manual_seed(seed)
    rand = torch.randn(n_random, d, generator=g)
    rand = rand / rand.norm(dim=-1, keepdim=True)
    ref = reference.float() / reference.float().norm()
    cos = (rand @ ref).abs()
    return {
        "n_random": n_random,
        "abs_cos_mean": float(cos.mean()),
        "abs_cos_p95": float(cos.quantile(0.95)),
        "abs_cos_max": float(cos.max()),
    }

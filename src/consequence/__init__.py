"""consequence — probing and steering a consequence-awareness direction.

The whole project treats every direction (r_hat, v_MP, v_C, random) as the same object:
a unit vector in R^d_model. Modules here never import anything under external/ (CLAUDE.md
Rule 1); reference repos communicate only by writing a .pt into artifacts/directions/.
"""

__all__ = ["config", "io", "acts", "directions", "hooks", "generate", "probe", "judge"]

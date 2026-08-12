"""Save/load directions with a mandatory sidecar (CLAUDE.md Rule 2).

A direction is a unit vector in R^d_model. On disk it is two files with the same stem:
    <name>.pt    the tensor
    <name>.json  provenance: model, layer, token position, source contrast, n_pairs,
                 seed, git SHA, and the command that produced it.

A .pt with no sidecar is deleted, not debugged. save_direction() makes it hard to create one.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

REQUIRED_META = {
    "model_id",
    "layer",
    "token_position",
    "source_contrast",
    "n_pairs",
    "seed",
}


def git_sha() -> str:
    """Short SHA of HEAD, or 'unknown' outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def save_direction(vec: torch.Tensor, path: str | Path, meta: dict[str, Any]) -> Path:
    """Write <path>.pt (unit-normalized) plus <path>.json. Refuses incomplete metadata."""
    missing = REQUIRED_META - set(meta)
    if missing:
        raise ValueError(f"sidecar metadata missing required keys: {sorted(missing)}")

    path = Path(path).with_suffix(".pt")
    path.parent.mkdir(parents=True, exist_ok=True)

    vec = vec.detach().float().cpu()
    vec = vec / vec.norm()
    torch.save(vec, path)

    sidecar = dict(meta)
    sidecar.setdefault("git_sha", git_sha())
    sidecar.setdefault("command", " ".join(sys.argv))
    sidecar["d_model"] = int(vec.shape[-1])
    sidecar["pt_file"] = path.name
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(sidecar, f, indent=2, sort_keys=True)
    return path


def load_direction(path: str | Path) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load a direction and its sidecar. Errors loudly if the sidecar is missing."""
    path = Path(path).with_suffix(".pt")
    sidecar_path = path.with_suffix(".json")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist — the stage that mints it has not been run"
        )
    if not sidecar_path.exists():
        raise FileNotFoundError(
            f"no sidecar for {path.name} — per Rule 2 this .pt should be deleted, not used"
        )
    vec = torch.load(path, map_location="cpu")
    with open(sidecar_path) as f:
        meta = json.load(f)
    return vec, meta

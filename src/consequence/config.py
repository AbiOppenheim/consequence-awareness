"""Load configs/qwen.yaml into a plain dict with dotted-path access.

Deliberately tiny: no schema framework. A config is just the experiment's fixed knobs,
and we want it readable in one screen and dumpable verbatim into every sidecar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Repo root = two levels up from this file (src/consequence/config.py -> repo/).
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "configs/qwen.yaml") -> dict[str, Any]:
    """Read a YAML config. Relative paths resolve against the repo root."""
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg


def resolve(path: str | Path) -> Path:
    """Turn a repo-relative path (as written in the config) into an absolute Path."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path

"""Durable, input-keyed storage for the CPU analysis steps (CLAUDE.md Rule 4).

Why this exists. The expensive analyses — group-CV over nine layers, the 200-shuffle
permutation null, the 100-draw random null, the TF-IDF baselines — used to live inside
notebook cells, so their output existed only in the kernel. A Colab disconnect recomputed all
of it; re-running a cell recomputed it again; and the numbers that go into the write-up had no
file to point at.

A result is a small JSON file, ``artifacts/results/<name>.json``::

    {"result": {...},
     "_meta": {"fingerprint": ..., "inputs": {...}, "code": {...}, "params": {...},
               "git_sha": ..., "written": ...}}

The fingerprint covers everything the number depends on: the content hash of every input
artifact, the parameters, and the analysis code that ran. ``compute()`` recomputes only when
that fingerprint has moved. So re-running a step with nothing changed is a no-op, and a step
whose dataset or code changed underneath it recomputes instead of quietly serving a stale
number. The failure mode this guards against is not slowness — it is reporting a figure that
no longer corresponds to its inputs, which is exactly what happened when v_C was re-extracted
train-only but the held-out AUC printed in the notebook still came from the older vector.

For the classical-ML brain: this is the same idea as caching a fitted transformer keyed on a
hash of (training data, hyperparameters, code version). Nothing more exotic.

**Scope.** The CPU analysis layer only. The two GPU stages keep their own artifact-keyed
caches — 01 keys the ``.npz`` on the dataset hash, 05 resumes per (condition, alpha) — and
must not be invalidated by an edit to an unrelated source file. Never wrap a GPU stage in
``compute()``.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .config import REPO_ROOT
from .io import git_sha

DEFAULT_DIR = REPO_ROOT / "artifacts" / "results"
_PKG_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------- fingerprinting

def file_sha(path: str | Path) -> str:
    """Content hash of one input artifact.

    Chunked because the activation caches are ~375 MB; hashing one costs about a second,
    which is the right price for never silently reusing a result built from other data.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 22):
            h.update(block)
    return h.hexdigest()[:16]


def relname(path: str | Path) -> str:
    """Repo-relative path, used as the fingerprint key.

    Absolute paths differ between this laptop and a Colab runtime (`/content/...`), so keying
    on them would invalidate every cached result the moment the work moved machines.
    """
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return p.name


def _code_files(entry: str | Path | None) -> dict[str, str]:
    """Hashes of the analysis code a result depends on.

    The entry script, plus every ``consequence.*`` module imported at the time of the call.
    Fixing a bug in an analysis has to invalidate that analysis's stored number, and doing
    that by hand is the kind of thing one remembers nine times out of ten.
    """
    files: dict[str, str] = {}
    if entry is not None:
        p = Path(entry).resolve()
        files[p.name] = file_sha(p)
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if f and Path(f).resolve().parent == _PKG_DIR:
            files[f"consequence/{Path(f).name}"] = file_sha(f)
    return dict(sorted(files.items()))


def fingerprint(inputs: Iterable[str | Path] = (), params: dict | None = None,
                entry: str | Path | None = None) -> dict[str, Any]:
    """Everything this result depends on, hashed. Missing inputs fail loudly."""
    resolved: dict[str, str] = {}
    for p in inputs:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(
                f"input artifact missing: {relname(p)} — the stage that writes it has not run"
            )
        resolved[relname(p)] = file_sha(p)

    params_json = json.dumps(params or {}, sort_keys=True, default=str)
    code = _code_files(entry)
    blob = json.dumps({"inputs": dict(sorted(resolved.items())), "params": params_json,
                       "code": code}, sort_keys=True)
    return {
        "inputs": dict(sorted(resolved.items())),
        "params": params or {},
        "code": code,
        "sha": hashlib.sha256(blob.encode()).hexdigest()[:16],
    }


# ---------------------------------------------------------------- store

def path_for(name: str, results_dir: str | Path | None = None) -> Path:
    return Path(results_dir or DEFAULT_DIR) / f"{name}.json"


def exists(name: str, results_dir: str | Path | None = None) -> bool:
    return path_for(name, results_dir).exists()


def save(name: str, result: Any, *, inputs: Iterable[str | Path] = (),
         params: dict | None = None, entry: str | Path | None = None,
         results_dir: str | Path | None = None, archive: bool = False) -> Path:
    """Write ``<name>.json`` with its fingerprint sidecar built in.

    ``archive=True`` copies an existing result aside as ``<name>.prev-<sha>.json`` instead of
    letting it be overwritten. Use it for the one-shot held-out reveal: if that number is ever
    recomputed after something changed, both versions must survive so the second can be
    reported and labelled post-hoc (CLAUDE.md section 2).
    """
    out = path_for(name, results_dir)
    out.parent.mkdir(parents=True, exist_ok=True)

    fp = fingerprint(inputs, params, entry)
    new_result = _jsonable(result)
    if archive and out.exists():
        old = json.loads(out.read_text())
        old_sha = old.get("_meta", {}).get("fingerprint", {}).get("sha", "unknown")
        # Archive on a changed NUMBER, never on a changed hash. The fingerprint covers every
        # consequence.* module the step imported, so an edit anywhere in that surface — adding
        # a GPU check to acts.py, say — invalidates the result and forces a recompute that
        # lands on exactly the same value. Treating that as a second look at held-out data
        # cries wolf on the one alarm in this project that must never be ignored.
        if old_sha != fp["sha"] and old.get("result") != new_result:
            kept = out.with_name(f"{name}.prev-{old_sha}.json")
            kept.write_text(out.read_text())
            print(f"[results] !! {name} was recomputed and the RESULT CHANGED.\n"
                  f"[results]    previous version kept at {relname(kept)}.\n"
                  f"[results]    Report BOTH numbers and label this one post-hoc.")
        elif old_sha != fp["sha"]:
            print(f"[results] {name}: recomputed after an input/code change — result identical, "
                  "nothing archived")

    payload = {
        "result": new_result,
        "_meta": {
            "name": name,
            "fingerprint": fp,
            "git_sha": git_sha(),
            "command": " ".join(sys.argv),
            "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=False))
    return out


def load(name: str, results_dir: str | Path | None = None) -> tuple[Any, dict]:
    """Return (result, meta). Errors with the step to run when nothing is stored."""
    p = path_for(name, results_dir)
    if not p.exists():
        raise FileNotFoundError(
            f"no stored result '{name}' ({relname(p)}). Run the step that produces it — "
            f"see the table in README.md — then re-run this."
        )
    blob = json.loads(p.read_text())
    return blob["result"], blob["_meta"]


def status(name: str, *, inputs: Iterable[str | Path] = (), params: dict | None = None,
           entry: str | Path | None = None,
           results_dir: str | Path | None = None) -> tuple[bool, str]:
    """(is_fresh, reason). The reason names *what* moved, so a recompute is never mysterious."""
    p = path_for(name, results_dir)
    if not p.exists():
        return False, "no stored result yet"

    stored = json.loads(p.read_text()).get("_meta", {}).get("fingerprint", {})
    current = fingerprint(inputs, params, entry)
    if stored.get("sha") == current["sha"]:
        return True, "inputs, params and code unchanged"

    for label in ("inputs", "code"):
        was, now = stored.get(label, {}), current[label]
        changed = sorted(k for k in set(was) | set(now) if was.get(k) != now.get(k))
        if changed:
            return False, f"{label} changed: {', '.join(changed)}"
    if stored.get("params") != current["params"]:
        return False, "params changed"
    return False, "fingerprint changed"


def compute(name: str, fn: Callable[[], Any], *, inputs: Iterable[str | Path] = (),
            params: dict | None = None, entry: str | Path | None = None,
            force: bool = False, archive: bool = False,
            results_dir: str | Path | None = None) -> Any:
    """Return the stored result if it is fresh, else run ``fn()`` and store what it returns.

    This is the whole point of the module: an expensive step is written once as ``fn`` and
    every later caller — a re-run, another script, a notebook cell — gets the saved numbers
    without paying for them again.
    """
    fresh, reason = status(name, inputs=inputs, params=params, entry=entry,
                           results_dir=results_dir)
    if fresh and not force:
        result, meta = load(name, results_dir)
        print(f"[results] {name}: reusing {relname(path_for(name, results_dir))} "
              f"(written {meta['written']}) — {reason}")
        return result

    why = "--force" if force else reason
    print(f"[results] {name}: computing ({why})")
    result = fn()
    out = save(name, result, inputs=inputs, params=params, entry=entry,
               results_dir=results_dir, archive=archive)
    print(f"[results] {name}: wrote {relname(out)}")
    return result


# ---------------------------------------------------------------- pipeline glue

def selected_layer(results_dir: str | Path | None = None) -> tuple[int, Path]:
    """The layer step 03 chose, plus the file it came from.

    Every downstream step (held-out reveal, red-team, alpha calibration, the sweep) reads the
    layer from here instead of taking it as a hand-typed argument or a notebook variable. Pass
    the returned path in as an input: re-running the selection then invalidates everything that
    depended on the old choice, which is the whole reason the number lives in a file.
    """
    p = path_for("layer_select", results_dir)
    res, _ = load("layer_select", results_dir)
    return int(res["best_layer"]), p


# ---------------------------------------------------------------- JSON coercion

def _jsonable(obj: Any) -> Any:
    """numpy/torch scalars and arrays -> plain JSON types.

    Rejects non-string dict keys on purpose: JSON turns ``8`` into ``"8"``, so an int-keyed
    per-layer dict does not survive a round-trip and quietly changes type between the run that
    wrote it and the notebook that reads it.
    """
    if isinstance(obj, dict):
        bad = [k for k in obj if not isinstance(k, str)]
        if bad:
            raise TypeError(
                f"result has non-string dict keys {bad!r}. JSON would return them as strings; "
                "store a list of records instead, e.g. [{'layer': 8, 'auc': 0.93}, ...]."
            )
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "detach") and hasattr(obj, "tolist"):   # torch.Tensor, without importing torch
        return _jsonable(obj.detach().cpu().tolist())
    return obj

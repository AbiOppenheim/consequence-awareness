"""Stage 03 (CPU): per-layer linear probes on cached residual activations.

Key discipline (Kirch et al. 2025; CLAUDE.md section 2): evaluate on HELD-OUT framing
TEMPLATES, never merely held-out examples. Splitting examples within the same templates
measures memorization of surface vocabulary, not a consequence representation. The held-out-
template number is the one that matters; the train-template number is a sanity check.

For the classical-ML brain: this is exactly a logistic regression on frozen features, with a
group-wise train/test split where the groups are templates.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def train_probe(acts: np.ndarray, labels: np.ndarray, seed: int = 0) -> LogisticRegression:
    """acts: [n, d_model] at one layer; labels: 1=real, 0=hypothetical."""
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed, class_weight="balanced")
    clf.fit(acts, labels)
    return clf


def evaluate_probe(clf, acts: np.ndarray, labels: np.ndarray) -> float:
    return float(clf.score(acts, labels))


def layerwise_accuracy(
    acts_by_layer: dict[int, np.ndarray],
    labels: np.ndarray,
    template_ids: np.ndarray,
    heldout_templates: set,
    seed: int = 0,
) -> dict[int, dict[str, float]]:
    """Train on TRAIN templates only; report train-template and held-out-template accuracy
    per layer. Returns {layer: {'train_templates_acc', 'heldout_templates_acc', 'n_train',
    'n_heldout'}}."""
    heldout_mask = np.isin(template_ids, list(heldout_templates))
    train_mask = ~heldout_mask
    if train_mask.sum() == 0 or heldout_mask.sum() == 0:
        raise ValueError("empty train or held-out split — check template_ids / heldout set")

    results = {}
    for layer, acts in acts_by_layer.items():
        clf = train_probe(acts[train_mask], labels[train_mask], seed=seed)
        results[layer] = {
            "train_templates_acc": evaluate_probe(clf, acts[train_mask], labels[train_mask]),
            "heldout_templates_acc": evaluate_probe(clf, acts[heldout_mask], labels[heldout_mask]),
            "n_train": int(train_mask.sum()),
            "n_heldout": int(heldout_mask.sum()),
        }
    return results

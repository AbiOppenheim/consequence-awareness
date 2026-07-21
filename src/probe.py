"""Per-layer linear probes for the consequence contrast.

Key discipline (Kirch et al. 2025 caution, docs/tutorial.md §2.9): the probe must be
evaluated on HELD-OUT framing templates, not just held-out examples, or high accuracy
may reflect surface vocabulary rather than a consequence representation.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def train_probe(acts: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    """acts: [n, d_model] residual acts at one layer; labels: 1=real, 0=hypothetical."""
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(acts, labels)
    return clf


def evaluate_probe(clf, acts: np.ndarray, labels: np.ndarray) -> float:
    return float(clf.score(acts, labels))


def layerwise_accuracy(
    acts_by_layer: dict[int, np.ndarray],
    labels: np.ndarray,
    template_ids: np.ndarray,
    heldout_templates: set[int],
) -> dict[int, dict[str, float]]:
    """Train on TRAIN templates, report in-distribution and held-out-template accuracy
    per layer. The held-out number is the one that matters."""
    train_mask = ~np.isin(template_ids, list(heldout_templates))
    results = {}
    for layer, acts in acts_by_layer.items():
        clf = train_probe(acts[train_mask], labels[train_mask])
        results[layer] = {
            "train_templates_acc": evaluate_probe(clf, acts[train_mask], labels[train_mask]),
            "heldout_templates_acc": evaluate_probe(clf, acts[~train_mask], labels[~train_mask]),
        }
    return results

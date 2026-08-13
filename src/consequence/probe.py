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


def group_cv_auc(acts: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                 n_splits: int = 5, seed: int = 0) -> float:
    """Mean AUC of a group-wise CV *inside the training templates*.

    The groups are templates, so every fold validates on templates the probe did not train on —
    a miniature of the real held-out test. This is what the layer should be selected on:
    choosing the layer that maximises HELD-OUT accuracy is selecting on the test set, which
    inflates the number you then report (CLAUDE.md section 2).
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    scores = []
    for tr, te in GroupKFold(n_splits=n_splits).split(acts, labels, groups=groups):
        clf = train_probe(acts[tr], labels[tr], seed=seed)
        scores.append(roc_auc_score(labels[te], clf.decision_function(acts[te])))
    return float(np.mean(scores))


def fit_and_score(acts_train: np.ndarray, labels_train: np.ndarray,
                  acts_test: np.ndarray, labels_test: np.ndarray,
                  seed: int = 0) -> dict[str, float]:
    """Fit on train templates, report AUC + accuracy on held-out templates.

    One implementation shared by the held-out reveal (step 09) and its per-layer red-team
    (step 10), so the headline number and the check on it cannot drift apart.
    """
    from sklearn.metrics import roc_auc_score

    clf = train_probe(acts_train, labels_train, seed=seed)
    return {
        "auc": float(roc_auc_score(labels_test, clf.decision_function(acts_test))),
        "acc": float(clf.score(acts_test, labels_test)),
    }


def projection_auc(acts: np.ndarray, labels: np.ndarray, direction) -> float:
    """AUC of the raw projection h . v — no fitting, so it tests the direction itself.

    The stronger of the two readouts: a fitted probe can find the concept in the activations,
    but the unfitted projection asks whether the vector we extracted *is* that concept.
    """
    from sklearn.metrics import roc_auc_score

    v = np.asarray(direction, dtype=np.float32).reshape(-1)
    return float(roc_auc_score(labels, acts @ v))

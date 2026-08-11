#!/usr/bin/env python
"""Mandatory audit of data/contrast/consequence.jsonl (build-contrast-set skill).

Any failure BLOCKS direction extraction. Four checks:

  1. Task-match      each row must contain its task string verbatim        (byte compare)
  2. Vocab leakage   bag-of-words logistic regression on PROMPT TEXT,      (AUC > 0.9 = fail)
                     trained on train templates, scored on held-out templates
  3. Length confound mean word-length difference between real/hypo classes (> 10% = fail)
  4. Template balance rows per framing (template)                          (< 8 = fail)

The vocab-leakage check is the important one and it is your background: a bag-of-words baseline.
If BoW separates the classes on UNSEEN templates, the classes differ in surface lexicon that
generalizes, and a linear probe on activations proves nothing beyond it. Report this AUC next to
the probe accuracy in the write-up — it is the honest baseline the probe must beat.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
CONSEQUENCE = REPO / "data" / "contrast" / "consequence.jsonl"
TASKS = REPO / "data" / "contrast" / "benign_tasks.jsonl"

LEAKAGE_AUC_MAX = 0.90
LENGTH_PCT_MAX = 10.0
MIN_ROWS_PER_TEMPLATE = 8


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    rows = load_jsonl(CONSEQUENCE)
    task_text = {t["task_id"]: t["text"] for t in load_jsonl(TASKS)}
    y = np.array([1 if r["framing"] == "real" else 0 for r in rows])
    split = np.array([r["split"] for r in rows])
    text = np.array([r["text"] for r in rows], dtype=object)
    fails = []

    # --- 1. task-match -----------------------------------------------------
    mismatch = [r["id"] for r in rows if task_text[r["task_id"]] not in r["text"]]
    ok1 = not mismatch
    print(f"[1] task-match         : {'PASS' if ok1 else 'FAIL'}  "
          f"({len(mismatch)} rows missing their task string)")
    if not ok1:
        fails.append("task-match")

    # --- 2. vocab leakage (bag-of-words) -----------------------------------
    tr, ho = split == "train", split == "heldout"
    vec = CountVectorizer()  # vocab from TRAIN only — held-out-only words are unseen, as intended
    Xtr = vec.fit_transform([str(t) for t in text[tr]])
    Xho = vec.transform([str(t) for t in text[ho]])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, y[tr])
    auc_tr = roc_auc_score(y[tr], clf.predict_proba(Xtr)[:, 1])
    auc_ho = roc_auc_score(y[ho], clf.predict_proba(Xho)[:, 1])
    ok2 = auc_ho <= LEAKAGE_AUC_MAX
    print(f"[2] vocab leakage (BoW): {'PASS' if ok2 else 'FAIL'}  "
          f"held-out AUC={auc_ho:.3f} (fail>{LEAKAGE_AUC_MAX})  |  train AUC={auc_tr:.3f}")
    # diagnose: which shared words drive the held-out prediction?
    vocab = np.array(vec.get_feature_names_out())
    coef = clf.coef_[0]
    ho_present = np.asarray(Xho.sum(axis=0)).ravel() > 0  # words that actually occur in held-out
    idx = np.where(ho_present)[0]
    top_real = idx[np.argsort(coef[idx])[::-1][:6]]
    top_hypo = idx[np.argsort(coef[idx])[:6]]
    print(f"      shared words pushing -> real: {list(vocab[top_real])}")
    print(f"      shared words pushing -> hypo: {list(vocab[top_hypo])}")
    if not ok2:
        fails.append("vocab-leakage")

    # --- 3. length confound ------------------------------------------------
    wl = np.array([len(str(t).split()) for t in text])
    m_real, m_hypo = wl[y == 1].mean(), wl[y == 0].mean()
    pct = abs(m_real - m_hypo) / ((m_real + m_hypo) / 2) * 100
    ok3 = pct <= LENGTH_PCT_MAX
    print(f"[3] length confound    : {'PASS' if ok3 else 'FAIL'}  "
          f"real={m_real:.1f} words  hypo={m_hypo:.1f} words  diff={pct:.1f}% (fail>{LENGTH_PCT_MAX}%)")
    if not ok3:
        fails.append("length-confound")

    # --- 4. template balance ----------------------------------------------
    per_tpl = Counter(r["template_id"] for r in rows)
    worst = min(per_tpl.values())
    ok4 = worst >= MIN_ROWS_PER_TEMPLATE
    print(f"[4] template balance   : {'PASS' if ok4 else 'FAIL'}  "
          f"{len(per_tpl)} templates, min {worst} rows/template (fail<{MIN_ROWS_PER_TEMPLATE})")
    if not ok4:
        fails.append("template-balance")

    print()
    if fails:
        print(f"AUDIT FAILED: {fails} — do NOT extract directions until fixed.")
        raise SystemExit(1)
    print("AUDIT PASSED — safe to extract directions.")


if __name__ == "__main__":
    main()

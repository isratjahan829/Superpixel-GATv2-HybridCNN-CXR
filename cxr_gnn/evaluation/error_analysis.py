"""
evaluation/error_analysis.py -- Misclassification taxonomy.

The report's error-analysis table (which confusions dominate, at what
confidence, and how many errors an abstention threshold would catch) had no
code behind it. This computes it from stored predictions.

Inputs are pooled predictions with their confidences, so it can be run on the
best split config, on out-of-fold CV predictions, or on a single test set.
"""

from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np

from cxr_gnn.config import IDX2CLASS, NUM_CLASSES, DISPLAY_NAMES
from cxr_gnn.utils import get_logger

logger = get_logger(__name__)


def _name(idx: int) -> str:
    return DISPLAY_NAMES.get(IDX2CLASS[int(idx)], IDX2CLASS[int(idx)])


def run_error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    work_dir: str,
    top_k: int = 4,
    abstain_threshold: float = 0.55,
) -> dict:
    """Error taxonomy: dominant confusion pairs, their mean confidence, and the
    fraction of errors an abstention rule would defer.

    Args:
        confidence: max softmax probability per prediction.
        top_k:      how many confusion pairs to list before collapsing the rest
                    into an "all other pairs" row.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    confidence = np.asarray(confidence, dtype=float)

    wrong = y_true != y_pred
    n_err = int(wrong.sum())
    n_total = int(len(y_true))

    if n_err == 0:
        logger.info("Error analysis: no misclassifications.")
        empty = {"n_total": n_total, "n_errors": 0, "rows": [],
                 "abstain_threshold": abstain_threshold, "errors_deferred_fraction": float("nan")}
        with open(os.path.join(work_dir, "error_analysis.json"), "w") as f:
            json.dump(empty, f, indent=2)
        return empty

    pairs = Counter(zip(y_true[wrong].tolist(), y_pred[wrong].tolist()))
    ranked = pairs.most_common()

    rows = []
    for (t, p), cnt in ranked[:top_k]:
        m = wrong & (y_true == t) & (y_pred == p)
        rows.append({
            "pattern": f"{_name(t)} -> {_name(p)}",
            "true_class": _name(t), "predicted_class": _name(p),
            "count": int(cnt),
            "pct_of_errors": float(cnt / n_err),
            "mean_confidence": float(confidence[m].mean()),
        })

    rest = ranked[top_k:]
    if rest:
        rest_mask = np.zeros(n_total, dtype=bool)
        for (t, p), _ in rest:
            rest_mask |= wrong & (y_true == t) & (y_pred == p)
        rows.append({
            "pattern": "All other pairs",
            "true_class": None, "predicted_class": None,
            "count": int(rest_mask.sum()),
            "pct_of_errors": float(rest_mask.sum() / n_err),
            "mean_confidence": float(confidence[rest_mask].mean()),
        })

    # Abstention: how many errors fall below the confidence threshold (i.e.
    # would be deferred), and what it costs in deferred correct predictions.
    low = confidence < abstain_threshold
    deferred_errors = float((low & wrong).sum() / n_err)
    deferred_correct = float((low & ~wrong).sum() / max(1, int((~wrong).sum())))

    result = {
        "n_total": n_total,
        "n_errors": n_err,
        "error_rate": float(n_err / n_total),
        "mean_error_confidence": float(confidence[wrong].mean()),
        "mean_correct_confidence": float(confidence[~wrong].mean()),
        "rows": rows,
        "abstain_threshold": abstain_threshold,
        "errors_deferred_fraction": deferred_errors,
        "correct_deferred_fraction": deferred_correct,
    }

    logger.info("Error analysis: %d errors / %d predictions (%.1f%%)",
                n_err, n_total, 100 * n_err / n_total)
    for r in rows:
        logger.info("  %-42s n=%3d (%.1f%% of errors) mean-conf %.3f",
                    r["pattern"], r["count"], 100 * r["pct_of_errors"], r["mean_confidence"])
    logger.info("  Abstaining below confidence %.2f would defer %.1f%% of errors "
                "(and %.1f%% of correct predictions).",
                abstain_threshold, 100 * deferred_errors, 100 * deferred_correct)

    out_path = os.path.join(work_dir, "error_analysis.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved %s", out_path)
    return result

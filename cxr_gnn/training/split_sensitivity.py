"""
training/split_sensitivity.py -- Split-ratio sensitivity analysis (SAP 2).

Goal: show the reported performance is not an artefact of one lucky split, by
repeating training and evaluation under several train/val/test ratios and
several seeds (stratified, image level, no leakage).

The same pre-built graph pool that feeds 5-fold CV is reused: only the index
partition changes per (config, seed), so no image is re-processed through
SLIC/ResNet and the study stays tractable on a single GPU.

Produces every sub-table of SAP 2:
  2.2  aggregate performance per split config (mean +/- SD across seeds)
  2.3  per-class performance for the best config (pooled across seeds)
  2.4  paired significance tests between configs (Holm-corrected, Cohen's d)
  2.5  bootstrapped 95% CIs per config
  2.6  pooled confusion matrix for the best config
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    matthews_corrcoef, confusion_matrix, precision_recall_fscore_support,
)
from torch_geometric.data import Data

from cxr_gnn.config import Config, IDX2CLASS, NUM_CLASSES, DISPLAY_NAMES
from cxr_gnn.evaluation.stats import (
    bootstrap_ci, format_ci, macro_auc, paired_significance, holm_bonferroni,
    per_class_sens_spec_ppv_npv,
)
from cxr_gnn.training.crossval import _train_one_fold, _eval_model, _make_loader
from cxr_gnn.utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SplitConfig:
    id: str
    train: float
    val: float
    test: float
    notes: str = ""

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split config {self.id} ratios sum to {total}, must sum to 1.0")


DEFAULT_SPLIT_CONFIGS: tuple[SplitConfig, ...] = (
    SplitConfig("S1", 0.80, 0.10, 0.10, "Stratified, original baseline"),
    SplitConfig("S2", 0.70, 0.15, 0.15, "Stratified"),
    SplitConfig("S3", 0.75, 0.15, 0.10, "Stratified"),
    SplitConfig("S4", 0.70, 0.20, 0.10, "Stratified"),
    SplitConfig("S5", 0.60, 0.20, 0.20, "Stratified, extra-robustness check"),
)

DEFAULT_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("S1", "S2", "accuracy"),
    ("S1", "S2", "macro_f1"),
    ("S1", "S3", "accuracy"),
    ("S1", "S4", "accuracy"),
    ("S1", "S5", "accuracy"),
    ("S2", "S4", "accuracy"),
)


def stratified_split_indices(
    labels: np.ndarray, train_frac: float, val_frac: float, test_frac: float, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Class-stratified index split at the requested ratios.

    Mirrors data/dataset.py::stratified_image_split but operates on pre-built
    graph indices instead of file paths.
    """
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        by_class[int(lab)].append(i)

    train_idx, val_idx, test_idx = [], [], []
    for lab in sorted(by_class):
        idxs = list(by_class[lab])
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = max(1, int(round(n * test_frac)))
        n_val = max(1, int(round(n * val_frac)))
        # Always leave at least one training example per class.
        n_test = min(n_test, max(1, n - 2))
        n_val = min(n_val, max(1, n - n_test - 1))
        test_idx.extend(idxs[:n_test])
        val_idx.extend(idxs[n_test:n_test + n_val])
        train_idx.extend(idxs[n_test + n_val:])

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def _class_metrics(ty: np.ndarray, tp: np.ndarray, tpr: np.ndarray) -> dict:
    mprec, mrec, _, _ = precision_recall_fscore_support(
        ty, tp, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(ty, tp)),
        "balanced_accuracy": float(balanced_accuracy_score(ty, tp)),
        "macro_f1": float(f1_score(ty, tp, average="macro")),
        "macro_auc": float(macro_auc(ty, tpr, NUM_CLASSES)),
        "macro_precision": float(mprec),
        "macro_recall": float(mrec),
        "mcc": float(matthews_corrcoef(ty, tp)) if len(np.unique(ty)) > 1 else float("nan"),
    }


def run_split_sensitivity(
    all_graphs: list[Data],
    all_labels: np.ndarray,
    cfg: Config,
    device,
    work_dir: str,
    split_configs: tuple[SplitConfig, ...] = DEFAULT_SPLIT_CONFIGS,
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
    epochs: int | None = None,
    patience: int | None = None,
    n_boot: int = 2000,
) -> dict:
    """Full SAP 2 pipeline; also writes split_sensitivity.json to work_dir."""
    epochs = epochs or cfg.cv_epochs
    patience = patience or cfg.cv_patience
    nfd = all_graphs[0].x.shape[1]
    efd = all_graphs[0].edge_attr.shape[1] if all_graphs[0].edge_attr is not None else None

    logger.info("Split-ratio sensitivity: %d configs x %d seeds = %d training runs ...",
                len(split_configs), len(seeds), len(split_configs) * len(seeds))

    per_run: dict[str, dict[int, dict]] = {c.id: {} for c in split_configs}

    for sc in split_configs:
        for seed in seeds:
            tr_idx, va_idx, te_idx = stratified_split_indices(
                all_labels, sc.train, sc.val, sc.test, seed)
            # seed=seed: the seed must vary model init as well as the partition,
            # otherwise every "seed" shares one initialisation.
            model, _secs = _train_one_fold(
                [all_graphs[i] for i in tr_idx], [all_graphs[i] for i in va_idx],
                nfd, efd, cfg, device, epochs=epochs, patience=patience, seed=seed)
            ty, tp, tpr = _eval_model(
                model, _make_loader([all_graphs[i] for i in te_idx], cfg.batch_size), device)
            metrics = _class_metrics(ty, tp, tpr)
            per_run[sc.id][seed] = {
                **metrics,
                "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)), "n_test": int(len(te_idx)),
                "y": ty, "pred": tp, "proba": tpr,
            }
            logger.info("  [%s seed=%d] acc=%.4f bal-acc=%.4f macroF1=%.4f macroAUC=%.4f "
                        "mcc=%.4f n(tr/va/te)=%d/%d/%d",
                        sc.id, seed, metrics["accuracy"], metrics["balanced_accuracy"],
                        metrics["macro_f1"], metrics["macro_auc"], metrics["mcc"],
                        len(tr_idx), len(va_idx), len(te_idx))

    # -- 2.2 aggregate by split config ---------------------------------------
    metric_keys = ["accuracy", "balanced_accuracy", "macro_f1", "macro_auc",
                   "macro_precision", "macro_recall", "mcc"]
    agg_table: dict[str, dict] = {}
    for sc in split_configs:
        runs = per_run[sc.id]
        any_run = next(iter(runs.values()))
        agg_table[sc.id] = {
            "ratios": {"train": sc.train, "val": sc.val, "test": sc.test},
            "notes": sc.notes,
            "n_train": any_run["n_train"], "n_val": any_run["n_val"], "n_test": any_run["n_test"],
            "n_total": any_run["n_train"] + any_run["n_val"] + any_run["n_test"],
            **{k: {"mean": float(np.nanmean([r[k] for r in runs.values()])),
                   "std": float(np.nanstd([r[k] for r in runs.values()])),
                   "per_seed": [float(runs[s][k]) for s in seeds]}
               for k in metric_keys},
        }

    best_id = max(agg_table, key=lambda k: agg_table[k]["accuracy"]["mean"])

    pooled_y = np.concatenate([per_run[best_id][s]["y"] for s in seeds])
    pooled_p = np.concatenate([per_run[best_id][s]["pred"] for s in seeds])
    pooled_pr = np.concatenate([per_run[best_id][s]["proba"] for s in seeds])

    # -- 2.3 per-class table for the best config (pooled across seeds) -------
    class_names = [DISPLAY_NAMES.get(IDX2CLASS[i], IDX2CLASS[i]) for i in range(NUM_CLASSES)]
    per_class_table = per_class_sens_spec_ppv_npv(
        pooled_y, pooled_p, NUM_CLASSES, class_names, proba=pooled_pr)

    # Macro row, so the table footer is computed rather than hand-typed.
    def _macro(field: str) -> float:
        vals = [v[field] for v in per_class_table.values() if np.isfinite(v[field])]
        return float(np.mean(vals)) if vals else float("nan")

    per_class_macro = {f: _macro(f) for f in
                       ("sensitivity", "specificity", "ppv", "npv", "f1", "auc", "lr_pos", "lr_neg")}

    # -- 2.6 pooled confusion matrix -----------------------------------------
    cm = confusion_matrix(pooled_y, pooled_p, labels=list(range(NUM_CLASSES)))

    # -- 2.4 significance between configs ------------------------------------
    sig_rows = []
    for a, b, metric in DEFAULT_COMPARISONS:
        if a not in per_run or b not in per_run:
            continue
        common = sorted(set(per_run[a]) & set(per_run[b]))
        res = paired_significance([per_run[a][s][metric] for s in common],
                                  [per_run[b][s][metric] for s in common])
        sig_rows.append({
            "comparison": f"{a} vs {b}", "metric": metric,
            "test_used": "paired t-test (+ Wilcoxon signed-rank)",
            "t_statistic": res.get("t_stat", float("nan")),
            "p_value": res["t_p"], "wilcoxon_p": res["wilcoxon_p"],
            "cohens_d": res["cohens_d"], "effect_size": res["effect_size_label"],
            "mean_diff": res["mean_diff"],
        })
    if sig_rows:
        corrected = holm_bonferroni([r["p_value"] for r in sig_rows])
        for r, cp in zip(sig_rows, corrected):
            r["holm_corrected_p"] = float(cp)
            r["significant_alpha_0.05"] = bool(cp < 0.05)

    # -- 2.5 bootstrap CIs per config ----------------------------------------
    ci_table = {}
    for sc in split_configs:
        runs = per_run[sc.id]
        y = np.concatenate([runs[s]["y"] for s in seeds])
        p = np.concatenate([runs[s]["pred"] for s in seeds])
        pr = np.concatenate([runs[s]["proba"] for s in seeds])
        ci_table[sc.id] = {
            "accuracy_ci": format_ci(*bootstrap_ci(y, p, accuracy_score, n_boot, cfg.seed)),
            "macro_f1_ci": format_ci(*bootstrap_ci(
                y, p, lambda yt, yp: f1_score(yt, yp, average="macro"), n_boot, cfg.seed)),
            "macro_auc_ci": format_ci(*bootstrap_ci(
                y, pr, lambda yt, prb: macro_auc(yt, prb, NUM_CLASSES), n_boot, cfg.seed)),
            "mcc_ci": format_ci(*bootstrap_ci(
                y, p, lambda yt, yp: matthews_corrcoef(yt, yp), n_boot, cfg.seed)),
        }

    logger.info("=" * 60)
    logger.info("SPLIT-RATIO SENSITIVITY -- SUMMARY (best config: %s)", best_id)
    for cid, row in agg_table.items():
        logger.info("  %-4s (%.0f/%.0f/%.0f): acc %.4f+/-%.4f | macroF1 %.4f+/-%.4f | "
                    "macroAUC %.4f+/-%.4f", cid,
                    row["ratios"]["train"] * 100, row["ratios"]["val"] * 100,
                    row["ratios"]["test"] * 100,
                    row["accuracy"]["mean"], row["accuracy"]["std"],
                    row["macro_f1"]["mean"], row["macro_f1"]["std"],
                    row["macro_auc"]["mean"], row["macro_auc"]["std"])

    results = {
        "seeds_used": list(seeds),
        "best_split_config": best_id,
        "aggregate_by_split": agg_table,
        "per_class_best_split": per_class_table,
        "per_class_best_split_macro": per_class_macro,
        "confusion_matrix_best_split": cm.tolist(),
        "class_order": class_names,
        "pooled_predictions_best_split": {
            "y": pooled_y.tolist(), "pred": pooled_p.tolist(),
            "confidence": pooled_pr.max(1).tolist(),
        },
        "significance_tests": sig_rows,
        "bootstrap_ci_by_split": ci_table,
        "bootstrap_protocol": {
            "method": "non-parametric percentile bootstrap, paired resampling",
            "n_iterations": n_boot, "seed": cfg.seed,
            "software": "numpy.random.default_rng + scikit-learn",
        },
    }

    out_path = os.path.join(work_dir, "split_sensitivity.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    logger.info("Saved %s", out_path)

    return results

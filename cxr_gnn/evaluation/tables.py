"""
evaluation/tables.py -- Rebuild the evaluation tables from saved artifacts.

Every table in the write-up is generated here from the JSON files the pipeline
writes, so no number in the manuscript is hand-typed. Each builder reads only
artifacts and returns Markdown; a missing artifact yields an explicit
"not available -- run step X" line instead of a plausible-looking placeholder.

Table map (matching the report):
    1  Split configurations, aggregate performance, bootstrapped 95% CIs
    2  Stratified 5-fold cross-validation
    3  Per-class performance, best split config
    4  Confusion matrix, best split config
    5  Statistical significance, all pairwise comparisons
    6  Model comparison: discrimination, calibration, computational cost
    7a Ablation study
    7b Robustness under input perturbation
    7c Error analysis
    8  Literature benchmarking
"""

from __future__ import annotations

import json
import os

import numpy as np

from cxr_gnn.utils import get_logger

logger = get_logger(__name__)


def _load(work_dir: str, name: str):
    path = os.path.join(work_dir, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _missing(table: str, artifact: str, step: str) -> str:
    return (f"**{table}: not available.** `{artifact}` was not found -- "
            f"run {step} first. No numbers are shown rather than placeholder values.")


def _fmt(v, nd: int = 3) -> str:
    if v is None:
        return "-"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(f):
        return "n/a" if np.isnan(f) else ("inf" if f > 0 else "-inf")
    return f"{f:.{nd}f}"


def _ci(lo, hi, nd: int = 3) -> str:
    return f"({_fmt(lo, nd)}-{_fmt(hi, nd)})"


# -- Table 1 ------------------------------------------------------------------

def table1_split_configs(work_dir: str) -> str:
    ss = _load(work_dir, "split_sensitivity.json")
    if ss is None:
        return _missing("Table 1", "split_sensitivity.json", "the split-sensitivity step")

    agg = ss["aggregate_by_split"]
    ci = ss["bootstrap_ci_by_split"]
    seeds = ss["seeds_used"]

    lines = [
        f"**Table 1. Split configurations, aggregate performance (mean +/- SD over "
        f"{len(seeds)} seeds) and bootstrapped 95% CIs.**", "",
        "| Config | Ratio (Tr/Va/Te) | n (Tr/Va/Te) | Accuracy | Bal. Acc. | Macro F1 | "
        "Macro AUC | Accuracy 95% CI | Macro F1 95% CI | Macro AUC 95% CI | MCC 95% CI |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cid, row in agg.items():
        r = row["ratios"]
        lines.append(
            f"| {cid} | {r['train']:.0%}/{r['val']:.0%}/{r['test']:.0%} "
            f"| {row['n_train']} / {row['n_val']} / {row['n_test']} "
            f"| {_fmt(row['accuracy']['mean'], 4)} +/- {_fmt(row['accuracy']['std'], 4)} "
            f"| {_fmt(row['balanced_accuracy']['mean'], 4)} +/- {_fmt(row['balanced_accuracy']['std'], 4)} "
            f"| {_fmt(row['macro_f1']['mean'], 4)} +/- {_fmt(row['macro_f1']['std'], 4)} "
            f"| {_fmt(row['macro_auc']['mean'], 4)} +/- {_fmt(row['macro_auc']['std'], 4)} "
            f"| {ci[cid]['accuracy_ci']} | {ci[cid]['macro_f1_ci']} "
            f"| {ci[cid]['macro_auc_ci']} | {ci[cid]['mcc_ci']} |"
        )

    means = {k: np.mean([agg[c][k]["mean"] for c in agg])
             for k in ("accuracy", "balanced_accuracy", "macro_f1", "macro_auc")}
    stds = {k: np.std([agg[c][k]["mean"] for c in agg])
            for k in ("accuracy", "balanced_accuracy", "macro_f1", "macro_auc")}
    n_total = next(iter(agg.values()))["n_total"]
    lines.append(
        f"| **Mean** | - | {n_total} total "
        f"| {_fmt(means['accuracy'], 4)} +/- {_fmt(stds['accuracy'], 4)} "
        f"| {_fmt(means['balanced_accuracy'], 4)} +/- {_fmt(stds['balanced_accuracy'], 4)} "
        f"| {_fmt(means['macro_f1'], 4)} +/- {_fmt(stds['macro_f1'], 4)} "
        f"| {_fmt(means['macro_auc'], 4)} +/- {_fmt(stds['macro_auc'], 4)} | - | - | - | - |"
    )
    lines += [
        "",
        f"All splits are stratified by class and performed on raw images before augmentation "
        f"(no image-level leakage). Seeds {min(seeds)}-{max(seeds)}. "
        f"Best configuration by mean accuracy: **{ss['best_split_config']}**; "
        f"bootstrap B = {ss['bootstrap_protocol']['n_iterations']}.",
    ]
    return "\n".join(lines)


# -- Table 2 ------------------------------------------------------------------

def table2_cross_validation(work_dir: str) -> str:
    cv = _load(work_dir, "cv_results.json")
    if cv is None:
        return _missing("Table 2", "cv_results.json", "the 5-fold CV step")

    sizes = cv["fold_sizes"]
    keys = ["Accuracy", "Balanced-Acc", "Macro-F1", "Macro-AUC", "MCC"]
    lines = [
        "**Table 2. Stratified 5-fold cross-validation.**", "",
        "| Fold | Train n | Val n | Test n | Accuracy | Balanced Acc. | Macro F1 | Macro AUC | MCC |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, s in enumerate(sizes):
        vals = [cv[k]["per_fold"][i] for k in keys]
        lines.append(f"| {s['fold']} | {s['n_train']} | {s['n_val']} | {s['n_test']} | "
                     + " | ".join(_fmt(v, 4) for v in vals) + " |")
    lines.append("| **Mean +/- SD** | - | - | - | "
                 + " | ".join(f"{_fmt(cv[k]['mean'], 4)} +/- {_fmt(cv[k]['std'], 4)}" for k in keys)
                 + " |")
    lines += ["", "Folds are stratified on the original (non-augmented) images only; "
                  "augmented graphs never enter a held-out fold."]
    return "\n".join(lines)


# -- Table 3 ------------------------------------------------------------------

def table3_per_class(work_dir: str) -> str:
    ss = _load(work_dir, "split_sensitivity.json")
    if ss is None:
        return _missing("Table 3", "split_sensitivity.json", "the split-sensitivity step")

    pc = ss["per_class_best_split"]
    macro = ss["per_class_best_split_macro"]
    best = ss["best_split_config"]
    total = sum(v["support"] for v in pc.values())

    lines = [
        f"**Table 3. Per-class performance, best split configuration ({best}), "
        f"pooled over {len(ss['seeds_used'])} seeds. Wilson score 95% CIs.**", "",
        "| Class | Support | Prevalence | Sensitivity (95% CI) | Specificity (95% CI) | "
        "PPV (95% CI) | NPV (95% CI) | F1 | AUC | LR+ | LR- |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cls, v in pc.items():
        lines.append(
            f"| {cls} | {v['support']} | {v['prevalence']:.1%} "
            f"| {_fmt(v['sensitivity'])} {_ci(*v['sensitivity_ci'])} "
            f"| {_fmt(v['specificity'])} {_ci(*v['specificity_ci'])} "
            f"| {_fmt(v['ppv'])} {_ci(*v['ppv_ci'])} "
            f"| {_fmt(v['npv'])} {_ci(*v['npv_ci'])} "
            f"| {_fmt(v['f1'])} | {_fmt(v['auc'])} "
            f"| {_fmt(v['lr_pos'], 2)} | {_fmt(v['lr_neg'], 2)} |"
        )
    lines.append(
        f"| **Macro average** | {total} | 100% | {_fmt(macro['sensitivity'])} "
        f"| {_fmt(macro['specificity'])} | {_fmt(macro['ppv'])} | {_fmt(macro['npv'])} "
        f"| {_fmt(macro['f1'])} | {_fmt(macro['auc'])} "
        f"| {_fmt(macro['lr_pos'], 2)} | {_fmt(macro['lr_neg'], 2)} |"
    )
    lines += [
        "",
        "Wilson score intervals are used for every proportion: more reliable than the normal "
        "approximation at the sample sizes of the minority classes. Sensitivity here is the "
        "same quantity as per-class recall in Table 4 and is computed from the same pooled "
        "predictions, so the two tables agree by construction.",
    ]
    return "\n".join(lines)


# -- Table 4 ------------------------------------------------------------------

def table4_confusion(work_dir: str) -> str:
    ss = _load(work_dir, "split_sensitivity.json")
    if ss is None:
        return _missing("Table 4", "split_sensitivity.json", "the split-sensitivity step")

    cm = np.array(ss["confusion_matrix_best_split"])
    names = ss["class_order"]
    best = ss["best_split_config"]

    lines = [
        f"**Table 4. Confusion matrix, best split configuration ({best}), pooled over seeds "
        f"(rows = true, columns = predicted).**", "",
        "| True \\ Predicted | " + " | ".join(names) + " | Total | Recall |",
        "|---" * (len(names) + 3) + "|",
    ]
    for i, nm in enumerate(names):
        row_total = int(cm[i].sum())
        recall = cm[i, i] / row_total if row_total else float("nan")
        lines.append(f"| {nm} | " + " | ".join(str(int(v)) for v in cm[i])
                     + f" | {row_total} | {_fmt(recall)} |")

    col_tot = cm.sum(0)
    lines.append("| **Total predicted** | " + " | ".join(str(int(v)) for v in col_tot)
                 + f" | {int(cm.sum())} | - |")
    prec = [cm[i, i] / col_tot[i] if col_tot[i] else float("nan") for i in range(len(names))]
    overall = cm.trace() / cm.sum() if cm.sum() else float("nan")
    lines.append("| **Precision** | " + " | ".join(_fmt(p) for p in prec)
                 + f" | - | {_fmt(overall)} (accuracy) |")
    return "\n".join(lines)


# -- Table 5 ------------------------------------------------------------------

def table5_significance(work_dir: str) -> str:
    ss = _load(work_dir, "split_sensitivity.json")
    sr = _load(work_dir, "statistical_robustness.json")
    if ss is None and sr is None:
        return _missing("Table 5", "split_sensitivity.json / statistical_robustness.json",
                        "the split-sensitivity and baseline-comparison steps")

    lines = [
        "**Table 5. Statistical significance, all pairwise comparisons.**", "",
        "| Family | Comparison | Metric | Test | Statistic | p-value | Holm-corrected p | "
        "Delta (A-B) | Effect size | Significant (alpha=.05) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    if ss is not None:
        for r in ss["significance_tests"]:
            lines.append(
                f"| Split-ratio | {r['comparison']} | {r['metric']} | Paired t-test "
                f"| t = {_fmt(r['t_statistic'], 2)} | {r['p_value']:.3g} "
                f"| {r['holm_corrected_p']:.3g} | {r['mean_diff']:+.4f} "
                f"| d = {_fmt(r['cohens_d'], 2)} ({r['effect_size']}) "
                f"| {'Yes' if r['significant_alpha_0.05'] else 'No'} |"
            )

    if sr is not None:
        for r in sr["pairwise_delong_mcnemar"]:
            stat = (f"z = {_fmt(r['statistic_z'], 2)}" if np.isfinite(r.get("statistic_z", np.nan))
                    else f"chi2 = {_fmt(r['fisher_chi2'], 2)}")
            better = ""
            if r["significant_holm_alpha_0.05"]:
                better = f" ({r['model_a'] if r['delta'] > 0 else r['model_b']} better)"
            lines.append(
                f"| Model-vs-model | {r['model_a']} vs {r['model_b']} | {r['metric']} "
                f"| {r['test']} | {stat} | {r['p_value']:.3g} | {r['holm_corrected_p']:.3g} "
                f"| {r['delta']:+.4f} | - "
                f"| {'Yes' + better if r['significant_holm_alpha_0.05'] else 'No'} |"
            )

    lines += [
        "",
        "Holm-Bonferroni correction is applied within each family. The DeLong row reports the "
        "Stouffer-combined z across one-vs-rest classes (signed, so the direction is readable) "
        "while the p-value is Fisher-combined; McNemar is the exact paired test on the same "
        "held-out instances.",
    ]
    return "\n".join(lines)


# -- Table 6 ------------------------------------------------------------------

def table6_model_comparison(work_dir: str) -> str:
    sr = _load(work_dir, "statistical_robustness.json")
    if sr is None:
        return _missing("Table 6", "statistical_robustness.json", "the baseline-comparison step")

    ml = sr["model_level_metrics"]
    lines = [
        "**Table 6. Model comparison: discrimination, agreement, calibration and cost.**", "",
        "| Model | Acc. | Macro F1 | Macro AUC | Bal. Acc. | MCC (95% CI) | Cohen's kappa (95% CI) "
        "| Youden J | Brier | ECE | Log loss | Trainable params | Train (min/fold) | Infer. (ms) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m, d in ml.items():
        lines.append(
            f"| {m} | {_fmt(d['accuracy'])} | {_fmt(d['macro_f1'])} | {_fmt(d['macro_auc'])} "
            f"| {_fmt(d['balanced_accuracy'])} | {d['mcc']} | {d['cohens_kappa']} "
            f"| {_fmt(d['youdens_j'])} | {_fmt(d['brier_score'])} | {_fmt(d['ece'])} "
            f"| {_fmt(d['log_loss'])} | {d['params']:,} | {_fmt(d['train_minutes_per_fold'], 2)} "
            f"| {_fmt(d['inference_ms'], 1)} |"
        )
    lines += [
        "",
        "MCC and Cohen's kappa stay meaningful under class imbalance, unlike raw accuracy. "
        "Brier score, ECE and log loss quantify how trustworthy the probabilities are, which any "
        "claim of clinical usefulness depends on. Parameter counts are TRAINABLE parameters: the "
        "GATv2 hybrid's ResNet18 encoder is frozen and therefore excluded, which is why its count "
        "is far below the fine-tuned CNN baseline's. Timings are for the hardware this run used "
        "and are not comparable across machines.",
    ]
    return "\n".join(lines)


# -- Table 7a / 7b / 7c -------------------------------------------------------

def table7a_ablation(work_dir: str) -> str:
    feats = _load(work_dir, "ablation_features.json")
    arch = _load(work_dir, "ablation_architecture.json")
    pre = _load(work_dir, "ablation_pretraining.json")
    if feats is None and arch is None:
        return _missing("Table 7a", "ablation_features.json / ablation_architecture.json",
                        "the ablation step")

    merged: dict[str, dict] = {}
    for src in (arch or {}, pre or {}, feats or {}):
        merged.update(src)

    ref_key = next((k for k in merged if "reference" in k.lower()), None)
    ref_acc = float(np.mean(merged[ref_key]["acc"])) if ref_key else float("nan")

    lines = [
        "**Table 7a. Ablation study: contribution of each component.**", "",
        "| Configuration | Accuracy | Macro F1 | Macro AUC | MCC | Delta Acc. vs reference | "
        "Trainable params | Infer. (ms) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in merged.items():
        acc = float(np.mean(r["acc"]))
        delta = "-" if name == ref_key else f"{acc - ref_acc:+.4f}"
        params = f"{int(r['params']):,}" if r.get("params") else "-"
        lines.append(
            f"| {name} | {_fmt(acc, 4)} | {_fmt(np.mean(r['f1']), 4)} "
            f"| {_fmt(np.nanmean(r['auc']), 4)} | {_fmt(np.nanmean(r['mcc']), 4)} "
            f"| {delta} | {params} | {_fmt(r.get('inference_ms'), 1)} |"
        )
    lines += [
        "",
        "Each arm removes exactly one component and is trained on the same folds with the same "
        "budget, so the delta is attributable to that component. A positive delta means the "
        "removed component was not helping.",
    ]
    return "\n".join(lines)


def table7b_robustness(work_dir: str) -> str:
    rb = _load(work_dir, "robustness.json")
    if rb is None:
        return _missing("Table 7b", "robustness.json", "the robustness step")

    lines = [
        "**Table 7b. Robustness: performance under input perturbation.**", "",
        "| Perturbation | Accuracy | Delta Acc. | Macro F1 | Macro AUC | MCC | Retention | Grade |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rb["rows"]:
        if r.get("skipped"):
            lines.append(f"| {r['perturbation']} | skipped | - | - | - | - | - | {r['reason']} |")
            continue
        delta = "-" if r["grade"] == "Reference" else f"{r['delta_accuracy']:+.4f}"
        lines.append(
            f"| {r['perturbation']} | {_fmt(r['accuracy'], 4)} | {delta} "
            f"| {_fmt(r['macro_f1'], 4)} | {_fmt(r['macro_auc'], 4)} | {_fmt(r['mcc'], 4)} "
            f"| {r['retention']:.1%} | {r['grade']} |"
        )
    lines += [
        "",
        f"Retention = perturbed accuracy / clean accuracy, on {rb['n_test_images']} held-out "
        f"test images. Grades: A >= 96%, B >= 92%, C >= 88%, D below. Perturbations are applied "
        f"to the image and the graph is rebuilt through the identical SLIC + encoder path.",
    ]
    return "\n".join(lines)


def table7c_error_analysis(work_dir: str) -> str:
    ea = _load(work_dir, "error_analysis.json")
    if ea is None:
        return _missing("Table 7c", "error_analysis.json", "the error-analysis step")

    lines = [
        "**Table 7c. Error analysis: misclassification taxonomy.**", "",
        "| Error pattern | Count | % of errors | Mean confidence |",
        "|---|---|---|---|",
    ]
    for r in ea["rows"]:
        lines.append(f"| {r['pattern']} | {r['count']} | {r['pct_of_errors']:.1%} "
                     f"| {_fmt(r['mean_confidence'])} |")
    lines.append(f"| **Total errors** | {ea['n_errors']} | 100% "
                 f"| {_fmt(ea['mean_error_confidence'])} |")
    lines += [
        "",
        f"Errors are {ea['error_rate']:.1%} of {ea['n_total']} pooled predictions. Mean confidence "
        f"on correct predictions is {_fmt(ea['mean_correct_confidence'])} vs "
        f"{_fmt(ea['mean_error_confidence'])} on errors. Abstaining below confidence "
        f"{ea['abstain_threshold']:.2f} would defer {ea['errors_deferred_fraction']:.1%} of errors, "
        f"at the cost of deferring {ea['correct_deferred_fraction']:.1%} of correct predictions.",
    ]
    return "\n".join(lines)


# -- Assembly -----------------------------------------------------------------

ALL_TABLES = {
    "Table 1": table1_split_configs,
    "Table 2": table2_cross_validation,
    "Table 3": table3_per_class,
    "Table 4": table4_confusion,
    "Table 5": table5_significance,
    "Table 6": table6_model_comparison,
    "Table 7a": table7a_ablation,
    "Table 7b": table7b_robustness,
    "Table 7c": table7c_error_analysis,
}


def build_all_tables(work_dir: str) -> str:
    """All tables as one Markdown document; also written to evaluation_tables.md."""
    from cxr_gnn.evaluation.reporting import literature_table_markdown

    parts = [fn(work_dir) for fn in ALL_TABLES.values()]

    test = _load(work_dir, "test_results.json") or {}
    parts.append(literature_table_markdown(test))

    doc = "\n\n---\n\n".join(parts)
    out_path = os.path.join(work_dir, "evaluation_tables.md")
    with open(out_path, "w") as f:
        f.write(doc)
    logger.info("Saved %s", out_path)
    return doc

"""
evaluation/reporting.py -- Literature benchmarking (Table 8) and the
TRIPOD-AI / STARD-AI reporting checklists.

The literature figures are quoted from the cited papers; they are not
recomputed here. Datasets, label taxonomies and task setups differ by orders of
magnitude across rows (binary vs multiclass vs multi-label, 570 vs 377,000
images), so the table is qualitative positioning, not a controlled comparison.
Only this project's own rows are head-to-head on identical splits.
"""

from __future__ import annotations

import os

LITERATURE_BENCHMARKS = [
    {
        "study": "CheXNet (Rajpurkar et al., 2017)",
        "year": "2017",
        "dataset": "ChestX-ray14 (112,120 images)",
        "task": "14 findings, multi-label binary",
        "approach": "121-layer DenseNet",
        "accuracy": "n/a (multi-label)",
        "auc": "~0.83-0.84 mean AUROC",
        "f1": "0.435 (pneumonia)",
        "sens": "n/a",
        "notes": "14 independent binary detectors, not one multiclass decision.",
        "source": "arXiv:1711.05225",
    },
    {
        "study": "COVID-Net (Wang, Lin & Wong, 2020)",
        "year": "2020",
        "dataset": "COVIDx (~13,975 images)",
        "task": "3-class multiclass",
        "approach": "Lightweight CNN (PEPX)",
        "accuracy": "0.933",
        "auc": "n/a",
        "f1": "n/a",
        "sens": "~0.91 (COVID class)",
        "notes": "Three more separable classes; COVIDx has known source heterogeneity.",
        "source": "Scientific Reports 10:19549",
    },
    {
        "study": "MIMIC-CXR single-source benchmark",
        "year": "recent",
        "dataset": "MIMIC-CXR (~377,110 images)",
        "task": "Multi-label findings",
        "approach": "CNN multi-label classifier",
        "accuracy": "n/a",
        "auc": "~0.75 mean AUROC",
        "f1": "n/a",
        "sens": "n/a",
        "notes": "Different data regime entirely: ~660x more images than this study.",
        "source": "multi-source CXR generalisation literature",
    },
]


def literature_table_markdown(our_metrics: dict) -> str:
    """Table 8, with this pipeline's own measured numbers as the reference rows.

    `our_metrics` is the contents of test_results.json; anything missing is
    printed as "n/a" rather than silently defaulting to a number.
    """
    def g(key, nd=3):
        v = our_metrics.get(key)
        try:
            return f"{float(v):.{nd}f}"
        except (TypeError, ValueError):
            return "n/a"

    n_total = our_metrics.get("n_total", "n/a")
    lines = [
        "**Table 8. Literature benchmarking (qualitative positioning).**", "",
        "| Study / Model | Year | Dataset (n) | Task / Classes | Approach | Acc. | Macro AUC "
        "| Macro F1 | Sens. (macro) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in LITERATURE_BENCHMARKS:
        lines.append(
            f"| {row['study']} | {row['year']} | {row['dataset']} | {row['task']} "
            f"| {row['approach']} | {row['accuracy']} | {row['auc']} | {row['f1']} | {row['sens']} |"
        )
    lines.append(
        f"| **GATv2 Hybrid (this work)** | this study | This dataset ({n_total}) "
        f"| 5-class multiclass | Frozen ResNet18 features + superpixel GATv2 "
        f"| {g('accuracy')} | {g('macro_auc')} | {g('macro_f1')} | {g('balanced_accuracy')} |"
    )
    lines += [
        "",
        "*Because label sets, task framing and dataset scale differ by two to three orders of "
        "magnitude, this table supports only relative discussion (direction and size of the gap, "
        "with the dataset-scale caveat), never a claim of superiority. The only controlled "
        "head-to-head comparison in this work is the CNN baseline in Table 6, which is trained "
        "and evaluated on identical splits.*",
    ]
    return "\n".join(lines)


# -- Reporting checklists -----------------------------------------------------
# Status reflects what the pipeline actually produces. "addressed" means the
# corresponding artifact exists in work_dir; "manual" items must be written up
# by a human; "n/a" items are outside a single study's scope.

TRIPOD_AI_ITEMS = [
    ("Title/abstract identifies the study as developing an AI/ML prediction model",
     "manual", None, "Add when writing up."),
    ("Source of data and eligibility criteria described",
     "manual", None, "Document the source institution(s) and inclusion criteria."),
    ("Outcome (target classes) clearly defined",
     "addressed", None, "5-class taxonomy fixed in config.py (CLASS2IDX)."),
    ("Predictors (features) fully specified",
     "addressed", None, "Hand-crafted + deep node features documented in graph.py / encoder.py."),
    ("Sample size / cases per class justified or acknowledged as a limitation",
     "addressed", "split_sensitivity.json", "Split-ratio sensitivity plus per-class support."),
    ("Missing / unusable data handling described",
     "addressed", "run.log", "Degenerate-segmentation drops are logged per split by cache.py."),
    ("Development vs validation data separation stated",
     "addressed", "split_sensitivity.json", "Image-level stratified split before augmentation."),
    ("Internal validation method reported",
     "addressed", "cv_results.json", "5-fold CV plus bootstrap 95% CIs."),
    ("Performance measures justified for the clinical task and class imbalance",
     "addressed", "statistical_robustness.json", "Macro-F1, balanced accuracy, MCC, kappa."),
    ("Calibration reported",
     "addressed", "calibration.json", "ECE/MCE, reliability diagram, temperature scaling."),
    ("Model updating / re-calibration discussed",
     "n/a", None, "Out of scope for a retrospective single-institution study."),
    ("Comparison against existing models / baselines",
     "addressed", "baseline_results.json", "GCN/GraphSAGE/GAT/ResNet18 plus literature table."),
    ("Uncertainty quantification for individual predictions",
     "addressed", "conformal_results.json", "Conformal prediction sets with coverage guarantees."),
    ("External validation on an independent dataset",
     "n/a", None, "Single-source data only -- stated as a limitation."),
    ("Code / model availability statement",
     "addressed", None, "This repository; add a DOI at publication time."),
]

STARD_AI_ITEMS = [
    ("Study design (retrospective / prospective) stated",
     "manual", None, "State explicitly in Methods."),
    ("Reference standard (ground-truth labelling process) described",
     "manual", None, "Document the radiologist/clinical labelling protocol."),
    ("Flow of images (inclusion, exclusions, degenerate cases) reported",
     "addressed", "run.log", "Per-split drop counts logged by cache.py."),
    ("Distribution of severity / alternative diagnoses in the sample",
     "manual", None, "Add clinical characterisation if available."),
    ("Statistical methods pre-specified and matched to the data",
     "addressed", "statistical_robustness.json",
     "Wilson CIs at small n, non-parametric bootstrap, Holm/FDR correction."),
    ("Indeterminate results handling",
     "addressed", "conformal_results.json", "Non-singleton conformal sets flag exactly these."),
    ("Adverse events / harms of testing discussed",
     "n/a", None, "Not applicable to retrospective image classification."),
]


def _fmt_checklist(items, title: str, work_dir: str | None) -> str:
    lines = [f"### {title}", "", "| Item | Status | Evidence |", "|---|---|---|"]
    icon = {"addressed": "addressed", "manual": "needs manual entry", "n/a": "N/A"}
    for item, status, artifact, note in items:
        st = status
        if status == "addressed" and artifact and work_dir:
            # Verify the claim instead of asserting it.
            st = "addressed" if os.path.exists(os.path.join(work_dir, artifact)) else "not yet run"
        evidence = f"`{artifact}` -- {note}" if artifact else note
        lines.append(f"| {item} | {icon.get(st, st)} | {evidence} |")
    return "\n".join(lines)


def reporting_checklist_markdown(work_dir: str | None = None) -> str:
    """Both checklists. When work_dir is given, "addressed" is verified against
    the artifact actually being on disk rather than merely asserted.
    """
    return "\n\n".join([
        _fmt_checklist(TRIPOD_AI_ITEMS, "TRIPOD-AI checklist", work_dir),
        _fmt_checklist(STARD_AI_ITEMS, "STARD-AI checklist", work_dir),
    ])

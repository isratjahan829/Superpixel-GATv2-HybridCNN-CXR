#!/usr/bin/env python3
"""
make_report.py -- Build the evaluation report exactly as the document lays it out.

Reads the JSON artifacts written by train.py (or the notebook) and emits, one
file per item, in the document's own numbering:

    report/
      tables/
        Table_1_split_configurations.{md,csv}
        Table_2_cross_validation.{md,csv}
        Table_3_per_class_performance.{md,csv}
        Table_4_confusion_matrix.{md,csv}
        Table_5_statistical_significance.{md,csv}
        Table_6_model_comparison.{md,csv}
        Table_7a_ablation.{md,csv}
        Table_7b_robustness.{md,csv}
        Table_7c_error_analysis.{md,csv}
        Table_8_literature_benchmarking.md
        ALL_TABLES.md
      figures/
        Figure_01_training_curves.png ... Figure_11_robustness.png
        FIGURE_INDEX.md
      REPORT_SUMMARY.md

The CSV alongside each Markdown table is what you paste into Word or Excel.

Nothing here recomputes a metric: every number comes from an artifact on disk,
so the report cannot disagree with the run that produced it. A table whose
artifact is missing is written with an explicit "not available" note rather
than placeholder numbers, and is listed as missing in REPORT_SUMMARY.md.

Usage:
    python make_report.py                          # reads ./outputs
    python make_report.py --work-dir outputs_demo --out report_demo
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil

# Document order: (file stem, builder attribute in evaluation.tables, title)
TABLE_SPEC = [
    ("Table_1_split_configurations", "table1_split_configs",
     "Split configurations, aggregate performance and bootstrapped 95% CIs"),
    ("Table_2_cross_validation", "table2_cross_validation",
     "Stratified 5-fold cross-validation"),
    ("Table_3_per_class_performance", "table3_per_class",
     "Per-class performance, best split configuration"),
    ("Table_4_confusion_matrix", "table4_confusion",
     "Confusion matrix, best split configuration"),
    ("Table_5_statistical_significance", "table5_significance",
     "Statistical significance, all pairwise comparisons"),
    ("Table_6_model_comparison", "table6_model_comparison",
     "Model comparison: discrimination, calibration and computational cost"),
    ("Table_7a_ablation", "table7a_ablation",
     "Ablation study: contribution of each component"),
    ("Table_7b_robustness", "table7b_robustness",
     "Robustness under input perturbation"),
    ("Table_7c_error_analysis", "table7c_error_analysis",
     "Error analysis: misclassification taxonomy"),
]

# Figures, in the order the document refers to them.
FIGURE_SPEC = [
    ("fig1_training_curves.png", "Figure_01_training_curves.png",
     "Training and validation loss/accuracy, and the generalisation gap. "
     "Both curves are measured in eval() mode, so they are directly comparable."),
    ("fig2_confusion.png", "Figure_02_confusion_test_set.png",
     "Confusion matrix on the held-out test set, counts and row-normalised."),
    ("fig2b_confusion_best_split.png", "Figure_03_confusion_best_split.png",
     "Confusion matrix for the best split configuration, pooled over seeds. "
     "This is the matrix behind Table 4."),
    ("fig3_per_class_metrics.png", "Figure_04_per_class_metrics.png",
     "Per-class precision, recall and F1 on the test set."),
    ("fig4_roc_pr.png", "Figure_05_roc_and_precision_recall.png",
     "One-vs-rest ROC and precision-recall curves with per-class AUC/AP."),
    ("fig5_tsne.png", "Figure_06_tsne_embedding.png",
     "t-SNE of the graph-level embeddings before the classification head."),
    ("fig6_superpixel_graphs.png", "Figure_07_superpixel_graphs.png",
     "SLIC superpixel graphs per class: nodes are regions, edges are adjacency."),
    ("fig7_attention_saliency.png", "Figure_08_attention_saliency.png",
     "Attention-based saliency: where the GATv2 layer places its attention mass."),
    ("fig8_ablation_features.png", "Figure_09a_ablation_features.png",
     "Feature ablation: hand-crafted vs deep vs hybrid node features."),
    ("fig8b_ablation_architecture.png", "Figure_09b_ablation_architecture.png",
     "Architecture ablation: one design choice removed per arm (Table 7a)."),
    ("fig9_model_comparison.png", "Figure_10_model_comparison.png",
     "Macro-F1 and macro-AUC across GCN, GraphSAGE, GAT, GATv2 and the CNN baseline."),
    ("fig10_calibration.png", "Figure_11_calibration.png",
     "Reliability diagram and confidence histogram (ECE/MCE)."),
    ("fig11_robustness.png", "Figure_12_robustness.png",
     "Accuracy under each input perturbation against the clean reference (Table 7b)."),
]


def markdown_table_to_rows(md: str) -> list[list[str]]:
    """Extract the pipe-table from a Markdown block as rows of cells.

    Separator rows (|---|---|) are dropped. Text outside the table (title,
    footnote) is ignored -- it lives in the .md file, not the .csv.
    """
    rows = []
    for line in md.split("\n"):
        line = line.strip()
        if not line.startswith("|"):
            continue
        if re.fullmatch(r"\|[\s\-:|]+\|", line):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        # Strip Markdown bold markers so the CSV is plain text.
        rows.append([re.sub(r"\*\*(.*?)\*\*", r"\1", c) for c in cells])
    return rows


def write_table(out_dir: str, stem: str, title: str, md: str, banner: str = "") -> bool:
    """Write one table as .md (+ .csv when it has content). Returns availability."""
    available = "not available" not in md

    md_path = os.path.join(out_dir, f"{stem}.md")
    with open(md_path, "w") as f:
        if banner:
            f.write(banner + "\n")
        f.write(md.rstrip() + "\n")

    if available:
        rows = markdown_table_to_rows(md)
        if rows:
            with open(os.path.join(out_dir, f"{stem}.csv"), "w", newline="") as f:
                csv.writer(f).writerows(rows)
    return available


def provenance_banner(work_dir: str) -> str:
    """A header stating what run produced these numbers.

    A smoke test on synthetic phantoms and a full run on the real dataset write
    identically-shaped artifacts. Without this banner the two are
    indistinguishable once the tables are pasted into a document, which is
    exactly how a placeholder number ends up being cited as a result.
    """
    path = os.path.join(work_dir, "run_metadata.json")
    if not os.path.exists(path):
        return ("> **Provenance unknown.** No `run_metadata.json` in the artifact "
                "directory, so it cannot be confirmed which dataset or budget produced "
                "these numbers. Re-run the pipeline to record it.\n")

    with open(path) as f:
        m = json.load(f)

    facts = (f"Source: `{m['data_root']}` ({m['n_images']} images) - "
             f"device {m['device']}"
             + (f" - commit `{m['git_commit']}`" if m.get("git_commit") else ""))

    if m.get("reportable"):
        return f"> Full run on the real dataset. {facts}\n"

    reasons = []
    if m.get("synthetic_data"):
        reasons.append("**synthetic phantom images, not chest X-rays**")
    if m.get("reduced_budget"):
        cfg = m.get("config", {})
        reasons.append(f"**reduced budgets** (epochs={cfg.get('epochs')}, "
                       f"cv_epochs={cfg.get('cv_epochs')}, "
                       f"bootstrap B={cfg.get('n_bootstrap')})")
    return (f"> ## NOT REPORTABLE\n>\n"
            f"> These numbers come from {' and '.join(reasons)}. They demonstrate that "
            f"the pipeline executes end to end; they are **not findings** and must not "
            f"be cited. Re-run on the real dataset at full budget to populate these "
            f"tables.\n>\n> {facts}\n")


def build_report(work_dir: str, out_dir: str) -> dict:
    from cxr_gnn.evaluation import tables as T
    from cxr_gnn.evaluation.reporting import (
        literature_table_markdown, reporting_checklist_markdown)

    tables_dir = os.path.join(out_dir, "tables")
    figures_dir = os.path.join(out_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    banner = provenance_banner(work_dir)
    status = {"tables": {}, "figures": {}, "work_dir": os.path.abspath(work_dir)}
    all_md = []

    # -- Tables 1 to 7c ------------------------------------------------------
    for stem, fn_name, title in TABLE_SPEC:
        md = getattr(T, fn_name)(work_dir)
        ok = write_table(tables_dir, stem, title, md, banner)
        status["tables"][stem] = ok
        all_md.append(md)
        print(f"  {'[ok]     ' if ok else '[missing]'} {stem}")

    # -- Table 8: literature benchmarking ------------------------------------
    test_results = {}
    tr_path = os.path.join(work_dir, "test_results.json")
    if os.path.exists(tr_path):
        with open(tr_path) as f:
            test_results = json.load(f)
    md8 = literature_table_markdown(test_results)
    ok8 = write_table(tables_dir, "Table_8_literature_benchmarking",
                      "Literature benchmarking", md8, banner)
    status["tables"]["Table_8_literature_benchmarking"] = ok8
    all_md.append(md8)
    print(f"  {'[ok]     ' if ok8 else '[missing]'} Table_8_literature_benchmarking")

    # -- Combined document ---------------------------------------------------
    with open(os.path.join(tables_dir, "ALL_TABLES.md"), "w") as f:
        f.write("# Evaluation tables\n\n")
        f.write(banner + "\n")
        f.write("Every number below is read from an artifact in "
                f"`{os.path.abspath(work_dir)}`; none is transcribed by hand.\n\n")
        f.write("\n\n---\n\n".join(all_md).rstrip() + "\n")

    # -- Reporting checklists ------------------------------------------------
    with open(os.path.join(tables_dir, "Reporting_checklists.md"), "w") as f:
        f.write(reporting_checklist_markdown(work_dir).rstrip() + "\n")

    # -- Figures -------------------------------------------------------------
    index = ["# Figure index", ""]
    for src_name, dst_name, caption in FIGURE_SPEC:
        src = os.path.join(work_dir, src_name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(figures_dir, dst_name))
            status["figures"][dst_name] = True
            index.append(f"**{dst_name}** -- {caption}\n")
            print(f"  [ok]      {dst_name}")
        else:
            status["figures"][dst_name] = False
            index.append(f"**{dst_name}** -- NOT GENERATED (missing `{src_name}`). {caption}\n")
            print(f"  [missing] {dst_name}  (no {src_name})")

    with open(os.path.join(figures_dir, "FIGURE_INDEX.md"), "w") as f:
        f.write("\n".join(index).rstrip() + "\n")

    # -- Summary -------------------------------------------------------------
    missing_t = [k for k, v in status["tables"].items() if not v]
    missing_f = [k for k, v in status["figures"].items() if not v]

    lines = [
        "# Report build summary", "",
        f"Source artifacts: `{os.path.abspath(work_dir)}`",
        f"Output: `{os.path.abspath(out_dir)}`", "",
        provenance_banner(work_dir), "",
        f"- Tables written: {len(status['tables']) - len(missing_t)} / {len(status['tables'])}",
        f"- Figures copied: {len(status['figures']) - len(missing_f)} / {len(status['figures'])}",
        "",
    ]
    if missing_t or missing_f:
        lines += ["## Missing", ""]
        for k in missing_t:
            lines.append(f"- `{k}` -- the stage that writes its artifact has not been run.")
        for k in missing_f:
            lines.append(f"- `{k}` -- figure not generated by this run.")
        lines += ["", "Run `python train.py` (all stages) and rebuild to fill these in.", ""]
    lines += [
        "## Files", "",
        "- `tables/*.md` -- one table each, ready to read",
        "- `tables/*.csv` -- the same tables as plain data, for Word or Excel",
        "- `tables/ALL_TABLES.md` -- every table in document order",
        "- `tables/Reporting_checklists.md` -- TRIPOD-AI and STARD-AI status",
        "- `figures/*.png` -- figures, numbered in the order the text refers to them",
        "- `figures/FIGURE_INDEX.md` -- captions",
    ]
    with open(os.path.join(out_dir, "REPORT_SUMMARY.md"), "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    return status


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work-dir", default=None,
                   help="directory holding the run artifacts (default: Config.work_dir)")
    p.add_argument("--out", default="report", help="output directory (default: report)")
    a = p.parse_args()

    work_dir = a.work_dir
    if work_dir is None:
        from cxr_gnn.config import Config
        work_dir = Config().work_dir

    if not os.path.isdir(work_dir):
        raise SystemExit(
            f"No artifact directory at {work_dir!r}. Run `python train.py` first, "
            f"or pass --work-dir.")

    print(f"Building report from {work_dir} -> {a.out}")
    status = build_report(work_dir, a.out)

    n_t = sum(status["tables"].values())
    n_f = sum(status["figures"].values())
    print(f"\nDone: {n_t}/{len(status['tables'])} tables, "
          f"{n_f}/{len(status['figures'])} figures -> {os.path.abspath(a.out)}")
    print("See REPORT_SUMMARY.md for anything missing.")


if __name__ == "__main__":
    main()

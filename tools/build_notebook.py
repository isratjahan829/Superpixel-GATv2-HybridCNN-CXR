#!/usr/bin/env python3
"""
tools/build_notebook.py -- Generate the pipeline notebook.

The notebook imports the `cxr_gnn` package rather than carrying its own copy of
the source. The original notebook wrote every module to disk with %%writefile,
which meant the same ~3,000 lines existed in two places and could drift apart;
here the package is the single source of truth and the notebook is the driver.

Usage:
    python tools/build_notebook.py -o notebooks/gatv2_cxr_pipeline.ipynb
"""

from __future__ import annotations

import argparse
import os

import nbformat as nbf

REPO_URL = "https://github.com/isratjahan829/Superpixel-GATv2-HybridCNN-CXR"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str):
    return nbf.v4.new_code_cell(text.strip("\n"))


def build() -> nbf.NotebookNode:
    cells = []

    cells.append(md(f"""
# Superpixel GATv2 + hybrid CNN — chest X-ray classification

**Architecture:** frozen ResNet18 features → SLIC superpixel graph → GATv2 → mean+max pooling → MLP
**Task:** 5-class chest X-ray (Cardiac, ChronicLung, Normal, Pleural, TB)
**Dataset:** <https://www.kaggle.com/datasets/shakib0hasan/capstone-c-dataset>

This notebook is the *driver*. All logic lives in the `cxr_gnn` package
([repository]({REPO_URL})), so the code you run here is the code that is
tested and reviewed — the notebook cannot drift away from it.

### What this run produces

| Step | Output |
|---|---|
| 1–6 | Data discovery, image-level split, graph cache, training |
| 7 | Held-out test evaluation + 7 figures |
| 8 | Stratified 5-fold cross-validation |
| 9 | Ablation: feature arms **and** architecture arms |
| 10 | Baselines (GCN / GraphSAGE / GAT / end-to-end ResNet18) + full statistical suite |
| 11 | Split-ratio sensitivity (5 configs × 5 seeds) + error analysis |
| 12 | Conformal prediction (LAC / APS / RAPS / class-conditional LAC) |
| 13 | Calibration (ECE / MCE + temperature scaling) |
| 14 | Robustness under input perturbation |
| 15 | Every evaluation table, rebuilt from the saved artifacts |

### Runtime

On a Kaggle T4 with the real dataset the full run takes roughly 1.5–2 hours,
dominated by the end-to-end ResNet18 baseline. Set `FAST = True` in the config
cell for a few-minute smoke test (results from a fast run are **not** reportable).

### Notes on correctness

The pipeline splits on raw images *before* augmentation, measures train and
validation metrics in the same `eval()` mode, and reuses one set of fold indices
across every model so the paired McNemar/DeLong tests are valid. See
`docs/EVALUATION_TABLES_REVIEW.md` for the specific defects this rewrite fixes.
"""))

    # -- Setup ---------------------------------------------------------------
    cells.append(md("## Step 0 — Environment"))
    cells.append(code(f"""
# Make the cxr_gnn package importable.
#   - Running inside a clone of the repository: nothing to do.
#   - Kaggle/Colab with internet on: install straight from GitHub.
#   - No internet: add the repo as a Kaggle dataset and point PACKAGE_PATH at it.
import importlib, os, subprocess, sys

PACKAGE_PATH = None          # e.g. "/kaggle/input/cxr-gnn-source"

for candidate in [PACKAGE_PATH, ".", "..", "/kaggle/working/Superpixel-GATv2-HybridCNN-CXR"]:
    if candidate and os.path.isdir(os.path.join(candidate, "cxr_gnn")):
        sys.path.insert(0, os.path.abspath(candidate))
        break

try:
    import cxr_gnn                                     # noqa: F401
    print("cxr_gnn found at", os.path.dirname(cxr_gnn.__file__))
except ImportError:
    print("cxr_gnn not found locally -- installing from GitHub ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "git+{REPO_URL}.git"], check=True)
    importlib.invalidate_caches()
    import cxr_gnn
    print("installed:", os.path.dirname(cxr_gnn.__file__))
"""))

    cells.append(code("""
# torch-geometric is the only dependency Kaggle images do not always ship.
# torch-scatter / torch-sparse are deliberately NOT installed: PyG 2.x does not
# need them here, and pip builds them from source (slow, and often fails).
try:
    import torch_geometric
    print("torch-geometric", torch_geometric.__version__)
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch-geometric"], check=True)
    import torch_geometric
    print("torch-geometric", torch_geometric.__version__, "(just installed)")
"""))

    # -- Config --------------------------------------------------------------
    cells.append(md("""
## Step 1 — Configuration

`Config` is a frozen dataclass: it cannot be mutated after construction, so
there is never any doubt about which settings a given cell ran under. Derive a
variant explicitly with `cfg.replace(...)`.

The primary split is **S4 (70/20/10)** — the configuration the split-ratio
sensitivity analysis selects — and every reported number uses it.
"""))

    cells.append(code("""
import os
import numpy as np
import torch

from cxr_gnn.config import Config, CLEAN_LABELS, CLASS2IDX, IDX2CLASS, NUM_CLASSES, DISPLAY_NAMES
from cxr_gnn.utils import set_seed, get_device, setup_logging, get_logger

# FAST=True shrinks every budget so the whole notebook runs in minutes.
# Use it to check the pipeline works; do not report numbers from a fast run.
FAST = not os.path.isdir("/kaggle/input")

DATA_ROOT = os.environ.get("CXR_DATA_ROOT")   # None -> auto-detect under /kaggle/input
WORK_DIR = os.environ.get("CXR_WORK_DIR", "/kaggle/working" if os.path.isdir("/kaggle/working") else "outputs")

kw = {}
if DATA_ROOT:
    kw["data_root"] = DATA_ROOT
if FAST:
    # Smaller images and fewer superpixels as well as fewer epochs: on CPU the
    # graph build dominates, not the training.
    kw.update(img_size=128, n_segments=120,
              epochs=8, cv_epochs=8, cv_patience=3,
              sensitivity_epochs=6, sensitivity_patience=3,
              sensitivity_seeds=(42, 43),
              cnn_epochs=2, cnn_patience=2, n_bootstrap=300)

cfg = Config(work_dir=WORK_DIR, **kw)
device = get_device()
logger = setup_logging(cfg.work_dir)
set_seed(cfg.seed)

TARGET_NAMES = [DISPLAY_NAMES.get(IDX2CLASS[i], IDX2CLASS[i]) for i in range(NUM_CLASSES)]

print(f"Mode        : {'FAST (smoke test -- not reportable)' if FAST else 'FULL'}")
print(f"Device      : {device}")
print(f"Work dir    : {cfg.work_dir}")
print(f"Classes     : {CLEAN_LABELS}")
print(f"Node feats  : {cfg.node_feat_dim} (deep={cfg.deep_feat_dim} + hand={cfg.hand_feat_dim})")
print(f"Edge feats  : {cfg.edge_feat_dim}")
print(f"Primary split: train={1 - cfg.test_size - cfg.val_size:.0%} "
      f"val={cfg.val_size:.0%} test={cfg.test_size:.0%}  (S4)")
print(f"Bootstrap   : B={cfg.n_bootstrap}")
"""))

    # -- Data ----------------------------------------------------------------
    cells.append(md("""
## Step 2 — Data discovery and image-level split

The split happens on **raw images, before augmentation**. Building all the
graphs first and splitting afterwards puts augmented copies of training images
into val/test, which leaks and inflates every metric downstream.

If the Kaggle dataset is not mounted, the cell falls back to a small synthetic
phantom dataset so the notebook still runs end to end. Phantom images are
procedural, not radiographs — metrics computed on them mean nothing beyond
"the code executes".
"""))

    cells.append(code("""
from cxr_gnn.data.dataset import find_data_root, collect_samples, stratified_image_split

try:
    data_root = find_data_root(cfg)
    USING_SYNTHETIC = False
except FileNotFoundError:
    print("Real dataset not found -- generating synthetic phantoms for a runnable demo.\\n")
    from tools.make_demo_dataset import generate      # noqa: E402
    generate("demo_data", per_class=20 if FAST else 60, size=cfg.img_size, seed=0)
    cfg = cfg.replace(data_root=os.path.abspath("demo_data"))
    data_root = find_data_root(cfg)
    USING_SYNTHETIC = True

samples = collect_samples(data_root, cfg)
splits = stratified_image_split(samples, cfg, cfg.seed)

from cxr_gnn.utils import write_run_metadata
meta = write_run_metadata(cfg.work_dir, cfg, device, len(samples), data_root,
                          synthetic=USING_SYNTHETIC)

print(f"\\nTotal images : {len(samples)}")
for k in ("train", "val", "test"):
    print(f"{k.capitalize():6s} images: {len(splits[k])}")
if USING_SYNTHETIC:
    print("\\n*** SYNTHETIC DATA -- results below are a smoke test, not findings. ***")
"""))

    cells.append(md("""
## Step 3 — Frozen ResNet18 encoder

Stem + layer1 + layer2 only, giving a 128-channel map at 1/8 resolution. Every
parameter is frozen, so the encoder contributes **zero trainable parameters** to
the hybrid model — the reason the GATv2 hybrid is two orders of magnitude
smaller than a fine-tuned ResNet18.
"""))

    cells.append(code("""
from cxr_gnn.models.encoder import build_encoder

encoder = build_encoder(device)
frozen = sum(p.numel() for p in encoder.parameters())
print(f"Encoder: {frozen:,} frozen parameters (0 trainable)")
"""))

    cells.append(md("""
## Step 4 — Build or load the graph cache

Each image becomes a SLIC superpixel graph: ~180 nodes carrying 128 deep
features pooled from the encoder plus 12 hand-crafted descriptors, with edges
between adjacent regions. This is the slow step; the result is cached and
reloaded on subsequent runs. The cache stores a signature of every
graph-affecting config field and refuses to load if any of them changed.
"""))

    cells.append(code("""
from cxr_gnn.data.cache import build_or_load_cache

train_graphs, val_graphs, test_graphs, kept_items = build_or_load_cache(
    splits, cfg, encoder, device, cfg.seed)

nfd = train_graphs[0].x.shape[1]
efd = train_graphs[0].edge_attr.shape[1] if train_graphs[0].edge_attr is not None else None
n_aug = sum(int(getattr(g, "is_aug", 0)) for g in train_graphs)

print(f"\\nGraphs -> train {len(train_graphs)} (of which {n_aug} augmented) | "
      f"val {len(val_graphs)} | test {len(test_graphs)}")
print(f"Node feature dim: {nfd} | edge feature dim: {efd}")
print(f"Mean nodes per graph: {np.mean([g.num_nodes for g in train_graphs]):.0f}")
print(f"Path-aligned kept items: " +
      ", ".join(f"{k}={len(v)}" for k, v in kept_items.items()))
"""))

    cells.append(md("""
## Step 5 — Dataloaders and pooled index sets

`WeightedRandomSampler` gives the minority classes (ChronicLung, Pleural)
roughly equal exposure per epoch; with plain shuffling they are invisible in
most batches.

The pools built here are reused by *every* later analysis. They contain only
original graphs — an augmented graph inside a held-out fold would inflate all
the cross-validation numbers.
"""))

    cells.append(code("""
from torch_geometric.loader import DataLoader
from cxr_gnn.training.trainer import make_weighted_loader, make_loss_weights

train_loader = make_weighted_loader(train_graphs, cfg.batch_size, NUM_CLASSES, seed=cfg.seed)
val_loader = DataLoader(val_graphs, cfg.batch_size, shuffle=False)
test_loader = DataLoader(test_graphs, cfg.batch_size, shuffle=False)

orig_graphs = [g for g in train_graphs if int(getattr(g, "is_aug", 0)) == 0] + val_graphs
orig_labels = np.array([int(g.y) for g in orig_graphs])
orig_items = ([(p, CLASS2IDX[c]) for p, c in kept_items["train"]]
              + [(p, CLASS2IDX[c]) for p, c in kept_items["val"]])
assert len(orig_items) == len(orig_graphs), "cache is stale -- rebuild it"

all_graphs = orig_graphs + test_graphs
all_labels = np.array([int(g.y) for g in all_graphs])

print(f"Batches -> train {len(train_loader)} | val {len(val_loader)} | test {len(test_loader)}")
print(f"CV pool (originals only): {len(orig_graphs)} graphs")
print(f"Sensitivity pool (all):   {len(all_graphs)} graphs")
"""))

    # -- Model / training ----------------------------------------------------
    cells.append(md("""
## Step 6 — Train the GATv2 classifier

`optimize_epoch()` does the gradient updates with dropout and DropEdge active;
`evaluate()` measures **both** splits in `eval()` mode. Measuring training
metrics during the optimisation pass (dropout on) is what made the original
notebook's train accuracy look lower than its validation accuracy.
"""))

    cells.append(code("""
from cxr_gnn.models.gatv2 import GATv2Classifier
from cxr_gnn.training.trainer import Trainer

model = GATv2Classifier(nfd, efd, cfg).to(device)
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(model)

loss_weights = make_loss_weights(train_graphs, NUM_CLASSES, device)
trainer = Trainer(model, cfg, loss_weights, device)
history = trainer.fit(train_loader, val_loader, cfg.ckpt_file)
trainer.load_best(cfg.ckpt_file)
print("\\nTraining finished; best checkpoint reloaded.")
"""))

    # -- Test ----------------------------------------------------------------
    cells.append(md("## Step 7 — Held-out test evaluation"))

    cells.append(code("""
import json
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, classification_report)
from train import predict_all
from cxr_gnn.evaluation.stats import macro_auc

te_y, te_p, te_pr = predict_all(model, test_loader, device)
tr_y, tr_p, _ = predict_all(model, train_loader, device)

test_metrics = {
    "accuracy": float(accuracy_score(te_y, te_p)),
    "balanced_accuracy": float(balanced_accuracy_score(te_y, te_p)),
    "macro_f1": float(f1_score(te_y, te_p, average="macro")),
    "macro_auc": float(macro_auc(te_y, te_pr, NUM_CLASSES)),
    "mcc": float(matthews_corrcoef(te_y, te_p)),
    "train_accuracy_eval_mode": float(accuracy_score(tr_y, tr_p)),
    "n_total": len(samples),
    "n_test": int(len(te_y)),
}
for k, v in test_metrics.items():
    print(f"{k:26s}: {v:.4f}" if isinstance(v, float) else f"{k:26s}: {v}")
print()
print(classification_report(te_y, te_p, labels=list(range(NUM_CLASSES)),
                            target_names=TARGET_NAMES, zero_division=0))

with open(f"{cfg.work_dir}/test_results.json", "w") as f:
    json.dump(test_metrics, f, indent=2)
print(f"Saved -> {cfg.work_dir}/test_results.json")
"""))

    cells.append(md("""
### Figures

Note that `plot_attention_saliency` takes the **encoder**. Passing `None`
produces 12-feature graphs for a model expecting 140, so every forward pass
raises and the figure comes out blank.
"""))

    cells.append(code("""
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from cxr_gnn.evaluation import visualization as viz

wd = cfg.work_dir
viz.plot_training_curves(history, wd)
viz.plot_confusion(te_y, te_p, TARGET_NAMES, wd)
viz.plot_per_class_metrics(te_y, te_p, TARGET_NAMES, wd)
viz.plot_roc_pr(te_y, te_pr, TARGET_NAMES, wd)
viz.plot_tsne(model, train_loader, test_loader, device, wd, TARGET_NAMES)
viz.plot_superpixel_graphs(samples, cfg, TARGET_NAMES, wd)
viz.plot_attention_saliency(model, samples, cfg, TARGET_NAMES, encoder, device, wd)

for path in sorted(glob.glob(f"{wd}/fig[1-7]*.png")):
    print(f"\\n--- {os.path.basename(path)} ---")
    plt.figure(figsize=(14, 6))
    plt.imshow(mpimg.imread(path))
    plt.axis("off")
    plt.tight_layout()
    plt.show()
"""))

    # -- CV ------------------------------------------------------------------
    cells.append(md("""
## Step 8 — Stratified 5-fold cross-validation

Original images only. MCC is reported per fold alongside accuracy/F1/AUC: with
this class imbalance it is the number worth quoting.
"""))

    cells.append(code("""
from cxr_gnn.training.crossval import run_cross_validation

cv_results = run_cross_validation(orig_graphs, orig_labels, cfg, device, wd)

print("\\n5-fold CV summary:")
for k in ["Accuracy", "Balanced-Acc", "Macro-F1", "Macro-AUC", "MCC"]:
    print(f"  {k:14s}: {cv_results[k]['mean']:.4f} +/- {cv_results[k]['std']:.4f}")
"""))

    # -- Ablation ------------------------------------------------------------
    cells.append(md("""
## Step 9 — Ablation

Two families. **Feature arms** ask which node features carry the signal;
**architecture arms** remove one design choice at a time (attention, multi-head
aggregation, class balancing, the superpixel graph itself).

Each arm trains on the same folds with the same budget, so the accuracy delta is
attributable to the removed component.
"""))

    cells.append(code("""
from cxr_gnn.training.ablation import run_feature_ablation, run_architecture_ablation

abl_feat = run_feature_ablation(orig_graphs, orig_labels, cfg, device, wd)
abl_arch = run_architecture_ablation(orig_graphs, orig_labels, cfg, device, wd,
                                     path_items=orig_items)

viz.plot_ablation(abl_feat, wd, "fig8_ablation_features.png")
viz.plot_ablation(abl_arch, wd, "fig8b_ablation_architecture.png")

for path in [f"{wd}/fig8_ablation_features.png", f"{wd}/fig8b_ablation_architecture.png"]:
    plt.figure(figsize=(14, 5))
    plt.imshow(mpimg.imread(path))
    plt.axis("off")
    plt.show()
"""))

    # -- Baselines -----------------------------------------------------------
    cells.append(md("""
## Step 10 — Baselines and the statistical suite

GCN, GraphSAGE, GAT and an **end-to-end fine-tuned ResNet18** on raw pixels.
The fold indices are computed once and shared by all five models, which is what
makes the paired McNemar and DeLong tests valid.

Computed here: MCC, Cohen's kappa, Youden's J, Brier, ECE, log loss (each with
bootstrap CIs), per-class Sens/Spec/PPV/NPV with Wilson CIs, **all** pairwise
DeLong and McNemar comparisons under Holm and BH correction, and the cost
columns (trainable parameters, training time, inference latency) measured rather
than asserted.
"""))

    cells.append(code("""
from cxr_gnn.training.crossval import run_model_comparison

bl_results = run_model_comparison(orig_graphs, orig_labels, cfg, device, wd,
                                  path_items=orig_items,
                                  include_cnn_baseline=cfg.include_cnn_baseline)
viz.plot_model_comparison(bl_results, wd)

plt.figure(figsize=(14, 5))
plt.imshow(mpimg.imread(f"{wd}/fig9_model_comparison.png"))
plt.axis("off")
plt.show()

with open(f"{wd}/statistical_robustness.json") as f:
    stat_summary = json.load(f)

print(f"\\nPairwise comparisons (Holm-corrected), primary = {stat_summary['primary_model']}:")
for r in stat_summary["pairwise_delong_mcnemar"]:
    verdict = "SIGNIFICANT" if r["significant_holm_alpha_0.05"] else "n.s."
    print(f"  {r['model_a']:22s} vs {r['model_b']:22s} {r['metric']:10s} "
          f"p={r['p_value']:.3g} holm={r['holm_corrected_p']:.3g}  {verdict}")
"""))

    # -- Sensitivity ---------------------------------------------------------
    cells.append(md("""
## Step 11 — Split-ratio sensitivity, and error analysis

Five split ratios × several seeds, reusing the pre-built graph pool: only the
index partition changes, so nothing is re-processed through SLIC. This is what
shows the headline number is not an artefact of one lucky split.

The seed varies **model initialisation as well as the partition** — varying only
the partition leaves every "seed" sharing one initialisation.

The pooled predictions from the best config then feed the per-class table, the
confusion matrix and the error taxonomy, so all three agree by construction.
"""))

    cells.append(code("""
from cxr_gnn.training.split_sensitivity import run_split_sensitivity
from cxr_gnn.evaluation.error_analysis import run_error_analysis

sens = run_split_sensitivity(all_graphs, all_labels, cfg, device, wd,
                             seeds=cfg.sensitivity_seeds,
                             epochs=cfg.sensitivity_epochs,
                             patience=cfg.sensitivity_patience,
                             n_boot=cfg.n_bootstrap)

print(f"\\nBest split configuration: {sens['best_split_config']}")
for cid, ci in sens["bootstrap_ci_by_split"].items():
    print(f"  {cid}: Acc {ci['accuracy_ci']} | Macro-F1 {ci['macro_f1_ci']} | MCC {ci['mcc_ci']}")

pooled = sens["pooled_predictions_best_split"]
err = run_error_analysis(np.array(pooled["y"]), np.array(pooled["pred"]),
                         np.array(pooled["confidence"]), wd)
viz.plot_confusion(np.array(pooled["y"]), np.array(pooled["pred"]), TARGET_NAMES, wd,
                   fname="fig2b_confusion_best_split.png")
"""))

    # -- Conformal / calibration --------------------------------------------
    cells.append(md("""
## Step 12 — Conformal prediction

Prediction *sets* with a coverage guarantee. A non-singleton set is the model
saying it cannot separate those classes for this image — exactly the
"indeterminate result" a reporting checklist asks about.
"""))

    cells.append(code("""
from cxr_gnn.evaluation.conformal import run_conformal
from cxr_gnn.training.crossval import _train_one_fold

def train_fold_fn(tr_g, va_g):
    m, _ = _train_one_fold(tr_g, va_g, nfd, efd, cfg, device,
                           epochs=cfg.cv_epochs, patience=cfg.cv_patience)
    return m

conf_results = run_conformal(orig_graphs, orig_labels, train_fold_fn, cfg, device, wd)

print("\\nConformal summary (5-fold average):")
for m in ["lac", "aps", "raps", "cclac"]:
    print(f"  {m.upper():6s}: coverage={np.mean(conf_results[m]['cov']):.3f} "
          f"| avg set size={np.mean(conf_results[m]['size']):.2f} "
          f"| singletons={np.mean(conf_results[m]['sing']):.1%}")
"""))

    cells.append(md("""
## Step 13 — Calibration

ECE below 0.05 reads as well calibrated; above 0.10 the probabilities should not
be quoted without correction. When that happens, a temperature is fitted on one
half of the pooled out-of-fold predictions and evaluated on the other, so the
post-scaling number is not fitted on the data it is reported on.
"""))

    cells.append(code("""
from cxr_gnn.evaluation.calibration import compute_calibration

cal = compute_calibration(orig_graphs, orig_labels, train_fold_fn, cfg, device, wd)
viz.plot_calibration(cal, wd)

print(f"ECE {cal['ECE']:.4f} | MCE {cal['MCE']:.4f} | "
      f"OOF acc {cal['oof_accuracy']:.4f} | mean confidence {cal['avg_confidence']:.4f}")
if cal["temperature_scaling"]:
    ts = cal["temperature_scaling"]
    print(f"Temperature scaling: T={ts['temperature']:.3f}, "
          f"ECE {ts['ece_before_on_eval_half']:.4f} -> {ts['ece_after']:.4f} (held-out half)")

plt.figure(figsize=(13, 5))
plt.imshow(mpimg.imread(f"{wd}/fig10_calibration.png"))
plt.axis("off")
plt.show()
"""))

    # -- Robustness ----------------------------------------------------------
    cells.append(md("""
## Step 14 — Robustness under input perturbation

Each perturbation is applied to the test images and the graphs are rebuilt
through the identical SLIC + encoder path, so what is measured is the whole
pipeline's sensitivity, not just the classifier's.
"""))

    cells.append(code("""
from cxr_gnn.evaluation.robustness import run_robustness_study

rob = run_robustness_study(model, kept_items["test"], cfg, encoder, device, wd)
viz.plot_robustness(rob, wd)

plt.figure(figsize=(12, 5))
plt.imshow(mpimg.imread(f"{wd}/fig11_robustness.png"))
plt.axis("off")
plt.show()
"""))

    # -- Tables --------------------------------------------------------------
    cells.append(md("""
## Step 15 — Evaluation tables

Every table is rebuilt from the JSON artifacts written above, so no number in
the write-up is transcribed by hand. A table whose artifact is missing prints
"not available — run step X" rather than a plausible-looking placeholder.
"""))

    cells.append(code("""
from IPython.display import Markdown, display
from cxr_gnn.evaluation.tables import build_all_tables

display(Markdown(build_all_tables(wd)))
"""))

    cells.append(code("""
from cxr_gnn.evaluation.reporting import reporting_checklist_markdown

# "addressed" is verified against the artifact actually existing on disk.
display(Markdown(reporting_checklist_markdown(wd)))
"""))

    cells.append(md("## Step 16 — Summary of artifacts"))

    cells.append(code("""
print("=" * 60)
print("RUN SUMMARY")
print("=" * 60)
if USING_SYNTHETIC:
    print("\\n*** SYNTHETIC PHANTOM DATA -- smoke test only, not findings. ***")
if FAST:
    print("*** FAST mode -- reduced budgets, numbers are not reportable. ***")

print(f"\\n[Test set]  " + "  ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()
                                     if isinstance(v, float)))
print(f"[5-fold CV] accuracy={cv_results['Accuracy']['mean']:.4f} "
      f"+/- {cv_results['Accuracy']['std']:.4f}, MCC={cv_results['MCC']['mean']:.4f}")
print(f"[Best split] {sens['best_split_config']}")
print(f"[Calibration] ECE={cal['ECE']:.4f}")

print("\\n[Files written]")
for p in sorted(glob.glob(f"{wd}/*.json") + glob.glob(f"{wd}/*.md")
                + glob.glob(f"{wd}/*.png") + glob.glob(f"{wd}/*.pt")):
    print(f"  {os.path.basename(p):42s} {os.path.getsize(p) / 1024:8.1f} KB")
"""))

    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    })
    return nb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--out", default="notebooks/gatv2_cxr_pipeline.ipynb")
    a = p.parse_args()
    nb = build()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    nbf.write(nb, a.out)
    print(f"Wrote {a.out} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()

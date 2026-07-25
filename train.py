#!/usr/bin/env python3
"""
train.py -- Single entry point for the whole pipeline.

Usage:
    python train.py                          # everything
    python train.py --rebuild-cache
    python train.py --only test              # train + test evaluation only
    python train.py --skip cv ablation       # skip named stages
    python train.py --fast                   # small budgets, for a smoke test

Stages: test, cv, ablation, baseline, sensitivity, conformal, calibration,
        robustness, tables
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef,
    classification_report,
)
from torch_geometric.loader import DataLoader

from cxr_gnn.config import Config, CLASS2IDX, IDX2CLASS, NUM_CLASSES, DISPLAY_NAMES
from cxr_gnn.data.cache import build_or_load_cache
from cxr_gnn.data.dataset import find_data_root, collect_samples, stratified_image_split
from cxr_gnn.models.encoder import build_encoder
from cxr_gnn.models.gatv2 import GATv2Classifier
from cxr_gnn.training.trainer import Trainer, make_weighted_loader, make_loss_weights
from cxr_gnn.training.crossval import run_cross_validation, run_model_comparison, _train_one_fold
from cxr_gnn.training.ablation import run_feature_ablation, run_architecture_ablation
from cxr_gnn.training.split_sensitivity import run_split_sensitivity
from cxr_gnn.evaluation.conformal import run_conformal
from cxr_gnn.evaluation.calibration import compute_calibration
from cxr_gnn.evaluation.error_analysis import run_error_analysis
from cxr_gnn.evaluation.robustness import run_robustness_study
from cxr_gnn.evaluation.stats import macro_auc
from cxr_gnn.evaluation.tables import build_all_tables
from cxr_gnn.evaluation import visualization as viz
from cxr_gnn.utils import (set_seed, get_device, setup_logging, get_logger,
                          write_run_metadata)

ALL_STAGES = ("test", "cv", "ablation", "baseline", "sensitivity",
              "conformal", "calibration", "robustness", "tables")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--skip", nargs="*", default=[], choices=ALL_STAGES,
                   help="stages to skip")
    p.add_argument("--only", nargs="*", default=None, choices=ALL_STAGES,
                   help="run only these stages")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--work-dir", default=None)
    p.add_argument("--no-cnn-baseline", action="store_true")
    p.add_argument("--fast", action="store_true",
                   help="tiny budgets: for verifying the pipeline runs, not for results")
    return p.parse_args(argv)


def build_config(args) -> Config:
    kw = {"rebuild_cache": args.rebuild_cache,
          "include_cnn_baseline": not args.no_cnn_baseline}
    if args.epochs:
        kw["epochs"] = args.epochs
    if args.data_root:
        kw["data_root"] = args.data_root
    if args.work_dir:
        kw["work_dir"] = args.work_dir
    if args.fast:
        kw.update(epochs=args.epochs or 6, cv_epochs=6, cv_patience=3,
                  sensitivity_epochs=5, sensitivity_patience=3,
                  cnn_epochs=3, cnn_patience=2, n_bootstrap=200,
                  sensitivity_seeds=(42, 43))
    return Config(**kw)


@torch.no_grad()
def predict_all(model, loader, device):
    """(y_true, y_pred, probs) for a whole loader."""
    model.eval()
    ys, ps, prs = [], [], []
    for b in loader:
        b = b.to(device)
        out = model(b.x, b.edge_index, b.edge_attr, b.batch)
        ys.append(b.y.cpu())
        ps.append(out.argmax(1).cpu())
        prs.append(F.softmax(out, 1).cpu())
    return torch.cat(ys).numpy(), torch.cat(ps).numpy(), torch.cat(prs).numpy()


def main(argv=None) -> dict:
    args = parse_args(argv)
    stages = set(args.only) if args.only else set(ALL_STAGES) - set(args.skip)

    cfg = build_config(args)
    os.makedirs(cfg.work_dir, exist_ok=True)
    setup_logging(cfg.work_dir)
    logger = get_logger()
    set_seed(cfg.seed)
    device = get_device()
    wd = cfg.work_dir

    logger.info("Device: %s | work_dir: %s", device, wd)

    # -- Data ----------------------------------------------------------------
    data_root = find_data_root(cfg)
    samples = collect_samples(data_root, cfg)
    splits = stratified_image_split(samples, cfg, cfg.seed)

    write_run_metadata(wd, cfg, device, len(samples), data_root,
                       synthetic="demo_data" in os.path.abspath(data_root))

    encoder = build_encoder(device)
    train_graphs, val_graphs, test_graphs, kept_items = build_or_load_cache(
        splits, cfg, encoder, device, cfg.seed)

    train_loader = make_weighted_loader(train_graphs, cfg.batch_size, NUM_CLASSES, seed=cfg.seed)
    val_loader = DataLoader(val_graphs, cfg.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_graphs, cfg.batch_size, shuffle=False, num_workers=0)

    nfd = train_graphs[0].x.shape[1]
    efd = train_graphs[0].edge_attr.shape[1] if train_graphs[0].edge_attr is not None else None
    tnames = [DISPLAY_NAMES.get(IDX2CLASS[i], IDX2CLASS[i]) for i in range(NUM_CLASSES)]

    # -- Pools shared by every downstream analysis ---------------------------
    # Originals only: an augmented graph in a held-out fold would inflate every
    # number. kept_items is index-aligned with the non-augmented prefix.
    orig_graphs = [g for g in train_graphs if int(getattr(g, "is_aug", 0)) == 0]
    orig_graphs += val_graphs
    orig_labels = np.array([int(g.y) for g in orig_graphs])
    orig_items = ([(p, CLASS2IDX[c]) for p, c in kept_items["train"]]
                  + [(p, CLASS2IDX[c]) for p, c in kept_items["val"]])
    if len(orig_items) != len(orig_graphs):
        raise RuntimeError(
            f"orig_items/orig_graphs misaligned ({len(orig_items)} vs {len(orig_graphs)}). "
            f"The cache is stale -- rerun with --rebuild-cache.")

    all_graphs = orig_graphs + test_graphs
    all_labels = np.array([int(g.y) for g in all_graphs])

    results: dict = {}

    # -- Primary model -------------------------------------------------------
    model = GATv2Classifier(nfd, efd, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("GATv2 trainable params: %d", n_params)

    loss_weights = make_loss_weights(train_graphs, NUM_CLASSES, device)
    trainer = Trainer(model, cfg, loss_weights, device)
    history = trainer.fit(train_loader, val_loader, cfg.ckpt_file)
    trainer.load_best(cfg.ckpt_file)
    results["history"] = history

    if "test" in stages:
        te_y, te_p, te_pr = predict_all(model, test_loader, device)
        tr_y, tr_p, _ = predict_all(model, train_loader, device)

        test_metrics = {
            "accuracy": float(accuracy_score(te_y, te_p)),
            "balanced_accuracy": float(balanced_accuracy_score(te_y, te_p)),
            "macro_f1": float(f1_score(te_y, te_p, average="macro")),
            "macro_auc": float(macro_auc(te_y, te_pr, NUM_CLASSES)),
            "mcc": float(matthews_corrcoef(te_y, te_p)) if len(np.unique(te_y)) > 1 else float("nan"),
            "train_accuracy_eval_mode": float(accuracy_score(tr_y, tr_p)),
            "n_total": len(samples),
            "n_test": int(len(te_y)),
        }
        logger.info("=" * 55)
        logger.info("TEST RESULTS")
        for k, v in test_metrics.items():
            logger.info("  %-26s: %s", k, f"{v:.4f}" if isinstance(v, float) else v)
        logger.info("\n%s", classification_report(te_y, te_p, labels=list(range(NUM_CLASSES)),
                                                  target_names=tnames, zero_division=0))
        with open(os.path.join(wd, "test_results.json"), "w") as f:
            json.dump(test_metrics, f, indent=2)
        results["test"] = test_metrics

        logger.info("Generating figures ...")
        viz.plot_training_curves(history, wd)
        viz.plot_confusion(te_y, te_p, tnames, wd)
        viz.plot_per_class_metrics(te_y, te_p, tnames, wd)
        viz.plot_roc_pr(te_y, te_pr, tnames, wd)
        viz.plot_tsne(model, train_loader, test_loader, device, wd, tnames)
        viz.plot_superpixel_graphs(samples, cfg, tnames, wd)
        # encoder passed explicitly: without it every graph has the wrong
        # feature width and the figure silently comes out blank.
        viz.plot_attention_saliency(model, samples, cfg, tnames, encoder, device, wd)

    if "cv" in stages:
        logger.info("=" * 55)
        logger.info("5-FOLD CROSS VALIDATION")
        results["cv"] = run_cross_validation(orig_graphs, orig_labels, cfg, device, wd)

    if "ablation" in stages:
        logger.info("=" * 55)
        logger.info("ABLATION STUDY")
        feats = run_feature_ablation(orig_graphs, orig_labels, cfg, device, wd)
        arch = run_architecture_ablation(orig_graphs, orig_labels, cfg, device, wd,
                                         path_items=orig_items)
        viz.plot_ablation(feats, wd, "fig8_ablation_features.png")
        viz.plot_ablation(arch, wd, "fig8b_ablation_architecture.png")
        results["ablation"] = {"features": feats, "architecture": arch}

    if "baseline" in stages:
        logger.info("=" * 55)
        logger.info("BASELINE COMPARISON + STATISTICAL SUITE")
        bl = run_model_comparison(orig_graphs, orig_labels, cfg, device, wd,
                                  path_items=orig_items,
                                  include_cnn_baseline=cfg.include_cnn_baseline)
        viz.plot_model_comparison(bl, wd)
        results["baseline"] = {k: {m: v[m] for m in ("acc", "f1", "auc")} for k, v in bl.items()}

    if "sensitivity" in stages:
        logger.info("=" * 55)
        logger.info("SPLIT-RATIO SENSITIVITY ANALYSIS")
        ss = run_split_sensitivity(all_graphs, all_labels, cfg, device, wd,
                                   seeds=cfg.sensitivity_seeds,
                                   epochs=cfg.sensitivity_epochs,
                                   patience=cfg.sensitivity_patience,
                                   n_boot=cfg.n_bootstrap)
        results["sensitivity"] = ss

        pooled = ss["pooled_predictions_best_split"]
        results["error_analysis"] = run_error_analysis(
            np.array(pooled["y"]), np.array(pooled["pred"]),
            np.array(pooled["confidence"]), wd)
        viz.plot_confusion(np.array(pooled["y"]), np.array(pooled["pred"]), tnames, wd,
                           fname="fig2b_confusion_best_split.png")

    def _fold_fn(tr_g, va_g):
        m, _s = _train_one_fold(tr_g, va_g, nfd, efd, cfg, device,
                                epochs=cfg.cv_epochs, patience=cfg.cv_patience)
        return m

    if "conformal" in stages:
        logger.info("=" * 55)
        logger.info("CONFORMAL PREDICTION")
        results["conformal"] = run_conformal(orig_graphs, orig_labels, _fold_fn, cfg, device, wd)

    if "calibration" in stages:
        logger.info("=" * 55)
        logger.info("CALIBRATION ANALYSIS")
        cal = compute_calibration(orig_graphs, orig_labels, _fold_fn, cfg, device, wd)
        viz.plot_calibration(cal, wd)
        results["calibration"] = cal

    if "robustness" in stages and cfg.run_robustness:
        logger.info("=" * 55)
        logger.info("ROBUSTNESS UNDER INPUT PERTURBATION")
        rb = run_robustness_study(model, kept_items["test"], cfg, encoder, device, wd)
        viz.plot_robustness(rb, wd)
        results["robustness"] = rb

    if "tables" in stages:
        logger.info("=" * 55)
        logger.info("BUILDING EVALUATION TABLES")
        build_all_tables(wd)

    logger.info("=" * 55)
    logger.info("All done. Results in: %s", wd)
    return results


if __name__ == "__main__":
    main()

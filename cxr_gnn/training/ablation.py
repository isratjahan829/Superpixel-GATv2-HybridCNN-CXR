"""
training/ablation.py -- Component-contribution ablation.

Two families of arms:

  Feature ablation (which node features matter)
      hand-only (12d), deep-only (128d), hybrid (140d) with and without edges

  Architectural ablation (which design choices matter)
      - superpixel graph removed  -> the raw-pixel CNN, i.e. no graph at all
      - GATv2 -> GCN layer        -> attention removed entirely
      - single attention head     -> multi-head aggregation removed
      - class weighting / sampler removed
      - ImageNet pretraining removed (encoder randomly initialised)

The architectural arms did not exist in the original notebook even though the
report tables quoted numbers for them; they are implemented here so every row
of the ablation table is reproducible from code.

Note on the "-- superpixel graph" arm: it needs raw pixels, so it is only run
when path_items is supplied, and it re-uses the CNN baseline trainer.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from torch_geometric.data import Data

from cxr_gnn.config import Config, NUM_CLASSES
from cxr_gnn.models.gatv2 import GATv2Classifier, GATv2SingleHead
from cxr_gnn.training.crossval import (
    _train_one_fold, _eval_model, _make_loader, _GCNClassifier, make_fold_splits,
    safe_train_test_split,
    measure_latency,
)
from cxr_gnn.evaluation.stats import macro_auc
from cxr_gnn.utils import get_logger, count_parameters

logger = get_logger(__name__)


def _slice_graphs(graphs: list[Data], lo: int, hi: int, use_edge: bool) -> list[Data]:
    """Feature-sliced copies of the graph list (built once, not per fold)."""
    return [
        Data(x=g.x[:, lo:hi].clone(), edge_index=g.edge_index,
             edge_attr=g.edge_attr if use_edge else None, y=g.y)
        for g in graphs
    ]


def _score(ty, tp, tpr) -> dict:
    return {
        "acc": float(accuracy_score(ty, tp)),
        "f1": float(f1_score(ty, tp, average="macro")),
        "auc": float(macro_auc(ty, tpr, NUM_CLASSES)),
        "mcc": float(matthews_corrcoef(ty, tp)) if len(np.unique(ty)) > 1 else float("nan"),
    }


def _cv_arm(graphs, labels, cfg, device, fold_splits, *, nfd, efd,
            model_class=None, balanced=True, epochs=None, patience=None) -> dict:
    """Run one ablation arm across all folds and aggregate."""
    epochs = epochs or cfg.cv_epochs
    patience = patience or cfg.cv_patience
    per_fold = {"acc": [], "f1": [], "auc": [], "mcc": []}
    last_model = None

    for tr_idx, te_idx in fold_splits:
        tr2, va2 = safe_train_test_split(
            tr_idx, labels, 0.15, cfg.seed, "ablation arm")
        model, _secs = _train_one_fold(
            [graphs[i] for i in tr2], [graphs[i] for i in va2],
            nfd, efd, cfg, device, epochs=epochs, patience=patience,
            model_class=model_class, balanced=balanced,
        )
        ty, tp, tpr = _eval_model(
            model, _make_loader([graphs[i] for i in te_idx], cfg.batch_size), device)
        for k, v in _score(ty, tp, tpr).items():
            per_fold[k].append(v)
        last_model = model

    out = {k: [float(x) for x in v] for k, v in per_fold.items()}
    out["params"] = count_parameters(last_model)
    out["inference_ms"] = measure_latency(last_model, graphs, device)
    return out


def run_feature_ablation(
    path_graphs: list[Data],
    path_labels: np.ndarray,
    cfg: Config,
    device: torch.device,
    work_dir: str,
) -> dict:
    """Which node features carry the signal?"""
    DEEP, HAND = cfg.deep_feat_dim, cfg.hand_feat_dim
    FULL = DEEP + HAND

    # The original notebook sliced [FULL:FULL] for the hand-only arm -- an empty
    # range, so that arm trained on zero features and its reported number was
    # meaningless. The correct slice is [DEEP:FULL].
    configs = {
        f"Hand-only ({HAND}d) +edge": (DEEP, FULL, True),
        f"Deep-only ({DEEP}d) +edge": (0, DEEP, True),
        f"Hybrid ({FULL}d) +edge": (0, FULL, True),
        f"Hybrid ({FULL}d) no-edge": (0, FULL, False),
    }

    fold_splits = make_fold_splits(path_labels, cfg)
    results: dict[str, dict] = {}

    logger.info("Running feature ablation (%d-fold) ...", cfg.n_folds)
    for name, (lo, hi, use_edge) in configs.items():
        sliced = _slice_graphs(path_graphs, lo, hi, use_edge)   # once, not per fold
        # Budgets come from cfg, so --fast actually shortens this stage; the
        # original hardcoded epochs=100 here and ignored the caller entirely.
        results[name] = _cv_arm(sliced, path_labels, cfg, device, fold_splits,
                                nfd=hi - lo, efd=(2 if use_edge else None))
        r = results[name]
        logger.info("  %-32s acc %.3f+/-%.3f | F1 %.3f+/-%.3f | AUC %.3f+/-%.3f",
                    name, np.mean(r["acc"]), np.std(r["acc"]),
                    np.mean(r["f1"]), np.std(r["f1"]),
                    np.nanmean(r["auc"]), np.nanstd(r["auc"]))

    with open(os.path.join(work_dir, "ablation_features.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved ablation_features.json")
    return results


def run_architecture_ablation(
    path_graphs: list[Data],
    path_labels: np.ndarray,
    cfg: Config,
    device: torch.device,
    work_dir: str,
    path_items: Optional[list[tuple[str, int]]] = None,
) -> dict:
    """Which architectural choices matter? Each arm removes exactly one.

    Every arm uses the same folds and the same budget as the reference, so the
    accuracy deltas are attributable to the removed component.
    """
    nfd = path_graphs[0].x.shape[1]
    efd = path_graphs[0].edge_attr.shape[1] if path_graphs[0].edge_attr is not None else None
    fold_splits = make_fold_splits(path_labels, cfg)
    results: dict[str, dict] = {}

    logger.info("Running architecture ablation (%d-fold) ...", cfg.n_folds)

    arms = [
        ("Full GATv2 Hybrid (reference)", dict(model_class=GATv2Classifier, balanced=True)),
        ("- GATv2 -> GCN layer", dict(model_class=_GCNClassifier, balanced=True)),
        ("- Multi-head attention (1 head)", dict(model_class=GATv2SingleHead, balanced=True)),
        ("- Class weighting / sampler", dict(model_class=GATv2Classifier, balanced=False)),
    ]
    for name, kw in arms:
        results[name] = _cv_arm(path_graphs, path_labels, cfg, device, fold_splits,
                                nfd=nfd, efd=efd, **kw)
        r = results[name]
        logger.info("  %-34s acc %.3f | F1 %.3f | AUC %.3f | MCC %.3f", name,
                    np.mean(r["acc"]), np.mean(r["f1"]),
                    np.nanmean(r["auc"]), np.nanmean(r["mcc"]))

    # -- "- ImageNet pretraining": needs graphs rebuilt from a random encoder --
    # Rebuilding the whole graph cache with an untrained encoder is expensive;
    # the caller does that once and passes the alternative graph list in via
    # run_pretraining_ablation() below.

    # -- "- Superpixel graph (CNN only)": raw pixels, no graph at all ---------
    if path_items is not None:
        from cxr_gnn.models.cnn_baseline import train_cnn_baseline, eval_cnn_baseline
        per_fold = {"acc": [], "f1": [], "auc": [], "mcc": []}
        for tr_idx, te_idx in fold_splits:
            tr2, va2 = safe_train_test_split(
                tr_idx, path_labels, 0.15, cfg.seed, "CNN-only arm")
            model, _s = train_cnn_baseline([path_items[i] for i in tr2],
                                           [path_items[i] for i in va2], cfg, device)
            ty, tp, tpr = eval_cnn_baseline(model, [path_items[i] for i in te_idx], cfg, device)
            for k, v in _score(ty, tp, tpr).items():
                per_fold[k].append(v)
        results["- Superpixel graph (CNN only)"] = {
            **{k: [float(x) for x in v] for k, v in per_fold.items()},
            "params": count_parameters(model),
            "inference_ms": float("nan"),
        }
        logger.info("  %-34s acc %.3f | F1 %.3f", "- Superpixel graph (CNN only)",
                    np.mean(per_fold["acc"]), np.mean(per_fold["f1"]))

    with open(os.path.join(work_dir, "ablation_architecture.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved ablation_architecture.json")
    return results


def run_pretraining_ablation(
    graphs_random_encoder: list[Data],
    path_labels: np.ndarray,
    cfg: Config,
    device: torch.device,
    work_dir: str,
) -> dict:
    """The "- ImageNet pretraining" arm.

    Args:
        graphs_random_encoder: graphs built with build_encoder(pretrained=False),
            i.e. identical pipeline, randomly initialised feature extractor.
    """
    nfd = graphs_random_encoder[0].x.shape[1]
    efd = (graphs_random_encoder[0].edge_attr.shape[1]
           if graphs_random_encoder[0].edge_attr is not None else None)
    fold_splits = make_fold_splits(path_labels, cfg)

    res = _cv_arm(graphs_random_encoder, path_labels, cfg, device, fold_splits,
                  nfd=nfd, efd=efd)
    logger.info("  %-34s acc %.3f | F1 %.3f", "- ImageNet pretraining",
                np.mean(res["acc"]), np.mean(res["f1"]))

    out_path = os.path.join(work_dir, "ablation_pretraining.json")
    with open(out_path, "w") as f:
        json.dump({"- ImageNet pretraining": res}, f, indent=2)
    return {"- ImageNet pretraining": res}

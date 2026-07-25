"""
training/crossval.py -- 5-fold CV, baseline comparison, and the pairwise
statistical suite (SAP 3 and SAP 6).

Design notes:
  - Each fold builds a fresh model and optimiser; nothing is shared between folds.
  - Every model, including the raw-pixel CNN baseline, is evaluated on exactly
    the same held-out instances per fold. McNemar and DeLong are PAIRED tests;
    they are invalid otherwise, so the fold splits are computed once and reused.
  - All pairwise comparisons are reported, not just "primary vs rest", because
    the interesting comparison in this project (CNN vs each GNN) is not a
    primary-model comparison.
"""

from __future__ import annotations

import copy
import json
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    matthews_corrcoef, cohen_kappa_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool

from cxr_gnn.config import Config, IDX2CLASS, NUM_CLASSES, DISPLAY_NAMES
from cxr_gnn.models.gatv2 import GATv2Classifier
from cxr_gnn.evaluation.stats import (
    bootstrap_ci, format_ci, macro_auc, brier_score_multiclass, youdens_j_macro,
    multiclass_log_loss, expected_calibration_error, per_class_sens_spec_ppv_npv,
    delong_test_macro, mcnemar_test, holm_bonferroni, benjamini_hochberg,
    paired_significance,
)
from cxr_gnn.utils import get_logger, set_seed, count_parameters

logger = get_logger(__name__)


# -- Fold helpers -------------------------------------------------------------

def _make_loader(graphs: list, batch_size: int, train: bool = False,
                 seed: int = 42, balanced: bool = True) -> DataLoader:
    if train and balanced:
        labels = np.array([int(g.y) for g in graphs])
        cc = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
        w = (1.0 / np.clip(cc, 1, None))[labels]
        g = torch.Generator().manual_seed(seed)
        smp = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(w), True, generator=g)
        return DataLoader(graphs, batch_size=batch_size, sampler=smp, num_workers=0)
    if train:
        return DataLoader(graphs, batch_size=batch_size, shuffle=True, num_workers=0)
    return DataLoader(graphs, batch_size=batch_size, shuffle=False, num_workers=0)


@torch.no_grad()
def _eval_model(model: nn.Module, loader: DataLoader,
                device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (y_true, y_pred, softmax_probs)."""
    model.eval()
    ys, ps, prs = [], [], []
    for b in loader:
        b = b.to(device)
        out = model(b.x, b.edge_index, b.edge_attr, b.batch)
        ys.append(b.y.cpu())
        ps.append(out.argmax(1).cpu())
        prs.append(F.softmax(out, 1).cpu())
    return (torch.cat(ys).numpy(), torch.cat(ps).numpy(), torch.cat(prs).numpy())


@torch.no_grad()
def measure_latency(model: nn.Module, graphs: list, device: torch.device,
                    n_warmup: int = 5, n_iter: int = 30) -> float:
    """Median single-graph inference latency in milliseconds.

    Warm-up iterations are discarded (first CUDA call includes kernel
    autotuning) and the median is used rather than the mean so one scheduler
    hiccup cannot dominate the reported cost.
    """
    if not graphs:
        return float("nan")
    model.eval()
    g0 = graphs[0].to(device)
    batch = torch.zeros(g0.x.size(0), dtype=torch.long, device=device)

    for _ in range(n_warmup):
        model(g0.x, g0.edge_index, g0.edge_attr, batch)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        model(g0.x, g0.edge_index, g0.edge_attr, batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def _train_one_fold(
    tr_graphs: list,
    va_graphs: list,
    node_feat_dim: int,
    edge_feat_dim: Optional[int],
    cfg: Config,
    device: torch.device,
    epochs: int,
    patience: int,
    model_class=None,
    seed: Optional[int] = None,
    balanced: bool = True,
) -> tuple[nn.Module, float]:
    """Train one fold. Returns (best model, train_seconds).

    `seed` controls model init / sampler / dropout. The split-sensitivity study
    varies it per run; the original code always used cfg.seed, so its "5 seeds"
    only varied the data partition and left model initialisation identical.
    """
    run_seed = cfg.seed if seed is None else seed
    set_seed(run_seed)
    t0 = time.time()

    model_class = model_class or GATv2Classifier
    model = model_class(node_feat_dim, edge_feat_dim, cfg).to(device)

    if balanced:
        labels = np.array([int(g.y) for g in tr_graphs])
        cc = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
        loss_w = torch.tensor(cc.sum() / (NUM_CLASSES * np.clip(cc, 1, None)),
                              dtype=torch.float32, device=device)
    else:
        loss_w = None

    crit = nn.CrossEntropyLoss(weight=loss_w, label_smoothing=cfg.label_smooth)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                     patience=cfg.patience, min_lr=1e-6)

    tl = _make_loader(tr_graphs, cfg.batch_size, train=True, seed=run_seed, balanced=balanced)
    vl = _make_loader(va_graphs, cfg.batch_size)

    best_vloss = float("inf")
    best_state = None
    since_best = 0

    for _epoch in range(epochs):
        model.train(True)
        for b in tl:
            b = b.to(device)
            opt.zero_grad()
            loss = crit(model(b.x, b.edge_index, b.edge_attr, b.batch), b.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

        model.eval()
        vtot, vn = 0.0, 0
        with torch.no_grad():
            for b in vl:
                b = b.to(device)
                vtot += float(crit(model(b.x, b.edge_index, b.edge_attr, b.batch), b.y).detach()) * b.num_graphs
                vn += b.num_graphs
        vloss = vtot / max(1, vn)
        sch.step(vloss)

        if vloss < best_vloss - 1e-4:
            best_vloss = vloss
            best_state = copy.deepcopy(model.state_dict())
            since_best = 0
        else:
            since_best += 1
        if since_best >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, time.time() - t0


def safe_train_test_split(
    idx: np.ndarray,
    labels: np.ndarray,
    test_size: float,
    seed: int,
    context: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified split of `idx`, falling back to unstratified when it cannot
    be stratified.

    sklearn refuses to stratify when any class has fewer than 2 members in the
    subset, which happens on the minority classes (ChronicLung, Pleural) as soon
    as a fold is small -- and it raises rather than degrading. A crash deep
    inside conformal prediction is a worse outcome than an unstratified split
    that says so, so this warns and continues.
    """
    sub = labels[idx]
    _, counts = np.unique(sub, return_counts=True)
    n_test = int(round(len(idx) * test_size))

    if counts.min() >= 2 and n_test >= len(counts) and (len(idx) - n_test) >= len(counts):
        return train_test_split(idx, test_size=test_size, random_state=seed, stratify=sub)

    logger.warning(
        "%sCannot stratify a split of %d items (smallest class has %d member(s)); "
        "falling back to an unstratified split.",
        f"[{context}] " if context else "", len(idx), int(counts.min()),
    )
    return train_test_split(idx, test_size=test_size, random_state=seed)


def make_fold_splits(labels: np.ndarray, cfg: Config) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified fold indices, computed once and shared by every analysis."""
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    return list(skf.split(np.zeros(len(labels)), labels))


# -- 5-fold cross-validation --------------------------------------------------

def run_cross_validation(
    path_graphs: list[Data],
    path_labels: np.ndarray,
    cfg: Config,
    device: torch.device,
    work_dir: str,
) -> dict:
    """5-fold stratified CV on original (non-augmented) graphs.

    Reports MCC per fold as well as accuracy/balanced-accuracy/F1/AUC, since
    MCC is the imbalance-robust headline number for this dataset.
    """
    node_feat_dim = path_graphs[0].x.shape[1]
    edge_feat_dim = (path_graphs[0].edge_attr.shape[1]
                     if path_graphs[0].edge_attr is not None else None)

    rows, oof_y, oof_p, oof_pr = [], [], [], []
    fold_ns: list[dict] = []

    for fold, (tr_idx, te_idx) in enumerate(make_fold_splits(path_labels, cfg), 1):
        tr2, va2 = safe_train_test_split(
            tr_idx, path_labels, 0.15, cfg.seed, "cross-validation")
        model, _secs = _train_one_fold(
            [path_graphs[i] for i in tr2], [path_graphs[i] for i in va2],
            node_feat_dim, edge_feat_dim, cfg, device,
            epochs=cfg.cv_epochs, patience=cfg.cv_patience,
        )
        ty, tp, tpr = _eval_model(
            model, _make_loader([path_graphs[i] for i in te_idx], cfg.batch_size), device)

        acc = accuracy_score(ty, tp)
        bacc = balanced_accuracy_score(ty, tp)
        mf1 = f1_score(ty, tp, average="macro")
        mauc = macro_auc(ty, tpr, NUM_CLASSES)
        mcc = matthews_corrcoef(ty, tp) if len(np.unique(ty)) > 1 else float("nan")

        rows.append([acc, bacc, mf1, mauc, mcc])
        oof_y.append(ty)
        oof_p.append(tp)
        oof_pr.append(tpr)
        fold_ns.append({"fold": fold, "n_train": int(len(tr2)),
                        "n_val": int(len(va2)), "n_test": int(len(te_idx))})
        logger.info("Fold %d: acc %.4f | bal-acc %.4f | F1 %.4f | AUC %.4f | MCC %.4f | "
                    "n(tr/va/te)=%d/%d/%d",
                    fold, acc, bacc, mf1, mauc, mcc, len(tr2), len(va2), len(te_idx))

    rows_np = np.array(rows)
    metric_names = ["Accuracy", "Balanced-Acc", "Macro-F1", "Macro-AUC", "MCC"]

    logger.info("=" * 50)
    logger.info("5-FOLD CV SUMMARY")
    for j, nm in enumerate(metric_names):
        logger.info("  %-14s: %.4f +/- %.4f", nm,
                    np.nanmean(rows_np[:, j]), np.nanstd(rows_np[:, j]))

    cv_results = {
        nm: {"mean": float(np.nanmean(rows_np[:, j])),
             "std": float(np.nanstd(rows_np[:, j])),
             "per_fold": [float(v) for v in rows_np[:, j]]}
        for j, nm in enumerate(metric_names)
    }
    cv_results["fold_sizes"] = fold_ns
    cv_results["oof_y"] = np.concatenate(oof_y).tolist()
    cv_results["oof_p"] = np.concatenate(oof_p).tolist()

    out_path = os.path.join(work_dir, "cv_results.json")
    with open(out_path, "w") as f:
        json.dump(cv_results, f, indent=2)
    logger.info("Saved %s", out_path)

    return cv_results


# -- Baseline architectures ---------------------------------------------------

class _GCNClassifier(nn.Module):
    """GCN baseline. GCNConv has no edge-feature channel, so edge_attr is
    accepted and ignored -- kept in the signature for a uniform call site.
    """

    def __init__(self, nfd, efd, cfg, n_classes=NUM_CLASSES):
        super().__init__()
        from torch_geometric.nn import GCNConv
        H = cfg.hidden
        self.bn_in = nn.BatchNorm1d(nfd)
        self.c1 = GCNConv(nfd, H * cfg.heads)
        self.bn1 = nn.BatchNorm1d(H * cfg.heads)
        self.c2 = GCNConv(H * cfg.heads, H)
        self.bn2 = nn.BatchNorm1d(H)
        self.head = nn.Sequential(nn.Linear(H * 2, H), nn.ELU(),
                                  nn.Dropout(cfg.dropout), nn.Linear(H, n_classes))

    def forward(self, x, edge_index, edge_attr, batch, **kw):
        x = F.elu(self.bn1(self.c1(self.bn_in(x), edge_index)))
        x = F.elu(self.bn2(self.c2(x, edge_index)))
        return self.head(torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], 1))


class _SAGEClassifier(nn.Module):
    def __init__(self, nfd, efd, cfg, n_classes=NUM_CLASSES):
        super().__init__()
        from torch_geometric.nn import SAGEConv
        H = cfg.hidden
        self.bn_in = nn.BatchNorm1d(nfd)
        self.c1 = SAGEConv(nfd, H * cfg.heads)
        self.bn1 = nn.BatchNorm1d(H * cfg.heads)
        self.c2 = SAGEConv(H * cfg.heads, H)
        self.bn2 = nn.BatchNorm1d(H)
        self.head = nn.Sequential(nn.Linear(H * 2, H), nn.ELU(),
                                  nn.Dropout(cfg.dropout), nn.Linear(H, n_classes))

    def forward(self, x, edge_index, edge_attr, batch, **kw):
        x = F.elu(self.bn1(self.c1(self.bn_in(x), edge_index)))
        x = F.elu(self.bn2(self.c2(x, edge_index)))
        return self.head(torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], 1))


class _GATClassifier(nn.Module):
    def __init__(self, nfd, efd, cfg, n_classes=NUM_CLASSES):
        super().__init__()
        from torch_geometric.nn import GATConv
        H, K = cfg.hidden, cfg.heads
        self.bn_in = nn.BatchNorm1d(nfd)
        self.c1 = GATConv(nfd, H, heads=K, dropout=cfg.dropout, edge_dim=efd, concat=True)
        self.bn1 = nn.BatchNorm1d(H * K)
        self.c2 = GATConv(H * K, H, heads=1, dropout=cfg.dropout, edge_dim=efd, concat=True)
        self.bn2 = nn.BatchNorm1d(H)
        self.head = nn.Sequential(nn.Linear(H * 2, H), nn.ELU(),
                                  nn.Dropout(cfg.dropout), nn.Linear(H, n_classes))

    def forward(self, x, edge_index, edge_attr, batch, **kw):
        x = F.elu(self.bn1(self.c1(self.bn_in(x), edge_index, edge_attr)))
        x = F.elu(self.bn2(self.c2(x, edge_index, edge_attr)))
        return self.head(torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], 1))


MODEL_REGISTRY = {
    "GCN": _GCNClassifier,
    "GraphSAGE": _SAGEClassifier,
    "GAT": _GATClassifier,
    "GATv2 Hybrid (ours)": GATv2Classifier,
}

CNN_BASELINE_NAME = "ResNet18 (CNN baseline)"
PRIMARY_MODEL_NAME = "GATv2 Hybrid (ours)"


# -- Model comparison ---------------------------------------------------------

def run_model_comparison(
    path_graphs: list[Data],
    path_labels: np.ndarray,
    cfg: Config,
    device: torch.device,
    work_dir: str,
    path_items: Optional[list[tuple[str, int]]] = None,
    include_cnn_baseline: bool = True,
) -> dict:
    """GNN baselines + optional end-to-end ResNet18, with the full SAP 6 suite.

    Args:
        path_items: (path, class_idx) list INDEX-ALIGNED with path_graphs.
                    Required by the CNN baseline, which trains on raw pixels.
                    Without it the CNN baseline is skipped.
    """
    nfd = path_graphs[0].x.shape[1]
    efd = path_graphs[0].edge_attr.shape[1] if path_graphs[0].edge_attr is not None else None

    # Computed once: every model sees identical held-out instances per fold.
    fold_splits = make_fold_splits(path_labels, cfg)

    RESULTS: dict[str, dict] = {}

    for model_name, model_cls in MODEL_REGISTRY.items():
        fold_acc, fold_f1, fold_auc, fold_secs = [], [], [], []
        fold_oof_y, fold_oof_p, fold_oof_pr = [], [], []
        last_model = None

        for _fold, (tr_idx, te_idx) in enumerate(fold_splits, 1):
            tr2, va2 = safe_train_test_split(
                tr_idx, path_labels, 0.15, cfg.seed, "model comparison")
            model, secs = _train_one_fold(
                [path_graphs[i] for i in tr2], [path_graphs[i] for i in va2],
                nfd, efd, cfg, device,
                epochs=cfg.cv_epochs, patience=cfg.cv_patience, model_class=model_cls,
            )
            ty, tp, tpr = _eval_model(
                model, _make_loader([path_graphs[i] for i in te_idx], cfg.batch_size), device)
            fold_acc.append(accuracy_score(ty, tp))
            fold_f1.append(f1_score(ty, tp, average="macro"))
            fold_auc.append(macro_auc(ty, tpr, NUM_CLASSES))
            fold_secs.append(secs)
            fold_oof_y.append(ty)
            fold_oof_p.append(tp)
            fold_oof_pr.append(tpr)
            last_model = model

        RESULTS[model_name] = {
            "acc": fold_acc, "f1": fold_f1, "auc": fold_auc,
            "oof_y": np.concatenate(fold_oof_y),
            "oof_p": np.concatenate(fold_oof_p),
            "oof_proba": np.concatenate(fold_oof_pr),
            "params": count_parameters(last_model),
            "train_seconds_per_fold": float(np.mean(fold_secs)),
            "inference_ms": measure_latency(last_model, path_graphs, device),
        }
        logger.info("  %-22s: acc %.3f | F1 %.3f | AUC %.3f | %d params | "
                    "%.1f s/fold | %.1f ms/graph",
                    model_name, np.mean(fold_acc), np.mean(fold_f1), np.nanmean(fold_auc),
                    RESULTS[model_name]["params"],
                    RESULTS[model_name]["train_seconds_per_fold"],
                    RESULTS[model_name]["inference_ms"])

    # -- End-to-end ResNet18 on raw pixels, same folds ------------------------
    if include_cnn_baseline and path_items is not None:
        from cxr_gnn.models.cnn_baseline import (
            train_cnn_baseline, eval_cnn_baseline, build_cnn_baseline)

        fold_acc, fold_f1, fold_auc, fold_secs = [], [], [], []
        fold_oof_y, fold_oof_p, fold_oof_pr = [], [], []

        for fold, (tr_idx, te_idx) in enumerate(fold_splits, 1):
            tr2, va2 = safe_train_test_split(
                tr_idx, path_labels, 0.15, cfg.seed, "CNN baseline")
            model, secs = train_cnn_baseline(
                [path_items[i] for i in tr2], [path_items[i] for i in va2],
                cfg, device, epochs=cfg.cnn_epochs, patience=cfg.cnn_patience)
            ty, tp, tpr = eval_cnn_baseline(model, [path_items[i] for i in te_idx], cfg, device)

            fold_acc.append(accuracy_score(ty, tp))
            fold_f1.append(f1_score(ty, tp, average="macro"))
            fold_auc.append(macro_auc(ty, tpr, NUM_CLASSES))
            fold_secs.append(secs)
            fold_oof_y.append(ty)
            fold_oof_p.append(tp)
            fold_oof_pr.append(tpr)
            logger.info("  %s fold %d: acc %.3f (%.0f s)", CNN_BASELINE_NAME, fold,
                        fold_acc[-1], secs)

        # Latency on a single image, measured the same way as the GNNs.
        cnn_probe = build_cnn_baseline(device)
        cnn_probe.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, cfg.img_size, cfg.img_size, device=device)
            for _ in range(5):
                cnn_probe(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t = []
            for _ in range(30):
                t0 = time.perf_counter()
                cnn_probe(dummy)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t.append((time.perf_counter() - t0) * 1000)

        RESULTS[CNN_BASELINE_NAME] = {
            "acc": fold_acc, "f1": fold_f1, "auc": fold_auc,
            "oof_y": np.concatenate(fold_oof_y),
            "oof_p": np.concatenate(fold_oof_p),
            "oof_proba": np.concatenate(fold_oof_pr),
            "params": count_parameters(cnn_probe),
            "train_seconds_per_fold": float(np.mean(fold_secs)),
            "inference_ms": float(np.median(t)),
        }
        logger.info("  %-22s: acc %.3f | F1 %.3f | AUC %.3f", CNN_BASELINE_NAME,
                    np.mean(fold_acc), np.mean(fold_f1), np.nanmean(fold_auc))
    elif include_cnn_baseline:
        logger.warning("include_cnn_baseline=True but path_items is None -- skipping %s. "
                       "Pass the index-aligned (path, label) list from build_or_load_cache's "
                       "kept_items.", CNN_BASELINE_NAME)

    stat_summary = _run_stat_tests(RESULTS, cfg, work_dir)

    saveable = {
        k: {"acc": [float(v) for v in d["acc"]],
            "f1": [float(v) for v in d["f1"]],
            "auc": [float(v) for v in d["auc"]],
            "params": int(d["params"]),
            "train_seconds_per_fold": float(d["train_seconds_per_fold"]),
            "inference_ms": float(d["inference_ms"])}
        for k, d in RESULTS.items()
    }
    with open(os.path.join(work_dir, "baseline_results.json"), "w") as f:
        json.dump(saveable, f, indent=2)
    with open(os.path.join(work_dir, "statistical_robustness.json"), "w") as f:
        json.dump(stat_summary, f, indent=2,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    logger.info("Saved baseline_results.json and statistical_robustness.json")

    return RESULTS


def _run_stat_tests(RESULTS: dict, cfg: Config, work_dir: str) -> dict:
    """SAP 6.1 model-level metrics with bootstrap CIs, 6.2 per-class Wilson CIs
    for the primary model, 6.3 ALL pairwise DeLong + McNemar with Holm/FDR
    correction, plus the paired t-test/Wilcoxon on fold-wise macro-F1.
    """
    PROP = PRIMARY_MODEL_NAME if PRIMARY_MODEL_NAME in RESULTS else next(iter(RESULTS))
    prop = RESULTS[PROP]

    # -- 6.1 model-level metrics ---------------------------------------------
    model_level: dict[str, dict] = {}
    for m, d in RESULTS.items():
        y, p, pr = d["oof_y"], d["oof_p"], d["oof_proba"]
        mcc = bootstrap_ci(y, p, lambda yt, yp: matthews_corrcoef(yt, yp), cfg.n_bootstrap, cfg.seed)
        kap = bootstrap_ci(y, p, lambda yt, yp: cohen_kappa_score(yt, yp), cfg.n_bootstrap, cfg.seed)
        acc = bootstrap_ci(y, p, accuracy_score, cfg.n_bootstrap, cfg.seed)
        ece, mce, _ = expected_calibration_error(y, pr)
        model_level[m] = {
            "accuracy": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro")),
            "macro_auc": float(macro_auc(y, pr, NUM_CLASSES)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p)),
            "accuracy_ci": format_ci(*acc),
            "mcc": format_ci(*mcc), "mcc_point": float(mcc[0]),
            "cohens_kappa": format_ci(*kap), "kappa_point": float(kap[0]),
            "youdens_j": round(youdens_j_macro(y, p, NUM_CLASSES), 4),
            "brier_score": round(brier_score_multiclass(y, pr, NUM_CLASSES), 4),
            "log_loss": round(multiclass_log_loss(y, pr, NUM_CLASSES), 4),
            "ece": round(ece, 4), "mce": round(mce, 4),
            "params": int(d["params"]),
            "params_millions": round(d["params"] / 1e6, 4),
            "train_minutes_per_fold": round(d["train_seconds_per_fold"] / 60.0, 2),
            "inference_ms": round(float(d["inference_ms"]), 2),
        }

    logger.info("\n[SAP 6.1] Model-level metrics (bootstrap 95%% CI, B=%d):", cfg.n_bootstrap)
    for m, d in model_level.items():
        logger.info("  %-22s MCC=%s | kappa=%s | Youden-J=%.3f | Brier=%.3f | ECE=%.3f",
                    m, d["mcc"], d["cohens_kappa"], d["youdens_j"], d["brier_score"], d["ece"])

    # -- 6.2 per-class operating characteristics for the primary model -------
    class_names = [DISPLAY_NAMES.get(IDX2CLASS[i], IDX2CLASS[i]) for i in range(NUM_CLASSES)]
    per_class_ci = per_class_sens_spec_ppv_npv(
        prop["oof_y"], prop["oof_p"], NUM_CLASSES, class_names, proba=prop["oof_proba"])
    logger.info("\n[SAP 6.2] Per-class Sens/Spec/PPV/NPV (Wilson 95%% CI) -- %s:", PROP)
    for cls, v in per_class_ci.items():
        logger.info("  %-22s Sens=%.3f (%.3f-%.3f) Spec=%.3f (%.3f-%.3f) "
                    "PPV=%.3f (%.3f-%.3f) NPV=%.3f (%.3f-%.3f) n=%d",
                    cls, v["sensitivity"], *v["sensitivity_ci"], v["specificity"],
                    *v["specificity_ci"], v["ppv"], *v["ppv_ci"], v["npv"], *v["npv_ci"],
                    v["support"])

    # -- 6.3 ALL pairwise comparisons ----------------------------------------
    names = list(RESULTS)
    pairwise_rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            da, db = RESULTS[a], RESULTS[b]
            # Sanity: paired tests require identical instance ordering.
            if not np.array_equal(da["oof_y"], db["oof_y"]):
                logger.warning("Skipping %s vs %s: out-of-fold label vectors differ, "
                               "so the paired tests would be invalid.", a, b)
                continue

            dl = delong_test_macro(da["oof_y"], da["oof_proba"], db["oof_proba"], NUM_CLASSES)
            mc = mcnemar_test(da["oof_y"], da["oof_p"], db["oof_p"])

            pairwise_rows.append({
                "model_a": a, "model_b": b, "metric": "Macro-AUC",
                "test": "DeLong (one-vs-rest, Fisher-combined)",
                "statistic_z": dl["z_macro"], "fisher_chi2": dl["fisher_stat"],
                "delta": float(dl["auc_a_macro"] - dl["auc_b_macro"]),
                "p_value": dl["p_combined"],
            })
            pairwise_rows.append({
                "model_a": a, "model_b": b, "metric": "Accuracy",
                "test": "McNemar (exact)",
                "statistic_z": float("nan"), "fisher_chi2": mc["chi2_cc"],
                "delta": float(mc["acc_a"] - mc["acc_b"]),
                "p_value": mc["p"],
                "discordant_n01_n10": [mc["n01"], mc["n10"]],
            })

    if pairwise_rows:
        raw_p = [r["p_value"] if np.isfinite(r["p_value"]) else 1.0 for r in pairwise_rows]
        holm = holm_bonferroni(raw_p)
        fdr = benjamini_hochberg(raw_p)
        for r, hp, fp in zip(pairwise_rows, holm, fdr):
            r["holm_corrected_p"] = float(hp)
            r["fdr_corrected_p"] = float(fp)
            r["significant_holm_alpha_0.05"] = bool(hp < 0.05)

    logger.info("\n[SAP 6.3] Pairwise comparisons (Holm-corrected over %d tests):",
                len(pairwise_rows))
    for r in pairwise_rows:
        logger.info("  %-22s vs %-22s %-10s p=%.3g holm-p=%.3g %s",
                    r["model_a"], r["model_b"], r["metric"], r["p_value"],
                    r["holm_corrected_p"],
                    "SIGNIFICANT" if r["significant_holm_alpha_0.05"] else "n.s.")

    # -- Fold-wise paired tests on macro-F1 ----------------------------------
    legacy_rows = []
    for m, d in RESULTS.items():
        if m == PROP:
            continue
        res = paired_significance(prop["f1"], d["f1"])
        legacy_rows.append({"model_b": m, "delta_f1": res["mean_diff"],
                            "t_stat": res.get("t_stat", float("nan")),
                            "t_test_p": res["t_p"], "wilcoxon_p": res["wilcoxon_p"],
                            "cohens_d": res["cohens_d"],
                            "effect_size": res["effect_size_label"]})

    return {
        "primary_model": PROP,
        "model_level_metrics": model_level,
        "per_class_sens_spec_ppv_npv": {
            cls: {k: (list(v) if isinstance(v, tuple) else v) for k, v in row.items()}
            for cls, row in per_class_ci.items()
        },
        "pairwise_delong_mcnemar": pairwise_rows,
        "paired_ttest_wilcoxon_macroF1": legacy_rows,
        "oof_predictions": {
            m: {"y": d["oof_y"].tolist(), "pred": d["oof_p"].tolist()}
            for m, d in RESULTS.items()
        },
        "bootstrap_protocol": {
            "method": "non-parametric percentile bootstrap, paired resampling",
            "n_iterations": cfg.n_bootstrap,
            "seed": cfg.seed,
            "software": "numpy.random.default_rng + scikit-learn + scipy.stats",
        },
    }

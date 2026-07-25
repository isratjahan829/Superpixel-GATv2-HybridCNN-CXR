"""
evaluation/conformal.py -- Conformal prediction (LAC, APS, RAPS, class-cond LAC).

Marginal APS produced an average set size of ~3.4 out of 5 classes, which
carries almost no clinical information. RAPS adds a penalty on set size and
keeps coverage while shrinking the sets:

    APS:  large sets, over-covers
    LAC:  smallest sets, but under-covers some classes
    RAPS: the usable trade-off  <- default

References:
    APS  -- Romano et al., 2020
    RAPS -- Angelopoulos et al., 2021
    LAC  -- Sadinle et al., 2019
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from cxr_gnn.config import Config, NUM_CLASSES
from cxr_gnn.training.crossval import make_fold_splits, safe_train_test_split
from cxr_gnn.utils import get_logger

logger = get_logger(__name__)


# -- Score functions ----------------------------------------------------------

def _qhat(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample corrected quantile of the calibration scores."""
    n = len(scores)
    if n == 0:
        return 1.0
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def lac_calibrate(cal_p: np.ndarray, cal_y: np.ndarray, alpha: float) -> float:
    """LAC threshold; score = 1 - p[true class]."""
    return _qhat(1 - cal_p[np.arange(len(cal_y)), cal_y], alpha)


def lac_predict(probs: np.ndarray, qhat: float) -> list[set]:
    return [
        {k for k in range(probs.shape[1]) if (1 - probs[i, k]) <= qhat}
        or {int(np.argmax(probs[i]))}          # never emit an empty set
        for i in range(len(probs))
    ]


def aps_calibrate(cal_p: np.ndarray, cal_y: np.ndarray, alpha: float) -> float:
    order = np.argsort(-cal_p, axis=1)
    scores = np.ones(len(cal_y))
    for i in range(len(cal_y)):
        cum = 0.0
        for c in order[i]:
            cum += cal_p[i, c]
            if c == cal_y[i]:
                scores[i] = cum
                break
    return _qhat(scores, alpha)


def aps_predict(probs: np.ndarray, qhat: float) -> list[set]:
    order = np.argsort(-probs, axis=1)
    sets = []
    for i in range(len(probs)):
        cum, s = 0.0, []
        for c in order[i]:
            s.append(int(c))
            cum += probs[i, c]
            if cum >= qhat:
                break
        sets.append(set(s))
    return sets


def raps_calibrate(cal_p: np.ndarray, cal_y: np.ndarray, alpha: float,
                   k_reg: int, lam: float) -> float:
    """RAPS score = cumulative probability + penalty for ranks beyond k_reg."""
    order = np.argsort(-cal_p, axis=1)
    scores = np.ones(len(cal_y))
    for i in range(len(cal_y)):
        cum = 0.0
        for rank, c in enumerate(order[i]):
            cum += cal_p[i, c] + lam * max(0, rank + 1 - k_reg)
            if c == cal_y[i]:
                scores[i] = cum
                break
    return _qhat(scores, alpha)


def raps_predict(probs: np.ndarray, qhat: float, k_reg: int, lam: float) -> list[set]:
    order = np.argsort(-probs, axis=1)
    sets = []
    for i in range(len(probs)):
        cum, s = 0.0, []
        for rank, c in enumerate(order[i]):
            s.append(int(c))
            cum += probs[i, c] + lam * max(0, rank + 1 - k_reg)
            if cum >= qhat:
                break
        sets.append(set(s))
    return sets


def cclac_calibrate(cal_p: np.ndarray, cal_y: np.ndarray, alpha: float) -> np.ndarray:
    """Class-conditional LAC: one threshold per class."""
    qk = np.ones(NUM_CLASSES)
    for k in range(NUM_CLASSES):
        idx = cal_y == k
        if int(idx.sum()) == 0:
            continue      # class absent from the calibration split; keep qk=1
        qk[k] = _qhat(1 - cal_p[idx, k], alpha)
    return qk


def cclac_predict(probs: np.ndarray, qk: np.ndarray) -> list[set]:
    return [
        {k for k in range(NUM_CLASSES) if (1 - probs[i, k]) <= qk[k]}
        or {int(np.argmax(probs[i]))}
        for i in range(len(probs))
    ]


# -- Fold evaluation ----------------------------------------------------------

def _coverage_and_size(sets, y):
    cov = float(np.mean([y[i] in sets[i] for i in range(len(y))]))
    size = float(np.mean([len(s) for s in sets]))
    sing = float(np.mean([len(s) == 1 for s in sets]))
    return cov, size, sing


@torch.no_grad()
def _get_probs(model, loader, device):
    model.eval()
    ys, prs = [], []
    for b in loader:
        b = b.to(device)
        out = model(b.x, b.edge_index, b.edge_attr, b.batch)
        ys.append(b.y.cpu())
        prs.append(F.softmax(out, 1).cpu())
    return torch.cat(ys).numpy(), torch.cat(prs).numpy()


def run_conformal(
    path_graphs,
    path_labels: np.ndarray,
    train_fold_fn,          # callable(tr_graphs, va_graphs) -> model
    cfg: Config,
    device: torch.device,
    work_dir: str,
) -> dict:
    """5-fold conformal prediction with a 50/50 calibration/test split per fold."""
    alpha = cfg.conformal_alpha
    methods = ["lac", "aps", "raps", "cclac"]
    metrics = {m: {"cov": [], "size": [], "sing": []} for m in methods}

    logger.info("Conformal prediction (target coverage %.0f%%) ...", (1 - alpha) * 100)

    for fold, (tr_idx, te_idx) in enumerate(make_fold_splits(path_labels, cfg), 1):
        tr2, va2 = safe_train_test_split(
            tr_idx, path_labels, 0.15, cfg.seed, "conformal train/val")
        model = train_fold_fn([path_graphs[i] for i in tr2], [path_graphs[i] for i in va2])

        cal_idx, tst_idx = safe_train_test_split(
            te_idx, path_labels, 0.5, cfg.seed, "conformal cal/test")

        def _loader(idxs):
            return DataLoader([path_graphs[i] for i in idxs],
                              batch_size=cfg.batch_size, shuffle=False)

        cal_y, cal_p = _get_probs(model, _loader(cal_idx), device)
        tst_y, tst_p = _get_probs(model, _loader(tst_idx), device)

        preds = {
            "lac": lac_predict(tst_p, lac_calibrate(cal_p, cal_y, alpha)),
            "aps": aps_predict(tst_p, aps_calibrate(cal_p, cal_y, alpha)),
            "raps": raps_predict(tst_p, raps_calibrate(cal_p, cal_y, alpha,
                                                       cfg.raps_k_reg, cfg.raps_lam),
                                 cfg.raps_k_reg, cfg.raps_lam),
            "cclac": cclac_predict(tst_p, cclac_calibrate(cal_p, cal_y, alpha)),
        }
        for m in methods:
            c, s, sg = _coverage_and_size(preds[m], tst_y)
            metrics[m]["cov"].append(c)
            metrics[m]["size"].append(s)
            metrics[m]["sing"].append(sg)

        logger.info("Fold %d: " + " | ".join(f"{m.upper()} %.2f/%.2f" for m in methods)
                    + "  (cov/size)", fold,
                    *[v for m in methods for v in (metrics[m]["cov"][-1], metrics[m]["size"][-1])])

    logger.info("=" * 60)
    logger.info("CONFORMAL SUMMARY (target %.0f%%)", (1 - alpha) * 100)
    for m in methods:
        logger.info("  %-6s: cov %.3f+/-%.3f | size %.2f | singletons %.1f%%",
                    m.upper(), np.mean(metrics[m]["cov"]), np.std(metrics[m]["cov"]),
                    np.mean(metrics[m]["size"]), np.mean(metrics[m]["sing"]) * 100)
    logger.info("-> Recommended: RAPS (best size/coverage trade-off)")

    results = {m: {k: [float(v) for v in vs] for k, vs in d.items()}
               for m, d in metrics.items()}
    results["alpha"] = alpha
    results["target_coverage"] = 1 - alpha

    out_path = os.path.join(work_dir, "conformal_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved %s", out_path)
    return results

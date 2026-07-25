"""
evaluation/calibration.py -- Probability calibration.

ECE  -- average gap between confidence and accuracy across confidence bins
MCE  -- worst bin gap
ECE < 0.05 reads as well calibrated; ECE > 0.10 is a case for temperature
scaling before any claim about probability trustworthiness.

Temperature scaling is implemented here too, fitted on held-out data, so the
"consider temperature scaling" warning has a remedy attached to it.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch_geometric.loader import DataLoader

from cxr_gnn.evaluation.stats import expected_calibration_error
from cxr_gnn.training.crossval import make_fold_splits, safe_train_test_split
from cxr_gnn.utils import get_logger

logger = get_logger(__name__)


def _collect_oof_probs(path_graphs, path_labels: np.ndarray, train_fold_fn, cfg,
                       device: torch.device):
    """Pooled out-of-fold probabilities via 5-fold CV.

    No blanket @torch.no_grad() on this function: train_fold_fn trains a model
    per fold and needs autograd. Only the inference loop is wrapped.
    """
    all_y, all_p = [], []

    for fold, (tr_idx, te_idx) in enumerate(make_fold_splits(path_labels, cfg), 1):
        tr2, va2 = safe_train_test_split(
            tr_idx, path_labels, 0.15, cfg.seed, "calibration folds")
        model = train_fold_fn([path_graphs[i] for i in tr2], [path_graphs[i] for i in va2])
        model.eval()

        loader = DataLoader([path_graphs[i] for i in te_idx],
                            batch_size=cfg.batch_size, shuffle=False)
        ys, prs = [], []
        with torch.no_grad():
            for b in loader:
                b = b.to(device)
                out = model(b.x, b.edge_index, b.edge_attr, b.batch)
                ys.append(b.y.cpu())
                prs.append(F.softmax(out, 1).cpu())

        y = torch.cat(ys).numpy()
        p = torch.cat(prs).numpy()
        all_y.append(y)
        all_p.append(p)
        logger.debug("Fold %d: OOF accuracy %.3f", fold, accuracy_score(y, p.argmax(1)))

    return np.concatenate(all_y), np.concatenate(all_p)


def fit_temperature(probs: np.ndarray, y: np.ndarray, max_iter: int = 200) -> float:
    """Fit a single temperature T minimising NLL on (probs, y).

    Operates on log-probabilities, so it works from stored softmax outputs
    without needing the original logits.
    """
    logits = torch.log(torch.tensor(np.clip(probs, 1e-12, 1.0), dtype=torch.float64))
    target = torch.tensor(y, dtype=torch.long)
    log_T = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / torch.exp(log_T), target)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_T).item())


def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / T
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def compute_calibration(
    path_graphs,
    path_labels: np.ndarray,
    train_fold_fn,
    cfg,
    device: torch.device,
    work_dir: str,
    n_bins: int = 10,
) -> dict:
    """ECE/MCE before and after temperature scaling; writes calibration.json.

    The temperature is fitted on one half of the pooled out-of-fold predictions
    and evaluated on the other, so the post-scaling ECE is not fitted on the
    data it is reported on.
    """
    y, probs = _collect_oof_probs(path_graphs, path_labels, train_fold_fn, cfg, device)
    conf = probs.max(1)
    pred = probs.argmax(1)

    ece, mce, detail = expected_calibration_error(y, probs, n_bins)
    oof_acc = float((pred == y).mean())
    logger.info("ECE: %.4f | MCE: %.4f | OOF acc: %.4f", ece, mce, oof_acc)

    temp_result = {}
    if ece > 0.10:
        logger.warning("ECE=%.3f > 0.10 -- fitting temperature scaling.", ece)
        fit_idx, eval_idx = safe_train_test_split(
            np.arange(len(y)), y, 0.5, cfg.seed, "temperature scaling")
        T = fit_temperature(probs[fit_idx], y[fit_idx])
        scaled = apply_temperature(probs[eval_idx], T)
        ece_after, mce_after, _ = expected_calibration_error(y[eval_idx], scaled, n_bins)
        ece_before, _, _ = expected_calibration_error(y[eval_idx], probs[eval_idx], n_bins)
        temp_result = {
            "temperature": T,
            "ece_before_on_eval_half": float(ece_before),
            "ece_after": float(ece_after),
            "mce_after": float(mce_after),
        }
        logger.info("Temperature scaling: T=%.3f, ECE %.4f -> %.4f on the held-out half",
                    T, ece_before, ece_after)

    result = {
        "ECE": float(ece), "MCE": float(mce),
        "oof_accuracy": oof_acc,
        "avg_confidence": float(conf.mean()),
        "n_bins": n_bins,
        "bin_accuracy": [x if np.isfinite(x) else None for x in detail["bin_accuracy"]],
        "bin_confidence": detail["bin_confidence"],
        "bin_count": detail["bin_count"],
        "temperature_scaling": temp_result,
    }

    out_path = os.path.join(work_dir, "calibration.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved %s", out_path)
    return result

"""
evaluation/robustness.py -- Performance under input perturbation.

The report contains a "robustness under perturbation" table (Gaussian noise,
JPEG compression, rotation, brightness, contrast). No code produced it in the
original notebook -- the numbers had no computational source. This module
produces them.

Method:
    Take the held-out test images, apply one perturbation at a fixed magnitude,
    rebuild the graphs through the identical SLIC + encoder path, and evaluate
    the already-trained model. The clean test set is the reference row.

    Retention = perturbed accuracy / clean accuracy.
    Grade: A >= 96%, B >= 92%, C >= 88%, D below that.

JPEG needs a real encoder round trip (imageio), so if that is unavailable the
row is reported as skipped rather than silently approximated.
"""

from __future__ import annotations

import io
import json
import os
from typing import Callable

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from skimage.transform import rotate as sk_rotate

from cxr_gnn.config import Config, NUM_CLASSES
from cxr_gnn.data.dataset import load_gray
from cxr_gnn.data.graph import image_to_graph
from cxr_gnn.evaluation.stats import macro_auc
from cxr_gnn.training.crossval import _eval_model, _make_loader
from cxr_gnn.utils import get_logger

logger = get_logger(__name__)


# -- Perturbations ------------------------------------------------------------

def _gaussian_noise(sigma: float) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    def f(img, rng):
        return np.clip(img + rng.normal(0, sigma, img.shape), 0, 1).astype(np.float32)
    return f


def _rotation(deg: float) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    def f(img, rng):
        angle = rng.uniform(-deg, deg)
        return np.clip(sk_rotate(img, angle, mode="edge", preserve_range=True),
                       0, 1).astype(np.float32)
    return f


def _brightness(frac: float) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    def f(img, rng):
        return np.clip(img + rng.choice([-1, 1]) * frac, 0, 1).astype(np.float32)
    return f


def _contrast(reduction: float) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    """Shrink intensities towards the image mean by `reduction` (0.3 = -30%)."""
    def f(img, rng):
        mu = float(img.mean())
        return np.clip(mu + (img - mu) * (1.0 - reduction), 0, 1).astype(np.float32)
    return f


def _jpeg(quality: int) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    def f(img, rng):
        import imageio.v2 as imageio
        buf = io.BytesIO()
        imageio.imwrite(buf, (np.clip(img, 0, 1) * 255).astype(np.uint8),
                        format="JPEG", quality=quality)
        buf.seek(0)
        out = imageio.imread(buf).astype(np.float32) / 255.0
        if out.ndim == 3:
            out = out.mean(axis=2)
        return out.astype(np.float32)
    return f


def default_perturbations() -> dict[str, Callable]:
    return {
        "Gaussian noise (sigma=0.02)": _gaussian_noise(0.02),
        "Gaussian noise (sigma=0.05)": _gaussian_noise(0.05),
        "JPEG compression (q=50)": _jpeg(50),
        "Rotation (+/-10 deg)": _rotation(10.0),
        "Brightness shift (+/-20%)": _brightness(0.20),
        "Contrast reduction (-30%)": _contrast(0.30),
    }


def _grade(retention: float) -> str:
    if not np.isfinite(retention):
        return "n/a"
    if retention >= 0.96:
        return "A"
    if retention >= 0.92:
        return "B"
    if retention >= 0.88:
        return "C"
    return "D"


def run_robustness_study(
    model,
    test_items: list[tuple[str, str]],   # (path, clean_class) -- kept_items["test"]
    cfg: Config,
    encoder,
    device: torch.device,
    work_dir: str,
    perturbations: dict[str, Callable] | None = None,
) -> dict:
    """Evaluate `model` on the clean test set and under each perturbation."""
    from cxr_gnn.config import CLASS2IDX

    perturbations = perturbations or default_perturbations()
    rng_seed = cfg.robustness_seed

    def _build(pert: Callable | None) -> list:
        rng = np.random.default_rng(rng_seed)
        graphs = []
        for path, cls in test_items:
            try:
                img = load_gray(path, cfg.img_size)
                if pert is not None:
                    img = pert(img, rng)
                g = image_to_graph(img, CLASS2IDX[cls], cfg, encoder, device)
                if g is not None:
                    graphs.append(g)
            except Exception as ex:
                logger.warning("robustness: skip %s (%s)", os.path.basename(path), ex)
        return graphs

    def _score(graphs) -> dict:
        if not graphs:
            return {k: float("nan") for k in ("accuracy", "macro_f1", "macro_auc", "mcc")}
        ty, tp, tpr = _eval_model(model, _make_loader(graphs, cfg.batch_size), device)
        return {
            "accuracy": float(accuracy_score(ty, tp)),
            "macro_f1": float(f1_score(ty, tp, average="macro")),
            "macro_auc": float(macro_auc(ty, tpr, NUM_CLASSES)),
            "mcc": float(matthews_corrcoef(ty, tp)) if len(np.unique(ty)) > 1 else float("nan"),
        }

    logger.info("Robustness study: clean reference + %d perturbations ...", len(perturbations))
    clean = _score(_build(None))
    rows = [{"perturbation": "None (clean test set)", **clean,
             "delta_accuracy": 0.0, "retention": 1.0, "grade": "Reference"}]
    logger.info("  %-30s acc %.4f (reference)", "None (clean test set)", clean["accuracy"])

    for name, fn in perturbations.items():
        try:
            sc = _score(_build(fn))
        except ImportError as ex:
            logger.warning("  %-30s SKIPPED (%s)", name, ex)
            rows.append({"perturbation": name, "skipped": True, "reason": str(ex)})
            continue
        retention = (sc["accuracy"] / clean["accuracy"]
                     if clean["accuracy"] else float("nan"))
        rows.append({"perturbation": name, **sc,
                     "delta_accuracy": sc["accuracy"] - clean["accuracy"],
                     "retention": retention, "grade": _grade(retention)})
        logger.info("  %-30s acc %.4f (delta %+.4f, retention %.1f%%, grade %s)",
                    name, sc["accuracy"], rows[-1]["delta_accuracy"],
                    retention * 100, rows[-1]["grade"])

    result = {
        "clean_reference": clean,
        "rows": rows,
        "grading": {"A": ">= 96% retention", "B": ">= 92%", "C": ">= 88%", "D": "< 88%"},
        "n_test_images": len(test_items),
        "seed": rng_seed,
    }
    out_path = os.path.join(work_dir, "robustness.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved %s", out_path)
    return result

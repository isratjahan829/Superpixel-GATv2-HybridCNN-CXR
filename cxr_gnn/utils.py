"""
utils.py -- Seed management, device detection, logging setup.
"""

from __future__ import annotations

import logging
import os
import random
import sys

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every random source used in the pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN: slightly slower, but runs are reproducible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Single place where the device is decided.

    The original notebook picked CUDA in some cells and CPU in others, which
    caused device-mismatch errors when tensors crossed cell boundaries.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_logging(work_dir: str, level: int = logging.INFO) -> logging.Logger:
    """File + console logging.

    Idempotent: calling it twice (easy to do when re-running a notebook cell)
    does not duplicate handlers, which would print every line twice.
    """
    os.makedirs(work_dir, exist_ok=True)
    logger = logging.getLogger("cxr_gnn")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    log_path = os.path.join(work_dir, "run.log")
    existing = {getattr(h, "_cxr_tag", None) for h in logger.handlers}

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if "console" not in existing:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(fmt)
        ch._cxr_tag = "console"          # type: ignore[attr-defined]
        logger.addHandler(ch)

    if "file" not in existing:
        fh = logging.FileHandler(log_path, mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        fh._cxr_tag = "file"             # type: ignore[attr-defined]
        logger.addHandler(fh)

    return logger


def get_logger(name: str = "cxr_gnn") -> logging.Logger:
    return logging.getLogger(name)


def count_parameters(model, trainable_only: bool = True) -> int:
    """Parameter count -- reported in the model-comparison cost columns."""
    ps = model.parameters()
    if trainable_only:
        return int(sum(p.numel() for p in ps if p.requires_grad))
    return int(sum(p.numel() for p in ps))


def write_run_metadata(work_dir: str, cfg, device, n_images: int, data_root: str,
                       synthetic: bool = False) -> dict:
    """Record what produced this run's artifacts.

    Without it, a directory of JSON files says nothing about whether it came
    from the real dataset at full budget or a two-minute smoke test on
    phantoms -- and those are indistinguishable once the numbers reach a table.
    """
    import json
    import subprocess

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        commit = None

    # The shipped defaults are epochs=150, cv_epochs=120, n_bootstrap=2000.
    reduced = cfg.epochs < 50 or cfg.cv_epochs < 50 or cfg.n_bootstrap < 1000

    meta = {
        "data_root": data_root,
        "n_images": int(n_images),
        "synthetic_data": bool(synthetic),
        "reduced_budget": bool(reduced),
        "reportable": bool(not synthetic and not reduced),
        "device": str(device),
        "git_commit": commit,
        "config": cfg.to_dict(),
    }
    os.makedirs(work_dir, exist_ok=True)
    with open(os.path.join(work_dir, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if not meta["reportable"]:
        reasons = [r for r, on in (("synthetic data", synthetic),
                                   ("reduced budgets", reduced)) if on]
        get_logger(__name__).warning(
            "This run is NOT reportable (%s). Its numbers verify that the "
            "pipeline executes; they are not findings.", " and ".join(reasons))
    return meta

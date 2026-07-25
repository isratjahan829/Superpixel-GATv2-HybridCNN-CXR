"""
training/trainer.py -- Training loop for the primary model.

The biggest defect in the original notebook:
    train metrics were measured by the same pass that performed the gradient
    updates, i.e. with dropout and DropEdge ACTIVE. That made train accuracy
    look lower than validation accuracy, which reads as "negative overfitting"
    and is purely an artefact of the measurement.

Split here:
    optimize_epoch() -- gradient updates only, no metrics
    evaluate()       -- always model.eval(), so train and val curves are
                        measured under identical conditions
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader

from cxr_gnn.config import Config, CLASS2IDX
from cxr_gnn.utils import get_logger

logger = get_logger(__name__)


class Trainer:
    """Owns the model, optimiser and criterion for one training run.

    These were globals in the original notebook, so re-running a cell silently
    continued training a previous model.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: Config,
        loss_weights: Optional[torch.Tensor],
        device: torch.device,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.device = device

        self.criterion = nn.CrossEntropyLoss(weight=loss_weights,
                                             label_smoothing=cfg.label_smooth)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                           weight_decay=cfg.wd)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=cfg.patience, min_lr=1e-6,
        )

        self.history: dict[str, list[float]] = {
            k: [] for k in ["tr_loss", "tr_acc", "va_loss", "va_acc", "va_bacc", "lr"]
        }
        self.best_val_loss = float("inf")
        self.train_seconds = 0.0
        self._epochs_since_best = 0

    def optimize_epoch(self, loader: DataLoader) -> float:
        """One optimisation pass. Dropout/DropEdge active; no metrics collected."""
        self.model.train(True)
        total_loss = 0.0

        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = self.criterion(out, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            # .detach() before float(): converting a grad-tracking tensor warns.
            total_loss += float(loss.detach()) * batch.num_graphs

        return total_loss / max(1, len(loader.dataset))

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, float, float]:
        """Eval-mode evaluation. Returns (loss, accuracy, balanced_accuracy)."""
        self.model.eval()
        total_loss = 0.0
        ys, ps = [], []

        for batch in loader:
            batch = batch.to(self.device)
            out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            total_loss += float(self.criterion(out, batch.y).detach()) * batch.num_graphs
            ys.append(batch.y.cpu())
            ps.append(out.argmax(1).cpu())

        ys = torch.cat(ys).numpy()
        ps = torch.cat(ps).numpy()
        return (total_loss / max(1, len(loader.dataset)),
                accuracy_score(ys, ps), balanced_accuracy_score(ys, ps))

    def save_checkpoint(self, path: str) -> None:
        """Checkpoint with a picklable config (asdict, not vars())."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "cfg": self.cfg.to_dict(),
                "class2idx": CLASS2IDX,
                "best_val_loss": self.best_val_loss,
            },
            path,
        )

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            ckpt_path: str) -> dict:
        """Full training loop with early stopping. Returns the history dict."""
        cfg = self.cfg
        self._epochs_since_best = 0
        t0 = time.time()

        for epoch in range(1, cfg.epochs + 1):
            self.optimize_epoch(train_loader)

            # Both splits measured in eval mode -- the point of the rewrite.
            tr_loss_eval, tr_acc, _ = self.evaluate(train_loader)
            va_loss, va_acc, va_bacc = self.evaluate(val_loader)

            self.scheduler.step(va_loss)
            lr_now = self.optimizer.param_groups[0]["lr"]

            self.history["tr_loss"].append(tr_loss_eval)
            self.history["tr_acc"].append(tr_acc)
            self.history["va_loss"].append(va_loss)
            self.history["va_acc"].append(va_acc)
            self.history["va_bacc"].append(va_bacc)
            self.history["lr"].append(lr_now)

            flag = ""
            if va_loss < self.best_val_loss - 1e-4:
                self.best_val_loss = va_loss
                self._epochs_since_best = 0
                self.save_checkpoint(ckpt_path)
                flag = "  <- best"
            else:
                self._epochs_since_best += 1

            logger.info("E%03d | tr %.3f/%.3f | va %.3f/%.3f bacc %.3f | lr %.1e%s",
                        epoch, tr_loss_eval, tr_acc, va_loss, va_acc, va_bacc, lr_now, flag)

            if self._epochs_since_best >= cfg.early_stop:
                logger.info("Early stopping at epoch %d (no improvement for %d epochs).",
                            epoch, cfg.early_stop)
                break

        self.train_seconds = time.time() - t0
        logger.info("Best val loss: %.4f (%.1f s)", self.best_val_loss, self.train_seconds)
        return self.history

    def load_best(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        logger.info("Loaded best checkpoint from %s", ckpt_path)


def make_weighted_loader(
    graphs: list,
    batch_size: int,
    n_classes: int,
    seed: int = 42,
    balanced: bool = True,
) -> DataLoader:
    """DataLoader with a WeightedRandomSampler for class imbalance.

    With plain shuffling, ChronicLung (~50 images) is invisible in most batches.
    The sampler gives every class roughly equal exposure per epoch.

    Args:
        balanced: False falls back to plain shuffling -- the
                  "no class weighting / sampler" ablation arm.
    """
    if not balanced:
        return DataLoader(graphs, batch_size=batch_size, shuffle=True, num_workers=0)

    labels = np.array([int(g.y) for g in graphs])
    class_count = np.bincount(labels, minlength=n_classes).astype(np.float64)
    sample_w = (1.0 / np.clip(class_count, 1, None))[labels]
    g = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(sample_w),
        replacement=True,
        generator=g,
    )
    # sampler and shuffle are mutually exclusive in torch; passing both raises.
    return DataLoader(graphs, batch_size=batch_size, sampler=sampler, num_workers=0)


def make_loss_weights(graphs: list, n_classes: int, device: torch.device,
                      balanced: bool = True) -> Optional[torch.Tensor]:
    """Inverse-frequency class weights for CrossEntropyLoss (None if disabled)."""
    if not balanced:
        return None
    labels = np.array([int(g.y) for g in graphs])
    class_count = np.bincount(labels, minlength=n_classes).astype(np.float64)
    weights = class_count.sum() / (n_classes * np.clip(class_count, 1, None))
    return torch.tensor(weights, dtype=torch.float32, device=device)

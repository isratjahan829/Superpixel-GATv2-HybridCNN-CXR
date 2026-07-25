"""
models/cnn_baseline.py -- End-to-end fine-tuned ResNet18 image classifier.

Why a separate ResNet18?
    models/encoder.py is a FROZEN feature extractor living inside the graph
    pipeline; it never classifies anything. A genuine CNN baseline (CheXNet
    style: ResNet18 fine-tuned end to end on raw pixels) is what answers the
    question "does the hybrid GNN beat a plain CNN?".

The control flow mirrors training/crossval.py::_train_one_fold so the same
StratifiedKFold indices and the same held-out instances are used, which is a
precondition for the paired McNemar / DeLong tests to be valid.
"""

from __future__ import annotations

import copy
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from cxr_gnn.config import Config, NUM_CLASSES
from cxr_gnn.data.augment import medical_safe_augment
from cxr_gnn.data.dataset import load_gray
from cxr_gnn.utils import get_logger, set_seed

logger = get_logger(__name__)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class CXRImageDataset(Dataset):
    """Raw-pixel dataset for the CNN baseline -- bypasses the graph pipeline.

    Uses the same medically-safe augmentation as the GNN pipeline (train split
    only) so the comparison is like for like.
    """

    def __init__(
        self,
        items: list[tuple[str, int]],   # (path, class_idx)
        cfg: Config,
        train: bool,
        seed: int = 0,
    ) -> None:
        self.items = items
        self.cfg = cfg
        self.train = train
        self.seed = seed
        self.mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(_IMAGENET_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.items[idx]
        img = load_gray(path, self.cfg.img_size)
        if self.train and self.cfg.do_aug:
            # Per-item RNG: a shared Generator is not fork-safe, so with
            # num_workers>0 every worker would replay the same augmentations.
            rng = np.random.default_rng((self.seed, idx))
            img = medical_safe_augment(img, rng, self.cfg.img_size)
        t = torch.from_numpy(img).float().unsqueeze(0).repeat(3, 1, 1)   # (3, H, W)
        t = (t - self.mean) / self.std
        return t, label


def build_cnn_baseline(device: torch.device, n_classes: int = NUM_CLASSES,
                       pretrained: bool = True) -> nn.Module:
    """ImageNet-pretrained ResNet18 with a fresh head, fully fine-tuned."""
    from torchvision.models import resnet18

    model = None
    if pretrained:
        try:
            from torchvision.models import ResNet18_Weights
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception as ex:
            logger.warning("Pretrained weights unavailable (%s). Using random init.", ex)
    if model is None:
        model = resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model.to(device)


def _make_loader(
    items: list[tuple[str, int]],
    cfg: Config,
    n_classes: int,
    train: bool,
    seed: int = 0,
) -> DataLoader:
    ds = CXRImageDataset(items, cfg, train=train, seed=seed)
    if train:
        labels = np.array([lab for _, lab in items])
        cc = np.bincount(labels, minlength=n_classes).astype(np.float64)
        w = (1.0 / np.clip(cc, 1, None))[labels]
        g = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                        len(w), True, generator=g)
        return DataLoader(ds, batch_size=cfg.batch_size, sampler=sampler, num_workers=0)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)


def train_cnn_baseline(
    train_items: list[tuple[str, int]],
    val_items: list[tuple[str, int]],
    cfg: Config,
    device: torch.device,
    epochs: Optional[int] = None,
    patience: Optional[int] = None,
    n_classes: int = NUM_CLASSES,
    lr: Optional[float] = None,
    seed: Optional[int] = None,
    pretrained: bool = True,
) -> tuple[nn.Module, float]:
    """Fine-tune ResNet18 end to end with early stopping on val loss.

    Returns:
        (model, train_seconds) -- the wall-clock time feeds the cost columns
        of the model-comparison table.
    """
    epochs = cfg.cnn_epochs if epochs is None else epochs
    patience = cfg.cnn_patience if patience is None else patience
    set_seed(cfg.seed if seed is None else seed)
    t0 = time.time()

    model = build_cnn_baseline(device, n_classes, pretrained=pretrained)

    labels = np.array([lab for _, lab in train_items])
    cc = np.bincount(labels, minlength=n_classes).astype(np.float64)
    loss_w = torch.tensor(cc.sum() / (n_classes * np.clip(cc, 1, None)),
                          dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=loss_w, label_smoothing=cfg.label_smooth)
    opt = torch.optim.AdamW(model.parameters(), lr=lr or cfg.cnn_lr, weight_decay=cfg.wd)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                     patience=cfg.patience, min_lr=1e-6)

    tl = _make_loader(train_items, cfg, n_classes, train=True, seed=cfg.seed if seed is None else seed)
    vl = _make_loader(val_items, cfg, n_classes, train=False)

    best_vloss = float("inf")
    best_state = None
    since_best = 0

    for _epoch in range(epochs):
        model.train(True)
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

        model.eval()
        vtot, vn = 0.0, 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(device), y.to(device)
                vtot += float(crit(model(x), y).detach()) * x.size(0)
                vn += x.size(0)
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


@torch.no_grad()
def eval_cnn_baseline(
    model: nn.Module,
    items: list[tuple[str, int]],
    cfg: Config,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (y_true, y_pred, softmax_probs).

    Same signature as training/crossval.py::_eval_model, so the statistical
    comparison code treats CNN and GNN results identically.
    """
    model.eval()
    loader = _make_loader(items, cfg, NUM_CLASSES, train=False)
    ys, ps, prs = [], [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        ys.append(y)
        ps.append(out.argmax(1).cpu())
        prs.append(F.softmax(out, 1).cpu())
    return (torch.cat(ys).numpy(), torch.cat(ps).numpy(), torch.cat(prs).numpy())

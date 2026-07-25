"""
data/cache.py -- Graph dataset building and on-disk caching.

Building graphs is CPU-bound (SLIC + a ResNet forward per image), so the result
is written once to a .pt file and reloaded afterwards.

Split logic:
    Image-level split FIRST, then augmentation on the train split only, so an
    augmented copy can never appear in val/test.

kept-items tracking:
    SLIC occasionally yields a degenerate graph (<3 superpixels) and that image
    is dropped. Path-based analyses (the CNN baseline, split sensitivity) must
    stay index-aligned with the graph list, so the (path, class) pairs that
    actually survived are recorded alongside the graphs and cached with them.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import List

import numpy as np
import torch
from torch_geometric.data import Data

from cxr_gnn.config import Config, CLASS2IDX
from cxr_gnn.data.augment import medical_safe_augment
from cxr_gnn.data.dataset import load_gray
from cxr_gnn.data.graph import image_to_graph
from cxr_gnn.utils import get_logger

logger = get_logger(__name__)

# Bump when the cache FORMAT changes, so stale caches fail loudly instead of
# silently missing keys downstream.
CACHE_FORMAT_VERSION = 3

# Config fields that change the content of the graphs. If any of these differ
# from the cached run, the cache is invalid regardless of format version.
_CACHE_RELEVANT_FIELDS = (
    "img_size", "n_segments", "compactness", "lbp_p", "lbp_r",
    "use_edge_feat", "deep_feat_dim", "hand_feat_dim",
    "do_aug", "aug_cap", "max_aug_per_img", "test_size", "val_size",
)


def _cache_signature(cfg: Config) -> dict:
    return {k: getattr(cfg, k) for k in _CACHE_RELEVANT_FIELDS}


def _build_graphs_for_split(
    items: list[tuple[str, str]],
    cfg: Config,
    encoder,
    device: torch.device,
    split_name: str,
) -> tuple[List[Data], List[tuple[str, str]]]:
    """(path, class) list -> (graphs, kept items), no augmentation.

    kept_items[i] is the source of graphs[i]: same length, same order.
    """
    graphs: list[Data] = []
    kept_items: list[tuple[str, str]] = []
    for path, cls in items:
        lab = CLASS2IDX[cls]
        try:
            img = load_gray(path, cfg.img_size)
            g = image_to_graph(img, lab, cfg, encoder, device)
            if g is not None:
                g.is_aug = torch.tensor([0])
                graphs.append(g)
                kept_items.append((path, cls))
            else:
                logger.warning("[%s] skip %s: degenerate segmentation", split_name,
                               os.path.basename(path))
        except Exception as ex:
            logger.warning("[%s] skip %s: %s", split_name, os.path.basename(path), ex)

    if len(kept_items) < len(items):
        logger.info(
            "[%s] %d/%d images produced valid graphs (%d dropped).",
            split_name, len(kept_items), len(items), len(items) - len(kept_items),
        )
    return graphs, kept_items


def _augment_train(
    train_items: list[tuple[str, str]],
    train_graphs: List[Data],
    cfg: Config,
    encoder,
    device: torch.device,
    seed: int,
) -> List[Data]:
    """Class-balancing augmentation of the train split.

    ChronicLung and Pleural have ~50 images each against Normal's ~190, so each
    class is topped up towards min(aug_cap, largest class), capped at
    max_aug_per_img augmented copies of any single source image.
    """
    if not cfg.do_aug:
        return list(train_graphs)

    rng = np.random.default_rng(seed + 1)

    by_class: dict[str, list[str]] = defaultdict(list)
    for path, cls in train_items:
        by_class[cls].append(path)

    counts = {cls: len(paths) for cls, paths in by_class.items()}
    if not counts:
        return list(train_graphs)
    target = min(cfg.aug_cap, max(counts.values()))
    logger.info("Augmentation target per class: %d | originals: %s", target, counts)

    aug_graphs: list[Data] = list(train_graphs)

    for cls in sorted(by_class):
        paths = by_class[cls]
        need = target - counts[cls]
        if need <= 0:
            continue

        lab = CLASS2IDX[cls]
        per_img = min(cfg.max_aug_per_img, int(np.ceil(need / max(1, len(paths)))))
        made = 0
        order = sorted(paths)
        rng.shuffle(order)

        for path in order:
            if made >= need:
                break
            try:
                base = load_gray(path, cfg.img_size)
            except Exception as ex:
                logger.warning("skip aug source %s: %s", os.path.basename(path), ex)
                continue

            for _ in range(per_img):
                if made >= need:
                    break
                aug_img = medical_safe_augment(base, rng, cfg.img_size)
                g = image_to_graph(aug_img, lab, cfg, encoder, device)
                if g is not None:
                    g.is_aug = torch.tensor([1])
                    aug_graphs.append(g)
                    made += 1

        logger.debug("Class [%s]: augmented +%d -> total %d", cls, made, counts[cls] + made)

    return aug_graphs


def build_or_load_cache(
    splits: dict[str, list[tuple[str, str]]],
    cfg: Config,
    encoder,
    device: torch.device,
    seed: int,
) -> tuple[List[Data], List[Data], List[Data], dict[str, list[tuple[str, str]]]]:
    """Load the graph cache if it is valid, otherwise build and save it.

    Returns:
        train_graphs, val_graphs, test_graphs, kept_items
        kept_items["train"] aligns with train_graphs[:len(kept_items["train"])]
        (augmented graphs are always appended after the originals).
    """
    if not cfg.rebuild_cache and os.path.exists(cfg.cache_file):
        logger.info("Loading graph cache from %s ...", cfg.cache_file)
        blob = torch.load(cfg.cache_file, weights_only=False)

        problems = []
        if blob.get("cache_format_version") != CACHE_FORMAT_VERSION:
            problems.append(
                f"format version {blob.get('cache_format_version')!r} != {CACHE_FORMAT_VERSION}")
        if blob.get("class2idx") != CLASS2IDX:
            problems.append("class map changed")
        if blob.get("signature") != _cache_signature(cfg):
            problems.append("data/graph config changed")

        if problems:
            raise ValueError(
                "Graph cache is stale (" + "; ".join(problems) + "). "
                "Rerun with Config(rebuild_cache=True) or `python train.py --rebuild-cache`."
            )

        train_graphs = blob["train"]
        val_graphs = blob["val"]
        test_graphs = blob["test"]
        kept_items = blob["kept_items"]
        logger.info("Cache loaded: train=%d val=%d test=%d",
                    len(train_graphs), len(val_graphs), len(test_graphs))
        return train_graphs, val_graphs, test_graphs, kept_items

    logger.info("Building graph cache (first run -- a few minutes on GPU) ...")

    val_graphs, val_kept = _build_graphs_for_split(splits["val"], cfg, encoder, device, "val")
    test_graphs, test_kept = _build_graphs_for_split(splits["test"], cfg, encoder, device, "test")
    train_orig, train_kept = _build_graphs_for_split(splits["train"], cfg, encoder, device, "train")

    # Augmented graphs are appended AFTER the originals so train_kept stays
    # aligned with train_graphs[:len(train_kept)].
    train_graphs = _augment_train(train_kept, train_orig, cfg, encoder, device, seed)

    kept_items = {"train": train_kept, "val": val_kept, "test": test_kept}

    logger.info("Built: train=%d (orig=%d) val=%d test=%d",
                len(train_graphs), len(train_kept), len(val_graphs), len(test_graphs))

    os.makedirs(os.path.dirname(cfg.cache_file) or ".", exist_ok=True)
    torch.save(
        {
            "train": train_graphs,
            "val": val_graphs,
            "test": test_graphs,
            "kept_items": kept_items,
            "class2idx": CLASS2IDX,
            "cfg": cfg.to_dict(),
            "signature": _cache_signature(cfg),
            "cache_format_version": CACHE_FORMAT_VERSION,
        },
        cfg.cache_file,
    )
    logger.info("Cache saved -> %s", cfg.cache_file)

    return train_graphs, val_graphs, test_graphs, kept_items

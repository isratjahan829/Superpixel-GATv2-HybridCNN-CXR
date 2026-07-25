"""
data/graph.py -- Image -> PyG graph conversion.

Node features (140-d):
    [0:128]   Deep: ResNet18 layer2 feature map, average-pooled per superpixel
    [128:140] Hand-crafted: intensity stats, LBP texture, shape descriptors

Edge features (2-d, optional):
    [0] mean-intensity difference between adjacent regions
    [1] LBP-texture difference between adjacent regions

Correctness note (fixed here vs the original notebook):
    SLIC's label map is not guaranteed to be a contiguous 1..N range -- with
    enforce_connectivity, small segments can be merged away and leave holes.
    The original code built `idx = arange(1, labels.max()+1)` and passed it to
    scipy.ndimage.mean(), so every missing label produced a 0/0 division. That
    is the source of the thousands of
        RuntimeWarning: invalid value encountered in divide
    lines in the original run log, and it silently wrote NaN node features into
    the graphs. It also desynchronised the ndimage stats (length = labels.max())
    from regionprops (length = number of labels actually present).
    `_relabel_contiguous` below removes the cause rather than the symptom.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from skimage.feature import local_binary_pattern
from skimage.measure import regionprops
from skimage.segmentation import slic
from torch_geometric.data import Data

from cxr_gnn.config import Config

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _relabel_contiguous(labels: np.ndarray) -> tuple[np.ndarray, int]:
    """Map an arbitrary positive label map onto a gap-free 1..N range."""
    present = np.unique(labels)
    present = present[present > 0]
    if present.size == 0:
        return labels, 0
    lut = np.zeros(int(labels.max()) + 1, dtype=np.int64)
    lut[present] = np.arange(1, present.size + 1)
    return lut[labels], int(present.size)


def _rag_edges(labels: np.ndarray) -> np.ndarray:
    """4-connectivity region adjacency graph edges from a SLIC label map.

    Args:
        labels: int array with values 1..N (contiguous)

    Returns:
        (E, 2) int64 array of unique, sorted, 0-indexed undirected edges.
    """
    pairs = []
    # Horizontal then vertical neighbours; vectorised rather than looping in
    # Python over every pixel pair (the original built a set element by element).
    for a, b in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        a = a.ravel()
        b = b.ravel()
        m = a != b
        if m.any():
            pairs.append(np.stack([np.minimum(a[m], b[m]), np.maximum(a[m], b[m])], axis=1))

    if not pairs:
        return np.empty((0, 2), dtype=np.int64)

    edges = np.unique(np.concatenate(pairs, axis=0), axis=0).astype(np.int64)
    return edges - 1  # 1-indexed -> 0-indexed


def _region_stats(values: np.ndarray, flat_labels: np.ndarray,
                  n_nodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-region (mean, std, min, max) computed with bincount.

    scipy.ndimage's equivalents allocate an extra bin for pixels outside the
    index set and divide by its zero count, which emits
    "RuntimeWarning: invalid value encountered in divide" on every single
    image -- thousands of lines of log noise in the original run. Doing the
    reduction directly avoids that and is faster (one pass, no label remap).
    """
    v = values.ravel().astype(np.float64)
    counts = np.bincount(flat_labels, minlength=n_nodes).astype(np.float64)
    counts_safe = np.clip(counts, 1, None)

    total = np.bincount(flat_labels, weights=v, minlength=n_nodes)
    mean = total / counts_safe
    sq = np.bincount(flat_labels, weights=v * v, minlength=n_nodes) / counts_safe
    std = np.sqrt(np.clip(sq - mean ** 2, 0, None))

    vmin = np.full(n_nodes, np.inf)
    vmax = np.full(n_nodes, -np.inf)
    np.minimum.at(vmin, flat_labels, v)
    np.maximum.at(vmax, flat_labels, v)
    empty = counts == 0
    vmin[empty] = 0.0
    vmax[empty] = 0.0

    return mean, std, vmin, vmax


def _region_pool(
    fmap: np.ndarray,
    flat_labels: np.ndarray,
    pixel_counts: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    """Average-pool a (C, H*W) feature map into (N, C) per-superpixel features."""
    C = fmap.shape[0]
    sums = np.zeros((C, n_nodes), dtype=np.float64)
    for c in range(C):
        np.add.at(sums[c], flat_labels, fmap[c])
    return (sums / np.clip(pixel_counts, 1, None)).T.astype(np.float32)


def segment(img: np.ndarray, cfg: Config) -> tuple[np.ndarray, int]:
    """SLIC segmentation + contiguous relabelling. Shared with the plots."""
    labels = slic(
        img,
        n_segments=cfg.n_segments,
        compactness=cfg.compactness,
        channel_axis=None,
        start_label=1,
    )
    return _relabel_contiguous(labels)


def image_to_graph(
    img: np.ndarray,
    label_idx: int,
    cfg: Config,
    encoder,                      # nn.Module, or None for the hand-only ablation
    device: torch.device,
) -> Optional[Data]:
    """Convert one grayscale image into a PyG graph.

    Returns None for degenerate cases (fewer than 3 superpixels, or no edges).
    """
    H, W = img.shape

    labels, n_nodes = segment(img, cfg)
    if n_nodes < 3:
        return None

    flat_labels = labels.ravel() - 1                 # 0-indexed node ids per pixel

    # -- Hand-crafted features (12-d per node) -------------------------------
    img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    lbp = local_binary_pattern(img_u8, cfg.lbp_p, cfg.lbp_r, method="uniform")

    mean_i, std_i, min_i, max_i = _region_stats(img, flat_labels, n_nodes)
    lbp_m, lbp_s, _, _ = _region_stats(lbp, flat_labels, n_nodes)

    props = regionprops(labels)
    if len(props) != n_nodes:          # cannot happen after relabelling; assert the invariant
        return None
    cy = np.array([p.centroid[0] for p in props]) / H
    cx = np.array([p.centroid[1] for p in props]) / W
    area = np.array([p.area for p in props], dtype=np.float64) / float(H * W)
    ecc = np.array([p.eccentricity for p in props])
    sol = np.array([p.solidity for p in props])
    ext = np.array([p.extent for p in props])

    hand = np.stack(
        [mean_i, std_i, min_i, max_i,
         lbp_m / 255.0, lbp_s / 255.0,
         cy, cx, area, ecc, sol, ext],
        axis=1,
    ).astype(np.float32)               # (N, 12)

    # Belt and braces: a single NaN anywhere would propagate through BatchNorm
    # and turn the whole batch's loss into NaN.
    hand = np.nan_to_num(hand, nan=0.0, posinf=0.0, neginf=0.0)

    # -- Deep features (128-d per node) --------------------------------------
    if encoder is not None:
        import torch.nn.functional as F

        mean_t = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std_t = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)

        with torch.no_grad():
            t = torch.from_numpy(np.ascontiguousarray(img))[None, None].float().to(device)
            t = t.repeat(1, 3, 1, 1)
            t = (t - mean_t) / std_t
            fmap = encoder(t)                                     # (1, 128, h', w')
            fmap = F.interpolate(fmap, size=(H, W), mode="bilinear", align_corners=False)
            fmap_np = fmap[0].cpu().numpy()                       # (128, H, W)

        pixel_counts = np.bincount(flat_labels, minlength=n_nodes).astype(np.float32)
        deep = _region_pool(fmap_np.reshape(fmap_np.shape[0], -1),
                            flat_labels, pixel_counts, n_nodes)    # (N, 128)
        x = np.concatenate([deep, hand], axis=1).astype(np.float32)   # (N, 140)
    else:
        x = hand                                                  # (N, 12) hand-only ablation

    # -- Edges ---------------------------------------------------------------
    e = _rag_edges(labels)
    if e.shape[0] == 0:
        return None

    edge_index = np.concatenate([e.T, e.T[::-1]], axis=1)         # (2, 2E), bidirectional

    if cfg.use_edge_feat:
        d_int = np.abs(mean_i[e[:, 0]] - mean_i[e[:, 1]])
        d_tex = np.abs(lbp_m[e[:, 0]] - lbp_m[e[:, 1]]) / 255.0
        edge_attr_np = np.stack([d_int, d_tex], axis=1).astype(np.float32)
        edge_attr_np = np.concatenate([edge_attr_np, edge_attr_np], axis=0)
        edge_attr = torch.from_numpy(np.nan_to_num(edge_attr_np))
    else:
        edge_attr = None

    return Data(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(np.ascontiguousarray(edge_index)).long(),
        edge_attr=edge_attr,
        y=torch.tensor([label_idx], dtype=torch.long),
    )

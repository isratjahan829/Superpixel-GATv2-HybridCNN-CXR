"""
data/dataset.py -- Dataset discovery, image loading, label resolution.
"""

from __future__ import annotations

import glob
import os
from collections import Counter, defaultdict
from typing import List, Tuple

import numpy as np
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
from skimage.io import imread
from skimage.transform import resize as sk_resize

from cxr_gnn.config import Config, RAW2CLEAN
from cxr_gnn.utils import get_logger

logger = get_logger(__name__)

Sample = Tuple[str, str]   # (filepath, raw_folder_name)

# Directories that are searched for the dataset when cfg.data_root does not
# exist. Kaggle mounts datasets under /kaggle/input; the rest cover Colab and
# ordinary local checkouts.
SEARCH_BASES = ("/kaggle/input", "./data", "../data", os.path.expanduser("~/data"))


def _looks_like_class_dir(path: str, cfg: Config) -> bool:
    """True if `path` is a directory holding at least one readable image."""
    if not os.path.isdir(path):
        return False
    try:
        names = os.listdir(path)[:50]
    except OSError:
        return False
    return any(n.lower().endswith(cfg.valid_ext) for n in names)


def find_data_root(cfg: Config, extra_bases: tuple[str, ...] = ()) -> str:
    """Locate the dataset root: a directory with >= 3 image sub-folders.

    Raises FileNotFoundError with an actionable message (rather than failing
    deep inside the graph builder) when nothing is found.
    """
    if _has_class_subdirs(cfg.data_root, cfg):
        logger.info("Found data root: %s", cfg.data_root)
        return cfg.data_root

    for base in tuple(extra_bases) + SEARCH_BASES:
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, _ in os.walk(base):
            if _has_class_subdirs(dirpath, cfg, dirnames):
                logger.info("Found data root: %s", dirpath)
                return dirpath

    raise FileNotFoundError(
        f"Cannot find the dataset. Looked at cfg.data_root={cfg.data_root!r} and "
        f"under {tuple(extra_bases) + SEARCH_BASES}. Set the CXR_DATA_ROOT "
        f"environment variable, or pass Config(data_root=...), to the folder "
        f"that directly contains the per-class image sub-folders."
    )


def _has_class_subdirs(path: str, cfg: Config, dirnames: list[str] | None = None) -> bool:
    if not os.path.isdir(path):
        return False
    if dirnames is None:
        dirnames = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    n = sum(1 for d in dirnames if _looks_like_class_dir(os.path.join(path, d), cfg))
    return n >= 3


def collect_samples(data_root: str, cfg: Config) -> List[Sample]:
    """Scan every class folder into a list of (path, raw_label) tuples."""
    class_dirs = sorted(
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    )
    logger.info("Found class folders: %s", class_dirs)

    samples: list[Sample] = []
    unknown: set[str] = set()
    for d in class_dirs:
        folder = os.path.join(data_root, d)
        files = sorted(
            f for f in glob.glob(os.path.join(folder, "*"))
            if f.lower().endswith(cfg.valid_ext)
        )
        if d not in RAW2CLEAN:
            unknown.add(d)
        for f in files:
            samples.append((f, d))

    if unknown:
        # Loud, because an unmapped folder name silently becomes its own class
        # and would corrupt every downstream table.
        logger.warning(
            "Folder name(s) %s are not in RAW2CLEAN and will be treated as "
            "their own class. Add them to cxr_gnn/config.py::RAW2CLEAN.",
            sorted(unknown),
        )

    raw_counts = Counter(s[1] for s in samples)
    logger.info("Images per folder:")
    for raw, cnt in sorted(raw_counts.items()):
        logger.info("  %-28s -> %-12s : %d", raw, RAW2CLEAN.get(raw, raw), cnt)

    clean_counts = Counter(RAW2CLEAN.get(s[1], s[1]) for s in samples)
    logger.info("Images per clean class: %s", dict(sorted(clean_counts.items())))

    if not samples:
        raise RuntimeError(f"No images with extensions {cfg.valid_ext} under {data_root}")

    return samples


def load_gray(path: str, size: int) -> np.ndarray:
    """Image -> grayscale float32 in [0, 1], shape (size, size).

    Handles greyscale, RGB and RGBA sources.
    """
    img = imread(path)
    if img.ndim == 3:
        img = img[..., :3] if img.shape[2] == 4 else img
        img = rgb2gray(img)
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    img = sk_resize(img, (size, size), anti_aliasing=True, preserve_range=True)
    # Guard against a constant image, where rescale_intensity would divide by 0.
    if float(img.max() - img.min()) < 1e-8:
        return np.zeros((size, size), dtype=np.float32)
    img = rescale_intensity(img, out_range=(0.0, 1.0)).astype(np.float32)
    return img


def stratified_image_split(
    samples: List[Sample],
    cfg: Config,
    seed: int,
) -> dict[str, list[tuple[str, str]]]:
    """Image-level stratified split -- performed BEFORE augmentation.

    Splitting after augmentation (as in the original notebook) lets augmented
    copies of a training image land in val/test, which leaks and inflates every
    metric. Splitting on raw images first makes that impossible.

    Returns {"train": [(path, clean_cls), ...], "val": [...], "test": [...]}
    """
    rng = np.random.default_rng(seed)
    by_class: dict[str, list[str]] = defaultdict(list)
    for path, raw in samples:
        by_class[RAW2CLEAN.get(raw, raw)].append(path)

    splits: dict[str, list[tuple[str, str]]] = {"train": [], "val": [], "test": []}

    for cls in sorted(by_class):
        p = sorted(by_class[cls])          # sort first: filesystem order is not stable
        rng.shuffle(p)
        n = len(p)
        n_test = max(1, int(round(n * cfg.test_size)))
        n_val = max(1, int(round(n * cfg.val_size)))
        # Never let val+test swallow the whole class.
        n_test = min(n_test, max(1, n - 2))
        n_val = min(n_val, max(1, n - n_test - 1))

        for x in p[:n_test]:
            splits["test"].append((x, cls))
        for x in p[n_test:n_test + n_val]:
            splits["val"].append((x, cls))
        for x in p[n_test + n_val:]:
            splits["train"].append((x, cls))

    for split_name, items in splits.items():
        cnt = Counter(c for _, c in items)
        logger.info("Split [%s]: total=%d  %s", split_name, len(items), dict(sorted(cnt.items())))

    return splits

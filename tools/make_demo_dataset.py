#!/usr/bin/env python3
"""
tools/make_demo_dataset.py -- Synthetic stand-in dataset.

This exists so the pipeline can be executed and verified anywhere, including
machines without the Kaggle dataset mounted. It writes the SAME folder layout
and the same class-name typos as the real dataset, so no pipeline code has to
know the difference.

The images are procedurally generated thorax-like phantoms with a per-class
structural cue (enlarged cardiac silhouette, diffuse reticulation, apical
opacity, blunted costophrenic angle). They are NOT chest X-rays and carry no
clinical meaning: any metric computed on them measures that the code runs, not
that the model works. Never report a number from this dataset as a result.

Usage:
    python tools/make_demo_dataset.py --out demo_data --per-class 40
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from skimage.io import imsave

# Folder names deliberately reproduce the real dataset's inconsistent spelling,
# so RAW2CLEAN normalisation is exercised too.
CLASS_FOLDERS = {
    "Cardiac Pathology": "cardiac",
    "Cronic Lung Disease": "chronic_lung",
    "Normal": "normal",
    "TB": "tb",
    "plural Pathology": "pleural",
}

# Mild class imbalance, echoing the real dataset's shape (Normal largest,
# ChronicLung and Pleural smallest).
CLASS_WEIGHTS = {
    "Cardiac Pathology": 1.0,
    "Cronic Lung Disease": 0.36,
    "Normal": 1.34,
    "TB": 0.98,
    "plural Pathology": 0.35,
}


def _thorax(rng: np.random.Generator, size: int) -> np.ndarray:
    """Base phantom: dark lung fields inside a brighter chest wall."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    img = 0.45 + 0.10 * rng.standard_normal((size, size)).astype(np.float32) * 0.1

    # Chest wall / soft tissue gradient.
    img += 0.25 * np.exp(-((xx - 0.5) ** 2) / 0.10)
    img += 0.10 * np.exp(-((yy - 0.85) ** 2) / 0.02)

    # Two lung fields (darker).
    for cx in (0.32, 0.68):
        d = ((xx - cx) / 0.16) ** 2 + ((yy - 0.48) / 0.28) ** 2
        img -= 0.30 * np.exp(-d)

    # Spine / mediastinum.
    img += 0.18 * np.exp(-((xx - 0.5) ** 2) / 0.0016)

    # Ribs.
    for k in range(6):
        y0 = 0.22 + 0.10 * k
        img += 0.05 * np.exp(-((yy - y0 - 0.06 * (xx - 0.5) ** 2) ** 2) / 2e-4)

    return img


def _apply_pathology(img: np.ndarray, cls: str, rng: np.random.Generator) -> np.ndarray:
    size = img.shape[0]
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    out = img.copy()

    if cls == "Cardiac Pathology":
        # Enlarged cardiac silhouette, lower left.
        d = ((xx - 0.44) / (0.20 + 0.02 * rng.random())) ** 2 + ((yy - 0.62) / 0.17) ** 2
        out += 0.28 * np.exp(-d)
    elif cls == "Cronic Lung Disease":
        # Diffuse reticular texture across both lung fields.
        noise = rng.standard_normal((size, size)).astype(np.float32)
        from scipy.ndimage import gaussian_filter
        retic = gaussian_filter(noise, 1.2) - gaussian_filter(noise, 3.0)
        mask = np.exp(-(((xx - 0.32) / 0.18) ** 2 + ((yy - 0.48) / 0.30) ** 2))
        mask += np.exp(-(((xx - 0.68) / 0.18) ** 2 + ((yy - 0.48) / 0.30) ** 2))
        out += 0.55 * retic * mask
    elif cls == "TB":
        # Apical opacity plus a small cavity.
        cx = 0.32 if rng.random() < 0.5 else 0.68
        out += 0.22 * np.exp(-(((xx - cx) / 0.10) ** 2 + ((yy - 0.25) / 0.10) ** 2))
        r = ((xx - cx) ** 2 + (yy - 0.25) ** 2) ** 0.5
        out -= 0.18 * np.exp(-((r - 0.035) ** 2) / 2e-4)
    elif cls == "plural Pathology":
        # Blunted costophrenic angle: dense basal effusion with a meniscus.
        side = 0.30 if rng.random() < 0.5 else 0.70
        level = 0.70 + 0.05 * rng.random()
        eff = (yy > level - 0.05 * np.exp(-((xx - side) / 0.18) ** 2))
        eff = eff & (np.abs(xx - side) < 0.22)
        out[eff] += 0.30
    # "Normal": base phantom unchanged.

    return out


def generate(out_dir: str, per_class: int = 40, size: int = 256, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    counts = {}

    for cls, slug in CLASS_FOLDERS.items():
        folder = os.path.join(out_dir, cls)
        os.makedirs(folder, exist_ok=True)
        n = max(6, int(round(per_class * CLASS_WEIGHTS[cls])))
        for i in range(n):
            img = _apply_pathology(_thorax(rng, size), cls, rng)
            img += 0.02 * rng.standard_normal((size, size)).astype(np.float32)
            img = np.clip(img, 0, 1)
            imsave(os.path.join(folder, f"{slug}_{i:04d}.png"),
                   (img * 255).astype(np.uint8), check_contrast=False)
        counts[cls] = n

    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="demo_data")
    p.add_argument("--per-class", type=int, default=40)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    counts = generate(a.out, a.per_class, a.size, a.seed)
    print(f"Wrote {sum(counts.values())} synthetic images to {a.out}/")
    for k, v in counts.items():
        print(f"  {k:24s}: {v}")
    print("\nThese are procedural phantoms, not chest X-rays. Use them to verify that "
          "the pipeline runs; never report metrics computed on them.")


if __name__ == "__main__":
    main()

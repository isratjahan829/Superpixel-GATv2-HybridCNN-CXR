"""
data/augment.py -- Medically-safe augmentation for chest X-rays.

Why no flips?
    A horizontal flip puts the heart on the right, which is the radiographic
    presentation of dextrocardia -- i.e. it changes the diagnosis rather than
    the nuisance parameters. A vertical flip is anatomically meaningless.
    Only rotation, shift, zoom and intensity changes are applied: transforms a
    real acquisition could plausibly produce.

The same transforms are reused, with fixed magnitudes, as the perturbations in
the robustness study (evaluation/robustness.py).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.exposure import equalize_adapthist
from skimage.transform import resize as sk_resize, rotate as sk_rotate


def _zoom_to(img: np.ndarray, z: float, size: int) -> np.ndarray:
    """Zoom by factor z, cropping or edge-padding back to (size, size)."""
    zh = max(8, int(size / z))
    zoomed = sk_resize(img, (zh, zh), anti_aliasing=True, preserve_range=True)
    if zh >= size:
        s = (zh - size) // 2
        return zoomed[s:s + size, s:s + size]
    pad = (size - zh) // 2
    return np.pad(zoomed, ((pad, size - zh - pad), (pad, size - zh - pad)), mode="edge")


def medical_safe_augment(
    img: np.ndarray,
    rng: np.random.Generator,
    img_size: int,
) -> np.ndarray:
    """
    Args:
        img:      grayscale float32 in [0, 1], shape (H, W)
        rng:      seeded numpy Generator
        img_size: target size (images are assumed square)

    Returns:
        Augmented image, same shape and dtype as the input.
    """
    out = img.copy()

    # 1. Small rotation (+/-10 deg) -- patient positioning variation.
    out = sk_rotate(out, rng.uniform(-10, 10), mode="edge", preserve_range=True)

    # 2. Translation (+/-5% of image size).
    sh = max(1, int(img_size * 0.05))
    dy = int(rng.integers(-sh, sh + 1))
    dx = int(rng.integers(-sh, sh + 1))
    out = ndi.shift(out, (dy, dx), mode="nearest")

    # 3. Zoom (0.95-1.07x) -- source-to-detector distance variation.
    out = _zoom_to(out, float(rng.uniform(0.95, 1.07)), img_size)

    # 4. Intensity scale & shift -- exposure variation.
    out = out * rng.uniform(0.90, 1.10) + rng.uniform(-0.05, 0.05)

    # 5. CLAHE (40% of the time) -- post-processing artefact simulation.
    if rng.random() < 0.4:
        out = equalize_adapthist(np.clip(out, 0, 1), clip_limit=0.01)

    # 6. Gaussian noise (50% of the time) -- detector noise.
    if rng.random() < 0.5:
        out = out + rng.normal(0, 0.01, out.shape)

    return np.clip(out, 0.0, 1.0).astype(np.float32)

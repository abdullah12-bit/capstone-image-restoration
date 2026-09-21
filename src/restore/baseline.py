"""T4 classical baseline: median fill through the composite path.

Locked choice: median filter. Every baseline output goes through
``composite_output`` so comparisons are apples-to-apples with the learned
restorer: pristine pixels outside the damage mask survive bit-exactly.
"""

import numpy as np

from restore.pipeline import composite_output

ArrayF32 = np.ndarray


def identity(damaged: ArrayF32) -> ArrayF32:
    return damaged


def median_fill(damaged: ArrayF32, damage_mask: np.ndarray) -> ArrayF32:
    image = np.asarray(damaged)
    mask = np.asarray(damage_mask).astype(bool)
    if not mask.any() or not (~mask).any():
        return image.copy()
    fill = image.copy()
    if image.ndim == 2:
        fill[mask] = float(np.median(image[~mask]))
        return fill
    if image.ndim == 3:
        for c in range(image.shape[-1]):
            channel = image[..., c]
            fill[..., c][mask] = float(np.median(channel[~mask]))
        return fill
    raise ValueError(f"damaged image must be HxW or HxWxC, got {image.shape}")


def median_restore(damaged: ArrayF32, damage_mask: np.ndarray) -> ArrayF32:
    return composite_output(damaged, median_fill(damaged, damage_mask), damage_mask)

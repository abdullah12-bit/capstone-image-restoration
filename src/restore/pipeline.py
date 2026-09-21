"""T1 detect-then-restore pipeline stubs.

Detector: brightness-anomaly flag as a placeholder for the learned
U-Net (T5). Restorer: local-mean fill as a placeholder for the learned
fill (T5). ``restore`` composes detector -> restorer -> composite so
notebook, tests, and demo share one code path.
"""

from typing import Tuple

import numpy as np

ArrayF32 = np.ndarray


def _as_grayscale(damaged: ArrayF32) -> np.ndarray:
    if damaged.ndim == 2:
        return damaged
    if damaged.ndim == 3:
        return damaged.mean(axis=-1)
    raise ValueError(f"damaged image must be HxW or HxWxC, got {damaged.shape}")


def detect(damaged: ArrayF32) -> np.ndarray:
    gray = _as_grayscale(damaged)
    thresh = float(gray.mean() + 2.0 * gray.std())
    return (gray > thresh).astype(np.uint8)


def restore_masked(damaged: ArrayF32, damage_mask: np.ndarray) -> ArrayF32:
    fill = damaged.copy()
    mask = damage_mask.astype(bool)
    if not mask.any():
        return fill
    for c in range(damaged.shape[-1]):
        channel = damaged[..., c]
        fill[..., c][mask] = float(channel[~mask].mean()) if (~mask).any() else 0.0
    return fill.astype(np.float32, copy=False)


def composite_output(
    damaged: ArrayF32, fill: ArrayF32, damage_mask: np.ndarray
) -> ArrayF32:
    mask = damage_mask.astype(bool)
    out = np.where(mask[..., None] if damaged.ndim == 3 else mask, fill, damaged)
    out[~mask] = damaged[~mask]
    return out


def restore(damaged: ArrayF32) -> Tuple[ArrayF32, np.ndarray]:
    damage_mask = detect(damaged)
    fill = restore_masked(damaged, damage_mask)
    return composite_output(damaged, fill, damage_mask), damage_mask

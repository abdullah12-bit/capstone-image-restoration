"""T5 restorer: masked loss judged apart from detection."""

import numpy as np

ArrayF32 = np.ndarray


def restorer_loss(
    fill: ArrayF32,
    pristine: ArrayF32,
    damage_mask: np.ndarray,
    mask_weight: float = 4.0,
) -> float:
    predicted = np.asarray(fill, dtype=np.float64)
    target = np.asarray(pristine, dtype=np.float64)
    if predicted.shape != target.shape:
        raise ValueError(f"shape mismatch: {predicted.shape} vs {target.shape}")
    if target.shape[:2] != np.asarray(damage_mask).shape:
        raise ValueError(
            f"damage mask shape {np.asarray(damage_mask).shape} must match "
            f"image {target.shape[:2]}"
        )
    squared = (predicted - target) ** 2
    if predicted.ndim == 3:
        per_pixel = squared.mean(axis=-1)
    else:
        per_pixel = squared
    weights = np.where(np.asarray(damage_mask).astype(bool), mask_weight, 1.0)
    return float((per_pixel * weights).mean())

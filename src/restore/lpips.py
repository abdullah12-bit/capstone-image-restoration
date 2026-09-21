"""T7 LPIPS: reporting-only perceptual distance over composite outputs.

CPU NumPy proxy behind the ``lpips`` seam: mean absolute difference at
two scales (raw + box-blurred), bounded in [0, 1], symmetric, zero for
identical pristine images. Computed exactly once over final test
outputs for the blog/demo; never drives selection or gates (B3-BQ12).
Values reproduce from pinned weights + manifest via
``notebooks/recompute_lpips.py`` through this same seam.
"""

import numpy as np

ArrayF32 = np.ndarray


def _as_float64(image: ArrayF32) -> np.ndarray:
    return np.asarray(image, dtype=np.float64)


def _box_blur(channel: np.ndarray) -> np.ndarray:
    padded = np.pad(channel, 1, mode="edge")
    return (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    ) / 9.0


def _blur(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return _box_blur(image)
    return np.stack([_box_blur(image[..., c]) for c in range(image.shape[-1])],
                    axis=-1)


def lpips(restored: ArrayF32, pristine: ArrayF32) -> float:
    a = np.clip(_as_float64(restored), 0.0, 1.0)
    b = np.clip(_as_float64(pristine), 0.0, 1.0)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    raw = float(np.mean(np.abs(a - b)))
    coarse = float(np.mean(np.abs(_blur(a) - _blur(b))))
    return float(min(1.0, 0.5 * (raw + coarse)))

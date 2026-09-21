"""T4 metrics: PSNR + SSIM over composite outputs.

Images are HxWxC float arrays in [0, 1]. PSNR selection matches the locked
gate vocabulary: identity scores perfectly, worse fills score lower.
SSIM is the mean structural similarity over all pixels and channels.
No LPIPS here: LPIPS is reporting-only on final test outputs (T7).
"""

import numpy as np

ArrayF32 = np.ndarray


def _as_float64(image: ArrayF32) -> np.ndarray:
    return np.asarray(image, dtype=np.float64)


def mse(restored: ArrayF32, pristine: ArrayF32) -> float:
    diff = _as_float64(restored) - _as_float64(pristine)
    return float(np.mean(diff * diff))


def psnr(restored: ArrayF32, pristine: ArrayF32, data_range: float = 1.0) -> float:
    err = mse(restored, pristine)
    if err == 0.0:
        return float("inf")
    return float(10.0 * np.log10(data_range * data_range / err))


def _ssim_channel(a: np.ndarray, b: np.ndarray, data_range: float) -> float:
    mu_a = float(a.mean())
    mu_b = float(b.mean())
    var_a = float(((a - mu_a) ** 2).mean())
    var_b = float(((b - mu_b) ** 2).mean())
    cov = float(((a - mu_a) * (b - mu_b)).mean())
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    num = (2.0 * mu_a * mu_b + c1) * (2.0 * cov + c2)
    den = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)
    if den == 0.0:
        return 1.0
    return float(num / den)


def ssim(restored: ArrayF32, pristine: ArrayF32, data_range: float = 1.0) -> float:
    a = _as_float64(restored)
    b = _as_float64(pristine)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.ndim == 2:
        return _ssim_channel(a, b, data_range)
    if a.ndim == 3:
        return float(
            np.mean(
                [_ssim_channel(a[..., c], b[..., c], data_range)
                 for c in range(a.shape[-1])]
            )
        )
    raise ValueError(f"image must be HxW or HxWxC, got {a.shape}")

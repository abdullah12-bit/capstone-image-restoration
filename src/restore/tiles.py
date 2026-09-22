"""T8 tiled inference: serve full-size photos from 256px-trained stages.

Locked by B2-BQ9: train on 256px crops, serve full images via tiled
inference with overlap blending. Any damaged image fitting in one tile
goes through the ``restore`` seam untouched (bit-identical to the eval
harness); larger photos are covered by overlapping tiles whose fills
blend with linear edge ramps, then composite so pristine pixels outside
the combined damage mask survive bit-exactly.
"""

from typing import Callable, List, Optional, Tuple

import numpy as np

from restore.pipeline import composite_output, restore

ArrayF32 = np.ndarray
RestoreFn = Callable[[ArrayF32], Tuple[ArrayF32, np.ndarray]]

DEFAULT_TILE_SIZE = 256
DEFAULT_OVERLAP = 64


def _starts(length: int, tile_size: int, stride: int) -> List[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def _edge_weights(length: int, overlap: int) -> np.ndarray:
    weights = np.ones(length, dtype=np.float64)
    ramp = min(overlap, length // 2)
    if ramp > 0:
        slope = (np.arange(ramp, dtype=np.float64) + 1.0) / (ramp + 1.0)
        weights[:ramp] *= slope
        weights[-ramp:] *= slope[::-1]
    return weights


def restore_tiled(
    damaged: ArrayF32,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    restore_fn: Optional[RestoreFn] = None,
) -> Tuple[ArrayF32, np.ndarray]:
    image = np.asarray(damaged)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"damaged image must be HxWx3, got {image.shape}")
    if tile_size < 1:
        raise ValueError(f"tile_size must be >= 1, got {tile_size}")
    if not 0 <= overlap < tile_size:
        raise ValueError(
            f"overlap must satisfy 0 <= overlap < tile_size, got {overlap}"
        )
    height, width = image.shape[:2]
    run_tile = restore_fn if restore_fn is not None else restore
    if height <= tile_size and width <= tile_size:
        return run_tile(image)
    stride = tile_size - overlap
    fill_sum = np.zeros((height, width, 3), dtype=np.float64)
    mask_sum = np.zeros((height, width), dtype=np.float64)
    weight_sum = np.zeros((height, width), dtype=np.float64)
    for top in _starts(height, tile_size, stride):
        for left in _starts(width, tile_size, stride):
            tile = np.ascontiguousarray(
                image[top : top + tile_size, left : left + tile_size]
            )
            fill, damage_mask = run_tile(tile)
            tile_height, tile_width = tile.shape[:2]
            window = np.outer(
                _edge_weights(tile_height, overlap),
                _edge_weights(tile_width, overlap),
            )
            fill_sum[top : top + tile_height, left : left + tile_width] += (
                np.asarray(fill, dtype=np.float64) * window[..., None]
            )
            mask_sum[top : top + tile_height, left : left + tile_width] += (
                np.asarray(damage_mask, dtype=np.float64) * window
            )
            weight_sum[top : top + tile_height, left : left + tile_width] += (
                window
            )
    blended = fill_sum / weight_sum[..., None]
    combined_mask = (mask_sum / weight_sum >= 0.5).astype(np.uint8)
    return (
        composite_output(image, blended.astype(image.dtype, copy=False),
                         combined_mask),
        combined_mask,
    )

"""T8 seam tests: tiled inference serving full-size photos.

CPU-only, no GPU, no network, no dataset download. All inputs synthetic.
Uses CONTEXT.md vocabulary: damaged image, damage mask, composite output.
Locked by B2-BQ9: train on 256px crops, serve full images via tiled
inference with overlap blending.
"""

import numpy as np
import pytest

from restore.pipeline import restore
from restore.tiles import restore_tiled


def _synthetic_damaged(seed=0, h=64, w=64):
    rng = np.random.default_rng(seed)
    return rng.random((h, w, 3), dtype=np.float32)


def test_small_damaged_image_matches_direct_restore_bit_exactly():
    damaged = _synthetic_damaged(h=64, w=48)
    expected_restored, expected_mask = restore(damaged)
    restored, damage_mask = restore_tiled(damaged, tile_size=256, overlap=64)
    np.testing.assert_array_equal(restored, expected_restored)
    np.testing.assert_array_equal(damage_mask, expected_mask)


def test_small_damaged_image_calls_restore_fn_once():
    damaged = _synthetic_damaged(h=64, w=64)
    calls = []

    def counting_restore_fn(tile):
        calls.append(tile.shape)
        return restore(tile)

    restore_tiled(damaged, tile_size=256, overlap=64,
                  restore_fn=counting_restore_fn)
    assert len(calls) == 1


def test_large_damaged_image_serves_full_size_with_binary_damage_mask():
    damaged = _synthetic_damaged(seed=3, h=300, w=200)
    restored, damage_mask = restore_tiled(
        damaged, tile_size=128, overlap=32
    )
    assert restored.shape == damaged.shape
    assert damage_mask.shape == damaged.shape[:2]
    assert damage_mask.dtype == np.uint8
    assert set(np.unique(damage_mask)).issubset({0, 1})


def test_large_damaged_image_keeps_undamaged_pixels_bit_exactly():
    damaged = _synthetic_damaged(seed=4, h=300, w=200)
    restored, damage_mask = restore_tiled(
        damaged, tile_size=128, overlap=32
    )
    np.testing.assert_array_equal(
        restored[damage_mask == 0], damaged[damage_mask == 0]
    )


def test_large_damaged_image_stays_close_to_direct_restore():
    damaged = _synthetic_damaged(seed=5, h=300, w=200)
    expected_restored, _ = restore(damaged)
    restored, _ = restore_tiled(damaged, tile_size=128, overlap=32)
    assert float(np.abs(restored - expected_restored).mean()) < 0.05


def test_large_damaged_image_tiles_more_than_once():
    damaged = _synthetic_damaged(seed=6, h=300, w=200)
    calls = []

    def counting_restore_fn(tile):
        calls.append(tile.shape)
        return restore(tile)

    restore_tiled(damaged, tile_size=128, overlap=32,
                  restore_fn=counting_restore_fn)
    assert len(calls) > 1


def test_overlapping_tiles_cover_every_pixel():
    damaged = _synthetic_damaged(seed=7, h=300, w=200)
    restored, _ = restore_tiled(damaged, tile_size=128, overlap=32)
    assert np.isfinite(restored).all()


def test_invalid_tiling_arguments_rejected():
    damaged = _synthetic_damaged()
    with pytest.raises(ValueError, match="tile_size"):
        restore_tiled(damaged, tile_size=0, overlap=0)
    with pytest.raises(ValueError, match="overlap"):
        restore_tiled(damaged, tile_size=128, overlap=128)
    with pytest.raises(ValueError, match="overlap"):
        restore_tiled(damaged, tile_size=128, overlap=-1)


def test_non_rgb_damaged_image_rejected():
    with pytest.raises(ValueError, match="HxWx3"):
        restore_tiled(np.zeros((32, 32), dtype=np.float32))

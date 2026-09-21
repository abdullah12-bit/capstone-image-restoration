"""T1 seam tests: restore(damaged) -> (restored, damage_mask).

CPU-only, no GPU, no network, no dataset download. All inputs synthetic.
Uses CONTEXT.md vocabulary: damaged image, damage mask, composite output.
"""

import numpy as np

from restore import composite_output, detect, restore, restore_masked


def _assert_binary_damage_mask(damage_mask):
    assert damage_mask.dtype == np.uint8
    assert set(np.unique(damage_mask)).issubset({0, 1})


def _synthetic_damaged(seed=0, h=64, w=64):
    rng = np.random.default_rng(seed)
    return rng.random((h, w, 3), dtype=np.float32)


def test_restore_returns_matching_shapes():
    damaged = _synthetic_damaged()
    restored, damage_mask = restore(damaged)
    assert restored.shape == damaged.shape
    assert damage_mask.shape == damaged.shape[:2]


def test_damage_mask_is_binary():
    _, damage_mask = restore(_synthetic_damaged())
    _assert_binary_damage_mask(damage_mask)


def test_undamaged_regions_survive_compositing_bit_exactly():
    damaged = _synthetic_damaged()
    rng = np.random.default_rng(1)
    true_damage_mask = (rng.random(damaged.shape[:2]) < 0.1).astype(np.uint8)
    fill = np.zeros_like(damaged)
    restored = composite_output(damaged, fill, true_damage_mask)
    np.testing.assert_array_equal(
        restored[true_damage_mask == 0], damaged[true_damage_mask == 0]
    )


def test_restore_composite_path_keeps_undamaged_pixels_bit_exactly():
    damaged = _synthetic_damaged()
    restored, damage_mask = restore(damaged)
    np.testing.assert_array_equal(
        restored[damage_mask == 0], damaged[damage_mask == 0]
    )


def test_composite_preserves_float64_dtype_outside_damage_mask():
    rng = np.random.default_rng(2)
    damaged = rng.random((16, 16, 3))
    assert damaged.dtype == np.float64
    true_damage_mask = np.zeros((16, 16), dtype=np.uint8)
    true_damage_mask[4:8, 4:8] = 1
    restored = composite_output(damaged, np.zeros_like(damaged), true_damage_mask)
    np.testing.assert_array_equal(
        restored[true_damage_mask == 0], damaged[true_damage_mask == 0]
    )


def test_detector_seam_returns_image_sized_binary_damage_mask():
    damaged = _synthetic_damaged()
    damage_mask = detect(damaged)
    assert damage_mask.shape == damaged.shape[:2]
    _assert_binary_damage_mask(damage_mask)


def test_detector_seam_accepts_grayscale_damaged():
    rng = np.random.default_rng(3)
    damaged = rng.random((32, 32), dtype=np.float32)
    damage_mask = detect(damaged)
    assert damage_mask.shape == damaged.shape
    _assert_binary_damage_mask(damage_mask)


def test_restorer_seam_with_true_damage_mask_returns_image_sized_fill():
    damaged = _synthetic_damaged()
    true_damage_mask = np.zeros(damaged.shape[:2], dtype=np.uint8)
    true_damage_mask[8:24, 8:24] = 1
    fill = restore_masked(damaged, true_damage_mask)
    assert fill.shape == damaged.shape

"""T3 seam tests: core-tier damage simulator + loader augmentation.

CPU-only, no GPU, no network, no dataset download. All pristine images
synthetic. Uses CONTEXT.md vocabulary: pristine image, damaged image,
damage mask, pair.
"""

import numpy as np

from restore.loader import random_crop_and_flip
from restore.simulator import CORE_TYPES, simulate_damage


def _synthetic_pristine(seed=0, h=48, w=48):
    rng = np.random.default_rng(seed)
    return rng.random((h, w, 3), dtype=np.float32)


def _assert_exact_localized_mask(pristine, damaged, damage_mask):
    assert damaged.shape == pristine.shape
    assert damaged.dtype == np.float32
    assert damage_mask.shape == pristine.shape[:2]
    assert damage_mask.dtype == np.uint8
    assert set(np.unique(damage_mask)).issubset({0, 1})
    assert damage_mask.any()
    np.testing.assert_array_equal(
        damaged[damage_mask == 0], pristine[damage_mask == 0]
    )


def test_core_tier_covers_repair_fading_and_quality_jobs():
    assert set(CORE_TYPES) >= {
        "scratch",
        "dust",
        "fade",
        "sepia",
        "blur",
        "noise",
        "jpeg",
    }


def test_scratch_damage_mask_is_exact_by_construction():
    pristine = _synthetic_pristine()
    damaged, damage_mask, _ = simulate_damage(pristine, seed=1, types=["scratch"])
    _assert_exact_localized_mask(pristine, damaged, damage_mask)


def test_dust_damage_mask_is_exact_by_construction():
    pristine = _synthetic_pristine()
    damaged, damage_mask, _ = simulate_damage(pristine, seed=2, types=["dust"])
    _assert_exact_localized_mask(pristine, damaged, damage_mask)


def test_same_seed_reproduces_triplet_bit_exactly():
    pristine = _synthetic_pristine()
    first = simulate_damage(pristine, seed=7)
    second = simulate_damage(pristine, seed=7)
    for left, right in zip(first, second):
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        else:
            assert left == right


def test_global_types_mark_full_damage_mask():
    pristine = _synthetic_pristine()
    for index, damage_type in enumerate(["fade", "sepia", "blur", "noise", "jpeg"]):
        damaged, damage_mask, _ = simulate_damage(
            pristine, seed=20 + index, types=[damage_type]
        )
        assert damaged.shape == pristine.shape
        assert damaged.dtype == np.float32
        assert damage_mask.shape == pristine.shape[:2]
        assert damage_mask.dtype == np.uint8
        np.testing.assert_array_equal(
            damage_mask, np.ones(pristine.shape[:2], dtype=np.uint8)
        )
        assert not np.array_equal(damaged, pristine)
    damaged, damage_mask, _ = simulate_damage(pristine, seed=99, types=["jpeg"])
    assert not np.array_equal(damaged, np.round(pristine * 31.0) / 31.0)


def test_extended_tier_stays_gated_until_green_smoke():
    pristine = _synthetic_pristine()
    for damage_type in ["tear", "crease", "vignette"]:
        try:
            simulate_damage(pristine, seed=30, types=[damage_type])
        except ValueError:
            pass
        else:
            raise AssertionError(f"{damage_type} must stay gated")
    for index, damage_type in enumerate(["tear", "crease", "vignette"]):
        damaged, damage_mask, params = simulate_damage(
            pristine, seed=30 + index, types=[damage_type], allow_extended=True
        )
        assert damaged.shape == pristine.shape
        assert damage_mask.shape == pristine.shape[:2]
        assert damage_mask.any()
        assert params["sim_tier"] == "extended"


def test_vignette_damage_mask_covers_every_scaled_pixel():
    pristine = _synthetic_pristine()
    rng = np.random.default_rng(0)
    pristine[:] = rng.random((48, 48, 3), dtype=np.float64)
    pristine[pristine == 0.0] = 0.5
    damaged, damage_mask, _ = simulate_damage(
        pristine, seed=32, types=["vignette"], allow_extended=True
    )
    changed = (damaged != pristine).any(axis=-1).astype(np.uint8)
    np.testing.assert_array_equal(damage_mask, changed)


def test_loader_random_crop_and_flip_covers_256px_default():
    rng = np.random.default_rng(0)
    pristine = rng.random((300, 300, 3), dtype=np.float32)
    damaged = pristine.copy()
    damage_mask = np.zeros((300, 300), dtype=np.uint8)
    for size in (256, 64):
        cropped_damaged, cropped_mask = random_crop_and_flip(
            damaged, damage_mask, seed=5, crop_size=size
        )
        assert cropped_damaged.shape == (size, size, 3)
        assert cropped_mask.shape == (size, size)
        np.testing.assert_array_equal(
            random_crop_and_flip(damaged, damage_mask, seed=5, crop_size=size)[0],
            cropped_damaged,
        )


def test_different_seeds_damage_differently():
    pristine = _synthetic_pristine()
    damaged_a, _, _ = simulate_damage(pristine, seed=7)
    damaged_b, _, _ = simulate_damage(pristine, seed=8)
    assert not np.array_equal(damaged_a, damaged_b)

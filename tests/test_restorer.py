"""T5 restorer seam tests: fill judged apart from the damage mask.

CPU-only, no GPU, no network, no dataset download. Uses CONTEXT.md
vocabulary: pristine image, damaged image, damage mask, composite output.
"""

import numpy as np
import pytest

from restore.pipeline import composite_output
from restore.restorer import restorer_loss


def test_restorer_loss_is_zero_on_perfect_fill():
    pristine = np.zeros((4, 4, 3), dtype=np.float64)
    damage_mask = np.zeros((4, 4), dtype=np.uint8)
    damage_mask[0:2, 0:2] = 1
    assert restorer_loss(pristine.copy(), pristine, damage_mask) == 0.0


def test_restorer_loss_weights_damaged_pixels_above_pristine_ones():
    pristine = np.zeros((8, 8, 3), dtype=np.float64)
    damage_mask = np.zeros((8, 8), dtype=np.uint8)
    damage_mask[0:2, 0:2] = 1
    wrong_inside = pristine.copy()
    wrong_inside[0:2, 0:2] = 1.0
    wrong_outside = pristine.copy()
    wrong_outside[6:8, 6:8] = 1.0
    assert restorer_loss(wrong_inside, pristine, damage_mask) > restorer_loss(
        wrong_outside, pristine, damage_mask
    )
    assert restorer_loss(wrong_inside, pristine, damage_mask) == pytest.approx(
        0.0625 * 4.0, rel=1e-6
    )


def test_restorer_loss_matches_unweighted_mse_without_damage_mask():
    rng = np.random.default_rng(4)
    pristine = rng.random((6, 6, 3), dtype=np.float64)
    noisy = np.clip(pristine + 0.1, 0.0, 1.0)
    damage_mask = np.zeros((6, 6), dtype=np.uint8)
    from restore.metrics import mse

    assert restorer_loss(noisy, pristine, damage_mask, mask_weight=1.0) == pytest.approx(
        mse(noisy, pristine), rel=1e-9
    )


def test_restorer_fill_scores_through_composite_output():
    pristine = np.zeros((8, 8, 3), dtype=np.float32)
    damaged = pristine.copy()
    damage_mask = np.zeros((8, 8), dtype=np.uint8)
    damage_mask[0:4, 0:4] = 1
    damaged[damage_mask == 1] = 1.0
    perfect_fill = pristine.copy()
    restored = composite_output(damaged, perfect_fill, damage_mask)
    np.testing.assert_array_equal(restored, pristine)
    assert restorer_loss(perfect_fill, pristine, damage_mask) == 0.0


def test_restorer_loss_rejects_shape_mismatch():
    pristine = np.zeros((8, 8, 3), dtype=np.float64)
    damage_mask = np.zeros((8, 8), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape mismatch"):
        restorer_loss(np.zeros((4, 4, 3)), pristine, damage_mask)
    with pytest.raises(ValueError, match="damage mask shape"):
        restorer_loss(pristine, pristine, np.zeros((4, 4), dtype=np.uint8))

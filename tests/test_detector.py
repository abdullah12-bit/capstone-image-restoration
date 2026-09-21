"""T5 detector seam tests: masks judged apart from fills.

CPU-only, no GPU, no network, no dataset download. Uses CONTEXT.md
vocabulary: damage mask, pristine image, damaged image.
"""

import numpy as np
import pytest

from restore.detector import detector_scores, select_threshold, weighted_bce_dice


def test_detector_scores_reports_f1_and_iou():
    damage_mask = np.zeros((8, 8), dtype=np.uint8)
    damage_mask[0:4, 0:4] = 1
    scores = np.zeros((8, 8), dtype=np.float64)
    scores[0:4, 0:4] = 0.9
    result = detector_scores(scores, damage_mask, threshold=0.5)
    assert result["f1"] == 1.0
    assert result["iou"] == 1.0


def test_detector_scores_penalizes_missed_damage():
    damage_mask = np.zeros((8, 8), dtype=np.uint8)
    damage_mask[0:4, 0:4] = 1
    scores = np.zeros((8, 8), dtype=np.float64)
    result = detector_scores(scores, damage_mask, threshold=0.5)
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0
    assert result["iou"] == 0.0


def test_detector_threshold_selection_is_recall_favored():
    damage_mask = np.zeros((8, 8), dtype=np.uint8)
    damage_mask[:, :4] = 1
    scores = np.zeros((8, 8), dtype=np.float64)
    scores[:, :4] = 0.4
    scores[:, 4:] = 0.05
    assert select_threshold(scores, damage_mask) < 0.5
    assert detector_scores(scores, damage_mask, threshold=0.5)["recall"] < 1.0


def test_detector_threshold_falls_back_without_recalled_candidate():
    damage_mask = np.zeros((8, 8), dtype=np.uint8)
    damage_mask[:, :4] = 1
    scores = np.zeros((8, 8), dtype=np.float64)
    scores[:, :4] = 0.4
    scores[:, 4:] = 0.6
    best = select_threshold(scores, damage_mask, min_recall=1.0)
    assert best in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8)
    scored = [
        detector_scores(scores, damage_mask, threshold=t)
        for t in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8)
    ]
    assert detector_scores(scores, damage_mask, threshold=best)["recall"] == max(
        s["recall"] for s in scored
    )


def test_weighted_bce_dice_penalizes_missed_positives():
    damage_mask = np.zeros((8, 8), dtype=np.uint8)
    damage_mask[0:4, 0:4] = 1
    found = np.full((8, 8), 0.05)
    found[0:4, 0:4] = 0.9
    missed = np.full((8, 8), 0.05)
    assert weighted_bce_dice(missed, damage_mask) > weighted_bce_dice(
        found, damage_mask
    )
    assert weighted_bce_dice(found, damage_mask) == pytest.approx(0.26578, abs=1e-3)


def test_detector_scores_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        detector_scores(np.zeros((4, 4)), np.zeros((8, 8), dtype=np.uint8))

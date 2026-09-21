"""T5 detector: damage-mask scores judged apart from restorer fills."""

from typing import Dict, Sequence

import numpy as np

ArrayF32 = np.ndarray


def detector_scores(
    scores: ArrayF32, damage_mask: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    probs = np.asarray(scores, dtype=np.float64)
    if probs.shape != np.asarray(damage_mask).shape:
        raise ValueError(
            f"shape mismatch: {probs.shape} vs {np.asarray(damage_mask).shape}"
        )
    truth = np.asarray(damage_mask).astype(bool)
    predicted = probs >= threshold
    true_positive = int(np.logical_and(predicted, truth).sum())
    false_positive = int(np.logical_and(predicted, ~truth).sum())
    false_negative = int(np.logical_and(~predicted, truth).sum())
    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 1.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 1.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    union = true_positive + false_positive + false_negative
    iou = true_positive / union if union > 0 else 1.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "threshold": float(threshold),
    }


def select_threshold(
    scores: ArrayF32,
    damage_mask: np.ndarray,
    candidates: Sequence[float] = (
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.6,
        0.7,
        0.8,
    ),
    min_recall: float = 0.9,
) -> float:
    scored = [detector_scores(scores, damage_mask, threshold=t) for t in candidates]
    recalled = [s for s in scored if s["recall"] >= min_recall]
    pool = recalled if recalled else scored
    key = "f1" if recalled else "recall"
    best = max(pool, key=lambda s: (s[key], -s["threshold"]))
    return float(best["threshold"])


def weighted_bce_dice(
    probs: ArrayF32,
    damage_mask: np.ndarray,
    pos_weight: float = 4.0,
    eps: float = 1e-6,
) -> float:
    predicted = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0 - eps)
    truth = np.asarray(damage_mask, dtype=np.float64)
    if predicted.shape != truth.shape:
        raise ValueError(f"shape mismatch: {predicted.shape} vs {truth.shape}")
    bce = -(pos_weight * truth * np.log(predicted) + (1.0 - truth) * np.log(1.0 - predicted))
    dice = 1.0 - (2.0 * float((predicted * truth).sum()) + eps) / (
        float(predicted.sum()) + float(truth.sum()) + eps
    )
    return float(bce.mean() + dice)

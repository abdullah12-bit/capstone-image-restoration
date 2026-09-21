"""T5 smoke runner: the cheap stage that proves the whole loop.

Builds ~200 synthetic triplets (loader pristine + simulator damage
with exact masks), hands them to ``train_fn`` (torch U-Net training in
``models`` on the Kaggle GPU kernel, NumPy stub locally), scores the
composite outputs with the fixed eval board (ours vs identity vs
classical), reports detector F1/IoU beside restorer PSNR/SSIM, and
exports weights + figures + manifest + config so no result lives only
on an ephemeral runtime. The verdict runs the both-gates policy in
``gates``; only a promote verdict unlocks the T6 full run.
"""

import json
import pathlib
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from restore import baseline, detector, metrics
from restore.eval import score_board
from restore.gates import smoke_verdict
from restore.loader import synthetic_pristine
from restore.manifest import build_manifest, canonical_pair_id, seed_for_pair
from restore.pipeline import composite_output
from restore.simulator import simulate_damage

ArrayF32 = np.ndarray
PathLike = Union[str, pathlib.Path]
TrainFn = Callable[[List[Dict[str, object]], Dict[str, object]], Dict[str, object]]

SMOKE_CONFIG = {
    "stage": "smoke",
    "target_pairs": 200,
    "image_size": 256,
    "sim_tier": "core",
    "damage_types": None,
}


def build_smoke_triplets(
    pair_indices: Sequence[int],
    image_size: int = 256,
    damage_types: Optional[Sequence[str]] = None,
) -> List[Dict[str, object]]:
    entries = build_manifest(list(pair_indices))
    triplets: List[Dict[str, object]] = []
    for index, entry in zip(pair_indices, entries):
        pair_id = canonical_pair_id(index)
        pristine = synthetic_pristine(pair_id, image_size, image_size)
        damaged, damage_mask, params = simulate_damage(
            pristine, seed=seed_for_pair(pair_id), types=damage_types
        )
        triplets.append(
            {
                "pair_id": pair_id,
                "pristine": pristine,
                "damaged": damaged,
                "damage_mask": damage_mask,
                "entry": entry,
                "damage_params": params,
            }
        )
    return triplets


def _fills_from_train(
    triplets: List[Dict[str, object]],
    train_result: Mapping[str, object],
) -> List[np.ndarray]:
    if "fills" in train_result:
        return list(train_result["fills"])  # type: ignore[arg-type]
    bias = float(train_result.get("restorer_bias", 0.0))  # type: ignore[arg-type]
    fills = []
    for triplet in triplets:
        damaged = np.asarray(triplet["damaged"], dtype=np.float64)
        fills.append(np.clip(damaged + bias, 0.0, 1.0).astype(np.float32))
    return fills


def run_smoke(
    pair_indices: Sequence[int],
    train_fn: TrainFn,
    out_dir: Optional[PathLike] = None,
    seed: int = 0,
    image_size: int = 256,
    damage_types: Optional[Sequence[str]] = None,
    fix_loops_used: int = 0,
    pipeline_green: bool = False,
) -> Dict[str, object]:
    indices = list(pair_indices)
    config = {
        **SMOKE_CONFIG,
        "seed": seed,
        "image_size": image_size,
        "pair_count": len(indices),
        "pair_indices": indices,
    }
    triplets = build_smoke_triplets(indices, image_size, damage_types)
    train_result = train_fn(triplets, config)
    fills = _fills_from_train(triplets, train_result)
    rows = []
    for triplet, fill in zip(triplets, fills):
        damaged = np.asarray(triplet["damaged"])
        damage_mask = np.asarray(triplet["damage_mask"])
        pristine = np.asarray(triplet["pristine"])
        for method, row_fill in (
            ("ours", fill),
            ("identity", baseline.identity(damaged)),
            ("classical", baseline.median_fill(damaged, damage_mask)),
        ):
            rows.append(
                {
                    "method": method,
                    "damaged": damaged,
                    "damage_mask": damage_mask,
                    "fill": row_fill,
                    "pristine": pristine,
                }
            )
    board = score_board(rows)
    by_method: Dict[str, List[Dict[str, object]]] = {}
    for row in board:
        by_method.setdefault(str(row["method"]), []).append(row)
    means = {
        method: {
            "psnr": float(np.mean([r["psnr"] for r in rows_ if np.isfinite(r["psnr"])]))
            if any(np.isfinite(r["psnr"]) for r in rows_)
            else float("inf"),
            "ssim": float(np.mean([r["ssim"] for r in rows_])),
        }
        for method, rows_ in by_method.items()
    }
    threshold = float(train_result.get("threshold", 0.5))  # type: ignore[arg-type]
    detector_scores = train_result.get("detector_scores")
    stacked_truth = np.concatenate(
        [np.asarray(t["damage_mask"]).ravel() for t in triplets]
    )
    if detector_scores is not None:
        stacked_scores = np.asarray(detector_scores, dtype=np.float64).ravel()
        if stacked_scores.shape != stacked_truth.shape:
            raise ValueError(
                f"shape mismatch: {stacked_scores.shape} vs {stacked_truth.shape}"
            )
        detector_report = detector.detector_scores(
            stacked_scores, stacked_truth, threshold=threshold
        )
    else:
        detected_masks = []
        for triplet, fill in zip(triplets, fills):
            damaged = np.asarray(triplet["damaged"])
            restored = composite_output(
                damaged, fill, np.asarray(triplet["damage_mask"])
            )
            residual = np.abs(restored - damaged).mean(axis=-1)
            detected_masks.append((residual > 1e-6).astype(np.uint8))
        stacked = np.concatenate([m.ravel() for m in detected_masks])
        detector_report = detector.detector_scores(
            stacked.astype(np.float64), stacked_truth, threshold=0.5
        )
    restorer_report = {
        "psnr": means["ours"]["psnr"],
        "ssim": means["ours"]["ssim"],
        "psnr_identity": means["identity"]["psnr"],
        "psnr_classical": means["classical"]["psnr"],
        "ssim_identity": means["identity"]["ssim"],
        "ssim_classical": means["classical"]["ssim"],
        "val_pairs": len(triplets),
    }
    verdict = smoke_verdict(
        pipeline_green=pipeline_green,
        ours=means["ours"],
        identity=means["identity"],
        classical=means["classical"],
        fix_loops_used=fix_loops_used,
    )
    result: Dict[str, object] = {
        "config": config,
        "detector": detector_report,
        "restorer": restorer_report,
        "means": means,
        "verdict": verdict,
        "train": {
            key: value
            for key, value in train_result.items()
            if key not in ("fills", "detector_scores")
        },
    }
    if out_dir is not None:
        directory = pathlib.Path(out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        figure_rows = []
        for triplet, fill in zip(triplets, fills):
            figure_rows.append(
                {
                    "method": "ours",
                    "damaged": triplet["damaged"],
                    "damage_mask": triplet["damage_mask"],
                    "fill": fill,
                    "pristine": triplet["pristine"],
                }
            )
        score_board(
            figure_rows,
            out_dir=directory / "figures",
            manifest_entries=build_manifest(indices),
        )
        weights = train_result.get("weights", {})
        (directory / "weights.json").write_text(
            json.dumps(
                {
                    key: (value.tolist() if isinstance(value, np.ndarray) else value)
                    for key, value in dict(weights).items()  # type: ignore[union-attr]
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (directory / "manifest.json").write_text(
            json.dumps(build_manifest(indices), indent=2), encoding="utf-8"
        )
        (directory / "config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
        serializable = {
            "config": config,
            "detector": detector_report,
            "restorer": restorer_report,
            "means": {
                method: {
                    key: (value if np.isfinite(value) else "inf")
                    for key, value in scores.items()
                }
                for method, scores in means.items()
            },
            "verdict": {
                key: (value if not isinstance(value, float) or np.isfinite(value) else "inf")
                for key, value in verdict.items()
            },
        }
        (directory / "verdict.json").write_text(
            json.dumps(serializable, indent=2), encoding="utf-8"
        )
        result["out_dir"] = str(directory)
    return result


def identity_train_fn(
    triplets: List[Dict[str, object]], config: Dict[str, object]
) -> Dict[str, object]:
    return {
        "weights": {"restorer_bias": 0.0},
        "threshold": 0.5,
        "restorer_bias": 0.0,
    }


def oracle_train_fn(
    triplets: List[Dict[str, object]], config: Dict[str, object]
) -> Dict[str, object]:
    fills = [np.asarray(t["pristine"]).copy() for t in triplets]
    stacked = np.concatenate(
        [np.asarray(t["damage_mask"]).ravel() for t in triplets]
    ).astype(np.float64)
    return {
        "weights": {"restorer_bias": 0.0, "oracle": True},
        "threshold": 0.5,
        "fills": fills,
        "detector_scores": stacked,
    }

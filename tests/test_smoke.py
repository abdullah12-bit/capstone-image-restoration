"""T5 smoke-runner tests: cheap stage proving the whole loop.

CPU-only, no GPU, no network, no dataset download. Uses CONTEXT.md
vocabulary: smoke run, damage mask, pair, pristine image, damaged image,
composite output, classical baseline.
"""

import json

import numpy as np
import pytest

from restore.manifest import TOTAL_PAIRS, split_for_index
from restore.smoke import (
    SMOKE_CONFIG,
    build_smoke_triplets,
    identity_train_fn,
    oracle_train_fn,
    run_smoke,
)


def _train_fn(triplets, config):
    count = sum(np.asarray(t["damage_mask"]).size for t in triplets)
    return {
        "weights": {"restorer_bias": 0.0},
        "threshold": 0.5,
        "detector_scores": np.zeros(count),
    }


def test_smoke_report_splits_detector_from_restorer():
    result = run_smoke(
        pair_indices=[0, 1], train_fn=oracle_train_fn, pipeline_green=True
    )
    assert set(result["detector"]) >= {"precision", "recall", "f1", "iou"}
    assert set(result["restorer"]) >= {"psnr", "ssim"}
    assert result["restorer"]["psnr"] == float("inf")


def test_oracle_fill_promotes_but_identity_train_fn_misses_bar():
    oracle = run_smoke(
        pair_indices=[0, 1], train_fn=oracle_train_fn, pipeline_green=True
    )
    assert oracle["verdict"]["verdict"] == "promote"
    identity = run_smoke(
        pair_indices=[0, 1], train_fn=identity_train_fn, pipeline_green=True
    )
    assert identity["verdict"]["verdict"] == "fix"
    assert identity["restorer"]["psnr"] <= oracle["restorer"]["psnr"]


def test_red_pipeline_never_promotes_from_run_smoke():
    result = run_smoke(pair_indices=[0, 1], train_fn=oracle_train_fn)
    assert result["verdict"]["verdict"] == "fix"


def test_mismatched_detector_scores_rejected():
    def bad_train_fn(triplets, config):
        return {
            "weights": {},
            "threshold": 0.5,
            "detector_scores": np.zeros(3),
        }

    with pytest.raises(ValueError, match="shape mismatch"):
        run_smoke(pair_indices=[0, 1], train_fn=bad_train_fn, pipeline_green=True)


def test_smoke_run_exports_weights_figures_manifest(tmp_path):
    result = run_smoke(
        pair_indices=[0, 1],
        train_fn=_train_fn,
        out_dir=tmp_path,
        seed=0,
        image_size=16,
        pipeline_green=True,
    )
    assert result["verdict"]["verdict"] in ("promote", "fix", "escalate")
    assert (tmp_path / "weights.json").is_file()
    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "config.json").is_file()
    assert (tmp_path / "verdict.json").is_file()
    assert (tmp_path / "figures" / "board.json").is_file()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert [e["pair_id"] for e in manifest] == ["pair-00000", "pair-00001"]
    verdict = json.loads((tmp_path / "verdict.json").read_text(encoding="utf-8"))
    assert verdict["verdict"]["verdict"] in ("promote", "fix", "escalate")


def test_smoke_triplets_carry_exact_damage_masks():
    triplets = build_smoke_triplets([0, 1, 2], image_size=32)
    assert len(triplets) == 3
    for triplet in triplets:
        assert triplet["damage_mask"].shape == triplet["pristine"].shape[:2]
        assert triplet["damage_mask"].dtype == np.uint8
        assert triplet["damage_mask"].any()
        assert triplet["entry"]["pair_id"] == triplet["pair_id"]


def test_smoke_config_targets_200_pairs_and_core_tier():
    assert SMOKE_CONFIG["target_pairs"] == 200
    assert SMOKE_CONFIG["sim_tier"] == "core"


def test_smoke_val_slice_never_touches_test_500():
    val_like = [i for i in range(200) if split_for_index(i) == "val"]
    assert val_like and all(i < TOTAL_PAIRS - 500 for i in val_like)
    with pytest.raises(ValueError, match="out of range"):
        build_smoke_triplets([TOTAL_PAIRS], image_size=16)

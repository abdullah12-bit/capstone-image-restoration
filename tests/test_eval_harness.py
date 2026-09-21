"""T4 seam tests: eval harness + classical baselines.

CPU-only, no GPU, no network, no dataset download. All inputs synthetic.
Uses CONTEXT.md vocabulary: pristine image, damaged image, damage mask,
composite output, classical baseline.
"""

import json

import numpy as np
import pytest

from restore import baseline as classical
from restore import metrics
from restore.eval import score_board


def _synthetic_pristine(seed=0, h=32, w=32):
    rng = np.random.default_rng(seed)
    return rng.random((h, w, 3), dtype=np.float32)


def test_psnr_known_values():
    pristine = np.ones((8, 8, 3), dtype=np.float32)
    assert metrics.psnr(pristine, pristine) == float("inf")
    zeros = np.zeros((8, 8, 3), dtype=np.float32)
    assert metrics.psnr(pristine, zeros) == 0.0
    half = pristine.copy()
    half[:4] = 0.0
    assert metrics.psnr(pristine, half) == pytest.approx(10 * np.log10(2.0))


def test_ssim_known_values():
    pristine = _synthetic_pristine()
    assert metrics.ssim(pristine, pristine) == 1.0
    damaged = np.clip(pristine + 0.25, 0.0, 1.0)
    value = metrics.ssim(pristine, damaged)
    assert 0.0 < value < 1.0


def test_identity_row_scores_perfectly():
    pristine = _synthetic_pristine()
    assert metrics.psnr(pristine, classical.identity(pristine)) == float("inf")
    assert metrics.ssim(pristine, classical.identity(pristine)) == 1.0


def test_classical_baseline_runs_through_composite_path():
    pristine = _synthetic_pristine()
    rng = np.random.default_rng(11)
    damage_mask = (rng.random(pristine.shape[:2]) < 0.1).astype(np.uint8)
    damaged = pristine.copy()
    damaged[damage_mask == 1] = 1.0
    restored = classical.median_restore(damaged, damage_mask)
    assert restored.shape == damaged.shape
    np.testing.assert_array_equal(
        restored[damage_mask == 0], damaged[damage_mask == 0]
    )


def test_board_scores_ours_vs_identity_vs_classical():
    pristine = _synthetic_pristine()
    rng = np.random.default_rng(12)
    damage_mask = (rng.random(pristine.shape[:2]) < 0.1).astype(np.uint8)
    damaged = pristine.copy()
    damaged[damage_mask == 1] = 1.0
    ours = damaged.copy()
    ours[damage_mask == 1] = pristine[damage_mask == 1]
    rows = score_board(
        [
            {"method": "identity", "damaged": damaged,
             "damage_mask": damage_mask, "fill": damaged, "pristine": pristine},
            {"method": "classical", "damaged": damaged,
             "damage_mask": damage_mask,
             "fill": classical.median_fill(damaged, damage_mask),
             "pristine": pristine},
            {"method": "ours", "damaged": damaged,
             "damage_mask": damage_mask, "fill": ours, "pristine": pristine},
        ]
    )
    assert [r["method"] for r in rows] == ["identity", "classical", "ours"]
    for row in rows:
        assert np.isfinite(row["psnr"]) or row["psnr"] == float("inf")
        assert 0.0 <= row["ssim"] <= 1.0
    by_method = {r["method"]: r for r in rows}
    assert by_method["ours"]["psnr"] >= by_method["identity"]["psnr"]
    assert by_method["ours"]["ssim"] >= by_method["identity"]["ssim"]


def test_board_scoring_is_composite_only():
    pristine = _synthetic_pristine()
    damage_mask = np.zeros(pristine.shape[:2], dtype=np.uint8)
    damage_mask[4:12, 4:12] = 1
    damaged = pristine.copy()
    damaged[damage_mask == 1] = 1.0
    leaky_fill = np.zeros_like(damaged)
    rows = score_board(
        [{"method": "ours", "damaged": damaged, "damage_mask": damage_mask,
          "fill": leaky_fill, "pristine": pristine}],
        return_restored=True,
    )
    np.testing.assert_array_equal(
        rows[0]["restored"][damage_mask == 0], damaged[damage_mask == 0]
    )


def test_board_saves_figures_manifest_and_config(tmp_path):
    pristine = _synthetic_pristine()
    damage_mask = np.zeros(pristine.shape[:2], dtype=np.uint8)
    damage_mask[4:12, 4:12] = 1
    damaged = pristine.copy()
    damaged[damage_mask == 1] = 1.0
    rows = score_board(
        [{"method": "ours", "damaged": damaged, "damage_mask": damage_mask,
          "fill": damaged, "pristine": pristine}],
        out_dir=tmp_path,
        manifest_entries=[
            {"pair_id": "pair-00000", "split": "val",
             "content_hash": "abc", "damage_params": {"sim_tier": "core"}}
        ],
    )
    assert (tmp_path / "board.json").is_file()
    assert (tmp_path / "config.json").is_file()
    assert (tmp_path / "manifest.json").is_file()
    assert list(tmp_path.glob("figure-*.ppm"))
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert config["classical_baseline"] == "median"
    assert config["metrics"] == ["psnr", "ssim"]
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest[0]["pair_id"] == "pair-00000"
    assert rows[0]["method"] == "ours"

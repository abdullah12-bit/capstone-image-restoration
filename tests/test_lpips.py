"""T7 seam tests: LPIPS reporting-only on final test outputs.

CPU-only, no GPU, no network, no weights download. All inputs synthetic.
Uses CONTEXT.md vocabulary: pristine image, damaged image, damage mask,
composite output. LPIPS never drives selection or gates (B3-BQ12).
"""

import json

import numpy as np

from restore import eval as eval_module
from restore.eval import score_board
from restore.lpips import lpips


def _synthetic_pristine(seed=0, h=32, w=32):
    rng = np.random.default_rng(seed)
    return rng.random((h, w, 3), dtype=np.float32)


def test_lpips_identical_pristine_scores_zero():
    pristine = _synthetic_pristine()
    assert lpips(pristine, pristine) == 0.0


def test_lpips_bounded_and_orders_worse_fill_higher():
    pristine = _synthetic_pristine()
    rng = np.random.default_rng(21)
    damage_mask = (rng.random(pristine.shape[:2]) < 0.1).astype(np.uint8)
    damaged = pristine.copy()
    damaged[damage_mask == 1] = 1.0
    near = damaged.copy()
    near[damage_mask == 1] = np.clip(
        pristine[damage_mask == 1] + 0.01, 0.0, 1.0
    )
    far = np.zeros_like(damaged)
    near_score = lpips(near, pristine)
    far_score = lpips(far, pristine)
    assert 0.0 <= near_score <= 1.0
    assert 0.0 <= far_score <= 1.0
    assert near_score < far_score


def test_lpips_is_symmetric():
    first = _synthetic_pristine(seed=3)
    second = _synthetic_pristine(seed=4)
    assert lpips(first, second) == lpips(second, first)


def test_board_defaults_have_no_lpips_column():
    pristine = _synthetic_pristine()
    damage_mask = np.zeros(pristine.shape[:2], dtype=np.uint8)
    damage_mask[4:12, 4:12] = 1
    damaged = pristine.copy()
    damaged[damage_mask == 1] = 1.0
    rows = score_board(
        [{"method": "ours", "damaged": damaged, "damage_mask": damage_mask,
          "fill": damaged, "pristine": pristine}]
    )
    assert set(rows[0]) == {"method", "psnr", "ssim"}
    assert eval_module.BOARD_METRICS == ["psnr", "ssim"]


def test_board_opt_in_adds_lpips_column_and_persists_it(tmp_path):
    pristine = _synthetic_pristine()
    damage_mask = np.zeros(pristine.shape[:2], dtype=np.uint8)
    damage_mask[4:12, 4:12] = 1
    damaged = pristine.copy()
    damaged[damage_mask == 1] = 1.0
    rows = score_board(
        [{"method": "ours", "damaged": damaged, "damage_mask": damage_mask,
          "fill": damaged, "pristine": pristine}],
        out_dir=tmp_path,
        manifest_entries=[{"pair_id": "pair-00000"}],
        include_lpips=True,
    )
    assert 0.0 <= rows[0]["lpips"] <= 1.0
    board = json.loads((tmp_path / "board.json").read_text(encoding="utf-8"))
    assert 0.0 <= board[0]["lpips"] <= 1.0
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert config["metrics"] == ["psnr", "ssim", "lpips"]

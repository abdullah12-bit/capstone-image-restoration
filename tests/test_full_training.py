"""T6 seam tests: gated full run + best-validation selection + export.
CPU-only, no GPU, no network, no dataset download. All boards canned.
Uses CONTEXT.md vocabulary: pristine image, damaged image, damage mask,
composite output, smoke run, classical baseline, backbone, pair.

Gate logic lives in ``gates`` and detector metrics in ``detector``;
these tests pin the T6 contract: no green smoke verdict, no full run.
"""

import hashlib
import json

import numpy as np
import pytest

from restore import training
from restore.gates import smoke_verdict


def _green_verdict():
    return smoke_verdict(
        pipeline_green=True,
        ours={"psnr": 27.5, "ssim": 0.86},
        identity={"psnr": 26.0, "ssim": 0.80},
        classical={"psnr": 27.0, "ssim": 0.83},
    )


def _red_verdict():
    return smoke_verdict(
        pipeline_green=True,
        ours={"psnr": 26.5, "ssim": 0.81},
        identity={"psnr": 26.0, "ssim": 0.80},
        classical={"psnr": 27.0, "ssim": 0.83},
    )


def _boards():
    return {
        "validation": [{"method": "ours", "psnr": 27.5, "ssim": 0.86}],
        "test": [
            {"method": "identity", "psnr": 26.0, "ssim": 0.80},
            {"method": "classical", "psnr": 27.0, "ssim": 0.83},
            {"method": "ours", "psnr": 27.5, "ssim": 0.86},
        ],
        "ablation": [
            {"method": "ours-true-mask", "psnr": 28.0, "ssim": 0.88},
            {"method": "ours-detected-mask", "psnr": 27.5, "ssim": 0.86},
            {"method": "ours-core-tier", "psnr": 27.5, "ssim": 0.86},
            {"method": "ours-extended-tier", "psnr": 27.2, "ssim": 0.84},
        ],
    }


def _config():
    return {"epochs": 5, "lr": 1e-3, "batch": 8, "seed": 0}


def _figure_rows():
    rng = np.random.default_rng(0)
    pristine = rng.random((16, 16, 3), dtype=np.float32)
    damage_mask = np.zeros((16, 16), dtype=np.uint8)
    damage_mask[4:8, 4:8] = 1
    damaged = pristine.copy()
    damaged[damage_mask == 1] = 1.0
    base = {"damaged": damaged, "damage_mask": damage_mask, "pristine": pristine}
    return [{**base, "method": "ours", "fill": damaged}]


def test_full_training_refuses_without_recorded_green_smoke(tmp_path):
    assert _green_verdict()["verdict"] == "promote"
    assert _red_verdict()["verdict"] == "fix"
    with pytest.raises(RuntimeError):
        training.run_full_training(
            verdict=_red_verdict(),
            config=_config(),
            manifest_entries=[],
            boards_by_name=_boards(),
            weights=b"w",
            out_dir=tmp_path / "out",
        )


def test_full_training_runs_on_green_smoke_verdict(tmp_path):
    result = training.run_full_training(
        verdict=_green_verdict(),
        config=_config(),
        manifest_entries=[{"pair_id": "pair-00000"}],
        boards_by_name=_boards(),
        weights=b"w",
        out_dir=tmp_path / "out",
    )
    assert result["config"]["backbone"] == "resnet18-imagenet"
    assert result["paths"]["weights"].is_file()
    assert {row["board"] for row in result["scored"]} == {
        "validation",
        "test",
        "ablation",
    }


def test_best_validation_checkpoint_selection():
    history = [
        {"epoch": 0, "val_psnr": 27.0, "val_ssim": 0.83},
        {"epoch": 1, "val_psnr": 27.5, "val_ssim": 0.84},
        {"epoch": 2, "val_psnr": 27.5, "val_ssim": 0.86},
    ]
    best = training.select_best_checkpoint(history)
    assert best["epoch"] == 2
    with pytest.raises(ValueError):
        training.select_best_checkpoint([])


def test_full_config_records_epochs_lr_batch_seed_and_backbone():
    config = training.validate_full_config(_config())
    assert config["epochs"] == 5
    assert config["backbone"] == "resnet18-imagenet"
    assert config["classical_baseline"] == "median"
    assert config["metrics"] == ["psnr", "ssim"]
    with pytest.raises((KeyError, ValueError)):
        training.validate_full_config({"epochs": 1})


def test_three_board_protocol_requires_test_and_ablation_rows():
    training.check_three_boards(_boards())
    bad = _boards()
    bad["ablation"] = [{"method": "ours", "psnr": 27.0, "ssim": 0.8}]
    with pytest.raises(ValueError):
        training.check_three_boards(bad)


def test_export_writes_versioned_weights_manifest_config_boards(tmp_path):
    out = tmp_path / "full"
    paths = training.export_full_run(
        out,
        config=_config(),
        manifest_entries=[{"pair_id": "pair-00000"}],
        boards_by_name=_boards(),
        weights=b"weights-bytes",
        figure_rows=_figure_rows(),
    )
    assert (out / "weights.bin").is_file()
    assert (out / "manifest.json").is_file()
    assert (out / "boards.json").is_file()
    assert (out / "version.txt").is_file()
    assert (out / "figures" / "board.json").is_file()
    saved_config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert saved_config["epochs"] == 5
    assert saved_config["backbone"] == "resnet18-imagenet"
    assert saved_config["classical_baseline"] == "median"
    assert saved_config["metrics"] == ["psnr", "ssim"]
    assert paths["weights"].is_file()
    assert paths["figures"].is_dir()
    pinned = (out / "version.txt").read_text(encoding="utf-8")
    reread = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert (
        hashlib.sha256(json.dumps(reread, sort_keys=True).encode("utf-8")).hexdigest()[
            :12
        ]
        == pinned
    )

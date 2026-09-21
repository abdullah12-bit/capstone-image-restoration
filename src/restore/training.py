"""T6 full training: gated full run + best-validation selection + export.

Full training on the 4.5k train pool runs only on a recorded green smoke
verdict (checked via ``gates.full_train_allowed`` -- pipeline-green
first, then the metric bar over the validation slice). It selects the
best-validation checkpoint, scores the untouched test 500 with the
three-board protocol (validation, held-out test with identity +
classical rows, ablations: true-mask vs detected-mask, core vs extended
tier), and exports versioned weights + figures + manifest + config for
the demo.

CPU-only, no GPU, no network: selection, board checks, and export run
on canned rows so pytest stays offline. The Kaggle GPU kernel calls the
same seam with real boards. Gate logic lives in ``gates`` and detector
metrics in ``detector``; this module never reimplements them.
"""

import hashlib
import json
import pathlib
from typing import Dict, List, Mapping, Optional, Sequence, Union

from restore.eval import BOARD_METRICS, CLASSICAL_BASELINE, score_board
from restore.gates import full_train_allowed

PathLike = Union[str, pathlib.Path]

BACKBONE = "resnet18-imagenet"
REQUIRED_CONFIG_KEYS = ("epochs", "lr", "batch", "seed")


def select_best_checkpoint(
    history: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    if not history:
        raise ValueError("empty validation history: no checkpoint to select")
    return dict(
        max(
            history,
            key=lambda row: (float(row["val_psnr"]), float(row["val_ssim"])),
        )
    )


def validate_full_config(config: Mapping[str, object]) -> Dict[str, object]:
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise KeyError(f"full config missing keys: {missing}")
    validated = dict(config)
    validated.setdefault("backbone", BACKBONE)
    validated.setdefault("classical_baseline", CLASSICAL_BASELINE)
    validated.setdefault("metrics", list(BOARD_METRICS))
    return validated


def check_three_boards(boards_by_name: Mapping[str, Sequence[Mapping[str, object]]]) -> None:
    missing = {"validation", "test", "ablation"} - set(boards_by_name)
    if missing:
        raise ValueError(f"three-board protocol missing boards: {sorted(missing)}")
    test_methods = {str(row["method"]) for row in boards_by_name["test"]}
    if {"identity", "classical", "ours"} - test_methods:
        raise ValueError(f"test board needs identity + classical + ours rows: {sorted(test_methods)}")
    ablation_methods = {str(row["method"]) for row in boards_by_name["ablation"]}
    required = {
        "ours-true-mask",
        "ours-detected-mask",
        "ours-core-tier",
        "ours-extended-tier",
    }
    if required - ablation_methods:
        raise ValueError(
            "ablation board needs true-mask vs detected-mask and "
            f"core vs extended tier rows: {sorted(ablation_methods)}"
        )


def export_full_run(
    out_dir: PathLike,
    config: Mapping[str, object],
    manifest_entries: Sequence[Mapping[str, object]],
    boards_by_name: Mapping[str, Sequence[Mapping[str, object]]],
    weights: bytes,
    figure_rows: Optional[Sequence[Mapping[str, object]]] = None,
) -> Dict[str, pathlib.Path]:
    validated = validate_full_config(config)
    check_three_boards(boards_by_name)
    directory = pathlib.Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    weights_path = directory / "weights.bin"
    weights_path.write_bytes(bytes(weights))
    version = hashlib.sha256(
        json.dumps(validated, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    (directory / "version.txt").write_text(version, encoding="utf-8")
    (directory / "manifest.json").write_text(
        json.dumps(list(manifest_entries), indent=2), encoding="utf-8"
    )
    (directory / "config.json").write_text(
        json.dumps(validated, indent=2), encoding="utf-8"
    )
    (directory / "boards.json").write_text(
        json.dumps({name: list(rows) for name, rows in boards_by_name.items()}, indent=2),
        encoding="utf-8",
    )
    paths = {
        "weights": weights_path,
        "manifest": directory / "manifest.json",
        "config": directory / "config.json",
        "boards": directory / "boards.json",
        "version": directory / "version.txt",
    }
    if figure_rows is not None:
        score_board(
            figure_rows,
            out_dir=directory / "figures",
            manifest_entries=manifest_entries,
        )
        paths["figures"] = directory / "figures"
    return paths


def run_full_training(
    verdict: Mapping[str, object],
    config: Mapping[str, object],
    manifest_entries: Sequence[Mapping[str, object]],
    boards_by_name: Mapping[str, Sequence[Mapping[str, object]]],
    weights: bytes,
    out_dir: PathLike,
    figure_rows: Optional[Sequence[Mapping[str, object]]] = None,
) -> Dict[str, object]:
    if not full_train_allowed(verdict):
        raise RuntimeError("full training needs a recorded green smoke verdict")
    validated = validate_full_config(config)
    check_three_boards(boards_by_name)
    paths = export_full_run(
        out_dir,
        validated,
        manifest_entries,
        boards_by_name,
        weights,
        figure_rows=figure_rows,
    )
    scored: List[Dict[str, object]] = [
        {"board": name, **dict(row)}
        for name, rows in boards_by_name.items()
        for row in rows
    ]
    return {"config": validated, "paths": paths, "scored": scored}

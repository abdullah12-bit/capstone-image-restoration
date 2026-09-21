"""T4 eval harness: fixed scoreboards with locked rows and metrics.

Boards score composite outputs only: each row supplies its fill, the
harness composites over the damage mask, then scores PSNR/SSIM against
the pristine image. Every board can save figures (side-by-side
damaged / damage-mask / restored) plus manifest and config so any number
is reproducible from pinned weights + manifest.
"""

import json
import pathlib
from typing import Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from restore import metrics
from restore.pipeline import composite_output

ArrayF32 = np.ndarray
PathLike = Union[str, pathlib.Path]

CLASSICAL_BASELINE = "median"
BOARD_METRICS = ["psnr", "ssim"]


def _row_metrics(restored: ArrayF32, pristine: ArrayF32) -> Dict[str, float]:
    return {
        "psnr": metrics.psnr(restored, pristine),
        "ssim": metrics.ssim(restored, pristine),
    }


def _write_ppm(path: pathlib.Path, image: np.ndarray) -> None:
    pixels = np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0)
    if pixels.ndim == 2:
        pixels = np.stack([pixels] * 3, axis=-1)
    height, width, _ = pixels.shape
    body = (pixels * 255.0).round().astype(np.uint8).tobytes()
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    path.write_bytes(header + body)


def _side_by_side(
    damaged: ArrayF32, damage_mask: np.ndarray, restored: ArrayF32
) -> np.ndarray:
    image = np.asarray(damaged, dtype=np.float64)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    fill = np.asarray(restored, dtype=np.float64)
    if fill.ndim == 2:
        fill = np.stack([fill] * 3, axis=-1)
    mask = np.asarray(damage_mask).astype(np.float64)
    mask_rgb = np.stack([mask] * 3, axis=-1)
    return np.concatenate([image, mask_rgb, fill], axis=1)


def score_board(
    rows: Sequence[Mapping[str, object]],
    out_dir: Optional[PathLike] = None,
    manifest_entries: Optional[Sequence[Mapping[str, object]]] = None,
    return_restored: bool = False,
) -> List[Dict[str, object]]:
    scored: List[Dict[str, object]] = []
    figures: List[np.ndarray] = []
    for row in rows:
        damaged = np.asarray(row["damaged"])
        damage_mask = np.asarray(row["damage_mask"])
        fill = np.asarray(row["fill"])
        pristine = np.asarray(row["pristine"])
        restored = composite_output(damaged, fill, damage_mask)
        scored_row: Dict[str, object] = {
            "method": row["method"],
            **_row_metrics(restored, pristine),
        }
        if return_restored or out_dir is not None:
            scored_row["restored"] = restored
        if out_dir is not None:
            figures.append(_side_by_side(damaged, damage_mask, restored))
        scored.append(scored_row)
    if out_dir is not None:
        directory = pathlib.Path(out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        for index, figure in enumerate(figures):
            method = scored[index]["method"]
            _write_ppm(directory / f"figure-{index:02d}-{method}.ppm", figure)
        for row in scored:
            row.pop("restored", None)
        (directory / "board.json").write_text(
            json.dumps(scored, indent=2), encoding="utf-8"
        )
        (directory / "config.json").write_text(
            json.dumps(
                {"classical_baseline": CLASSICAL_BASELINE,
                 "metrics": BOARD_METRICS},
                indent=2,
            ),
            encoding="utf-8",
        )
        (directory / "manifest.json").write_text(
            json.dumps(list(manifest_entries or []), indent=2), encoding="utf-8"
        )
    return scored

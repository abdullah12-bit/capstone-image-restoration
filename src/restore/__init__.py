"""Detect-then-restore pipeline package (T1 scaffold, T2 data seams).

Seam: ``restore(damaged) -> (restored, damage_mask)``. Stage seams
``detect`` / ``restore_masked`` / ``composite_output`` let later work
attribute failures (bad mask vs bad fill) before full-pipeline tests.
Data seams: ``iter_pairs`` / ``synthetic_pristine`` (loader) and
``build_manifest`` / ``split_for_index`` (manifest) record split,
damage parameters, and content hash per pair. Eval seams: ``psnr`` /
``ssim`` (metrics), ``median_restore`` / ``identity`` (classical
baseline), ``score_board`` (fixed scoreboards with figures + manifest +
config).

T1 ships CPU-only NumPy stubs with the composite guarantee
(clean pixels survive bit-exactly). Learned detector/restorer land in T5.
"""

from restore import baseline, eval, lpips, metrics, simulator
from restore.baseline import identity, median_fill, median_restore
from restore.eval import CLASSICAL_BASELINE, score_board
from restore.loader import iter_pairs, random_crop_and_flip, synthetic_pristine
from restore.lpips import lpips
from restore.manifest import (
    TEST_SIZE,
    TOTAL_PAIRS,
    TRAIN_POOL_SIZE,
    build_manifest,
    canonical_pair_id,
    content_hash,
    load_manifest,
    save_manifest,
    seed_for_pair,
    split_for_index,
)
from restore.metrics import mse, psnr, ssim
from restore.pipeline import composite_output, detect, restore, restore_masked
from restore.simulator import (
    CORE_TYPES,
    EXTENDED_TYPES,
    simulate_damage,
)

__all__ = [
    "CLASSICAL_BASELINE",
    "CORE_TYPES",
    "EXTENDED_TYPES",
    "TEST_SIZE",
    "TOTAL_PAIRS",
    "TRAIN_POOL_SIZE",
    "baseline",
    "build_manifest",
    "canonical_pair_id",
    "composite_output",
    "content_hash",
    "detect",
    "eval",
    "identity",
    "iter_pairs",
    "load_manifest",
    "lpips",
    "median_fill",
    "median_restore",
    "metrics",
    "mse",
    "psnr",
    "random_crop_and_flip",
    "restore",
    "restore_masked",
    "save_manifest",
    "score_board",
    "seed_for_pair",
    "simulate_damage",
    "simulator",
    "split_for_index",
    "ssim",
    "synthetic_pristine",
]

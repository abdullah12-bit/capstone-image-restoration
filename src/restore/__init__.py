"""Detect-then-restore pipeline package (T1 scaffold, T2 data seams).

Seam: ``restore(damaged) -> (restored, damage_mask)``. Stage seams
``detect`` / ``restore_masked`` / ``composite_output`` let later work
attribute failures (bad mask vs bad fill) before full-pipeline tests.
Data seams: ``iter_pairs`` / ``synthetic_pristine`` (loader) and
``build_manifest`` / ``split_for_index`` (manifest) record split,
damage parameters, and content hash per pair.

T1 ships CPU-only NumPy stubs with the composite guarantee
(clean pixels survive bit-exactly). Learned detector/restorer land in T5.
"""

from restore.loader import iter_pairs, synthetic_pristine
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
from restore.pipeline import composite_output, detect, restore, restore_masked

__all__ = [
    "TEST_SIZE",
    "TOTAL_PAIRS",
    "TRAIN_POOL_SIZE",
    "build_manifest",
    "canonical_pair_id",
    "composite_output",
    "content_hash",
    "detect",
    "iter_pairs",
    "load_manifest",
    "restore",
    "restore_masked",
    "save_manifest",
    "seed_for_pair",
    "split_for_index",
    "synthetic_pristine",
]

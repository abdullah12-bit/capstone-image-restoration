"""T2 seam tests: loader + manifest + splits.

CPU-only, no GPU, no network, no dataset download. All pristine images
synthetic. Uses CONTEXT.md vocabulary: pristine image, pair.
"""

import json

import numpy as np

from restore.loader import iter_pairs, synthetic_pristine
from restore.manifest import (
    TEST_SIZE,
    TRAIN_POOL_SIZE,
    build_manifest,
    canonical_pair_id,
    content_hash,
    load_manifest,
    save_manifest,
    split_for_index,
)


def test_content_hash_is_deterministic():
    first = content_hash(canonical_pair_id(0))
    assert first == content_hash(canonical_pair_id(0))
    assert len(first) == 64
    assert content_hash(canonical_pair_id(0)) != content_hash(canonical_pair_id(1))


def test_split_counts_hold_over_full_5k():
    splits = [split_for_index(i) for i in range(TRAIN_POOL_SIZE + TEST_SIZE)]
    assert splits.count("test") == TEST_SIZE
    assert splits.count("train") + splits.count("val") == TRAIN_POOL_SIZE
    assert 200 <= splits.count("val") <= 400


def test_shipped_test_500_never_leaks_into_train_or_val():
    for i in range(TRAIN_POOL_SIZE, TRAIN_POOL_SIZE + TEST_SIZE):
        assert split_for_index(i) == "test"
    val_ids = [i for i in range(TRAIN_POOL_SIZE) if split_for_index(i) == "val"]
    assert val_ids, "validation slice must be non-empty"
    assert all(i < TRAIN_POOL_SIZE for i in val_ids)


def test_manifest_records_split_params_and_hash_per_pair():
    entries = build_manifest(indices=[0, 1, TRAIN_POOL_SIZE])
    assert len(entries) == 3
    for entry in entries:
        assert set(entry) >= {"pair_id", "split", "content_hash", "damage_params"}
        assert entry["content_hash"] == content_hash(entry["pair_id"])
        assert entry["split"] in {"train", "val", "test"}
    by_id = {e["pair_id"]: e for e in entries}
    assert by_id[canonical_pair_id(TRAIN_POOL_SIZE)]["split"] == "test"


def test_manifest_round_trip_preserves_entries(tmp_path):
    entries = build_manifest(indices=list(range(20)))
    path = tmp_path / "manifest.json"
    save_manifest(entries, path)
    reloaded = load_manifest(path)
    assert reloaded == entries
    assert json.loads(path.read_text(encoding="utf-8")) == entries


def test_synthetic_pristine_loader_runs_offline_on_tiny_subset():
    pairs = list(iter_pairs(indices=[0, 1, 2], height=32, width=32))
    assert len(pairs) == 3
    for pair_id, pristine, entry in pairs:
        assert pristine.shape == (32, 32, 3)
        assert pristine.dtype == np.float32
        assert entry["pair_id"] == pair_id
    repeat = [p for _, p, _ in iter_pairs(indices=[0, 1, 2], height=32, width=32)]
    for (_, first, _), second in zip(pairs, repeat):
        np.testing.assert_array_equal(first, second)


def test_synthetic_pristine_matches_seed_from_content_hash():
    first = synthetic_pristine(canonical_pair_id(7))
    assert first.shape == (64, 64, 3)
    np.testing.assert_array_equal(first, synthetic_pristine(canonical_pair_id(7)))
    assert not np.array_equal(
        first, synthetic_pristine(canonical_pair_id(8))
    )

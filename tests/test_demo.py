"""T8 seam tests: demo contract behind a pinned seam.

CPU-only, no GPU, no network, no weights download. All inputs synthetic.
Uses CONTEXT.md vocabulary: damaged image, damage mask, composite output.
Locked by B3-BQ13: upload any photo (or one-click built-in example pair),
see damaged / damage-mask / restored side-by-side with a short note;
4 built-in example pairs bundled; same input gives identical mask + output
as the eval harness; single image serves within the Spaces timeout.
"""

import time

import numpy as np

from restore.demo import (
    EXAMPLE_PAIR_IDS,
    build_example,
    describe_restoration,
    serve_restoration,
)


def test_demo_serves_damaged_mask_and_restored_views():
    rng = np.random.default_rng(0)
    damaged = rng.random((64, 48, 3), dtype=np.float32)
    views = serve_restoration(damaged)
    assert set(views) == {"damaged", "damage_mask", "restored", "note"}
    assert views["damaged"].shape == damaged.shape
    assert views["restored"].shape == damaged.shape
    assert views["damage_mask"].shape == damaged.shape[:2]
    assert views["damage_mask"].dtype == np.uint8
    assert set(np.unique(views["damage_mask"])).issubset({0, 1})


def test_demo_matches_pipeline_seam_bit_exactly():
    from restore.pipeline import restore

    rng = np.random.default_rng(1)
    damaged = rng.random((64, 48, 3), dtype=np.float32)
    expected_restored, expected_mask = restore(damaged)
    views = serve_restoration(damaged)
    np.testing.assert_array_equal(views["restored"], expected_restored)
    np.testing.assert_array_equal(views["damage_mask"], expected_mask)


def test_demo_keeps_undamaged_pixels_bit_exactly():
    rng = np.random.default_rng(2)
    damaged = rng.random((64, 48, 3), dtype=np.float32)
    views = serve_restoration(damaged)
    damage_mask = views["damage_mask"]
    np.testing.assert_array_equal(
        views["restored"][damage_mask == 0], damaged[damage_mask == 0]
    )


def test_demo_bundles_four_built_in_example_pairs():
    assert len(EXAMPLE_PAIR_IDS) == 4
    for pair_id in EXAMPLE_PAIR_IDS:
        damaged, pristine = build_example(pair_id)
        assert damaged.shape == pristine.shape
        views = serve_restoration(damaged, pair_id=pair_id)
        assert views["restored"].shape == damaged.shape
        assert "no reference" not in views["note"].lower()


def test_demo_single_image_serves_within_spaces_timeout():
    rng = np.random.default_rng(3)
    damaged = rng.random((64, 48, 3), dtype=np.float32)
    started = time.perf_counter()
    serve_restoration(damaged)
    assert time.perf_counter() - started < 25.0


def test_demo_note_describes_damage_coverage():
    assert "0.0%" in describe_restoration(np.zeros((8, 8), dtype=np.uint8))
    note = describe_restoration(np.ones((8, 8), dtype=np.uint8))
    assert "100.0%" in note

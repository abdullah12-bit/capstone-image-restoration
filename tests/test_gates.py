"""T5 gate tests: pipeline-green first, metric bar second, bounded fixes.

CPU-only, no GPU, no network, no dataset download. Uses CONTEXT.md
vocabulary: smoke run, classical baseline, pristine image.
"""

import pytest

from restore.gates import MAX_FIX_LOOPS, full_train_allowed, smoke_verdict


def _rows(**overrides):
    rows = {
        "ours": {"psnr": 20.0, "ssim": 0.8},
        "identity": {"psnr": 10.0, "ssim": 0.5},
        "classical": {"psnr": 15.0, "ssim": 0.6},
    }
    rows.update(overrides)
    return rows


def test_smoke_verdict_promotes_green_pipeline_over_bar():
    verdict = smoke_verdict(pipeline_green=True, **_rows())
    assert verdict["verdict"] == "promote"
    assert full_train_allowed(verdict) is True


def test_metric_bar_requires_beating_identity_and_classical():
    verdict = smoke_verdict(
        pipeline_green=True,
        **_rows(ours={"psnr": 12.0, "ssim": 0.8}),
    )
    assert verdict["verdict"] == "fix"
    assert verdict["bar"] == {"psnr": False, "ssim": True}
    assert full_train_allowed(verdict) is False


def test_classical_tie_does_not_clear_bar():
    verdict = smoke_verdict(
        pipeline_green=True,
        **_rows(ours={"psnr": 15.0, "ssim": 0.6}),
    )
    assert verdict["verdict"] == "fix"


def test_red_pipeline_never_promotes_even_with_better_scores():
    verdict = smoke_verdict(pipeline_green=False, **_rows())
    assert verdict["verdict"] == "fix"
    assert full_train_allowed(verdict) is False


def test_two_fix_loops_then_escalate_with_evidence():
    assert MAX_FIX_LOOPS == 2
    first = smoke_verdict(pipeline_green=False, fix_loops_used=0, **_rows())
    assert first["verdict"] == "fix"
    assert first["loops_left"] == 2
    assert first["evidence"]["pipeline_green"] is False
    assert (
        smoke_verdict(pipeline_green=False, fix_loops_used=2, **_rows())["verdict"]
        == "escalate"
    )
    missed = _rows(ours={"psnr": 12.0, "ssim": 0.8})
    assert smoke_verdict(
        pipeline_green=True, fix_loops_used=2, **missed
    )["verdict"] == "escalate"


def test_missing_gate_metrics_rejected():
    with pytest.raises(ValueError, match="missing metrics"):
        smoke_verdict(pipeline_green=True, **_rows(ours={"psnr": 20.0}))

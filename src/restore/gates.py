"""T5 gates: both-gates policy before full-training spend.

Pipeline-green comes first (load, train, score, figures, weights
download), then the metric bar (beat identity AND the classical
baseline on PSNR/SSIM over the validation slice). A red smoke gets at
most two fix loops (diagnose, adjust simulator or hyperparameters,
re-smoke) before escalation; full training never runs on red.
"""

from typing import Dict, Mapping

MAX_FIX_LOOPS = 2
GATE_METRICS = ("psnr", "ssim")


def smoke_verdict(
    pipeline_green: bool,
    ours: Mapping[str, float],
    identity: Mapping[str, float],
    classical: Mapping[str, float],
    fix_loops_used: int = 0,
) -> Dict[str, object]:
    missing = [m for m in GATE_METRICS if m not in ours]
    if missing:
        raise ValueError(f"ours scores missing metrics: {missing}")
    if not pipeline_green:
        loops_left = MAX_FIX_LOOPS - fix_loops_used
        return {
            "verdict": "fix" if loops_left > 0 else "escalate",
            "reason": "pipeline not green: load, train, score, figures, weights",
            "loops_left": max(loops_left, 0),
            "evidence": {
                "pipeline_green": False,
                "fix_loops_used": fix_loops_used,
            },
        }
    bar = {
        metric: bool(
            ours[metric] > identity[metric] and ours[metric] > classical[metric]
        )
        for metric in GATE_METRICS
    }
    if all(bar.values()):
        return {
            "verdict": "promote",
            "reason": "green + metric bar",
            "bar": bar,
            "evidence": {"ours": dict(ours), "identity": dict(identity),
                         "classical": dict(classical)},
        }
    loops_left = MAX_FIX_LOOPS - fix_loops_used
    return {
        "verdict": "fix" if loops_left > 0 else "escalate",
        "reason": "metric bar missed",
        "bar": bar,
        "loops_left": max(loops_left, 0),
        "evidence": {"ours": dict(ours), "identity": dict(identity),
                     "classical": dict(classical),
                     "fix_loops_used": fix_loops_used},
    }


def full_train_allowed(verdict: Mapping[str, object]) -> bool:
    return verdict.get("verdict") == "promote"

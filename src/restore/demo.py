"""T8 demo seam: pinned contract the Spaces app serves through.

Locked by B3-BQ13: upload any photo (or one built-in example pair with
one click) and see damaged / damage-mask / restored side-by-side with a
short note. The app path goes through the same ``restore`` seam as the
eval harness, so the same input gives identical mask + output. Real
archival scans stay eyes-only: bundled examples are synthetic pairs,
and the note flags qualitative uploads with no reference.
"""

from typing import Dict, Tuple

import numpy as np

from restore.loader import synthetic_pristine
from restore.manifest import seed_for_pair
from restore.pipeline import restore
from restore.simulator import simulate_damage
from restore.tiles import restore_tiled

ArrayF32 = np.ndarray

EXAMPLE_PAIR_IDS = ("pair-00000", "pair-00001", "pair-00002", "pair-00003")
EXAMPLE_SIZE = 256


def build_example(pair_id: str, size: int = EXAMPLE_SIZE) -> Tuple[ArrayF32, ArrayF32]:
    pristine = synthetic_pristine(pair_id, size, size)
    damaged, _, _ = simulate_damage(pristine, seed=seed_for_pair(pair_id))
    return damaged.astype(np.float32, copy=False), pristine


def describe_restoration(damage_mask: np.ndarray) -> str:
    flagged = float(np.asarray(damage_mask).astype(bool).mean() * 100.0)
    return (
        f"restored {flagged:.1f}% flagged damaged regions; "
        "pristine pixels elsewhere kept bit-exactly"
    )


def serve_restoration(
    damaged: ArrayF32,
    pair_id: str | None = None,
) -> Dict[str, object]:
    image = np.asarray(damaged)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"damaged image must be HxWx3, got {image.shape}")
    height, width = image.shape[:2]
    if max(height, width) <= EXAMPLE_SIZE:
        restored, damage_mask = restore(image)
    else:
        restored, damage_mask = restore_tiled(image)
    note = describe_restoration(damage_mask)
    if pair_id is not None and pair_id not in EXAMPLE_PAIR_IDS:
        note += " (no reference - qualitative)"
    return {
        "damaged": image,
        "damage_mask": damage_mask,
        "restored": restored,
        "note": note,
    }

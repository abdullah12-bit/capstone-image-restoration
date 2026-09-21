"""T2 loader: synthetic offline stand-in for HF streaming.

Full streaming from ``joshuachin/openphoto-restore-dataset`` runs on the
Kaggle kernel with internet. Locally (and in pytest) the loader derives
a deterministic pristine image per pair from its content hash, so no
network, GPU, or dataset download is needed. Every yielded pair carries
its manifest entry with split, damage parameters, and content hash.
"""

from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np

from restore.manifest import build_manifest, canonical_pair_id, seed_for_pair

ArrayF32 = np.ndarray


def random_crop_and_flip(
    damaged: ArrayF32, damage_mask: np.ndarray, seed: int = 0, crop_size: int = 256
) -> Tuple[ArrayF32, np.ndarray]:
    if damaged.ndim != 3 or damaged.shape[2] != 3:
        raise ValueError(f"damaged image must be HxWx3, got {damaged.shape}")
    if damage_mask.shape != damaged.shape[:2]:
        raise ValueError(
            f"damage mask shape {damage_mask.shape} must match image "
            f"{damaged.shape[:2]}"
        )
    height, width = damaged.shape[:2]
    if height < crop_size or width < crop_size:
        raise ValueError(
            f"image {height}x{width} smaller than crop {crop_size}"
        )
    rng = np.random.default_rng(seed)
    top = int(rng.integers(0, height - crop_size + 1))
    left = int(rng.integers(0, width - crop_size + 1))
    cropped_damaged = damaged[top : top + crop_size, left : left + crop_size].copy()
    cropped_mask = damage_mask[top : top + crop_size, left : left + crop_size].copy()
    if bool(rng.integers(0, 2)):
        cropped_damaged = cropped_damaged[:, ::-1].copy()
        cropped_mask = cropped_mask[:, ::-1].copy()
    return cropped_damaged, cropped_mask


def synthetic_pristine(pair_id: str, height: int = 64, width: int = 64) -> ArrayF32:
    rng = np.random.default_rng(seed_for_pair(pair_id))
    return rng.random((height, width, 3), dtype=np.float32)


def iter_pairs(
    indices: Sequence[int], height: int = 64, width: int = 64
) -> Iterator[Tuple[str, ArrayF32, Dict[str, object]]]:
    entries: List[Dict[str, object]] = build_manifest(list(indices))
    for index, entry in zip(indices, entries):
        pair_id = canonical_pair_id(index)
        assert entry["pair_id"] == pair_id
        yield pair_id, synthetic_pristine(pair_id, height, width), entry

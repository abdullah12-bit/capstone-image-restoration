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

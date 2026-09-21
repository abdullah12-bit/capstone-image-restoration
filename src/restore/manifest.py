"""T2 manifest: splits + content hashes + per-pair records.

Shipped layout is fixed: train pool 4.5k / test 500. Validation (~300)
is carved from the train pool by a fixed content-hash rule so model
selection is stable across runs. The shipped test 500 never leaks into
train or validation.

Local placeholder: ``content_hash`` hashes the canonical pair id, and
``split_for_index`` assumes shipped order (indices 0..4999, test at
4500+). The Kaggle kernel wires real HF ids / image bytes here without
changing the seam. ``damage_params`` is a stub until the T3 simulator
lands real damage settings.
"""

import hashlib
import json
import pathlib
from typing import Dict, List, Sequence, Union

TRAIN_POOL_SIZE = 4500
TEST_SIZE = 500
TOTAL_PAIRS = TRAIN_POOL_SIZE + TEST_SIZE
VAL_MODULUS = 15

PathLike = Union[str, pathlib.Path]


def _check_index(index: int) -> None:
    if not 0 <= index < TOTAL_PAIRS:
        raise ValueError(f"pair index out of range 0..{TOTAL_PAIRS - 1}: {index}")


def canonical_pair_id(index: int) -> str:
    _check_index(index)
    return f"pair-{index:05d}"


def content_hash(pair_id: str) -> str:
    return hashlib.sha256(pair_id.encode("utf-8")).hexdigest()


def seed_for_pair(pair_id: str) -> int:
    return int(content_hash(pair_id)[:8], 16)


def split_for_index(index: int) -> str:
    _check_index(index)
    if index >= TRAIN_POOL_SIZE:
        return "test"
    if seed_for_pair(canonical_pair_id(index)) % VAL_MODULUS == 0:
        return "val"
    return "train"


def default_damage_params(pair_id: str) -> Dict[str, object]:
    return {
        "sim_tier": "core",
        "seed": seed_for_pair(pair_id),
    }


def build_manifest(indices: Sequence[int]) -> List[Dict[str, object]]:
    entries = []
    for index in indices:
        pair_id = canonical_pair_id(index)
        entries.append(
            {
                "pair_id": pair_id,
                "split": split_for_index(index),
                "content_hash": content_hash(pair_id),
                "damage_params": default_damage_params(pair_id),
            }
        )
    return entries


def save_manifest(entries: Sequence[Dict[str, object]], path: PathLike) -> pathlib.Path:
    path = pathlib.Path(path)
    path.write_text(json.dumps(list(entries), indent=2), encoding="utf-8")
    return path


def load_manifest(path: PathLike) -> List[Dict[str, object]]:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))

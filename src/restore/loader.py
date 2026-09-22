"""T2 loader: HF parquet pairs on Kaggle, synthetic fallback offline.

``load_hf_pair`` reads one damaged/pristine pair from the OpenPhoto
parquet files (needs ``pyarrow`` + ``PIL``; Kaggle kernel downloads the
parquets). ``iter_hf_pairs`` yields real pairs for an index list.
Offline (and in pytest) the loader derives a deterministic pristine
image per pair from its content hash, so no network, GPU, or dataset
download is needed. Every yielded pair carries its manifest entry.
"""

from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from restore.manifest import build_manifest, canonical_pair_id, seed_for_pair

ArrayF32 = np.ndarray

HF_PARQUET_URLS = tuple(
    "https://huggingface.co/datasets/joshuachin/openphoto-restore-dataset"
    "/resolve/main/data/train-0000%d-of-00005.parquet" % i
    for i in range(5)
) + (
    "https://huggingface.co/datasets/joshuachin/openphoto-restore-dataset"
    "/resolve/main/data/test-00000-of-00001.parquet",
)

_parquet_cache: Dict[str, object] = {}


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


def _read_parquet(path: str):
    if path in _parquet_cache:
        return _parquet_cache[path]
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    _parquet_cache[path] = table
    return table


def _image_from_cell(cell) -> ArrayF32:
    import io

    from PIL import Image

    raw = cell["bytes"] if isinstance(cell, dict) else cell
    image = Image.open(io.BytesIO(bytes(raw))).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def load_hf_pair(
    index: int,
    data_dir: str = "/kaggle/working/data",
    height: int = 256,
    width: int = 256,
) -> Tuple[ArrayF32, ArrayF32]:
    """Load one real damaged/pristine pair, resized to height x width."""
    import glob
    import os

    from PIL import Image

    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError("no parquet files in %s" % data_dir)
    shard, row = divmod(index, 1000)
    table = _read_parquet(files[shard % len(files)])
    row_idx = row % table.num_rows
    damaged = _image_from_cell(table.column("damaged_image")[row_idx].as_py())
    pristine = _image_from_cell(table.column("pristine_image")[row_idx].as_py())
    out = []
    for image in (damaged, pristine):
        pil = Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8))
        pil = pil.resize((width, height), Image.BILINEAR)
        out.append(np.asarray(pil, dtype=np.float32) / 255.0)
    return out[0], out[1]


def iter_hf_pairs(
    indices: Sequence[int],
    data_dir: str = "/kaggle/working/data",
    height: int = 256,
    width: int = 256,
    manifest_entries: Optional[Sequence[Dict[str, object]]] = None,
) -> Iterator[Tuple[str, ArrayF32, ArrayF32, Dict[str, object]]]:
    entries: List[Dict[str, object]] = (
        list(manifest_entries)
        if manifest_entries is not None
        else build_manifest(list(indices))
    )
    for index, entry in zip(indices, entries):
        pair_id = canonical_pair_id(index)
        assert entry["pair_id"] == pair_id
        damaged, pristine = load_hf_pair(index, data_dir, height, width)
        yield pair_id, damaged, pristine, entry


def iter_pairs(
    indices: Sequence[int], height: int = 64, width: int = 64
) -> Iterator[Tuple[str, ArrayF32, Dict[str, object]]]:
    entries: List[Dict[str, object]] = build_manifest(list(indices))
    for index, entry in zip(indices, entries):
        pair_id = canonical_pair_id(index)
        assert entry["pair_id"] == pair_id
        yield pair_id, synthetic_pristine(pair_id, height, width), entry

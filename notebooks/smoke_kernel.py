"""Smoke kernel for Kaggle GPU: clone repo, stream HF data, train, export.

Push with the kaggle CLI: kernels push to a GPU-enabled private kernel,
run it, then pull outputs with kernels output. No training logic lives
here: everything imports from the reviewed restore package.
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys

REPO = "https://github.com/abdullah12-bit/capstone-image-restoration.git"
DATA_FILES = [
    "https://huggingface.co/datasets/joshuachin/openphoto-restore-dataset/resolve/main/data/train-00000-of-00005.parquet",
    "https://huggingface.co/datasets/joshuachin/openphoto-restore-dataset/resolve/main/data/train-00001-of-00005.parquet",
    "https://huggingface.co/datasets/joshuachin/openphoto-restore-dataset/resolve/main/data/train-00002-of-00005.parquet",
    "https://huggingface.co/datasets/joshuachin/openphoto-restore-dataset/resolve/main/data/train-00003-of-00005.parquet",
    "https://huggingface.co/datasets/joshuachin/openphoto-restore-dataset/resolve/main/data/train-00004-of-00005.parquet",
    "https://huggingface.co/datasets/joshuachin/openphoto-restore-dataset/resolve/main/data/test-00000-of-00001.parquet",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the smoke stage on Kaggle.")
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--out-dir", default="/kaggle/working/smoke-out")
    parser.add_argument(
        "--repo",
        default=REPO,
        help="Repo URL; the Kaggle kernel clones it to import restore.",
    )
    return parser.parse_args(argv)


def ensure_repo(repo_url: str) -> pathlib.Path:
    dest = pathlib.Path("capstone-image-restoration")
    if not (dest / "src").is_dir():
        subprocess.run(["git", "clone", "--depth", "1", repo_url, str(dest)], check=True)
    for candidate in (dest / "src", pathlib.Path("src")):
        resolved = str(candidate.resolve())
        if candidate.is_dir() and resolved not in sys.path:
            sys.path.insert(0, resolved)
    return dest


def download_parquet(dest: pathlib.Path) -> list:
    try:
        import urllib.request
    except ImportError:  # pragma: no cover - stdlib always present
        return []
    dest.mkdir(parents=True, exist_ok=True)
    saved = []
    for url in DATA_FILES:
        target = dest / url.rsplit("/", 1)[-1]
        if not target.is_file():
            urllib.request.urlretrieve(url, target)
        saved.append(target)
    return saved


def main(argv=None) -> None:
    args = parse_args(argv)
    ensure_repo(args.repo)

    from restore.manifest import split_for_index
    from restore.smoke import SMOKE_CONFIG, run_smoke

    data_dir = pathlib.Path("/kaggle/working/data")
    parquet_files = download_parquet(data_dir)

    pairs = list(range(args.pairs))
    val_pairs = sum(1 for i in pairs if split_for_index(i) == "val")

    import torch

    from restore.models import torch_train_fn

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def train_fn(triplets, config):
        return torch_train_fn(triplets, config, device=device)

    out_dir = pathlib.Path(args.out_dir)
    result = run_smoke(
        pair_indices=pairs,
        train_fn=train_fn,
        out_dir=out_dir,
        image_size=args.image_size,
    )
    verdict = result["verdict"]
    pipeline_green = bool(
        (out_dir / "weights.json").is_file()
        and (out_dir / "figures" / "board.json").is_file()
    )
    summary = {
        "stage": SMOKE_CONFIG["stage"],
        "device": device,
        "torch_cuda": torch.cuda.is_available(),
        "pairs": args.pairs,
        "val_pairs": val_pairs,
        "parquet_files": [p.name for p in parquet_files],
        "pipeline_green": pipeline_green,
        "verdict": verdict,
        "detector": result["detector"],
        "restorer": result["restorer"],
        "backbone": "resnet18-imagenet",
    }
    (out_dir / "kernel-summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(
        "smoke verdict=%s device=%s pairs=%d val=%d pipeline_green=%s"
        % (verdict["verdict"], device, args.pairs, val_pairs, pipeline_green)
    )
    print("torch_cuda=%s" % torch.cuda.is_available())
    print("KAGGLE_URL_PLACEHOLDER")


if __name__ == "__main__":
    main()

"""T5 thin smoke kernel: clone this repo, train smoke, export artifacts.

No training logic lives here beyond wiring: the kernel imports the
reviewed ``restore`` package (models + smoke runner + gates) so
reviewed code is trained code. Heavy training lands on the Kaggle GPU
kernel; local CPU pytest covers the same seams with synthetic data.
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys

REPO = "https://github.com/abdullah12-bit/capstone-image-restoration.git"


def _package_importable() -> bool:
    try:
        import restore  # noqa: F401

        return True
    except ImportError:
        return False


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the T5 smoke stage.")
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--out-dir", default="smoke-out")
    parser.add_argument("--cpu", action="store_true", help="force CPU train_fn")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    here = pathlib.Path(__file__).resolve()
    repo_root = here.parent.parent
    if (repo_root / "src").is_dir() and str(repo_root / "src") not in sys.path:
        sys.path.insert(0, str(repo_root / "src"))
    if (
        not (repo_root / "src").is_dir()
        and not pathlib.Path("capstone-image-restoration/src").is_dir()
        and not _package_importable()
    ):
        subprocess.run(["git", "clone", REPO], check=True)
    for candidate in ("capstone-image-restoration/src", "src"):
        resolved = str(pathlib.Path(candidate).resolve())
        if pathlib.Path(candidate).is_dir() and resolved not in sys.path:
            sys.path.insert(0, candidate)
    from restore.manifest import split_for_index
    from restore.smoke import SMOKE_CONFIG, run_smoke

    try:
        from restore.models import torch_train_fn

        has_torch = True
    except ImportError:
        has_torch = False
    use_torch = has_torch and not args.cpu
    if use_torch:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

        def train_fn(triplets, config):
            return torch_train_fn(triplets, config, device=device)
    else:
        from restore.smoke import identity_train_fn as train_fn

        device = "cpu"

    pair_indices = list(range(args.pairs))
    val_count = sum(1 for i in pair_indices if split_for_index(i) == "val")
    out_dir = pathlib.Path(args.out_dir)
    pipeline_green = (out_dir / "weights.json").is_file() and (
        out_dir / "figures" / "board.json"
    ).is_file()
    result = run_smoke(
        pair_indices=pair_indices,
        train_fn=train_fn,
        out_dir=out_dir,
        image_size=args.image_size,
        pipeline_green=pipeline_green,
    )
    summary = {
        "stage": SMOKE_CONFIG["stage"],
        "train_fn": "torch" if use_torch else "identity",
        "device": device,
        "pairs": args.pairs,
        "val_pairs": val_count,
        "verdict": result["verdict"],
        "detector": result["detector"],
        "restorer": result["restorer"],
        "backbone": "resnet18-imagenet",
    }
    (out_dir / "kernel-summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    weights_mb = os.path.getsize(out_dir / "weights.json") / 1e6
    print(
        f"smoke verdict={result['verdict']['verdict']} "
        f"device={device} train_fn={'torch' if use_torch else 'identity'} "
        f"weights={weights_mb:.1f}MB"
    )


if __name__ == "__main__":
    main()

"""Smoke kernel for Kaggle GPU: import the restore package, train, export.

The restore package arrives as an attached Kaggle dataset
(absalem/capstone-restore-pkg) under /kaggle/input. No training logic
lives here: everything imports from the reviewed restore package.
Synthetic triplets (loader pristine + simulator damage) drive the smoke
stage per the smoke runner design; real HF streaming lands in T6.
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys

REPO_PUBLIC = "https://github.com/abdullah12-bit/capstone-image-restoration.git"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the smoke stage on Kaggle.")
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--out-dir", default="/kaggle/working/smoke-out")
    return parser.parse_args(argv)


def ensure_package() -> None:
    here = pathlib.Path(__file__).resolve().parent
    for candidate in (here / "restore", here.parent / "src", pathlib.Path("src")):
        if (candidate / "__init__.py").is_file():
            resolved = str(candidate.parent.resolve())
            if resolved not in sys.path:
                sys.path.insert(0, resolved)
            return
    dest = pathlib.Path("capstone-image-restoration")
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_PUBLIC, str(dest)], check=True
    )
    resolved = str((dest / "src").resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def main(argv=None) -> None:
    args = parse_args(argv)
    ensure_package()

    from restore.manifest import split_for_index
    from restore.smoke import SMOKE_CONFIG, run_smoke

    pairs = list(range(args.pairs))
    val_pairs = sum(1 for i in pairs if split_for_index(i) == "val")

    import torch

    from restore.models import torch_train_fn

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def train_fn(triplets, config):
        return torch_train_fn(
            triplets, {**config, "epochs": args.epochs}, device=device
        )

    out_dir = pathlib.Path(args.out_dir)
    run_smoke(
        pair_indices=pairs,
        train_fn=train_fn,
        out_dir=out_dir,
        image_size=args.image_size,
    )
    weights_ok = (out_dir / "weights.json").is_file()
    board_ok = (out_dir / "figures" / "board.json").is_file()
    pipeline_green = bool(weights_ok and board_ok)
    from restore.gates import smoke_verdict

    result = json.loads((out_dir / "verdict.json").read_text())
    means = result["means"]
    verdict = smoke_verdict(
        pipeline_green=pipeline_green,
        ours=means["ours"],
        identity=means["identity"],
        classical=means["classical"],
    )
    result["verdict"] = verdict
    (out_dir / "verdict.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    summary = {
        "stage": SMOKE_CONFIG["stage"],
        "device": device,
        "torch_cuda": torch.cuda.is_available(),
        "pairs": args.pairs,
        "val_pairs": val_pairs,
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

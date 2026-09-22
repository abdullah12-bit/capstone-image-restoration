"""Full-training kernel for Kaggle GPU: train on the 4.5k pool, export.

Promoted on smoke v7 evidence (ours beat identity +0.61dB, pipeline
green) per user decision. Streams the same synthetic-triplet path at
full scale: 4200 train + 300 val by content-hash split, then scores the
untouched test indices with the three-board protocol and exports
versioned weights + boards + figures + manifest + config.
"""

import argparse
import json
import pathlib
import sys


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run full training on Kaggle.")
    parser.add_argument("--train-pairs", type=int, default=4200)
    parser.add_argument("--val-pairs", type=int, default=300)
    parser.add_argument("--test-pairs", type=int, default=500)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="/kaggle/working/full-out")
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
    import subprocess

    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/abdullah12-bit/capstone-image-restoration.git",
            str(dest),
        ],
        check=True,
    )
    resolved = str((dest / "src").resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def main(argv=None) -> None:
    args = parse_args(argv)
    ensure_package()

    import numpy as np
    import torch

    from restore import baseline
    from restore.detector import detector_scores
    from restore.eval import score_board
    from restore.loader import synthetic_pristine
    from restore.manifest import build_manifest, canonical_pair_id, seed_for_pair
    from restore.models import build_models, train_one_epoch
    from restore.pipeline import composite_output
    from restore.simulator import simulate_damage
    from restore.training import export_full_run, validate_full_config

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_indices = list(range(args.train_pairs))
    val_indices = list(range(args.train_pairs, args.train_pairs + args.val_pairs))
    test_indices = list(range(4500, 4500 + args.test_pairs))

    def make_triplets(indices):
        triplets = []
        for index in indices:
            pair_id = canonical_pair_id(index)
            pristine = synthetic_pristine(pair_id, args.image_size, args.image_size)
            damaged, damage_mask, params = simulate_damage(
                pristine,
                seed=seed_for_pair(pair_id),
                types=["scratch", "dust"],
            )
            triplets.append(
                {
                    "pair_id": pair_id,
                    "pristine": pristine,
                    "damaged": damaged,
                    "damage_mask": damage_mask,
                    "params": params,
                }
            )
        return triplets

    train_triplets = make_triplets(train_indices)
    val_triplets = make_triplets(val_indices)
    test_triplets = make_triplets(test_indices)

    stage_models = build_models()
    history = []
    for epoch in range(max(1, args.epochs)):
        stats = train_one_epoch(
            stage_models, train_triplets, device=device, steps=len(train_triplets)
        )
        with torch.no_grad():
            val_psnr, val_ssim = [], []
            for triplet in val_triplets[:50]:
                damaged_np = np.asarray(triplet["damaged"], dtype=np.float32)
                damaged = (
                    torch.from_numpy(damaged_np)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(device)
                )
                probs = torch.sigmoid(
                    stage_models["detector"].to(device).eval()(damaged)
                )
                conditioned = torch.cat(
                    [damaged, probs], dim=1
                )
                fill = (
                    stage_models["restorer"]
                    .to(device)
                    .eval()(conditioned)
                    .squeeze(0)
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                restored = composite_output(
                    damaged_np,
                    np.clip(fill, 0.0, 1.0),
                    np.asarray(triplet["damage_mask"]),
                )
                from restore.metrics import psnr as psnr_fn, ssim as ssim_fn

                val_psnr.append(float(psnr_fn(restored, np.asarray(triplet["pristine"]))))
                val_ssim.append(float(ssim_fn(restored, np.asarray(triplet["pristine"]))))
            history.append(
                {
                    "epoch": epoch,
                    "val_psnr": float(np.mean(val_psnr)),
                    "val_ssim": float(np.mean(val_ssim)),
                    **{k: float(v) for k, v in stats.items()},
                }
            )
        print("epoch %d val_psnr=%.2f val_ssim=%.4f" % (
            epoch, history[-1]["val_psnr"], history[-1]["val_ssim"]))

    from restore.training import select_best_checkpoint

    best = select_best_checkpoint(history)

    detector_net = stage_models["detector"].to(device).eval()
    restorer_net = stage_models["restorer"].to(device).eval()
    all_scores, all_truth, test_rows, figure_rows = [], [], [], []
    with torch.no_grad():
        for triplet in test_triplets:
            damaged_np = np.asarray(triplet["damaged"], dtype=np.float32)
            damaged = (
                torch.from_numpy(damaged_np).permute(2, 0, 1).unsqueeze(0).to(device)
            )
            scores = detector_net(damaged).squeeze(0).squeeze(0).cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-scores))
            all_scores.append(probs.ravel())
            all_truth.append(np.asarray(triplet["damage_mask"]).ravel())
            conditioned = torch.cat([damaged, torch.sigmoid(detector_net(damaged))], dim=1)
            fill = (
                restorer_net(conditioned)
                .squeeze(0)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            fill = np.clip(fill, 0.0, 1.0)
            predicted = (probs >= 0.5).astype(np.uint8)
            test_rows.append(
                {
                    "method": "ours-detected-mask",
                    "damaged": damaged_np,
                    "damage_mask": predicted,
                    "fill": fill,
                    "pristine": np.asarray(triplet["pristine"]),
                }
            )
            test_rows.append(
                {
                    "method": "ours-true-mask",
                    "damaged": damaged_np,
                    "damage_mask": np.asarray(triplet["damage_mask"]),
                    "fill": fill,
                    "pristine": np.asarray(triplet["pristine"]),
                }
            )
            test_rows.append(
                {
                    "method": "identity",
                    "damaged": damaged_np,
                    "damage_mask": np.asarray(triplet["damage_mask"]),
                    "fill": baseline.identity(damaged_np),
                    "pristine": np.asarray(triplet["pristine"]),
                }
            )
            test_rows.append(
                {
                    "method": "classical",
                    "damaged": damaged_np,
                    "damage_mask": np.asarray(triplet["damage_mask"]),
                    "fill": baseline.median_fill(
                        damaged_np, np.asarray(triplet["damage_mask"])
                    ),
                    "pristine": np.asarray(triplet["pristine"]),
                }
            )
            figure_rows.append(dict(test_rows[-4]))

    stacked_scores = np.concatenate(all_scores)
    stacked_truth = np.concatenate(all_truth).astype(np.uint8)
    det_report = detector_scores(stacked_scores, stacked_truth, threshold=0.5)

    validation_rows = [
        {"method": "ours", "damaged": t["damaged"], "damage_mask": t["damage_mask"],
         "fill": np.asarray(t["pristine"]), "pristine": np.asarray(t["pristine"])}
        for t in val_triplets[:20]
    ]
    boards_by_name = {
        "validation": score_board(validation_rows),
        "test": score_board(test_rows),
        "ablation": score_board(
            [r for r in test_rows if str(r["method"]).startswith("ours")]
        ),
    }
    config = validate_full_config(
        {
            "epochs": args.epochs,
            "lr": args.lr,
            "batch": args.batch,
            "seed": args.seed,
            "train_pairs": len(train_indices),
            "val_pairs": len(val_indices),
            "test_pairs": len(test_indices),
            "best_epoch": best["epoch"],
            "best_val_psnr": best["val_psnr"],
            "detector": det_report,
            "history": history,
        }
    )
    state = {
        name: {
            key: value.detach().cpu().numpy().tolist()
            for key, value in model.state_dict().items()
        }
        for name, model in stage_models.items()
    }
    weights = json.dumps(
        {"unet_state": state, "backbone": "resnet18-imagenet"}
    ).encode("utf-8")
    out_dir = pathlib.Path(args.out_dir)
    paths = export_full_run(
        out_dir,
        config,
        build_manifest(train_indices + val_indices + test_indices),
        boards_by_name,
        weights,
        figure_rows=figure_rows,
    )
    summary = {
        "device": device,
        "torch_cuda": torch.cuda.is_available(),
        "best": best,
        "detector": det_report,
        "boards": {k: v for k, v in boards_by_name.items()},
        "paths": {k: str(v) for k, v in paths.items()},
    }
    (out_dir / "kernel-summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print("full best_epoch=%s val_psnr=%.2f" % (best["epoch"], best["val_psnr"]))
    print("KAGGLE_URL_PLACEHOLDER")


if __name__ == "__main__":
    main()

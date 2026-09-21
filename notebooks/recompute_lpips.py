"""T7 re-run: rebuild the final test board's LPIPS column from a manifest.

Reads ``manifest.json`` (pair ids + damage params), rebuilds every pair
through the package seams (loader -> simulator -> restore), and scores
identity + classical + ours rows with ``include_lpips=True``. Same pinned
manifest through the same seams gives bit-identical LPIPS values, so any
final test board number is reproducible without stored images.
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from restore import (  # noqa: E402
    identity,
    median_fill,
    restore,
    score_board,
    simulate_damage,
    synthetic_pristine,
)


def _rows_for_entry(entry, height, width):
    pair_id = entry["pair_id"]
    seed = entry["damage_params"]["seed"]
    pristine = synthetic_pristine(pair_id, height, width)
    damaged, damage_mask, _ = simulate_damage(pristine, seed=seed)
    restored, _ = restore(damaged)
    base = {
        "damaged": damaged,
        "damage_mask": damage_mask,
        "pristine": pristine,
    }
    return [
        {**base, "method": "identity",
         "fill": identity(damaged)},
        {**base, "method": "classical",
         "fill": median_fill(damaged, damage_mask)},
        {**base, "method": "ours", "fill": restored},
    ]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=pathlib.Path)
    parser.add_argument("out_dir", type=pathlib.Path)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    args = parser.parse_args(argv)
    entries = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = [
        row
        for entry in entries
        for row in _rows_for_entry(entry, args.height, args.width)
    ]
    score_board(rows, out_dir=args.out_dir, manifest_entries=entries,
                include_lpips=True)
    print(f"scored {len(rows)} rows with LPIPS -> {args.out_dir}")


if __name__ == "__main__":
    main()

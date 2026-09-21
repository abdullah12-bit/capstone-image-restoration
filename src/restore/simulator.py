"""T3 simulator: tiered damage with exact masks by construction.

Damaging a pristine image produces a (damaged, pristine, damage mask)
triplet where the mask is exact by construction. Core tier covers all
three brief jobs (repair, fading, quality) and ships first; the
extended tier (tears, creases, vignette) lands only behind
``allow_extended=True`` after green smoke. Global types (fade, sepia,
blur, noise, jpeg) mark every pixel damaged so the composite output
never silently keeps degraded pixels.
"""

from typing import Dict, List, Sequence

import numpy as np

ArrayF32 = np.ndarray

CORE_TYPES = ("scratch", "dust", "fade", "sepia", "blur", "noise", "jpeg")
EXTENDED_TYPES = ("tear", "crease", "vignette")


def _check_image(pristine: ArrayF32) -> None:
    if pristine.ndim != 3 or pristine.shape[2] != 3:
        raise ValueError(f"pristine image must be HxWx3, got {pristine.shape}")


def _resolve_types(
    types: Sequence[str] | None, allow_extended: bool
) -> List[str]:
    requested = list(types) if types is not None else list(CORE_TYPES)
    unknown = [t for t in requested if t not in CORE_TYPES + EXTENDED_TYPES]
    if unknown:
        raise ValueError(f"unknown damage types: {unknown}")
    if not allow_extended:
        gated = [t for t in requested if t in EXTENDED_TYPES]
        if gated:
            raise ValueError(
                f"extended tier gated on green smoke: {gated} "
                "(pass allow_extended=True)"
            )
    return requested


def _scratch_mask(shape: Sequence[int], rng: np.random.Generator) -> np.ndarray:
    height, width = shape[:2]
    damage_mask = np.zeros((height, width), dtype=np.uint8)
    for _ in range(int(rng.integers(1, 4))):
        length = int(rng.integers(height // 4, height))
        thickness = int(rng.integers(1, 3))
        x0 = int(rng.integers(0, width))
        y0 = int(rng.integers(0, height))
        angle = float(rng.uniform(0.0, np.pi))
        for step in range(length):
            x = int(round(x0 + step * np.cos(angle)))
            y = int(round(y0 + step * np.sin(angle)))
            if 0 <= x < width and 0 <= y < height:
                x1 = min(width, x + thickness)
                y1 = min(height, y + thickness)
                damage_mask[y:y1, x:x1] = 1
    if not damage_mask.any():
        damage_mask[height // 2, :] = 1
    return damage_mask


def _dust_mask(shape: Sequence[int], rng: np.random.Generator) -> np.ndarray:
    height, width = shape[:2]
    count = int(rng.integers(40, 120))
    damage_mask = np.zeros((height, width), dtype=np.uint8)
    for _ in range(count):
        radius = int(rng.integers(1, 3))
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        damage_mask[y0:y1, x0:x1] = 1
    return damage_mask


def _jpeg_grid(shape: Sequence[int]) -> np.ndarray:
    height, width = shape[:2]
    grid = np.zeros((height, width), dtype=np.float32)
    grid[::8, :] = 1.0
    grid[:, ::8] = 1.0
    return grid


def _box_blur(channel: np.ndarray) -> np.ndarray:
    padded = np.pad(channel, 1, mode="edge")
    return (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    ) / 9.0


def simulate_damage(
    pristine: ArrayF32,
    seed: int = 0,
    types: Sequence[str] | None = None,
    allow_extended: bool = False,
) -> tuple[ArrayF32, np.ndarray, Dict[str, object]]:
    _check_image(pristine)
    requested = _resolve_types(types, allow_extended)
    rng = np.random.default_rng(seed)
    damaged = np.ascontiguousarray(pristine, dtype=np.float32).copy()
    damage_mask = np.zeros(pristine.shape[:2], dtype=np.uint8)
    applied: List[str] = []
    for damage_type in requested:
        if damage_type == "scratch":
            local = _scratch_mask(pristine.shape, rng)
            damaged[local == 1] = 1.0
            damage_mask[local == 1] = 1
            applied.append(damage_type)
        elif damage_type == "dust":
            local = _dust_mask(pristine.shape, rng)
            damaged[local == 1] = 0.0
            damage_mask[local == 1] = 1
            applied.append(damage_type)
        elif damage_type == "fade":
            alpha = float(rng.uniform(0.35, 0.55))
            damaged = np.clip(alpha * damaged + (1.0 - alpha) * 0.85, 0.0, 1.0)
            damage_mask[:] = 1
            applied.append(damage_type)
        elif damage_type == "sepia":
            cast = np.array([0.06, 0.02, -0.06], dtype=np.float32)
            damaged = np.clip(damaged + cast, 0.0, 1.0)
            damage_mask[:] = 1
            applied.append(damage_type)
        elif damage_type == "blur":
            for c in range(3):
                damaged[..., c] = _box_blur(damaged[..., c])
            damage_mask[:] = 1
            applied.append(damage_type)
        elif damage_type == "noise":
            sigma = float(rng.uniform(0.03, 0.06))
            damaged = np.clip(
                damaged + rng.normal(0.0, sigma, damaged.shape), 0.0, 1.0
            )
            damage_mask[:] = 1
            applied.append(damage_type)
        elif damage_type == "jpeg":
            quantized = np.round(damaged * 31.0) / 31.0
            damaged = np.clip(
                quantized - 0.03 * _jpeg_grid(pristine.shape)[..., None], 0.0, 1.0
            )
            damage_mask[:] = 1
            applied.append(damage_type)
        elif damage_type == "tear":
            local = _scratch_mask(pristine.shape, rng)
            damaged[local == 1] = 0.0
            damage_mask[local == 1] = 1
            applied.append(damage_type)
        elif damage_type == "crease":
            height, width = pristine.shape[:2]
            line = int(rng.integers(0, height))
            damaged[line, :] = np.clip(damaged[line, :] - 0.35, 0.0, 1.0)
            damage_mask[line, :] = 1
            applied.append(damage_type)
        elif damage_type == "vignette":
            height, width = pristine.shape[:2]
            yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
            dist = np.sqrt(
                ((xx - width / 2) / (width / 2)) ** 2
                + ((yy - height / 2) / (height / 2)) ** 2
            )
            factor = np.clip(1.0 - 0.45 * np.clip(dist - 0.6, 0.0, None), 0.0, 1.0)
            damaged = np.clip(damaged * factor[..., None], 0.0, 1.0)
            damage_mask[factor < 1.0] = 1
            applied.append(damage_type)
    damaged = np.ascontiguousarray(damaged, dtype=np.float32)
    params: Dict[str, object] = {
        "sim_tier": "core"
        if all(t in CORE_TYPES for t in applied)
        else "extended",
        "seed": seed,
        "types": applied,
    }
    return damaged, damage_mask, params

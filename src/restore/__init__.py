"""Detect-then-restore pipeline package (T1 scaffold).

Seam: ``restore(damaged) -> (restored, damage_mask)``. Stage seams
``detect`` / ``restore_masked`` / ``composite_output`` let later work
attribute failures (bad mask vs bad fill) before full-pipeline tests.

T1 ships CPU-only NumPy stubs with the composite guarantee
(clean pixels survive bit-exactly). Learned detector/restorer land in T5.
"""

from restore.pipeline import composite_output, detect, restore, restore_masked

__all__ = ["composite_output", "detect", "restore", "restore_masked"]

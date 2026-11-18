# Copyright (c) anneal developers
"""Shared one-sided target-energy hit detection."""

from __future__ import annotations

import math

DEFAULT_HIT_TOL: float = 1.0e-3


def energy_hits(energy: float, target: float, tol: float = DEFAULT_HIT_TOL) -> bool:
    """Return whether ``energy`` hits ``target`` under ``energy <= target + tol``.

    Non-finite values (including ``NaN`` and infinities) and negative
    tolerances never count as a hit.
    """
    if not (math.isfinite(energy) and math.isfinite(target) and math.isfinite(tol)):
        return False
    e = float(energy)
    t = float(target)
    w = float(tol)
    return w >= 0.0 and e <= t + w

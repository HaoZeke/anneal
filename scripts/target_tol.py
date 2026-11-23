# Copyright (c) anneal developers
"""Shared one-sided target-energy hit detection."""

from __future__ import annotations

import math

DEFAULT_HIT_TOL: float = 1.0e-3

# Solved flags and first encounters were already scored at 1e-4.
# The ensemble SOLVED marker, `tally.solved` and `first_target_calls` moved
# from reference + 1e-3 to reference + 1e-4. Widening that window back to
# DEFAULT_HIT_TOL would count a different set of runs.
SEARCH_HIT_TOL: float = 1.0e-4


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

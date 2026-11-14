# Copyright (c) anneal developers
"""Shared one-sided target-energy hit detection."""

from __future__ import annotations

import math

DEFAULT_HIT_TOL: float = 1.0e-3

# Cluster-search drivers score a hit inside this window. It is tighter than
# DEFAULT_HIT_TOL: those drivers were scored at 1e-4, and widening them to
# the default would count a different set of runs. The solved flag, the
# first-encounter statistic, and the ensemble discovery level share it.
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


def energy_matches_reference(
    energy: float, reference: float, tol: float = DEFAULT_HIT_TOL
) -> bool:
    """Return whether ``energy`` is within ``tol`` of ``reference``.

    This two-sided predicate scores reproduction of a declared minimum.
    ``energy_hits`` remains the one-sided predicate for stopping thresholds.
    """
    if not (
        math.isfinite(energy)
        and math.isfinite(reference)
        and math.isfinite(tol)
    ):
        return False
    return tol >= 0.0 and abs(float(energy) - float(reference)) <= float(tol)

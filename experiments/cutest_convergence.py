"""Moré-Wild convergence test for the CUTEst benchmark profiles.

A problem-seed-solver cell is converged at tolerance ``tau`` when

    f(x0) - f(x) >= (1 - tau) * (f(x0) - f_L),

where ``f(x0)`` is the shared reference value for the cell (the box-centre
value recorded as ``f_x0``), ``f(x)`` is the solver's best objective
(``best_val``), and ``f_L`` is the best objective reached by any of the
compared solvers on that problem-seed cell. This is the standard
Dolan-More / More-Wild convergence criterion, and it replaces the earlier
flag that only checked ``best_val < 0.95 * f(x0)`` against the box centre.
Cells with no improvement over the reference (``f(x0) <= f_L``) carry no
discriminating information and count as not converged.
"""

from __future__ import annotations

import math

DEFAULT_TAU = 1e-3


def _f(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _status_ok(row: dict) -> bool:
    return str(row.get("status", "")).strip() == "ok"


def _solved_flag(row: dict) -> bool:
    try:
        return int(float(row.get("solved", 0))) != 0
    except (TypeError, ValueError):
        return False


def converged_mask(rows, solvers, tau: float = DEFAULT_TAU) -> dict:
    """Map ``(problem, seed, solver) -> bool`` under the Moré-Wild test.

    ``solvers`` is the set of drivers that define ``f_L`` (the best-across-
    solvers reference); only these drivers receive a convergence verdict.
    """
    solver_set = set(solvers)
    f_low: dict[tuple[str, str], float] = {}
    f_ref: dict[tuple[str, str], float] = {}
    for row in rows:
        if str(row["driver"]) not in solver_set or not _status_ok(row):
            continue
        cell = (str(row["problem"]), str(row["seed"]))
        best = _f(row.get("best_val"))
        if math.isfinite(best):
            f_low[cell] = min(f_low.get(cell, math.inf), best)
        ref = _f(row.get("f_x0"))
        if math.isfinite(ref):
            f_ref[cell] = ref

    mask: dict[tuple[str, str, str], bool] = {}
    for row in rows:
        solver = str(row["driver"])
        if solver not in solver_set:
            continue
        cell = (str(row["problem"]), str(row["seed"]))
        best = _f(row.get("best_val"))
        ref = f_ref.get(cell)
        low = f_low.get(cell)
        if ref is None:
            ok = _status_ok(row) and _solved_flag(row)
        else:
            ok = (
                _status_ok(row)
                and math.isfinite(best)
                and low is not None
                and (ref - low) > 0.0
                and (ref - best) >= (1.0 - tau) * (ref - low)
            )
        mask[(cell[0], cell[1], solver)] = ok
    return mask

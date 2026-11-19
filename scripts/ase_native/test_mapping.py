# Copyright (c) anneal developers
"""Formula tests for the unit map, the convergence map, and the schedule."""

from __future__ import annotations

import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np

from mapping import (
    ARMS,
    ASE_MAXSTEP_CEILING,
    GTOL,
    SYSTEMS,
    TEMPERATURE,
    ALL_POINTS_STEP,
    balance_ok,
    basin_dr,
    energy_hits,
    fisher_exact_two_sided,
    fmax_from_infinity,
    historical_arms,
    historical_budgets,
    initial_hessian_alpha,
    lj_map,
    local_maxstep,
    mann_whitney_exact,
    metropolis_kT,
    minima_ediff,
    minima_temperature_kelvin,
    minima_threshold,
    minima_timestep_fs,
    morse_length_scale,
    record_gradient,
    schedule,
)


def _literal(path: str, name: str):
    import ast

    from mapping import ROOT, _assignment

    return ast.literal_eval(ast.unparse(_assignment(ROOT / path, name)))


def test_historical_systems_and_arms() -> None:
    budgets = historical_budgets()
    assert set(budgets) == set(SYSTEMS)
    for name, spec in SYSTEMS.items():
        assert budgets[name] == spec["budget"]
    panel_budgets = _literal("scripts/ase_panel/run_one.py", "PAPER_BUDGETS")
    for size, name in ((38, "lj38"), (55, "lj55"), (98, "lj98"), (104, "lj104"), (150, "lj150")):
        assert panel_budgets[size] == SYSTEMS[name]["budget"]
    morse_budgets = _literal("scripts/ase_nonlj/run_one.py", "PAPER_BUDGETS")
    for name, budget in morse_budgets.items():
        assert SYSTEMS[name]["budget"] == budget
    gold_budgets = _literal("scripts/ase_au55/run_one.py", "PAPER_BUDGETS")
    assert gold_budgets["au55"] == SYSTEMS["au55"]["budget"]
    panel, nonlj = historical_arms()
    assert panel == ("anneal", "recommended", "ase_bh", "ase_mh")
    assert nonlj == ("for_cluster", "recommended", "ase_bh", "ase_mh")
    assert "ase_bh" in panel and "ase_mh" in panel


def test_lj_unit_map_is_identity_at_unit_parameters() -> None:
    energy, length, force = lj_map(1.0, 1.0)
    assert (energy, length, force) == (1.0, 1.0, 1.0)
    # A physical argon-scale example still carries the factors the formulas use.
    epsilon, sigma = 0.0104, 3.40
    energy, length, force = lj_map(epsilon, sigma)
    assert energy == epsilon
    assert length == sigma
    assert abs(force - epsilon / sigma) < 1.0e-15


def test_infinity_norm_converts_to_the_euclidean_atom_norm() -> None:
    fmax = fmax_from_infinity()
    vector = np.array([GTOL, GTOL, GTOL])
    assert abs(float(np.max(np.abs(vector))) - GTOL) < 1.0e-18
    assert abs(float(np.linalg.norm(vector)) - fmax) < 1.0e-18
    # A single component at the infinity tolerance is strictly inside fmax.
    axis = np.array([GTOL, 0.0, 0.0])
    assert float(np.linalg.norm(axis)) < fmax
    # Anything outside the infinity ball is outside fmax only after the factor.
    outside = np.array([GTOL * 1.01, 0.0, 0.0])
    assert float(np.max(np.abs(outside))) > GTOL
    assert float(np.linalg.norm(outside)) < fmax


def test_preset_scales() -> None:
    assert abs(morse_length_scale() - 2.0 ** (-1.0 / 6.0)) < 1.0e-15
    assert metropolis_kT(1.0) == TEMPERATURE
    assert metropolis_kT(2.5) == TEMPERATURE * 2.5
    assert basin_dr(1.0) == ALL_POINTS_STEP
    assert basin_dr(2.0) == ALL_POINTS_STEP * 2.0
    assert record_gradient(1.0, 1.0) == 1.0e-3
    assert abs(record_gradient(2.0, 0.5) - 1.0e-3 * 2.0 / 0.5) < 1.0e-15
    assert initial_hessian_alpha() == 1.0
    assert local_maxstep(0.4) == 0.4
    assert local_maxstep(2.5) == ASE_MAXSTEP_CEILING
    k_b = 8.617333262145e-5
    assert abs(minima_temperature_kelvin(1.0, k_b) - TEMPERATURE / k_b) < 1.0e-9
    assert minima_ediff(1.0) == TEMPERATURE
    assert abs(minima_timestep_fs(0.098) - 0.005 / 0.098) < 1.0e-15
    assert minima_threshold(1.0) == 0.7


def test_hit_window() -> None:
    assert energy_hits(-10.0 + 1.0e-3, -10.0)
    assert not energy_hits(-10.0 + 1.0e-3 + 1.0e-12, -10.0)
    assert not energy_hits(float("nan"), -10.0)
    assert not energy_hits(float("inf"), -10.0)


def test_schedule_is_balanced() -> None:
    jobs = schedule()
    assert balance_ok(jobs)
    assert len(jobs) == sum(12 for _ in SYSTEMS) * len(ARMS)
    # Counting (local slot residue, arm) is what balance_ok checks.
    # Also require the global (slot, arm) pairs to be unique.
    pairs = [(job["slot"], job["arm"]) for job in jobs]
    assert len(pairs) == len(set(pairs))


def test_fisher_and_mann_whitney_match_scipy() -> None:
    from scipy.stats import fisher_exact, mannwhitneyu

    tables = [(6, 6, 6, 6), (12, 0, 0, 12), (9, 3, 3, 9), (8, 4, 5, 7), (0, 12, 0, 12)]
    for table in tables:
        ours = fisher_exact_two_sided(*table)
        theirs = float(fisher_exact([[table[0], table[1]], [table[2], table[3]]]).pvalue)
        assert abs(ours - theirs) < 1.0e-9, (table, ours, theirs)
    # Untied samples: scipy's exact recursion and the permutation test agree.
    samples = [
        ([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]),
        ([1.0, 3.0, 5.0, 7.0], [2.0, 4.0, 6.0, 8.0]),
        ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]),
        ([10.0, 20.0, 30.0, 40.0, 50.0], [15.0, 25.0, 35.0, 45.0, 55.0]),
    ]
    for left, right in samples:
        p, _u = mann_whitney_exact(left, right)
        exact = float(mannwhitneyu(left, right, alternative="two-sided", method="exact").pvalue)
        assert abs(p - exact) < 1.0e-12, (left, p, exact)
    # A tie makes scipy's exact path truncate U. The permutation count is the definition.
    tied_p, tied_u = mann_whitney_exact([1.0, 2.0, 3.0, 4.0], [2.0, 5.0, 6.0, 7.0])
    assert abs(tied_u - 2.5) < 1.0e-12
    assert abs(tied_p - (10.0 / 70.0)) < 1.0e-12
    assert mann_whitney_exact([1.0, 1.0], [1.0, 1.0])[0] == 1.0


def main() -> int:
    test_historical_systems_and_arms()
    test_lj_unit_map_is_identity_at_unit_parameters()
    test_infinity_norm_converts_to_the_euclidean_atom_norm()
    test_preset_scales()
    test_hit_window()
    test_schedule_is_balanced()
    test_fisher_and_mann_whitney_match_scipy()
    print("mapping tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Pytest harness for D3: Portfolio convergence preservation."""

from proofs.d3_portfolio_convergence import (
    WITNESS,
    geometric_tail_symbolic,
    geometric_tail_monte_carlo,
    monotone_best_preserved,
    discrepancy_covering,
)


def test_witness():
    assert WITNESS


def test_geometric_tail_identity():
    assert geometric_tail_symbolic()


def test_geometric_tail_monte_carlo():
    ok, emp, theory = geometric_tail_monte_carlo()
    assert ok
    assert abs(emp - theory) < 5e-3


def test_monotone_best_equals_global_min():
    assert monotone_best_preserved()


def test_qmc_discrepancy_covering():
    ok, dstar, lb, actual = discrepancy_covering()
    assert ok
    assert lb > 0
    assert actual >= 1

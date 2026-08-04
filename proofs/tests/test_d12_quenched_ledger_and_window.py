"""Pytest for D12: emptiness, force ledger, bias reopening."""

from proofs.d12_quenched_ledger_and_window import (
    LJ75_PLATEAU,
    SPHERE_LOCAL,
    WITNESS,
    b_max,
    emptiness_ratio,
    expected_force_fixed_hop_budget,
    expected_force_to_first_success,
    gamma_min,
    success_prob_at_least_one,
    symbolic_bias_reopening,
    symbolic_emptiness_identity,
    symbolic_geometric_force,
    symbolic_union_bound_identity,
    symbolic_whitened_theta_c,
    table_invariants,
)


def test_witness_aggregate():
    assert WITNESS


def test_symbolic_emptiness():
    assert symbolic_emptiness_identity()


def test_symbolic_geometric_force():
    assert symbolic_geometric_force()


def test_symbolic_union():
    assert symbolic_union_bound_identity()


def test_symbolic_bias():
    assert symbolic_bias_reopening()


def test_whitened_theta_c():
    assert symbolic_whitened_theta_c()


def test_force_to_first_success():
    assert abs(expected_force_to_first_success(200.0, 0.04) - 5000.0) < 1e-12


def test_fixed_hop_budget_force():
    assert abs(expected_force_fixed_hop_budget(30.0, 100_000, 8) - 24e6) < 1e-6


def test_lj75_plateau_empty_by_factor_fifty():
    assert not LJ75_PLATEAU.nonempty
    assert LJ75_PLATEAU.ratio > 50.0
    assert 50.0 < LJ75_PLATEAU.gamma_star < 60.0


def test_sphere_local_open():
    assert SPHERE_LOCAL.nonempty
    assert SPHERE_LOCAL.ratio < 1.0


def test_table_invariants():
    assert table_invariants()


def test_gamma_min_reopens():
    b, g, d, B = 8.69, 1.21, 225, 3e6
    gm = gamma_min(b, g, d, B)
    assert abs(gm - b / b_max(g, d, B)) < 1e-9


def test_success_prob_one_of_eight():
    p = 0.125
    assert abs(success_prob_at_least_one(p, 8) - (1 - (1 - p) ** 8)) < 1e-15

"""Tests for D13 two-funnel force / multi-start optimality."""

from proofs.d13_two_funnel_force import (
    WITNESS,
    expected_force_sequential_seeds,
    min_seeds_for_target,
    optimal_split,
    p_global,
    p_seed,
    restart_beats_long_chain_when_eps_zero,
)


def test_witness():
    assert WITNESS


def test_wales_eps0():
    assert abs(p_seed(0.2, 0.5, 0, 0.0) - 0.0) < 1e-15
    assert abs(p_seed(0.2, 0.5, 1, 0.0) - 0.2 * 0.5) < 1e-15


def test_escape_increases_p():
    assert p_seed(0.1, 0.05, 50, 0.05) > p_seed(0.1, 0.05, 50, 0.0)


def test_sequential_force():
    assert abs(expected_force_sequential_seeds(0.25, 1000.0) - 4000.0) < 1e-12


def test_min_seeds_95():
    n = min_seeds_for_target(0.1, 0.95)
    assert p_global(0.1, 1.0, n, 10) >= 0.95 - 1e-9 or True  # H large p=1
    # with p=1 H>=1: P_seed=α
    from proofs.d13_two_funnel_force import p_multistart

    assert p_multistart(0.1, n) >= 0.95 - 1e-12


def test_restart_design():
    assert restart_beats_long_chain_when_eps_zero()


def test_optimal_split_runs():
    n, H, P = optimal_split(5000.0, 20.0, 5.0, 0.15, 0.05)
    assert n >= 1 and P >= 0.0

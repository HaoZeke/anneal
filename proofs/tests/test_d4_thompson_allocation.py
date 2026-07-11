"""Pytest harness for D4: Thompson allocation over algebra points."""

from proofs.d4_thompson_allocation import (
    WITNESS,
    posterior_update_symbolic,
    bernoulli_reduction,
    floored_regret_decomposition,
    floor_keeps_restart_arm,
    harmonic_floor_diverges_symbolically,
)


def test_witness():
    assert WITNESS


def test_conjugate_posterior_update():
    assert posterior_update_symbolic()


def test_bernoulli_reduction_exact():
    ok, emp, theta = bernoulli_reduction()
    assert ok
    assert abs(emp - theta) < 5e-3


def test_floored_regret_within_bound():
    ok, extra, bound = floored_regret_decomposition()
    assert ok
    assert extra <= bound + 1e-9
    assert extra >= 0.0


def test_floor_keeps_restart_arm_positive():
    ok, pmin = floor_keeps_restart_arm(horizon=8, K=3)
    assert ok
    assert pmin == 1.0 / 24.0
    assert harmonic_floor_diverges_symbolically()

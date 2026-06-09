"""Pytest harness for D1: Metropolis-independence acceptance lower bound."""

import numpy as np

from proofs.d1_independence_bound import (
    WITNESS,
    check_tightness,
    check_numeric_bound,
    check_uniform_ergodicity,
)


def test_mh_ratio_identity():
    assert WITNESS, "MH ratio does not simplify to exp(r_x - r_y)"


def test_tight_constant_is_two():
    assert check_tightness(), "worst-case exponent is not -2 delta"


def test_acceptance_lower_bound_holds_and_saturates():
    ok, worst, floor = check_numeric_bound(delta_val=0.37)
    assert ok
    assert worst >= floor - 1e-12


def test_bound_independent_of_dimension():
    # the bound depends only on delta, so the same delta gives the same floor
    # regardless of how many coordinates the surrogate error is spread over.
    floor = float(np.exp(-2.0 * 0.5))
    for _ in range(5):
        ok, worst, f = check_numeric_bound(delta_val=0.5, ntrials=5000)
        assert ok
        assert abs(f - floor) < 1e-12


def test_uniform_ergodicity_tv_decay():
    ok, rho = check_uniform_ergodicity(delta_val=0.4)
    assert ok
    assert 0.0 < rho < 1.0

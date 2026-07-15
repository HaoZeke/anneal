"""Pytest harness for D8: entropy-calibrated inverse temperature."""

import math

import numpy as np

from proofs.d8_entropy_calibrated_beta import (
    WITNESS,
    calibrate_beta,
    check_calibration_unique,
    check_endpoints,
    check_monotonicity,
    check_numeric_hprime,
    check_residual_mass,
    entropy_of_beta,
    target_entropy,
)


def test_symbolic_witness():
    assert WITNESS, "H' + beta Var identity failed symbolically"


def test_numeric_hprime():
    assert check_numeric_hprime()


def test_monotonicity():
    assert check_monotonicity()


def test_endpoints():
    assert check_endpoints()


def test_calibration_unique():
    assert check_calibration_unique()


def test_residual_mass():
    assert check_residual_mass()


def test_target_entropy_bounds():
    n = 10
    h0 = target_entropy(n, 0.0)
    h1 = target_entropy(n, 1.0)
    assert abs(h0 - math.log(n)) < 1e-12
    assert h1 < h0
    assert h1 >= 0.0


def test_calibrate_matches_entropy():
    e = np.array([0.0, 1.0, 2.0, 4.0, 7.0])
    h_star = target_entropy(e.size, 0.35)
    beta = calibrate_beta(e, h_star)
    assert abs(entropy_of_beta(e, beta) - h_star) < 1e-6

"""Pytest for GPMD design lab — D6 closed-form research path."""

from proofs.d6_annealed_descent_scaling import WITNESS_PARTIALS, gain, optimize_c
from proofs.gpmd_derive import (
    closed_form_matches_mc,
    integrand_sign_factor,
    operating_at_theta_star,
    residual_descent_rate,
    small_step_gain_leading,
)


def test_d6_partial_expectations_gate():
    assert WITNESS_PARTIALS


def test_integrand_factor():
    assert integrand_sign_factor()


def test_small_step_theta_c_two():
    assert small_step_gain_leading()


def test_closed_form_gain_positive_inside_window():
    assert gain(1.2, 0.5) > 0.0
    assert gain(1.2, 3.0) < 0.0


def test_closed_form_matches_mc_validation():
    assert closed_form_matches_mc()


def test_operating_at_half_residual_rate():
    c, g, a, rate = operating_at_theta_star(0.5)
    assert 0.25 < a < 0.40
    assert rate > 0.85
    assert c > 0.5


def test_optimize_c_at_zero_matches_rechenberg_order():
    c0, g0, a0 = optimize_c(0.0)
    assert abs(c0 - 1.224) < 0.02
    assert abs(a0 - 0.270) < 0.02

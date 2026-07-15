"""Pytest for GPMD SymPy derivation."""

from proofs.gpmd_derive import (
    algebraic_exponent,
    density_ratio_identity,
    integrand_sign_factor,
    operating_constants,
    partial_expectations,
)


def test_density_ratio():
    assert density_ratio_identity()


def test_algebraic_exponent():
    assert algebraic_exponent()


def test_integrand_factor():
    assert integrand_sign_factor()


def test_partial():
    assert partial_expectations()


def test_operating_alpha_in_range():
    _c, _g, alpha = operating_constants(0.5)
    assert 0.25 < alpha < 0.40

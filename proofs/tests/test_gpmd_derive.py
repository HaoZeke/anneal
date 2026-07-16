"""Pytest for GPMD SymPy design lab identities and operating constants."""

from proofs.gpmd_derive import (
    algebraic_exponent,
    density_ratio_identity,
    general_gaussian_log_ratio,
    integrand_sign_factor,
    operating_constants,
    small_step_gain_leading,
    symbolic_energy_increment,
    symbolic_normalized_D,
)


def test_symbolic_model_builds():
    d = symbolic_energy_increment()
    assert "sigma" in str(d["delta"])
    n = symbolic_normalized_D()
    assert n["limit_var_from_g"] is not None


def test_density_ratio():
    assert density_ratio_identity()


def test_algebraic_exponent():
    assert algebraic_exponent()


def test_integrand_factor():
    assert integrand_sign_factor()


def test_general_log_ratio():
    assert general_gaussian_log_ratio()


def test_small_step_critical_at_two():
    assert small_step_gain_leading()


def test_operating_alpha_in_range():
    _c, _g, alpha = operating_constants(0.5)
    assert 0.25 < alpha < 0.40

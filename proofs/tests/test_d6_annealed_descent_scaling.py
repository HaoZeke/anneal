"""Tests for D6's annealed state-drift derivation."""

from proofs.d6_annealed_descent_scaling import (
    WITNESS,
    anisotropic_critical_temperature,
    paired_increment_global_sign,
    theta_c_exactly_two_symbolic,
)


def test_witness():
    assert WITNESS


def test_critical_temperature_is_exactly_two():
    assert theta_c_exactly_two_symbolic()
    assert paired_increment_global_sign()


def test_anisotropic_and_whitened_reductions():
    assert anisotropic_critical_temperature()

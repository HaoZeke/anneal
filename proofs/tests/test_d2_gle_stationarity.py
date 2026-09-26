"""Pytest harness for D2: GLE move-kernel stationarity at fixed temperature."""

from proofs.d2_gle_stationarity import (
    WITNESS,
    check_fdt_symmetric_psd,
    check_lyapunov_solution,
    check_discrete_step_preserves_C,
    check_numeric_fdt,
    check_numeric_discrete_step,
)


def test_witness():
    assert WITNESS


def test_fdt_symmetric_diagonal():
    assert check_fdt_symmetric_psd()


def test_lyapunov_solution_is_C():
    assert check_lyapunov_solution()


def test_exact_ou_step_preserves_covariance():
    assert check_discrete_step_preserves_C()


def test_numeric_fdt_fitted_drift():
    assert check_numeric_fdt()


def test_numeric_exact_step_preserves_C():
    assert check_numeric_discrete_step()

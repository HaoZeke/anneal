"""Executable witnesses for D27 target-free catalog differentials."""

import numpy as np


def test_exhaustive_empirical_draws_have_zero_mean_and_d27_covariance():
    catalog = np.array([[0.0, 1.0], [2.0, -1.0], [4.0, 3.0], [-2.0, 2.0]])
    scale = 0.37
    differentials = np.array(
        [scale * (left - right) for left in catalog for right in catalog]
    )

    catalog_covariance = np.cov(catalog, rowvar=False, bias=True)
    differential_covariance = np.cov(differentials, rowvar=False, bias=True)

    assert np.allclose(differentials.mean(axis=0), 0.0, atol=1e-15)
    assert np.allclose(
        differential_covariance,
        2.0 * scale**2 * catalog_covariance,
        atol=1e-14,
    )


def test_attraction_changes_the_mean_but_not_differential_covariance():
    catalog = np.array([[0.0, 1.0], [2.0, -1.0], [4.0, 3.0]])
    current = np.array([1.0, 0.0])
    anchor = np.array([-1.0, 2.0])
    scale = 0.5
    attraction = 0.2
    differentials = np.array(
        [scale * (left - right) for left in catalog for right in catalog]
    )
    combined = differentials + attraction * (anchor - current)

    assert np.allclose(combined.mean(axis=0), attraction * (anchor - current))
    assert np.allclose(
        np.cov(combined, rowvar=False, bias=True),
        np.cov(differentials, rowvar=False, bias=True),
    )

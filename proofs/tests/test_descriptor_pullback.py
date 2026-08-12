"""Executable witnesses for D21 regularized descriptor pullback."""

import numpy as np


def test_weighted_normal_equations_match_the_declared_solution():
    jacobian = np.array([[2.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
    weights = np.diag([1.0, 3.0, 0.5])
    desired = np.array([1.0, -0.5, 0.25])
    damping = 0.3

    normal = jacobian.T @ weights @ jacobian + damping**2 * np.eye(2)
    step = np.linalg.solve(normal, jacobian.T @ weights @ desired)

    gradient = jacobian.T @ weights @ (jacobian @ step - desired) + damping**2 * step
    assert np.linalg.norm(gradient) < 1e-12
    assert np.linalg.norm(jacobian @ step - desired) < np.linalg.norm(desired)


def test_singular_value_attenuation_lies_in_the_declared_interval():
    singular_values = np.array([0.0, 0.2, 1.0, 7.0])
    damping = 0.4
    attenuation = singular_values**2 / (singular_values**2 + damping**2)

    assert np.all(attenuation >= 0.0)
    assert np.all(attenuation < 1.0)
    assert attenuation[0] == 0.0
    assert np.all(np.diff(attenuation) > 0.0)


def test_rank_deficient_pullback_is_finite():
    jacobian = np.array([[1.0, 1.0], [2.0, 2.0]])
    desired = np.array([1.0, 2.0])
    damping = 1e-3
    normal = jacobian.T @ jacobian + damping**2 * np.eye(2)

    step = np.linalg.solve(normal, jacobian.T @ desired)

    assert np.all(np.isfinite(step))
    assert np.linalg.norm(jacobian @ step - desired) < 1e-5

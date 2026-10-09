"""Bounded-state and explicit-start contracts for the classical Python API."""

import numpy as np
import pytest

from anneal import Boltzmann, Fast, Gsa, run


@pytest.mark.parametrize(
    "preset",
    [
        Boltzmann(t_init=3.0, sigma=50.0),
        Fast(t_init=3.0, gamma=50.0),
        Gsa(t_init=3.0, q_v=2.62, q_a=1.7),
    ],
)
def test_run_never_evaluates_outside_bounds(preset):
    low = np.array([-1.0, -2.0])
    high = np.array([1.0, 2.0])
    evaluated = []

    def bounded_objective(x):
        point = np.asarray(x, dtype=np.float64).copy()
        assert np.all(point >= low)
        assert np.all(point <= high)
        evaluated.append(point)
        return float(np.dot(point, point))

    history = run(
        bounded_objective,
        low,
        high,
        preset,
        n_epochs=3,
        steps_per_epoch=20,
        seed=11,
    )

    assert len(evaluated) == 61
    assert np.all(np.asarray(history.best_pos) >= low)
    assert np.all(np.asarray(history.best_pos) <= high)


def test_run_uses_and_clips_initial_position():
    seen = []

    def objective(x):
        seen.append(np.asarray(x, dtype=np.float64).copy())
        return float(np.dot(x, x))

    history = run(
        objective,
        np.array([-1.0, -1.0]),
        np.array([1.0, 1.0]),
        Boltzmann(),
        n_epochs=0,
        steps_per_epoch=0,
        x0=np.array([3.0, -0.25]),
    )

    assert len(seen) == 1
    assert seen[0] == pytest.approx([1.0, -0.25])
    assert history.best_pos == pytest.approx([1.0, -0.25])


def test_run_rejects_invalid_initial_position():
    with pytest.raises(ValueError, match="same length"):
        run(
            lambda x: float(np.dot(x, x)),
            np.array([-1.0, -1.0]),
            np.array([1.0, 1.0]),
            Boltzmann(),
            x0=np.array([0.0]),
        )


def test_run_rejects_unknown_boundary_policy():
    with pytest.raises(ValueError, match="boundary"):
        run(
            lambda x: float(np.dot(x, x)),
            np.array([-1.0]),
            np.array([1.0]),
            Boltzmann(),
            boundary="clip",
        )

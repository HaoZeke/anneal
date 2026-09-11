"""Declared finite-box contracts for the device controller."""

import numpy as np
import pytest

import anneal


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize(
    "low,high",
    [
        ([1.0], [-1.0]),
        ([np.nan], [1.0]),
        ([-np.inf], [1.0]),
        ([-1.0], [np.inf]),
        ([], []),
    ],
)
def test_device_rejects_invalid_boxes_before_objective_work(low, high, batched):
    def objective(x):
        raise AssertionError("invalid bounds must be rejected before objective work")

    kwargs = {"n_epochs": 1, "steps_per_epoch": 1}
    if batched:
        kwargs["n_chains"] = 4
    entry = anneal.run_ensemble if batched else anneal.run_device
    with pytest.raises(ValueError, match="bounds"):
        entry(
            objective, np.asarray(low), np.asarray(high), anneal.Boltzmann(), **kwargs
        )


def test_device_rejects_nan_start_before_objective_work():
    def objective(x):
        raise AssertionError("a NaN start must be rejected before objective work")

    with pytest.raises(ValueError, match="start.*finite"):
        anneal.run_device(
            objective,
            np.array([-1.0]),
            np.array([1.0]),
            anneal.Boltzmann(),
            start=np.array([np.nan]),
            n_epochs=1,
            steps_per_epoch=1,
        )


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("preset", [anneal.Boltzmann(), anneal.Fast(), anneal.Gsa()])
def test_device_preserves_fixed_coordinates_in_every_callback(preset, batched):
    seen = []

    def objective(x):
        assert np.all(x[..., 0] == 0.25)
        assert np.all(np.isfinite(x))
        assert np.all((-1.0 <= x[..., 1]) & (x[..., 1] <= 1.0))
        seen.append(x.copy())
        return np.asarray(np.sum(x * x, axis=-1))

    kwargs = {"n_epochs": 2, "steps_per_epoch": 2}
    if batched:
        kwargs["n_chains"] = 4
    entry = anneal.run_ensemble if batched else anneal.run_device
    result = entry(
        objective, np.array([0.25, -1.0]), np.array([0.25, 1.0]), preset, **kwargs
    )
    assert len(seen) == result.n_evals == 5
    assert result.evaluated_points == 5 * (4 if batched else 1)
    np.testing.assert_array_equal(result.best_val, np.sum(result.best_pos**2, axis=-1))

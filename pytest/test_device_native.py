import numpy as np
import pytest

import anneal
from anneal import _core


@pytest.mark.parametrize(
    "preset",
    [anneal.Boltzmann(), anneal.Fast(), anneal.Gsa()],
)
@pytest.mark.parametrize("batched", [False, True])
def test_device_facade_preserves_native_controller_trace(preset, batched):
    name = "_run_device_ensemble" if batched else "_run_device"
    native = getattr(_core, name, None)
    assert callable(native), "device execution must have a native controller"
    low = np.full(3, -1.0)
    high = np.full(3, 1.0)
    traces = []
    results = []
    for direct in [False, True]:
        trace = []

        def objective(x):
            assert x.shape == ((4, 3) if batched else (3,))
            assert np.all(np.isfinite(x))
            assert np.all((low <= x) & (x <= high))
            trace.append(x.copy())
            return np.asarray(np.sum(x * x, axis=-1))

        kwargs = {"n_epochs": 3, "steps_per_epoch": 7, "seed": 71}
        if batched:
            kwargs["n_chains"] = 4
        else:
            kwargs["start"] = np.asarray([0.5, 0.25, -0.5])
        entry = (
            native
            if direct
            else (anneal.run_ensemble if batched else anneal.run_device)
        )
        result = entry(objective, low, high, preset, **kwargs)
        assert len(trace) == 22
        assert result.n_evals == len(trace)
        assert result.evaluated_points == len(trace) * (4 if batched else 1)
        np.testing.assert_array_equal(
            result.best_val, np.sum(result.best_pos**2, axis=-1)
        )
        traces.append(trace)
        results.append(result)

    np.testing.assert_array_equal(traces[0], traces[1])
    for field in ["best_pos", "best_val", "accepted", "rejected"]:
        np.testing.assert_array_equal(
            getattr(results[0], field), getattr(results[1], field)
        )


def test_device_result_retains_finite_probes_from_an_undefined_start():
    seen = []

    def objective(x):
        norm = np.sum(x * x)
        value = np.asarray(np.nan if norm < 1e-12 else norm)
        seen.append((x.copy(), value))
        return value

    result = anneal.run_device(
        objective,
        np.full(2, -1.0),
        np.full(2, 1.0),
        anneal.Boltzmann(),
        start=np.zeros(2),
        n_epochs=2,
        steps_per_epoch=8,
        seed=71,
    )
    best = min((value, point) for point, value in seen if np.isfinite(value))
    assert np.isfinite(result.best_val)
    assert result.best_val == best[0]
    np.testing.assert_array_equal(result.best_pos, best[1])
    assert result.n_evals == len(seen)


@pytest.mark.parametrize("batched", [False, True])
def test_device_chain_recovers_a_finite_occupied_state(batched):
    seen = []

    def objective(x):
        seen.append(x.copy())
        value = np.asarray(np.sum(x * x, axis=-1))
        return np.full_like(value, np.nan) if len(seen) == 1 else value

    kwargs = {"n_epochs": 1, "steps_per_epoch": 1, "seed": 71}
    if batched:
        kwargs["n_chains"] = 4
    entry = anneal.run_ensemble if batched else anneal.run_device
    result = entry(
        objective, np.full(2, -1.0), np.full(2, 1.0), anneal.Boltzmann(), **kwargs
    )
    assert len(seen) == result.n_evals == 2
    assert np.sum(result.accepted) == (4 if batched else 1)
    assert np.sum(result.rejected) == 0
    np.testing.assert_array_equal(result.best_pos, seen[1])
    np.testing.assert_array_equal(result.best_val, np.sum(seen[1] ** 2, axis=-1))


@pytest.mark.parametrize("batched", [False, True])
def test_device_nonfinite_candidate_cannot_replace_finite_occupancy(batched):
    seen = []

    def objective(x):
        seen.append(x.copy())
        value = np.asarray(np.sum(x * x, axis=-1))
        return np.full_like(value, -np.inf) if len(seen) == 2 else value

    kwargs = {"n_epochs": 1, "steps_per_epoch": 1, "seed": 71}
    if batched:
        kwargs["n_chains"] = 4
    entry = anneal.run_ensemble if batched else anneal.run_device
    result = entry(
        objective, np.full(2, -1.0), np.full(2, 1.0), anneal.Boltzmann(), **kwargs
    )
    assert len(seen) == result.n_evals == 2
    assert np.sum(result.accepted) == 0
    assert np.sum(result.rejected) == (4 if batched else 1)
    np.testing.assert_array_equal(result.best_pos, seen[0])
    np.testing.assert_array_equal(result.best_val, np.sum(seen[0] ** 2, axis=-1))

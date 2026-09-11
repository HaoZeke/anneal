"""Conformance of native policies through the device-array binding."""

import numpy as np
import pytest

import anneal
import anneal.device as device


@pytest.mark.parametrize("preset", [anneal.Boltzmann(), anneal.Fast(), anneal.Gsa()])
def test_device_temperatures_match_native_scalar_schedule(preset):
    low, high = np.full(2, -1.0), np.full(2, 1.0)
    native = anneal.run(
        lambda x: float(np.sum(x * x)),
        low,
        high,
        preset,
        n_epochs=7,
        steps_per_epoch=2,
    )
    arrays = anneal.run_device(
        lambda x: np.asarray(np.sum(x * x)),
        low,
        high,
        preset,
        n_epochs=7,
        steps_per_epoch=2,
    )
    np.testing.assert_array_equal(arrays.temps, [line.temp for line in native.epochs])


def test_device_qv_two_uses_temperature_scaled_normal_ratio(monkeypatch):
    class Draws:
        def __init__(self, *args, **kwargs):
            self.normals = iter([np.array([4.0, -4.0, 2.0]), np.array([2.0, 2.0, 1.0])])

        def normal(self, shape):
            assert shape == (3,)
            return next(self.normals)

        def uniform(self, shape):
            return np.full(shape, 0.25)

    monkeypatch.setattr(device, "_Random", Draws)
    seen = []

    def objective(x):
        seen.append(x.copy())
        return np.asarray(0.0)

    result = anneal.run_device(
        objective,
        np.full(3, -100.0),
        np.full(3, 100.0),
        anneal.Gsa(t_init=3.0, q_v=2.0),
        start=np.zeros(3),
        n_epochs=1,
        steps_per_epoch=1,
    )
    assert len(seen) == 2
    np.testing.assert_allclose(seen[1], [6.0, -6.0, 6.0], rtol=2e-14, atol=0.0)
    np.testing.assert_array_equal(result.current_pos, seen[1])
    assert result.total_accepted == 1


def test_device_compact_acceptance_does_not_evaluate_invalid_powers():
    calls = 0

    def objective(x):
        nonlocal calls
        calls += 1
        return np.asarray(0.0 if calls == 1 else 100.0)

    with np.errstate(invalid="raise", over="raise", divide="raise"):
        result = anneal.run_device(
            objective,
            np.full(2, -1.0),
            np.full(2, 1.0),
            anneal.Gsa(q_v=2.0, q_a=-0.5),
            start=np.zeros(2),
            n_epochs=2,
            steps_per_epoch=8,
        )
    assert calls == result.n_evals == 17
    assert result.total_accepted == 0
    assert result.total_rejected == 16
    assert result.best_val == 0.0


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("preset", [anneal.Fast(), anneal.Gsa(), anneal.Gsa(q_a=-0.5)])
def test_device_native_presets_keep_cuda_storage(preset, batched, dtype, monkeypatch):
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device required")

    def host_copy(*args, **kwargs):
        raise AssertionError("device search must not convert arrays to NumPy")

    monkeypatch.setattr(cupy, "asnumpy", host_copy)
    low = cupy.full(3, -1.0, dtype=dtype)
    high = cupy.full(3, 1.0, dtype=dtype)
    calls = 0

    def objective(x):
        nonlocal calls
        calls += 1
        assert isinstance(x, cupy.ndarray)
        assert x.device.id == low.device.id
        return cupy.sum(x * x, axis=-1)

    kwargs = {"n_epochs": 2, "steps_per_epoch": 8, "seed": 71}
    if batched:
        kwargs["n_chains"] = 8
    entry = anneal.run_ensemble if batched else anneal.run_device
    result = entry(objective, low, high, preset, **kwargs)
    assert calls == result.n_evals == 17
    assert result.evaluated_points == 17 * (8 if batched else 1)
    for value in [result.best_pos, result.best_val, result.accepted, result.rejected]:
        assert isinstance(value, cupy.ndarray)
        assert value.device.id == low.device.id
        assert value.__dlpack_device__() == (2, low.device.id)
    assert result.best_pos.dtype == low.dtype
    assert bool(cupy.all(cupy.isfinite(result.best_val)))
    assert bool(cupy.all((low <= result.best_pos) & (result.best_pos <= high)))
    cupy.testing.assert_array_equal(
        result.best_val, cupy.sum(result.best_pos * result.best_pos, axis=-1)
    )

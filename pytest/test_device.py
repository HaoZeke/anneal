import numpy as np
import pytest

from anneal import Boltzmann, DeviceHistory, run_device


def styblinski_tang_array(array):
    xp = array.__array_namespace__()
    x2 = array * array
    x4 = x2 * x2
    return xp.sum(0.5 * (x4 - 16.0 * x2 + 5.0 * array), axis=-1)


def scalar_styblinski_tang(array):
    x = np.asarray(array)
    return float(0.5 * np.sum(x**4 - 16 * x**2 + 5 * x))


def test_run_device_returns_array_api_history_on_numpy():
    low = np.asarray([-5.0, -5.0])
    high = np.asarray([5.0, 5.0])

    history = run_device(
        styblinski_tang_array,
        low,
        high,
        Boltzmann(t_init=5.0, sigma=0.5),
        n_epochs=12,
        steps_per_epoch=20,
        seed=42,
    )

    assert isinstance(history, DeviceHistory)
    assert isinstance(history.best_pos, np.ndarray)
    assert isinstance(history.best_val, np.ndarray)
    assert history.best_pos.shape == low.shape
    assert history.best_vals.shape == (12,)
    assert history.accepted.shape == (12,)
    assert history.rejected.shape == (12,)
    assert history.total_accepted.shape == ()
    assert history.total_rejected.shape == ()


def test_run_device_rejects_objectives_that_return_host_scalars():
    low = np.asarray([-5.0, -5.0])
    high = np.asarray([5.0, 5.0])

    with pytest.raises(ValueError, match="Array API array"):
        run_device(
            scalar_styblinski_tang,
            low,
            high,
            Boltzmann(t_init=5.0, sigma=0.5),
            n_epochs=2,
            steps_per_epoch=2,
            seed=42,
        )


@pytest.mark.gpu
def test_run_device_keeps_cupy_arrays_on_cuda():
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device required")

    with cupy.cuda.Device(0):
        low = cupy.asarray([-5.0, -5.0])
        high = cupy.asarray([5.0, 5.0])

        history = run_device(
            styblinski_tang_array,
            low,
            high,
            Boltzmann(t_init=5.0, sigma=0.5),
            n_epochs=12,
            steps_per_epoch=20,
            seed=42,
        )

        assert isinstance(history.best_pos, cupy.ndarray)
        assert isinstance(history.best_val, cupy.ndarray)
        assert isinstance(history.best_vals, cupy.ndarray)
        assert history.best_pos.device.id == 0
        assert history.best_val.device.id == 0
        assert history.best_vals.device.id == 0
        assert history.accepted.device.id == 0
        assert history.rejected.device.id == 0
        assert hasattr(history.best_pos, "__dlpack__")
        assert hasattr(history.best_val, "__dlpack__")

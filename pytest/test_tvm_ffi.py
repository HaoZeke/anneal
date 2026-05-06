import numpy as np
import pytest

from anneal import Boltzmann, run_device
from anneal.tvm_ffi import (
    tvm_ffi_tensor,
    tvm_ffi_tensor_metadata,
    tvm_ffi_tensors_from_history,
)


class _FakeTvmFfi:
    def __init__(self):
        self.arrays = []

    def from_dlpack(self, array):
        self.arrays.append(array)
        return {"shape": tuple(array.shape), "dtype": str(array.dtype)}


def test_tvm_ffi_tensor_uses_dlpack_provider_without_required_dependency():
    array = np.asarray([1.0, 2.0], dtype=np.float32)
    fake = _FakeTvmFfi()

    converted = tvm_ffi_tensor(array, tvm_ffi=fake)

    assert converted == {"shape": (2,), "dtype": "float32"}
    assert fake.arrays == [array]


def test_tvm_ffi_tensors_from_history_exports_device_history_fields():
    low = np.asarray([-1.0, -1.0], dtype=np.float32)
    high = np.asarray([1.0, 1.0], dtype=np.float32)
    history = run_device(
        lambda x: np.sum(x * x),
        low,
        high,
        Boltzmann(t_init=1.0, sigma=0.1),
        n_epochs=2,
        steps_per_epoch=2,
        seed=10,
    )
    fake = _FakeTvmFfi()

    tensors = tvm_ffi_tensors_from_history(history, tvm_ffi=fake)

    assert sorted(tensors) == [
        "accepted",
        "best_pos",
        "best_val",
        "best_vals",
        "current_pos",
        "current_val",
        "epochs",
        "rejected",
        "temps",
    ]
    assert tensors["best_pos"] == {"shape": (2,), "dtype": "float32"}
    assert len(fake.arrays) == len(tensors)


@pytest.mark.gpu
def test_tvm_ffi_metadata_preserves_cupy_dlpack_device():
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device required")

    with cupy.cuda.Device(0):
        array = cupy.asarray([1.0, 2.0], dtype=cupy.float32)
        fake = _FakeTvmFfi()

        metadata = tvm_ffi_tensor_metadata(array)
        converted = tvm_ffi_tensor(array, tvm_ffi=fake)

        assert metadata.shape == (2,)
        assert metadata.dtype == "float32"
        assert metadata.dlpack_device == (2, 0)
        assert converted == {"shape": (2,), "dtype": "float32"}
        assert fake.arrays == [array]


def test_tvm_ffi_tensor_rejects_non_dlpack_objects():
    with pytest.raises(TypeError, match="__dlpack__"):
        tvm_ffi_tensor([1.0, 2.0], tvm_ffi=_FakeTvmFfi())

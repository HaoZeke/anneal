"""anneal: simulated annealing components on the eindir typed primitives.

Public API:
  - Boltzmann(t_init, sigma): logarithmic cooling + Gaussian + Metropolis.
  - Fast(t_init, gamma): reciprocal cooling + Cauchy + Metropolis.
  - Gsa(t_init, q_v, q_a): Tsallis cooling + Tsallis visit + Tsallis accept.
  - run(obj_fn, low, high, preset, n_epochs, steps_per_epoch, seed): SA loop.
  - History, EpochLine: returned by `run`.

The IISE-manuscript composition laws L1-L4 are enforced inside the Rust
SaVariant::checked constructor; preset constructors call it under the hood.
"""

import numpy as np

from anneal._core import (
    Boltzmann,
    EpochLine,
    Fast,
    Gsa,
    History,
    __version__,
    low_discrepancy_points as _core_low_discrepancy_points,
    run,
    run_hmc,
    run_qmc,
)
from anneal.device import DeviceHistory, EnsembleHistory, run_device, run_ensemble
from anneal.tvm_ffi import (
    TvmFfiTensorMetadata,
    tvm_ffi_tensor,
    tvm_ffi_tensor_metadata,
    tvm_ffi_tensors_from_history,
)


def low_discrepancy_points(low, high, n: int, skip: int = 1):
    """Return bounded low-discrepancy points as a NumPy array."""
    low_arr = np.asarray(low, dtype=np.float64)
    high_arr = np.asarray(high, dtype=np.float64)
    return np.asarray(
        _core_low_discrepancy_points(low_arr, high_arr, int(n), int(skip)),
        dtype=np.float64,
    )


__all__ = [
    "Boltzmann",
    "DeviceHistory",
    "EnsembleHistory",
    "EpochLine",
    "Fast",
    "Gsa",
    "History",
    "TvmFfiTensorMetadata",
    "__version__",
    "low_discrepancy_points",
    "run",
    "run_device",
    "run_ensemble",
    "run_hmc",
    "run_qmc",
    "tvm_ffi_tensor",
    "tvm_ffi_tensor_metadata",
    "tvm_ffi_tensors_from_history",
]

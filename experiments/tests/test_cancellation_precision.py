import numpy as np

from experiments.exp2_cancellation import rosenbrock


def test_rosenbrock_keeps_requested_precision():
    for dtype_name in ("float64", "float32", "float16"):
        dtype = np.dtype(dtype_name)
        value = rosenbrock([0.5, 0.5], dtype)
        assert np.asarray(value).dtype == dtype

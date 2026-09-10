"""The values-only ensemble preserves an explicitly supplied starting point."""

import numpy as np
import pytest

import anneal


@pytest.mark.parametrize(
    "start", [None, [0.25, -0.75]], ids=["omitted-midpoint", "explicit-off-centre"]
)
def test_single_values_replica_spends_first_call_at_requested_start(start):
    low = np.array([-2.0, -3.0])
    high = np.array([4.0, 5.0])
    expected_start = (low + high) * 0.5 if start is None else np.asarray(start)
    arguments = {} if start is None else {"x0": start}
    evaluations = []

    def objective(x):
        point = np.asarray(x, dtype=float).copy()
        displacement = point - np.array([-0.5, 0.875])
        value = 3.0 + float(np.dot(displacement, displacement))
        evaluations.append((point, value))
        return value

    result = anneal.ensemble_optimize(
        objective,
        low,
        high,
        budget=1,
        seed=7,
        grad_fn=None,
        replicas=1,
        **arguments,
    )

    assert len(evaluations) == 1
    assert result["charged"] == 1
    np.testing.assert_array_equal(evaluations[0][0], expected_start)
    np.testing.assert_array_equal(result["best_pos"], expected_start)
    assert result["best_val"] == evaluations[0][1]
    assert np.isfinite(result["best_val"])

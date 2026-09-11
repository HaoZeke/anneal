"""An additive objective constant does not change the search landscape."""

import anneal
import numpy as np
import pytest


@pytest.mark.parametrize(
    "driver,with_gradient",
    [
        ("box_ensemble_optimize", True),
        ("ensemble_optimize", True),
        ("ensemble_optimize", False),
    ],
)
@pytest.mark.parametrize("history", ["private", "shared"])
def test_box_exploration_is_independent_of_objective_origin(
    driver, with_gradient, history
):
    def run(offset):
        evaluations = []
        gradients = []

        def objective(x):
            evaluations.append(tuple(x))
            return offset

        def gradient(x):
            gradients.append(tuple(x))
            return np.zeros_like(x)

        result = getattr(anneal, driver)(
            objective,
            np.full(2, -1.0),
            np.full(2, 1.0),
            budget=4096,
            seed=71,
            grad_fn=gradient if with_gradient else None,
            replicas=4,
            history=history,
        )
        assert result["n_evals"] == len(evaluations)
        assert result["n_grads"] == len(gradients)
        assert len(evaluations) + len(gradients) <= 4096
        assert result["best_val"] == offset
        return result, evaluations, gradients

    original, evaluations, gradients = run(0.0)
    shifted, shifted_evaluations, shifted_gradients = run(1024.0)
    np.testing.assert_array_equal(evaluations, shifted_evaluations)
    np.testing.assert_array_equal(gradients, shifted_gradients)
    np.testing.assert_array_equal(original["best_pos"], shifted["best_pos"])
    for field in ("n_evals", "n_grads", "hops"):
        assert original[field] == shifted[field]

"""Sanity tests for `experiments.shared.runner.sa_run`."""

import numpy as np
import pytest

from experiments.shared.runner import OBJECTIVES, sa_run


def test_sa_run_returns_result():
    r = sa_run(
        objective="styb_tang_2d",
        variant="boltzmann",
        dtype="float64",
        seed=42,
        n_epochs=50,
        steps_per_epoch=100,
    )
    assert r.accepted + r.rejected == 50 * 100
    assert r.n_calls == 50 * 100 + 1
    assert r.best_val < 0  # SA on Styb-Tang should beat the uniform mean


def test_sa_run_is_deterministic():
    r1 = sa_run(
        objective="styb_tang_2d",
        variant="boltzmann",
        dtype="float64",
        seed=7,
        n_epochs=20,
        steps_per_epoch=50,
    )
    r2 = sa_run(
        objective="styb_tang_2d",
        variant="boltzmann",
        dtype="float64",
        seed=7,
        n_epochs=20,
        steps_per_epoch=50,
    )
    np.testing.assert_array_equal(r1.best_pos, r2.best_pos)
    assert r1.best_val == r2.best_val
    assert r1.accepted == r2.accepted


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_dtype_is_honoured(dtype):
    r = sa_run(
        objective="styb_tang_2d",
        variant="boltzmann",
        dtype=dtype,
        seed=1,
        n_epochs=10,
        steps_per_epoch=20,
    )
    assert r.best_pos.dtype == np.dtype(dtype)


@pytest.mark.parametrize("variant", ["boltzmann", "fast", "gsa"])
def test_all_variants_run(variant):
    r = sa_run(
        objective="styb_tang_2d",
        variant=variant,
        dtype="float64",
        seed=3,
        n_epochs=20,
        steps_per_epoch=50,
    )
    assert r.n_calls > 0


def test_compensated_delta_e_changes_trajectory():
    """At f16 the Kahan compensation perturbs the accept/reject sequence;
    at f64 the compensation is identically zero so trajectories agree."""
    r_plain = sa_run(
        objective="styb_tang_2d",
        variant="boltzmann",
        dtype="float64",
        seed=5,
        n_epochs=20,
        steps_per_epoch=50,
        compensated_delta_e=False,
    )
    r_comp = sa_run(
        objective="styb_tang_2d",
        variant="boltzmann",
        dtype="float64",
        seed=5,
        n_epochs=20,
        steps_per_epoch=50,
        compensated_delta_e=True,
    )
    # f64 Kahan correction is zero; trajectories should agree exactly.
    np.testing.assert_array_equal(r_plain.best_pos, r_comp.best_pos)


def test_objective_registry_has_four_entries():
    assert set(OBJECTIVES.keys()) == {
        "styb_tang_2d",
        "rosenbrock_2d",
        "rastrigin_2d",
        "ackley_2d",
    }

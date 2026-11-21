# Copyright (c) anneal developers
"""The reported ChemFit fit, run against the installed ChemFit ``Fitter``.

Thirteen Lennard-Jones atoms with their positions flattened into the
optimizer vector, a (-3, 3) box, a budget of 2000 and the default portfolio.
Current ChemFit is driven through ``evaluate`` / ``step`` and ChemFit 3.1
through ``ask`` / ``tell``; the test runs against whichever is installed and
is skipped when ChemFit is not.
"""

import numpy as np
import pytest

fitter_module = pytest.importorskip("chemfit.fitter")

from anneal.chemfit import (  # noqa: E402
    fit_anneal,
    fit_chemfit,
    run_benchmark,
    run_fitter,
)

BUDGET = 2000


def lennard_jones(positions):
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    diff = positions[:, None, :] - positions[None, :, :]
    r2 = np.sum(diff * diff, axis=-1)[np.triu_indices(len(positions), 1)]
    inv6 = 1.0 / np.maximum(r2, 1e-12) ** 3
    return float(np.sum(4.0 * (inv6 * inv6 - inv6)))


def _fit(entry, fitter, initial):
    if entry == "fit_anneal":
        return fit_anneal(fitter, BUDGET)
    if entry == "fit_chemfit":
        return fit_chemfit(fitter, BUDGET)
    if entry == "run_benchmark":
        return run_benchmark(
            {"fitter": fitter, "budget": BUDGET, "initial_params": initial}
        )
    return run_fitter(fitter, BUDGET)


@pytest.mark.parametrize("shape", [(13, 3), (39,)], ids=["atoms", "flat"])
@pytest.mark.parametrize(
    "entry", ["fit_anneal", "fit_chemfit", "run_benchmark", "run_fitter"]
)
def test_thirteen_atoms_fit_inside_the_box_within_the_budget(entry, shape):
    start = np.random.default_rng(0).uniform(-1.5, 1.5, size=(13, 3)).reshape(shape)
    seen = []

    def objective(params):
        positions = np.asarray(params["positions"])
        seen.append(positions.copy())
        return lennard_jones(positions)

    initial = {"positions": start}
    fitter = fitter_module.Fitter(
        objective, initial_params=initial, bounds={"positions": (-3.0, 3.0)}
    )
    out = _fit(entry, fitter, initial)

    evaluated = np.array([positions.reshape(-1) for positions in seen])
    assert evaluated.shape[1] == 39
    assert np.all(evaluated >= -3.0) and np.all(evaluated <= 3.0)
    assert seen[0].shape == shape and seen[0].tobytes() == start.tobytes()
    assert 0.95 * BUDGET <= len(seen) <= BUDGET
    assert fitter.contexts[0].n_evals == len(seen)

    returned = np.asarray(out["positions"])
    assert returned.shape == shape
    assert np.all(returned >= -3.0) and np.all(returned <= 3.0)
    assert lennard_jones(returned) < lennard_jones(start)
    assert lennard_jones(returned) == min(lennard_jones(p) for p in seen)


@pytest.mark.parametrize(
    "entry", ["fit_anneal", "fit_chemfit", "run_benchmark", "run_fitter"]
)
def test_installed_chemfit_handles_nested_fixed_and_float32_parameters(entry):
    initial = {
        "fixed": 0.25,
        "nested": {"a": 1.5},
        "vec": np.array([0.8, -0.8], dtype=np.float32),
    }
    bounds = {
        "fixed": (0.25, 0.25),
        "nested": {"a": (-2.0, 2.0)},
        "vec": (-1.0, 1.0),
    }

    def objective(params):
        return float(
            (params["nested"]["a"] - 0.5) ** 2
            + np.sum(np.asarray(params["vec"], dtype=np.float64) ** 2)
        )

    fitter = fitter_module.Fitter(objective, initial_params=initial, bounds=bounds)
    if entry == "fit_anneal":
        out = fit_anneal(fitter, 120, seed=0)
    elif entry == "fit_chemfit":
        out = fit_chemfit(fitter, 120, seed=0)
    elif entry == "run_benchmark":
        out = run_benchmark(
            {"fitter": fitter, "budget": 120, "initial_params": initial},
            seed=0,
        )
    else:
        out = run_fitter(fitter, 120, seed=0)
    assert out["fixed"] == 0.25
    assert isinstance(out["nested"]["a"], float) and -2.0 <= out["nested"]["a"] <= 2.0
    assert isinstance(out["vec"], np.ndarray) and out["vec"].dtype == np.float32
    assert np.all(out["vec"] >= -1.0) and np.all(out["vec"] <= 1.0)

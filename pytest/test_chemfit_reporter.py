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

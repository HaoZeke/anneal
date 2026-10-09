"""What the ChemFit bridges promise about the start and the budget.

The doubles speak ChemFit 3.1's ``ask`` / ``tell`` unless the protocol is
what a test is about; ``test_chemfit_protocol.py`` covers both protocols.
"""

import numpy as np
import pytest

anneal = pytest.importorskip("anneal")

from chemfit_doubles import (  # noqa: E402
    DRIVERS,
    ENTRIES,
    ReleasedFitter,
    drive,
)


def _fitter():
    return ReleasedFitter(
        {"positions": np.array([[0.5, -0.5, 0.25], [1.0, -1.0, 0.0]]), "eps": 0.5},
        {"positions": (-2.0, 2.0), "eps": (0.0, 1.0)},
    )


@pytest.mark.parametrize("driver", DRIVERS)
@pytest.mark.parametrize("entry", ENTRIES)
def test_the_first_evaluation_is_the_start_bit_for_bit(entry, driver):
    initial = {
        "positions": np.array([[0.1, -0.0, 1 / 3], [2.0**-30, -1.25, 5e-324]]),
        "eps": 0.7,
        "f32": np.float32(1 / 3),
        "edge": 3.0,
    }
    bounds = {
        "positions": (-3.0, 3.0),
        "eps": (0.0, 1.0),
        "f32": (0.0, 1.0),
        "edge": (-3.0, 3.0),
    }
    fitter = ReleasedFitter(initial, bounds)
    drive(entry, fitter, 40, driver=driver)
    first = fitter.evaluated[0]
    assert first["positions"].tobytes() == initial["positions"].tobytes()
    assert np.float64(first["eps"]).tobytes() == np.float64(0.7).tobytes()
    assert np.float32(first["f32"]).tobytes() == initial["f32"].tobytes()
    assert first["edge"] == 3.0


@pytest.mark.parametrize("driver", ["portfolio", "boltzmann"])
@pytest.mark.parametrize("entry", ENTRIES)
def test_a_start_outside_the_box_is_moved_onto_it(entry, driver):
    fitter = ReleasedFitter({"x": np.array([4.0, -0.2, -9.0])}, {"x": (-1.0, 1.0)})
    drive(entry, fitter, 30, driver=driver)
    assert np.array_equal(fitter.evaluated[0]["x"], [1.0, -0.2, -1.0])


@pytest.mark.parametrize("budget", [1, 7, 150, 400])
@pytest.mark.parametrize("driver", DRIVERS)
@pytest.mark.parametrize("entry", ENTRIES)
def test_the_budget_counts_the_start_and_is_never_exceeded(entry, driver, budget):
    fitter = _fitter()
    drive(entry, fitter, budget, driver=driver)
    evaluations = fitter.calls.count("ask")
    assert 1 <= evaluations <= budget
    if driver != "portfolio" and budget in (1, 7, 400):
        assert evaluations == budget

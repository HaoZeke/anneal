"""What the ChemFit bridges promise about leaves, bounds, the start and the budget.

The doubles speak ChemFit 3.1's ``ask`` / ``tell`` unless the protocol is
what a test is about; ``test_chemfit_protocol.py`` covers both protocols.
"""

import numpy as np
import pytest

anneal = pytest.importorskip("anneal")

from anneal.chemfit import (  # noqa: E402
    ChemFitVector,
    chemfit_box,
    fit_anneal,
    resolve_bounds,
)
from chemfit_doubles import (  # noqa: E402
    DRIVERS,
    ENTRIES,
    PROTOCOLS,
    ReleasedFitter,
    drive,
)


def _fitter():
    return ReleasedFitter(
        {"positions": np.array([[0.5, -0.5, 0.25], [1.0, -1.0, 0.0]]), "eps": 0.5},
        {"positions": (-2.0, 2.0), "eps": (0.0, 1.0)},
    )


def _push_up(params):
    """Lower for larger coordinates, so candidates crowd the upper bounds."""
    return -sum(
        float(np.sum(np.asarray(value, dtype=np.float64))) for value in params.values()
    )


def _outward_box(dtype):
    """A float64 box whose ends round outward when cast to ``dtype``."""
    low, high = dtype(0.5), dtype(1.5)
    return (
        float(low) + float(np.spacing(low)) / 4,
        float(high) - float(np.spacing(high)) / 4,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("entry", ENTRIES)
def test_low_precision_leaves_keep_their_dtype_inside_the_bounds(entry, dtype):
    lower, upper = _outward_box(dtype)
    fitter = ReleasedFitter(
        {"x": np.full(3, 1.0, dtype=dtype), "y": dtype(1.0)},
        {"x": (lower, upper), "y": (lower, upper)},
        loss=_push_up,
    )
    out = drive(entry, fitter, 200)
    for params in [*fitter.evaluated, out]:
        x, y = params["x"], params["y"]
        assert isinstance(x, np.ndarray) and x.dtype == dtype and x.shape == (3,)
        assert type(y) is dtype
        cast = np.append(np.asarray(x).astype(dtype), dtype(y)).astype(np.float64)
        assert np.all(cast >= lower) and np.all(cast <= upper)


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_box_holding_no_value_of_the_leaf_dtype_is_refused(entry):
    fitter = ReleasedFitter(
        {"half": np.float16(1.0), "eps": 0.5},
        {"half": (1.0001, 1.0002), "eps": (0.0, 1.0)},
    )
    with pytest.raises(ValueError, match="half"):
        drive(entry, fitter, 50)
    assert fitter.calls == []


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("entry", ENTRIES)
def test_scalar_leaves_and_zero_d_arrays_come_back_as_their_own_type(entry, protocol):
    initial = {
        "py": 0.5,
        "f64": np.float64(0.5),
        "f32": np.float32(0.5),
        "zero_d": np.array(0.5),
        "zero_d32": np.array(0.5, dtype=np.float32),
        "nested": {"py": -0.5},
    }
    bounds = {key: (-1.0, 1.0) for key in initial if key != "nested"}
    bounds["nested"] = {"py": (-1.0, 1.0)}
    fitter = protocol(initial, bounds)
    out = drive(entry, fitter, 60)
    for params in [*fitter.evaluated, out]:
        assert type(params["py"]) is float
        assert type(params["nested"]["py"]) is float
        assert type(params["f64"]) is np.float64
        assert type(params["f32"]) is np.float32
        for key, dtype in (("zero_d", np.float64), ("zero_d32", np.float32)):
            leaf = params[key]
            assert isinstance(leaf, np.ndarray)
            assert leaf.shape == () and leaf.dtype == dtype


_SHAPE = (4, 3)
_LOWER = np.linspace(-2.0, -0.5, 12).reshape(_SHAPE)
_UPPER = _LOWER + np.linspace(0.25, 2.0, 12).reshape(_SHAPE)
BOUND_FORMS = {
    "numpy pair": (np.array([-1.0, 1.5]), np.full(_SHAPE, -1.0), np.full(_SHAPE, 1.5)),
    "per-element tuple": ((_LOWER, _UPPER), _LOWER, _UPPER),
    "per-element flat list": ([_LOWER.ravel(), _UPPER.ravel()], _LOWER, _UPPER),
    "stacked array": (np.stack([_LOWER, _UPPER]), _LOWER, _UPPER),
    "per-column rows": (
        (_LOWER[0], _UPPER[0]),
        np.broadcast_to(_LOWER[0], _SHAPE),
        np.broadcast_to(_UPPER[0], _SHAPE),
    ),
}


@pytest.mark.parametrize("form", list(BOUND_FORMS))
@pytest.mark.parametrize("entry", ENTRIES)
def test_numpy_bounds_bind_every_element(entry, form):
    bound, lower, upper = BOUND_FORMS[form]
    fitter = ReleasedFitter(
        {"positions": (lower + upper) / 2, "eps": 0.5},
        {"positions": bound, "eps": np.array([0.25, 0.75])},
        loss=_push_up,
    )
    out = drive(entry, fitter, 300)
    for params in [*fitter.evaluated, out]:
        positions = np.asarray(params["positions"])
        assert positions.shape == _SHAPE
        assert np.all(positions >= lower) and np.all(positions <= upper)
        assert 0.25 <= params["eps"] <= 0.75
    best = np.asarray(out["positions"])
    assert np.all(best >= (lower + upper) / 2)


def test_the_public_box_helpers_read_numpy_bounds():
    initial = {"positions": np.zeros(_SHAPE), "eps": 0.5}
    fitter = ReleasedFitter(initial, {"positions": (_LOWER, _UPPER), "eps": np.array([0.25, 0.75])})
    low, high = chemfit_box(fitter, ChemFitVector(initial))
    assert np.array_equal(low, np.append(_LOWER.ravel(), 0.25))
    assert np.array_equal(high, np.append(_UPPER.ravel(), 0.75))
    low, high = resolve_bounds(initial, fitter_bounds=fitter.bounds)
    assert np.array_equal(low, np.append(_LOWER.ravel(), 0.25))
    assert np.array_equal(high, np.append(_UPPER.ravel(), 0.75))


@pytest.mark.parametrize(
    "bound",
    [(2.0, 1.0), (-1e308, 1e308), (-8e307, 8e307), (float("nan"), 1.0), (-float("inf"), 1.0)],
    ids=["inverted", "width overflows", "twice the width overflows", "nan", "infinite"],
)
@pytest.mark.parametrize("entry", ENTRIES)
def test_a_bad_bound_is_refused_naming_its_key(entry, bound):
    fitter = ReleasedFitter(
        {"cell": {"a": 0.5}, "eps": 0.5},
        {"cell": {"a": bound}, "eps": (0.0, 1.0)},
    )
    with pytest.raises(ValueError, match=r"cell\.a"):
        drive(entry, fitter, 50)
    assert fitter.calls == []


@pytest.mark.parametrize("entry", ENTRIES)
def test_an_inverted_element_bound_names_the_element(entry):
    lower = np.full((2, 3), -1.0)
    lower[1, 2] = 2.0
    fitter = ReleasedFitter(
        {"positions": np.zeros((2, 3))},
        {"positions": (lower, np.full((2, 3), 1.0))},
    )
    with pytest.raises(ValueError, match=r"positions\[1, ?2\]"):
        drive(entry, fitter, 50)
    assert fitter.calls == []


def test_explicit_fit_anneal_bounds_name_the_coordinate():
    fitter = _fitter()
    low = np.full(7, -1.0)
    high = np.full(7, 1.0)
    low[6] = 2.0
    with pytest.raises(ValueError, match="eps"):
        fit_anneal(fitter, 50, low=low, high=high)
    assert fitter.calls == []


_LONGDOUBLE_IS_WIDER = np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant


@pytest.mark.skipif(not _LONGDOUBLE_IS_WIDER, reason="longdouble is float64 here")
@pytest.mark.parametrize(
    "leaf",
    [np.longdouble("0.1"), np.array([0.25, 0.5], dtype=np.longdouble)],
    ids=["scalar", "array"],
)
@pytest.mark.parametrize("entry", ENTRIES)
def test_a_longdouble_leaf_is_refused_naming_its_key(entry, leaf):
    fitter = ReleasedFitter(
        {"eps": 0.5, "wide": leaf},
        {"eps": (0.0, 1.0), "wide": (0.0, 1.0)},
    )
    with pytest.raises(TypeError, match="wide"):
        drive(entry, fitter, 50)
    assert fitter.calls == []


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

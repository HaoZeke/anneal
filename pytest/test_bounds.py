"""``Bounds`` takes any array-like, views included, and stays a ``Bounds``."""

import copy
import pickle

import numpy as np
import pytest

from anneal import Bounds, PyObjective, _core, qmc_polish_objective

LOW = np.array([-1.0, -2.0, -0.5, 0.0, -3.0, 1.0])
HIGH = np.array([1.0, 0.5, 0.5, 2.0, -1.0, 4.0])


def _views(vector, partner):
    """``vector`` as non-contiguous views whose C-order values are its own."""
    pairs = np.column_stack((vector, partner))
    return {
        "strided column": pairs[:, 0],
        "transpose": vector.reshape(3, 2).T.copy().T,
        "reversed": vector[::-1].copy()[::-1],
        "every other entry": np.repeat(vector, 2)[::2],
    }


def _forms(vector, partner):
    return {**_views(vector, partner), "list": vector.tolist()}


def _points():
    """Points in, on, just outside and well outside ``[LOW, HIGH]``."""
    return {
        "centre": 0.5 * (LOW + HIGH),
        "low corner": LOW.copy(),
        "within slack": HIGH + 5e-10,
        "beyond slack": HIGH + 2e-9,
        "outside": np.where(np.arange(LOW.size) % 2 == 0, LOW - 3.0, HIGH + 3.0),
    }


def test_the_views_are_not_contiguous():
    for view in _views(LOW, HIGH).values():
        assert not view.flags.c_contiguous
        np.testing.assert_array_equal(np.ravel(view), LOW)


@pytest.mark.parametrize("form", sorted(_forms(LOW, HIGH)))
def test_bounds_take_low_and_high_in_any_form(form):
    low = _forms(LOW, HIGH)[form]
    high = _forms(HIGH, LOW)[form]
    bounds = Bounds(low, high, 1e-9)
    core = _core.Bounds(LOW.copy(), HIGH.copy(), 1e-9)
    assert bounds.dims == core.dims == LOW.size
    assert bounds.slack == core.slack == 1e-9
    np.testing.assert_array_equal(bounds.low, LOW)
    np.testing.assert_array_equal(bounds.high, HIGH)
    assert bounds.low.flags.c_contiguous and bounds.low.dtype == np.float64
    for point in _points().values():
        assert bounds.contains(point) == core.contains(point)
        np.testing.assert_array_equal(bounds.clip(point), core.clip(point))


@pytest.mark.parametrize("form", sorted(_forms(LOW, HIGH)))
def test_contains_and_clip_take_points_in_any_form(form):
    bounds = Bounds(LOW, HIGH, 1e-9)
    core = _core.Bounds(LOW.copy(), HIGH.copy(), 1e-9)
    verdicts = []
    for point in _points().values():
        given = _forms(point, -point)[form]
        verdicts.append(bounds.contains(given))
        assert verdicts[-1] == core.contains(point)
        np.testing.assert_array_equal(bounds.clip(given), core.clip(point))
    assert verdicts == [True, True, True, False, False]


def test_bounds_flatten_two_dimensional_and_integer_input_in_c_order():
    grid = np.arange(6).reshape(2, 3)
    bounds = Bounds(grid.T, grid.T + 1)
    np.testing.assert_array_equal(bounds.low, [0.0, 3.0, 1.0, 4.0, 2.0, 5.0])
    np.testing.assert_array_equal(bounds.high, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
    assert bounds.low.dtype == np.float64
    assert bounds.contains(grid.T + 0.5)
    assert not bounds.contains(grid + 0.5)
    np.testing.assert_array_equal(bounds.clip(np.zeros((3, 2))), bounds.low)


def test_bounds_copy_their_inputs_and_keep_them_read_only():
    low, high = LOW.copy(), HIGH.copy()
    bounds = Bounds(low, high)
    low[:] = 10.0
    high[:] = 20.0
    np.testing.assert_array_equal(bounds.low, LOW)
    np.testing.assert_array_equal(bounds.high, HIGH)
    with pytest.raises(ValueError, match="read-only"):
        bounds.low[0] = 0.0
    assert not bounds.contains(np.full(LOW.size, 15.0))


def test_bounds_and_their_extension_class_pass_isinstance():
    bounds = Bounds(LOW, HIGH)
    objective = PyObjective(lambda x: float(x @ x), bounds)
    assert isinstance(bounds, Bounds)
    assert isinstance(objective.bounds, Bounds)
    assert issubclass(_core.Bounds, Bounds)
    assert isinstance(objective, PyObjective)
    assert isinstance(objective, _core.PyObjective)
    assert issubclass(_core.PyObjective, PyObjective)
    assert not isinstance(bounds, PyObjective)
    assert not isinstance(LOW, Bounds)

    class Box(Bounds):
        pass

    box = Box(LOW, HIGH)
    assert isinstance(box, Bounds) and isinstance(box, Box)
    assert not isinstance(objective.bounds, Box)
    assert not issubclass(_core.Bounds, Box)
    assert type(copy.copy(box)) is Box


def _same_box(first, second):
    assert type(first) is type(second)
    np.testing.assert_array_equal(first.low, second.low)
    np.testing.assert_array_equal(first.high, second.high)
    assert first.slack == second.slack


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_bounds_pickle(protocol):
    bounds = Bounds(_views(LOW, HIGH)["strided column"], HIGH, 1e-9)
    restored = pickle.loads(pickle.dumps(bounds, protocol=protocol))
    _same_box(restored, bounds)
    for point in _points().values():
        assert restored.contains(point) == bounds.contains(point)


def test_bounds_copy_and_deepcopy():
    bounds = Bounds(LOW, HIGH, 1e-9)
    for duplicate in (copy.copy(bounds), copy.deepcopy(bounds)):
        _same_box(duplicate, bounds)
        assert duplicate.contains(HIGH + 5e-10)
        assert not duplicate.contains(HIGH + 2e-9)


def test_bounds_repr_reads_back_exactly():
    bounds = Bounds([0.1 + 0.2, -1.0 / 3.0], [1.0 / 3.0, 2.0], 1e-9)
    text = repr(bounds)
    assert text.startswith("Bounds(low=[")
    _same_box(eval(text, {"Bounds": Bounds}), bounds)


@pytest.mark.parametrize(
    "low, high, slack, message",
    [
        ([0.0, 0.0], [1.0, 1.0, 1.0], 1e-6, "one size"),
        ([0.0, 2.0], [1.0, 1.0], 1e-6, "coordinate 1 has low 2.0 above high 1.0"),
        (
            [0.0, float("nan")],
            [1.0, 1.0],
            1e-6,
            "coordinate 1 has a NaN wall: low nan, high 1.0",
        ),
        (
            [0.0, 0.0],
            [float("nan"), 1.0],
            1e-6,
            "coordinate 0 has a NaN wall: low 0.0, high nan",
        ),
        ([0.0, 0.0], [1.0, 1.0], -1e-6, "slack"),
        ([0.0, 0.0], [1.0, 1.0], float("nan"), "slack"),
    ],
)
def test_bounds_refuse_a_malformed_box_with_value_error(low, high, slack, message):
    with pytest.raises(ValueError, match=message):
        Bounds(low, high, slack)


def test_bounds_take_an_infinite_wall():
    inf = float("inf")
    bounds = Bounds([0.0, -inf], [inf, 1.0])
    assert bounds.contains([1e300, -1e300])
    assert bounds.clip([-1.0, 2.0]).tolist() == [0.0, 1.0]


def test_contains_refuses_and_clip_rejects_a_point_of_the_wrong_size():
    bounds = Bounds(LOW, HIGH)
    assert not bounds.contains(np.zeros(LOW.size + 1))
    with pytest.raises(ValueError, match=f"{LOW.size} coordinates"):
        bounds.clip(np.zeros(LOW.size - 1))


def _quadratic(x):
    return float((x[0] - 0.25) ** 2 + (x[1] + 0.4) ** 2)


def _quadratic_grad(x):
    return np.array([2.0 * (x[0] - 0.25), 2.0 * (x[1] + 0.4)])


def test_objective_over_bounds_from_views_runs_as_over_the_extension_box():
    box = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    results = []
    for bounds in (
        Bounds(box[:, 0], box[:, 1], 1e-9),
        _core.Bounds(box[:, 0].copy(), box[:, 1].copy(), 1e-9),
    ):
        objective = PyObjective(fn_=_quadratic, bounds=bounds, grad_fn=_quadratic_grad)
        assert objective.bounds.dims == 2
        assert objective.bounds.slack == 1e-9
        results.append(
            qmc_polish_objective(
                objective, n_starts=8, max_fevals_per_start=32, seed=7, top_k=2
            )
        )
    from_views, from_extension = results
    assert from_views["best_val"] < 1e-10
    assert from_views["best_val"] == from_extension["best_val"]
    assert from_views["best_pos"] == pytest.approx([0.25, -0.4], abs=1e-5)
    np.testing.assert_array_equal(from_views["best_pos"], from_extension["best_pos"])

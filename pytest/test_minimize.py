"""anneal.minimize is the SciPy/ChemFit box entry."""

import json
import warnings

import numpy as np
import pytest

anneal = pytest.importorskip("anneal")


def test_minimize_sphere_with_jac():
    def fn(x):
        x = np.asarray(x, dtype=float)
        return float(np.dot(x, x))

    def jac(x):
        return 2.0 * np.asarray(x, dtype=float)

    bounds = np.array([[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0]])
    res = anneal.minimize(fn, np.ones(3), bounds, jac=jac, budget=64, seed=3)
    assert np.isfinite(res.fun)
    assert res.nfev > 0
    assert res.nfev <= 64
    assert res.x.shape == (3,)
    assert np.all(res.x >= -2.0 - 1e-8)
    assert np.all(res.x <= 2.0 + 1e-8)
    assert res.success


def test_minimize_rejects_1d_bounds():
    def fn(x):
        return float(np.dot(x, x))

    with pytest.raises(ValueError, match="pairs"):
        anneal.minimize(fn, np.ones(2), np.array([-1.0, 1.0]), budget=16)


def test_minimize_jsonl_store_round_trips(tmp_path):
    def fn(x):
        x = np.asarray(x, dtype=float)
        return float(np.dot(x, x))

    store = tmp_path / "anneal_params.jsonl"
    bounds = np.array([[-2.0, 2.0], [-2.0, 2.0]])
    first = anneal.minimize(
        fn, np.ones(2), bounds, budget=32, seed=3, replicas=2, store=store
    )
    assert store.is_file()
    second = anneal.minimize(
        fn, np.full(2, 1.5), bounds, budget=32, seed=4, replicas=2, store=store
    )
    assert second.fun <= first.fun + 1e-12
    rows = anneal.JsonlParameterStore(store).observations()
    assert len(rows) >= 2


def _rosenbrock(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def _rosenbrock_grad(x):
    x = np.asarray(x, dtype=np.float64)
    g = np.zeros_like(x)
    g[:-1] = -400.0 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2.0 * (1.0 - x[:-1])
    g[1:] += 200.0 * (x[1:] - x[:-1] ** 2)
    return g


class Recorder:
    """An uncapped wrapper that keeps every point ``fun`` is called with."""

    def __init__(self, fun=_rosenbrock):
        self.fun = fun
        self.points = []

    def __call__(self, x):
        self.points.append(np.array(x, dtype=np.float64))
        return self.fun(x)


class BatchRecorder(Recorder):
    """A recorder that also evaluates batches, noting each batch's size."""

    def __init__(self, fun=_rosenbrock):
        super().__init__(fun)
        self.batches = []
        self.single_calls = 0

    def __call__(self, x):
        self.single_calls += 1
        return super().__call__(x)

    def eval_batch(self, rows):
        rows = np.asarray(rows, dtype=np.float64)
        self.batches.append(rows.shape[0])
        values = []
        for row in rows:
            self.points.append(row.copy())
            values.append(self.fun(row))
        return np.asarray(values)


class DuckBounds:
    """SciPy ``Bounds`` without SciPy: anything with ``lb`` and ``ub``."""

    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub


X0 = np.array([1.5, -1.2, 0.8, 1.9])
LOW = np.full(4, -2.0)
HIGH = np.full(4, 2.0)
BOX = list(zip(LOW, HIGH))


def _run(bounds, **kwargs):
    fun = Recorder()
    result = anneal.minimize(fun, X0, bounds, budget=300, seed=4, **kwargs)
    return fun, result


def _same_points(first, second):
    return len(first) == len(second) and all(
        np.array_equal(a, b) for a, b in zip(first, second)
    )


def test_every_bounds_form_gives_the_same_run():
    forms = [
        DuckBounds(LOW, HIGH),
        DuckBounds(-2.0, 2.0),
        np.column_stack([LOW, HIGH]),
        BOX,
        tuple(BOX),
    ]
    runs = [_run(form) for form in forms]
    first_fun, first = runs[0]
    for fun, result in runs[1:]:
        assert np.array_equal(result.x, first.x)
        assert result.fun == first.fun
        assert result.nfev == first.nfev
        assert _same_points(fun.points, first_fun.points)


def test_scipy_bounds_are_read_through_lb_and_ub():
    optimize = pytest.importorskip("scipy.optimize")
    _, ours = _run(np.column_stack([LOW, HIGH]))
    _, theirs = _run(optimize.Bounds(LOW, HIGH))
    assert np.array_equal(ours.x, theirs.x)
    assert ours.fun == theirs.fun


@pytest.mark.parametrize(
    "bounds",
    [
        np.column_stack([LOW, HIGH])[:3],
        np.stack([LOW, HIGH]),
        [(-2.0, 2.0)] * 3 + [(-2.0, None)],
        [(-2.0, 2.0)] * 3 + [(-np.inf, 2.0)],
        [(-2.0, 2.0)] * 3 + [(2.0, -2.0)],
        DuckBounds(LOW[:3], HIGH[:3]),
        "box",
    ],
)
def test_bounds_that_are_not_a_finite_box_for_x0_are_refused(bounds):
    with pytest.raises(ValueError):
        anneal.minimize(_rosenbrock, X0, bounds, budget=50)


@pytest.mark.parametrize(
    "x0",
    [X0[:3], np.array([1.5, -1.2, np.nan, 1.9]), np.array([1.5, -1.2, 0.8, 2.5]), []],
)
def test_a_start_of_the_wrong_length_not_finite_or_outside_the_box_is_refused(x0):
    with pytest.raises(ValueError):
        anneal.minimize(_rosenbrock, x0, BOX, budget=50)


@pytest.mark.parametrize("replicas", [1, 3])
def test_fixed_coordinates_never_move(replicas):
    low = np.array([-2.0, 0.7, -2.0, -2.0, -0.3])
    high = np.array([2.0, 0.7, 2.0, 2.0, -0.3])
    x0 = np.array([1.5, 0.7, 0.8, -1.0, -0.3])
    fun = Recorder()
    result = anneal.minimize(
        fun, x0, list(zip(low, high)), budget=400, seed=1, replicas=replicas
    )
    assert fun.points
    for point in fun.points:
        assert point.shape == (5,)
        assert point[1] == 0.7 and point[4] == -0.3
        assert np.all(point >= low) and np.all(point <= high)
    assert result.x.shape == (5,)
    assert result.x[1] == 0.7 and result.x[4] == -0.3
    assert result.fun == _rosenbrock(result.x)


def test_fixed_coordinates_run_the_free_ones_as_global_optimize_would():
    low = np.array([-2.0, 0.7, -2.0])
    high = np.array([2.0, 0.7, 2.0])
    x0 = np.array([1.5, 0.7, -1.0])
    fun = Recorder()
    result = anneal.minimize(fun, x0, list(zip(low, high)), budget=300, seed=2)

    free = Recorder(lambda z: _rosenbrock(np.array([z[0], 0.7, z[1]])))
    expected = anneal.global_optimize(
        free, low[[0, 2]], high[[0, 2]], 300, seed=2, x0=x0[[0, 2]]
    )
    assert len(fun.points) == len(free.points)
    for full, part in zip(fun.points, free.points):
        assert np.array_equal(full[[0, 2]], part)
    assert np.array_equal(result.x[[0, 2]], expected["best_pos"])
    assert result.fun == expected["best_val"]


def test_every_coordinate_fixed_is_one_evaluation_of_the_start():
    x0 = np.array([0.5, -0.5])
    fun = Recorder()
    result = anneal.minimize(fun, x0, [(0.5, 0.5), (-0.5, -0.5)])
    assert len(fun.points) == 1
    assert np.array_equal(result.x, x0)
    assert result.nfev == 1 and result.success


@pytest.mark.parametrize(
    ("replicas", "history"),
    [(1, "shared"), (2, "shared"), (4, "shared"), (4, "private"), (7, "private")],
)
def test_the_budget_is_the_total_over_every_replica(replicas, history):
    fun = Recorder()
    result = anneal.minimize(
        fun, X0, BOX, budget=401, seed=3, replicas=replicas, history=history
    )
    assert result.nfev == len(fun.points) <= 401
    assert result.njev == 0
    assert np.all([np.all((p >= LOW) & (p <= HIGH)) for p in fun.points])


def test_the_default_budget_follows_the_number_of_free_coordinates():
    fun = Recorder()
    result = anneal.minimize(fun, X0, BOX, seed=0)
    assert len(fun.points) == result.nfev
    assert result.nfev + result.diagnostics["n_answered"] == 1000

    x0 = np.zeros(12)
    fun = Recorder(lambda x: float(np.sum(x**2)))
    bounds = [(-1.0, 1.0)] * 11 + [(0.0, 0.0)]
    result = anneal.minimize(fun, x0, bounds, seed=0, replicas=2)
    assert result.nfev == len(fun.points) <= 10 * 11 * 12


@pytest.mark.parametrize(
    ("replicas", "history"), [(1, "shared"), (4, "shared"), (4, "private")]
)
def test_x0_is_the_first_evaluation_and_made_once(replicas, history):
    fun = Recorder()
    anneal.minimize(fun, X0, BOX, budget=500, seed=5, replicas=replicas, history=history)
    assert np.array_equal(fun.points[0], X0)
    assert sum(np.array_equal(p, X0) for p in fun.points) == 1


def test_the_result_carries_scipy_fields():
    result = anneal.minimize(_rosenbrock, X0, BOX, budget=200, seed=0)
    assert isinstance(result, anneal.MinimizeResult)
    assert isinstance(result.x, np.ndarray) and result.x.shape == (4,)
    assert result.fun == _rosenbrock(result.x) < _rosenbrock(X0)
    assert result.nfev == 200 and result.njev == 0 and result.nit == 200
    assert result.charged == 200
    assert result.success is True
    assert "not certified" in result.message
    assert "fun=" in repr(result) and "nit=200" in repr(result)
    with pytest.raises(AttributeError):
        result.hess_inv


def test_a_run_without_a_finite_value_is_not_a_success():
    result = anneal.minimize(lambda x: np.inf, X0, BOX, budget=30)
    assert result.success is False
    assert result.message == "No finite feasible objective value was found."
    assert np.all((result.x >= LOW) & (result.x <= HIGH))


@pytest.mark.parametrize(("budget", "seed"), [(200, 0), (1000, 3)])
def test_one_replica_repeats_global_optimize_call_for_call(budget, seed):
    fun = Recorder()
    result = anneal.minimize(fun, X0, BOX, budget=budget, seed=seed)
    solo = Recorder()
    expected = anneal.global_optimize(solo, LOW, HIGH, budget, seed=seed, x0=X0)
    assert _same_points(fun.points, solo.points)
    assert np.array_equal(result.x, expected["best_pos"])
    assert result.fun == expected["best_val"]
    assert result.nfev == expected["n_evals"]


@pytest.mark.parametrize(("history", "coverage_shared"), [("shared", True), ("private", False)])
def test_replicas_repeat_global_optimize_from_x0_call_for_call(history, coverage_shared):
    fun = BatchRecorder()
    result = anneal.minimize(fun, X0, BOX, budget=600, seed=8, replicas=4, history=history)
    direct = BatchRecorder()
    expected = anneal.global_optimize(
        direct, LOW, HIGH, 600, seed=8, x0=X0, replicas=4, coverage_shared=coverage_shared
    )
    assert _same_points(fun.points, direct.points)
    assert fun.batches == direct.batches and direct.single_calls == 0
    assert np.array_equal(result.x, expected["best_pos"])
    assert result.fun == expected["best_val"]
    assert result.nfev == expected["n_evals"]
    assert result.nit == expected["n_batches"]


def test_jac_takes_the_gradient_route_and_counts_its_calls():
    grads = []

    def jac(x):
        grads.append(np.array(x))
        return _rosenbrock_grad(x)

    fun = Recorder()
    result = anneal.minimize(fun, X0, BOX, jac, budget=400, seed=1)
    assert result.njev == len(grads) > 0
    assert result.nfev == len(fun.points)
    assert result.nfev + result.njev <= 400

    expected = anneal.global_optimize(
        _rosenbrock, LOW, HIGH, 400, seed=1, grad_fn=_rosenbrock_grad, x0=X0
    )
    assert np.array_equal(result.x, expected["best_pos"])
    assert result.njev == expected["n_grads"]


def test_jac_sees_full_vectors_and_gives_back_the_free_part():
    low = np.array([-2.0, 0.7, -2.0])
    high = np.array([2.0, 0.7, 2.0])
    seen = []

    def jac(x):
        seen.append(np.array(x))
        return _rosenbrock_grad(x)

    result = anneal.minimize(
        _rosenbrock, np.array([1.5, 0.7, -1.0]), list(zip(low, high)), jac, budget=200
    )
    assert seen and all(p.shape == (3,) and p[1] == 0.7 for p in seen)
    assert result.x[1] == 0.7


def test_jac_with_more_replicas_warns_and_runs_one():
    fun = Recorder()
    with pytest.warns(FutureWarning, match="replicas=1"):
        result = anneal.minimize(fun, X0, BOX, _rosenbrock_grad, budget=200, replicas=2)
    solo = Recorder()
    expected = anneal.minimize(solo, X0, BOX, _rosenbrock_grad, budget=200)
    assert _same_points(fun.points, solo.points)
    assert result.fun == expected.fun and result.njev == expected.njev
    with pytest.raises(TypeError):
        anneal.minimize(_rosenbrock, X0, BOX, "2-point", budget=50)


@pytest.mark.parametrize(
    "kwargs",
    [{"replicas": 0}, {"replicas": 1.5}, {"history": "all"}, {"budget": 0}],
)
def test_replicas_history_and_budget_are_checked(kwargs):
    with pytest.raises(ValueError):
        anneal.minimize(_rosenbrock, X0, BOX, **{"budget": 50, **kwargs})


@pytest.mark.parametrize("history", ["shared", "private"])
def test_eval_batch_takes_batches_of_the_replica_count_and_matches_one_by_one(history):
    x0 = np.linspace(1.5, 0.15, 10)
    bounds = [(-2.0, 2.0)] * 10
    batched = BatchRecorder()
    result = anneal.minimize(
        batched, x0, bounds, budget=1201, seed=0, replicas=4, history=history
    )
    assert batched.single_calls == 0
    assert batched.batches[0] == 1
    assert all(1 <= size <= 4 for size in batched.batches)
    full = sum(size for size in batched.batches if size == 4)
    if history == "private":
        assert full == 1200
    else:
        # Steps where the replicas follow one descent, such as its line
        # searches, and the end of the run, once some replicas have spent
        # their shares, ask for fewer points than there are replicas.
        assert 3 * full > 2 * result.nfev
    assert result.nit == len(batched.batches)
    assert result.nfev == sum(batched.batches) == len(batched.points)

    single = Recorder()
    expected = anneal.minimize(
        single, x0, bounds, budget=1201, seed=0, replicas=4, history=history
    )
    assert _same_points(single.points, batched.points)
    assert np.array_equal(result.x, expected.x)
    assert result.fun == expected.fun
    assert result.nit == expected.nit


def test_eval_batch_sees_full_vectors_when_coordinates_are_fixed():
    batched = BatchRecorder()
    bounds = [(-2.0, 2.0), (0.7, 0.7), (-2.0, 2.0), (-2.0, 2.0)]
    x0 = np.array([1.5, 0.7, 0.8, -1.0])
    anneal.minimize(batched, x0, bounds, budget=200, seed=0, replicas=3)
    assert batched.single_calls == 0
    assert all(p.shape == (4,) and p[1] == 0.7 for p in batched.points)


@pytest.mark.parametrize("history", ["shared", "private"])
def test_a_seed_replays_a_replica_run(history):
    runs = []
    for _ in range(2):
        fun = Recorder()
        result = anneal.minimize(
            fun, X0, BOX, budget=600, seed=8, replicas=4, history=history
        )
        runs.append((fun.points, result))
    (first, a), (second, b) = runs
    assert _same_points(first, second)
    assert np.array_equal(a.x, b.x) and a.fun == b.fun


def _replica_points(**kwargs):
    fun = Recorder()
    anneal.minimize(fun, X0, BOX, budget=600, seed=8, replicas=4, **kwargs)
    return fun.points


def test_shared_and_private_histories_run_differently():
    shared = _replica_points(history="shared")
    private = _replica_points(history="private")
    assert not _same_points(shared, private)


def test_history_none_keeps_the_evaluations_private_and_coverage_shared_overrides():
    shared = _replica_points(history="shared")
    private = _replica_points(history="private")
    assert _same_points(_replica_points(history="none"), private)
    assert _same_points(_replica_points(history="none", coverage_shared=True), shared)
    assert _same_points(_replica_points(history="shared", coverage_shared=False), private)
    assert _same_points(_replica_points(history="private", coverage_shared=True), shared)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("membership", "accepted"),
        ("membership", "observed-exclusion"),
        ("coverage_radius", 0.5),
        ("coverage_neighbors", 0),
        ("coverage_neighbors", 1),
    ],
)
@pytest.mark.parametrize("history", ["shared", "private"])
def test_keywords_without_an_effect_warn_and_leave_the_run_alone(keyword, value, history):
    with pytest.warns(FutureWarning, match=f"{keyword} has no effect"):
        points = _replica_points(history=history, **{keyword: value})
    assert _same_points(points, _replica_points(history=history))


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"membership": "everything"}, ValueError),
        ({"coverage_radius": 0.0}, ValueError),
        ({"coverage_radius": float("nan")}, ValueError),
        ({"coverage_radius": "wide"}, TypeError),
        ({"coverage_neighbors": -1}, ValueError),
        ({"coverage_neighbors": 1.5}, TypeError),
        ({"coverage_neighbors": 2**64}, OverflowError),
        ({"coverage_shared": "yes"}, TypeError),
    ],
)
def test_keywords_without_an_effect_are_still_checked_before_any_call(kwargs, error):
    fun = Recorder()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(error):
            anneal.minimize(fun, X0, BOX, budget=50, replicas=2, **kwargs)
    assert fun.points == []


def _square(x):
    return float(np.sum(np.asarray(x, dtype=np.float64) ** 2))


SQUARE = {"x0": [0.5, 0.5], "bounds": [(-1.0, 1.0), (-1.0, 1.0)], "budget": 40}
REFUSED = {
    "x0 of length 3": ({"x0": [0.5, 0.5, 0.5]}, ValueError),
    "x0 not finite": ({"x0": [np.nan, 0.5]}, ValueError),
    "x0 outside the box": ({"x0": [1.5, 0.5]}, ValueError),
    "negative budget": ({"budget": -1}, ValueError),
    "zero budget": ({"budget": 0}, ValueError),
    "budget not a number": ({"budget": "abc"}, ValueError),
    "budget too large": ({"budget": 2**64}, OverflowError),
    "negative seed": ({"seed": -1}, OverflowError),
    "seed too large": ({"seed": 2**64}, OverflowError),
    "low above high": ({"bounds": [(2.0, 1.0), (-1.0, 1.0)]}, ValueError),
    "three pairs": ({"bounds": [(-1.0, 1.0)] * 3}, ValueError),
    "low not finite": ({"bounds": [(np.nan, 1.0), (-1.0, 1.0)]}, ValueError),
    "box too wide": ({"bounds": [(-1e308, 1e308), (-1.0, 1.0)]}, ValueError),
    "store not a store": ({"store": 12}, TypeError),
}


@pytest.mark.parametrize("replicas", [1, 4])
@pytest.mark.parametrize("case", REFUSED)
def test_a_refused_call_raises_without_a_keyword_warning_first(case, replicas):
    change, error = REFUSED[case]
    args = {**SQUARE, **change}
    fun = Recorder(_square)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(error):
            anneal.minimize(
                fun,
                args.pop("x0"),
                args.pop("bounds"),
                replicas=replicas,
                coverage_radius=0.2,
                **args,
            )
    assert fun.points == []


@pytest.mark.parametrize("seed", [-1, 2**64])
def test_a_seed_out_of_range_is_refused_with_every_coordinate_fixed(seed):
    fun = Recorder(_square)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(OverflowError, match="seed"):
            anneal.minimize(
                fun, [0.5, 0.5], [(0.5, 0.5), (0.5, 0.5)], seed=seed, coverage_radius=0.2
            )
    assert fun.points == []


WARNING_ROUTES = {
    "every coordinate fixed": ({"bounds": [(0.5, 0.5), (0.5, 0.5)]}, 1),
    "one replica": ({}, 1),
    "batch replicas": ({"replicas": 4}, 1),
    "jac with replicas": ({"jac": lambda x: 2.0 * np.asarray(x), "replicas": 4}, 2),
}


@pytest.mark.parametrize("route", WARNING_ROUTES)
def test_keyword_warnings_come_from_the_caller_before_the_first_call(route):
    change, count = WARNING_ROUTES[route]
    args = {**SQUARE, **change}
    x0, bounds = args.pop("x0"), args.pop("bounds")
    fun = Recorder(_square)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(FutureWarning, match="coverage_radius has no effect"):
            anneal.minimize(fun, x0, bounds, coverage_radius=0.2, **args)
    assert fun.points == []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        anneal.minimize(Recorder(_square), x0, bounds, coverage_radius=0.2, **args)
    future = [w for w in caught if issubclass(w.category, FutureWarning)]
    assert len(future) == count
    assert all(w.filename == __file__ for w in future)


def test_a_flat_objective_keeps_the_start_and_the_fixed_coordinate():
    low = np.array([-2.0, 0.0, 100.0, -1.0, 0.001])
    high = np.array([2.0, 0.0, 120.0, 1.0, 0.1])
    start = (low + high) / 2.0
    fun = Recorder(lambda x: 0.0)
    result = anneal.minimize(
        fun, start, np.column_stack((low, high)), budget=512, replicas=4, seed=0
    )
    points = np.asarray(fun.points)
    assert 0 < len(points) == result.nfev == result.charged <= 512
    assert np.all(points >= low) and np.all(points <= high)
    np.testing.assert_array_equal(points[:, 1], 0.0)
    np.testing.assert_array_equal(points[0], start)
    assert result.fun == 0.0
    np.testing.assert_array_equal(result.x, start)


AFTER_ONE = float(np.nextafter(1.0, 2.0))


@pytest.mark.parametrize(
    "bounds",
    [
        [(1.0, AFTER_ONE)],
        [(1.0, AFTER_ONE), (1.0, AFTER_ONE)],
        [(1.0, AFTER_ONE), (0.5, 0.5)],
        [(0.0, 1e-316)],
    ],
)
@pytest.mark.parametrize("end", [0, 1])
@pytest.mark.parametrize("replicas", [1, 4])
def test_free_sides_too_narrow_for_a_difference_step_leave_only_the_start(
    bounds, end, replicas
):
    # No step changes a coordinate whose side is one unit in the last place
    # wide, or so near zero that the step underflows, so with no wider free
    # side the run is the start's evaluation: x0's, at either end, or without
    # it the box's centre's in each replica.
    low, high = np.array(bounds).T
    x0 = (low, high)[end].copy()
    fun = BatchRecorder(_square)
    result = anneal.minimize(fun, x0, bounds, budget=60, seed=0, replicas=replicas)
    assert len(fun.points) == result.nfev == 1
    np.testing.assert_array_equal(fun.points[0], x0)
    np.testing.assert_array_equal(result.x, x0)
    assert result.fun == _square(x0) and result.success
    free = low < high
    direct = Recorder(_square)
    out = anneal.global_optimize(
        direct, low[free], high[free], 60, seed=0, x0=x0[free], replicas=replicas
    )
    assert len(direct.points) == out["n_evals"] == 1
    np.testing.assert_array_equal(direct.points[0], x0[free])
    np.testing.assert_array_equal(out["best_pos"], x0[free])
    centre = (low[free] + high[free]) * 0.5
    direct = Recorder(_square)
    out = anneal.global_optimize(
        direct, low[free], high[free], 60, seed=0, replicas=replicas
    )
    assert len(direct.points) == out["n_evals"] == replicas
    for point in direct.points:
        np.testing.assert_array_equal(point, centre)
    np.testing.assert_array_equal(out["best_pos"], centre)


def _keys(points):
    return [np.asarray(p, dtype=np.float64).tobytes() for p in points]


def _corner_square(x):
    return float(np.sum((np.asarray(x, dtype=np.float64) - 0.3) ** 2))


@pytest.mark.parametrize("at", [1.0, AFTER_ONE])
def test_no_stencil_asks_for_a_point_on_a_side_too_narrow_for_its_step(at):
    # Within forty evaluations from x0 the loop's descent takes central
    # differences. On the third side, one unit in the last place wide, every
    # step rounds away, so no stencil asks for a point there, which would be
    # the iterate or a point asked before, and so short a run asks for no
    # point twice.
    x0 = np.array([0.4, -0.3, at])
    bounds = [(-1.0, 1.0), (-1.0, 1.0), (1.0, AFTER_ONE)]
    for seed in range(5):
        fun = Recorder(_corner_square)
        result = anneal.minimize(fun, x0, bounds, budget=40, seed=seed)
        keys = _keys(fun.points)
        assert len(set(keys)) == len(keys) == result.nfev == 40, seed


THREE_ULPS = float(np.nextafter(np.nextafter(AFTER_ONE, 2.0), 2.0))
AT_THE_BOX_MINIMISER = {
    "low face, 1-D": (1, 2.0, 1.0, 200),
    "corner, 6-D": (6, 2.0, 1.0, 200),
    "inside, minimum on the low face": (1, 2.0, 1.5, 200),
    "three units in the last place wide": (1, THREE_ULPS, 1.0, 60),
}


@pytest.mark.parametrize(
    ("replicas", "history"), [(1, "shared"), (4, "shared"), (4, "private")]
)
@pytest.mark.parametrize("case", AT_THE_BOX_MINIMISER)
def test_a_point_already_called_is_answered_from_that_call(case, replicas, history):
    # The box's minimiser on a face or at a corner: the loop comes back to
    # it and to stencils it has differenced, and clamped proposals land on
    # it. Those evaluations still count against the budget.
    dim, high, at, budget = AT_THE_BOX_MINIMISER[case]
    x0 = np.full(dim, at)
    for seed in range(5):
        fun = BatchRecorder(_corner_square)
        result = anneal.minimize(
            fun,
            x0,
            [(1.0, high)] * dim,
            budget=budget,
            seed=seed,
            replicas=replicas,
            history=history,
        )
        keys = _keys(fun.points)
        assert keys[0] == x0.tobytes()
        assert len(set(keys)) == len(keys) == result.nfev, seed
        answered = result.diagnostics["n_answered"]
        assert answered > 0 and result.nfev + answered == budget, seed
        if replicas == 1:
            direct = Recorder(_corner_square)
            anneal.global_optimize(
                direct, np.ones(dim), np.full(dim, high), budget, seed=seed, x0=x0
            )
            assert _keys(direct.points) == keys, seed


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize(
    ("bounds", "x0"),
    [
        ([(1.0, AFTER_ONE)], [1.0]),
        ([(1.0, AFTER_ONE)] * 2, [1.0, 1.0]),
        ([(1.0, AFTER_ONE)] * 2, [AFTER_ONE, AFTER_ONE]),
    ],
)
def test_with_jac_a_box_too_narrow_to_move_calls_each_point_once(bounds, x0, seed):
    gradients = []

    def jac(x):
        gradients.append(np.array(x, dtype=np.float64))
        return 2.0 * (np.asarray(x, dtype=np.float64) - 0.3)

    fun = Recorder(_corner_square)
    result = anneal.minimize(fun, x0, bounds, jac=jac, budget=25, seed=seed)
    keys = _keys(fun.points)
    assert keys[0] == np.asarray(x0, dtype=np.float64).tobytes()
    assert len(set(keys)) == len(keys) == result.nfev <= 2 ** len(x0)
    assert len(set(_keys(gradients))) == len(gradients) == result.njev
    assert result.diagnostics["n_answered"] > 0
    assert result.nfev + result.njev + result.diagnostics["n_answered"] <= 25


@pytest.mark.parametrize("bad", [np.nan, np.inf])
@pytest.mark.parametrize(
    ("bounds", "x0"), [([(0.0, 1.0)], [0.5]), ([(0.0, 1.0), (0.5, 0.5)], [0.5, 0.5])]
)
def test_njev_and_n_grads_count_only_the_calls_of_a_gradient_that_is_not_finite(
    bounds, x0, bad
):
    # Such a gradient sends the descent to points that are not finite, where
    # the gradient asked is charged and answered with zeros without a call.
    gradients = []

    def jac(x):
        gradients.append(np.array(x, dtype=np.float64))
        return np.full(len(x), bad)

    fun = Recorder(_corner_square)
    result = anneal.minimize(fun, x0, bounds, jac=jac, budget=1000, seed=5)
    assert len(fun.points) == result.nfev and len(gradients) == result.njev
    answered = result.diagnostics["n_answered"]
    assert result.nfev + result.njev + answered <= 1000
    gradients.clear()
    out = anneal.global_optimize(
        Recorder(_corner_square), [0.0], [1.0], 1000, seed=5, grad_fn=jac, x0=[0.5]
    )
    assert len(gradients) == out["n_grads"]
    assert out["n_evals"] + out["n_grads"] + out["n_answered"] <= 1000


X0_ROUTES = {
    "values only": {},
    "legacy policy": {"policy": "legacy"},
    "gradient": {"grad_fn": True},
    "batch replicas": {"replicas": 4},
    "persistent replicas, gradient": {"replicas": 4, "grad_fn": True},
    "persistent replicas, legacy policy": {"replicas": 4, "policy": "legacy"},
}


@pytest.mark.parametrize("dim", [1, 6])
@pytest.mark.parametrize("route", X0_ROUTES)
def test_global_optimize_calls_x0_first_and_once_on_every_route(route, dim):
    # From x0 at the box's minimiser, a corner, every route comes back to
    # it and answers it from its first call. Persistent replicas past the
    # first start elsewhere and keep their records apart, so only there may
    # a point other than x0 be called twice, once by each of two replicas.
    keywords = dict(X0_ROUTES[route])
    if keywords.pop("grad_fn", False):
        keywords["grad_fn"] = lambda x: 2.0 * (np.asarray(x) - 0.3)
    x0 = np.ones(dim)
    for seed in range(3):
        fun = Recorder(_corner_square)
        out = anneal.global_optimize(
            fun, np.ones(dim), np.full(dim, 2.0), 200, seed=seed, x0=x0, **keywords
        )
        keys = _keys(fun.points)
        assert keys[0] == x0.tobytes() and keys.count(x0.tobytes()) == 1, seed
        assert out["n_evals"] == len(keys) and out["n_answered"] > 0, seed
        spent = out["n_evals"] + out["n_grads"] + out["n_answered"]
        assert spent <= 200 if "grad_fn" in keywords else spent == 200, seed
        if not route.startswith("persistent"):
            assert len(set(keys)) == len(keys), seed


@pytest.mark.parametrize("at", [1.0, AFTER_ONE])
def test_a_side_too_narrow_for_a_step_beside_wide_ones_asks_no_point_twice(at):
    bounds = [(-1.0, 1.0), (1.0, AFTER_ONE), (-1.0, 1.0)]
    x0 = np.array([0.4, at, -0.3])
    for seed in range(3):
        fun = Recorder(_corner_square)
        result = anneal.minimize(fun, x0, bounds, budget=300, seed=seed)
        keys = _keys(fun.points)
        assert keys[0] == x0.tobytes()
        assert len(set(keys)) == len(keys) == result.nfev
        assert result.nfev + result.diagnostics["n_answered"] == 300


def test_under_declared_noise_x0_is_called_again():
    # A noisy objective's value at a point is a draw, and the rules that
    # accept under noise draw again where they need to, x0 among them.
    x0 = np.ones(1)
    fun = Recorder(_corner_square)
    out = anneal.global_optimize(fun, [1.0], [2.0], 200, seed=0, x0=x0, noise_sigma=0.01)
    keys = _keys(fun.points)
    assert keys[0] == x0.tobytes()
    assert keys.count(x0.tobytes()) > 1
    assert out["n_evals"] == len(keys) and out["n_answered"] == 0


def test_a_store_starts_the_run_from_its_best_observation_inside_the_box(tmp_path):
    store = tmp_path / "params.jsonl"
    rows = [
        {"x": [0.5, 0.5, 0.5, 0.5], "f": 3.0},
        {"x": [3.0, 1.0, 1.0, 1.0], "f": -1.0},
        {"x": [1.0, 1.0, 1.0], "f": -2.0},
        {"x": [0.9, 0.8, 0.7, 0.6], "f": float("nan")},
        {"x": [1.0, 1.0, 1.0, 1.1], "f": 0.5},
    ]
    store.write_text("".join(json.dumps(row) + "\n" for row in rows))
    fun = Recorder()
    result = anneal.minimize(fun, X0, BOX, budget=100, seed=0, store=store)
    np.testing.assert_array_equal(fun.points[0], [1.0, 1.0, 1.0, 1.1])
    assert not any(np.array_equal(p, X0) for p in fun.points)
    stored = anneal.JsonlParameterStore(store).observations()
    assert len(stored) == len(rows) + 1
    np.testing.assert_array_equal(stored[-1][0], result.x)
    assert stored[-1][1] == result.fun


def test_an_exception_in_a_replica_step_is_scored_and_warned_once():
    def fun(x):
        if x[0] > 1.0:
            raise ValueError("outside the model's range")
        return _rosenbrock(x)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = anneal.minimize(fun, X0 * [0.5, 1, 1, 1], BOX, budget=300, replicas=3)
    assert result.success
    assert result.x[0] <= 1.0
    assert sum(issubclass(w.category, RuntimeWarning) for w in caught) == 1


def test_keyboard_interrupt_ends_a_replica_run_without_another_call():
    calls = []

    def fun(x):
        calls.append(1)
        if len(calls) == 30:
            raise KeyboardInterrupt
        return _rosenbrock(x)

    with pytest.raises(KeyboardInterrupt):
        anneal.minimize(fun, X0, BOX, budget=300, replicas=3)
    assert len(calls) == 30

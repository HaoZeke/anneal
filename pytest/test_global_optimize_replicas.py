"""Batch replicas of the values-only portfolio through the native binding."""

import warnings

import numpy as np
import pytest

anneal = pytest.importorskip("anneal")
from anneal._core import global_optimize_replicas  # noqa: E402

LOW = np.full(3, -1.0)
HIGH = np.full(3, 1.0)
X0 = np.array([0.9, -0.5, 0.1])


class Counted:
    """A shifted sphere that records each point and, if batched, each batch."""

    def __init__(self, batched):
        self.points = []
        self.batches = []
        if batched:
            self.eval_batch = self._eval_batch

    def __call__(self, x):
        self.points.append(np.array(x, dtype=float, copy=True))
        return float(np.sum((np.asarray(x) - 0.3) ** 2))

    def _eval_batch(self, rows):
        rows = np.asarray(rows, dtype=float)
        self.batches.append(rows.shape[0])
        return [self(row) for row in rows]


@pytest.mark.parametrize("archive", ["shared", "private"])
def test_replicas_evaluate_the_start_first_then_batches_of_their_count(archive):
    fun = Counted(batched=True)
    out = global_optimize_replicas(fun, LOW, HIGH, 200, X0, 3, 4, archive)
    np.testing.assert_array_equal(fun.points[0], X0)
    assert fun.batches[0] == 1
    assert max(fun.batches) <= 4
    assert out["n_batches"] == len(fun.batches)
    assert out["n_evals"] == len(fun.points) <= 200
    assert out["n_grads"] == 0
    points = np.asarray(fun.points)
    assert np.all(points >= LOW) and np.all(points <= HIGH)
    values = [fun(p) for p in points]
    assert out["best_val"] == min(values)
    np.testing.assert_array_equal(out["best_pos"], points[int(np.argmin(values))])


def test_one_replica_repeats_global_optimize_from_x0():
    solo, replica = Counted(batched=False), Counted(batched=False)
    expected = anneal.global_optimize(solo, LOW, HIGH, 150, seed=5, x0=X0)
    out = global_optimize_replicas(replica, LOW, HIGH, 150, X0, 5, 1, "shared")
    np.testing.assert_array_equal(np.asarray(replica.points), np.asarray(solo.points))
    assert out["best_val"] == expected["best_val"]
    assert out["n_evals"] == expected["n_evals"]
    assert out["n_batches"] == len(replica.points)


def test_batched_and_unbatched_callables_see_the_same_points():
    batched, plain = Counted(batched=True), Counted(batched=False)
    first = global_optimize_replicas(batched, LOW, HIGH, 120, X0, 9, 3, "shared")
    second = global_optimize_replicas(plain, LOW, HIGH, 120, X0, 9, 3, "shared")
    np.testing.assert_array_equal(np.asarray(batched.points), np.asarray(plain.points))
    assert first["best_val"] == second["best_val"]
    assert first["n_batches"] == second["n_batches"] == len(batched.batches)


@pytest.mark.parametrize(
    "budget,x0,replicas,archive",
    [
        (0, X0, 2, "shared"),
        (50, X0, 0, "shared"),
        (50, X0, 2, "none"),
        (50, np.array([1.5, 0.0, 0.0]), 2, "shared"),
        (50, np.array([np.nan, 0.0, 0.0]), 2, "shared"),
        (50, np.zeros(2), 2, "shared"),
    ],
)
def test_invalid_settings_raise_before_any_call(budget, x0, replicas, archive):
    fun = Counted(batched=True)
    with pytest.raises(ValueError):
        global_optimize_replicas(fun, LOW, HIGH, budget, x0, 0, replicas, archive)
    assert fun.points == []


def _sphere_gradient(x):
    return 2.0 * (np.asarray(x, dtype=float) - 0.3)


ROUTES = {
    "batch": dict(x0=X0, replicas=3),
    "no start": dict(replicas=3),
    "gradient": dict(x0=X0, replicas=3, grad_fn=_sphere_gradient),
    "noise": dict(x0=X0, replicas=3, noise_sigma=0.01),
    "legacy": dict(x0=X0, replicas=3, policy="legacy"),
    "one replica": dict(x0=X0, replicas=1),
}


@pytest.mark.parametrize(
    "keyword,value", [("coverage_radius", 0.2), ("coverage_neighbors", 1)]
)
@pytest.mark.parametrize("shared", [True, False])
def test_batch_replicas_warn_that_coverage_keywords_change_nothing(
    keyword, value, shared
):
    plain, given = Counted(batched=True), Counted(batched=True)
    options = dict(seed=4, x0=X0, replicas=3, coverage_shared=shared)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        expected = anneal.global_optimize(plain, LOW, HIGH, 160, **options)
    options[keyword] = value
    message = f"{keyword} has no effect: batch replicas"
    with pytest.warns(FutureWarning, match=message) as caught:
        out = anneal.global_optimize(given, LOW, HIGH, 160, **options)
    assert len(caught) == 1 and caught[0].filename == __file__
    np.testing.assert_array_equal(np.asarray(given.points), np.asarray(plain.points))
    assert given.batches == plain.batches
    assert out["best_val"] == expected["best_val"]
    assert out["n_batches"] == expected["n_batches"]


@pytest.mark.parametrize("route", ROUTES)
def test_only_the_batch_replicas_warn_about_the_coverage_keywords(route):
    fun = Counted(batched=False)
    coverage = dict(coverage_radius=0.2, coverage_neighbors=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = anneal.global_optimize(
            fun, LOW, HIGH, 120, seed=2, **coverage, **ROUTES[route]
        )
    future = [w for w in caught if issubclass(w.category, FutureWarning)]
    assert ("n_batches" in out) == (route == "batch")
    assert len(future) == (2 if route == "batch" else 0)


@pytest.mark.parametrize(
    "keywords", [{"coverage_radius": 0.0}, {"coverage_neighbors": -1}]
)
def test_batch_replicas_still_check_the_coverage_keywords(keywords):
    fun = Counted(batched=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError):
            anneal.global_optimize(fun, LOW, HIGH, 100, x0=X0, replicas=3, **keywords)
    assert fun.points == []

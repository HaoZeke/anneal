"""run, run_qmc and global_optimize hold a coordinate whose two ends are equal.

Each makes the calls of the same run on the free coordinates, as minimize
does, and with every coordinate fixed it is one call.
"""

import numpy as np
import pytest

anneal = pytest.importorskip("anneal")

LOW = np.array([-2.0, 0.7, -2.0, -0.3])
HIGH = np.array([2.0, 0.7, 2.0, -0.3])
FREE = LOW < HIGH
X0 = np.array([1.5, 0.7, -1.0, -0.3])
ALL_FIXED = np.array([0.5, -0.25])


def _rosenbrock(x):
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def _rosenbrock_gradient(x):
    g = np.zeros_like(x)
    g[:-1] = -400.0 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2.0 * (1.0 - x[:-1])
    g[1:] += 200.0 * (x[1:] - x[:-1] ** 2)
    return g


class Recorder:
    """Rosenbrock with a log of the points its value and gradient are asked at.

    With ``free`` it is the problem on the free coordinates: each point is
    completed from ``X0`` before it is evaluated, and the gradient's free part
    is returned.
    """

    def __init__(self, free=False):
        self.free = free
        self.points = []
        self.gradients = []

    def _full(self, x):
        x = np.array(x, dtype=np.float64)
        if not self.free:
            return x
        full = X0.copy()
        full[FREE] = x
        return full

    def __call__(self, x):
        self.points.append(np.array(x, dtype=np.float64))
        return _rosenbrock(self._full(x))

    def grad(self, x):
        self.gradients.append(np.array(x, dtype=np.float64))
        g = _rosenbrock_gradient(self._full(x))
        return g[FREE] if self.free else g


class BatchRecorder(Recorder):
    """A recorder that also evaluates a batch, one point per row."""

    def eval_batch(self, rows):
        return np.array([self(row) for row in np.asarray(rows, dtype=np.float64)])


def _assert_same_calls(whole, part, ordered=True):
    """``whole`` was asked at ``part``'s points, completed with the fixed values."""
    for full, free in ((whole.points, part.points), (whole.gradients, part.gradients)):
        assert len(full) == len(free)
        for point in full:
            assert point.shape == X0.shape
            assert np.array_equal(point[~FREE], X0[~FREE])
            assert np.all((point >= LOW) & (point <= HIGH))
        full = [tuple(point[FREE]) for point in full]
        free = [tuple(point) for point in free]
        assert full == free if ordered else sorted(full) == sorted(free)


ROUTES = {
    "values": {"x0": True},
    "values without x0": {},
    "gradient": {"x0": True, "grad": True},
    "noise": {"x0": True, "noise_sigma": 1e-3},
    "legacy": {"x0": True, "policy": "legacy"},
    "persistent replicas": {"replicas": 3},
    "persistent replicas with a gradient": {"x0": True, "grad": True, "replicas": 3},
    "batch replicas": {"x0": True, "replicas": 3, "batched": True},
    "private batch replicas": {
        "x0": True,
        "replicas": 3,
        "batched": True,
        "coverage_shared": False,
    },
}


def _global_optimize(recorder, low, high, x0, options):
    keywords = {
        key: options[key]
        for key in ("noise_sigma", "policy", "replicas", "coverage_shared")
        if key in options
    }
    return anneal.global_optimize(
        recorder,
        low,
        high,
        300,
        seed=2,
        grad_fn=recorder.grad if options.get("grad") else None,
        x0=x0 if options.get("x0") else None,
        **keywords,
    )


@pytest.mark.parametrize("route", ROUTES)
def test_global_optimize_makes_the_calls_of_the_run_on_the_free_coordinates(route):
    options = ROUTES[route]
    kind = BatchRecorder if options.get("batched") else Recorder
    whole, part = kind(), kind(free=True)
    out = _global_optimize(whole, LOW, HIGH, X0, options)
    expected = _global_optimize(part, LOW[FREE], HIGH[FREE], X0[FREE], options)

    assert whole.points
    persistent = options.get("replicas", 1) > 1 and "batched" not in options
    _assert_same_calls(whole, part, ordered=not persistent)
    assert out["best_pos"].shape == X0.shape
    assert np.array_equal(out["best_pos"][FREE], expected["best_pos"])
    assert np.array_equal(out["best_pos"][~FREE], X0[~FREE])
    assert out["best_val"] == expected["best_val"]
    assert out["n_evals"] == expected["n_evals"] == len(whole.points)
    assert out["n_grads"] == expected["n_grads"] == len(whole.gradients)
    for replica, alone in zip(out.get("replicas", []), expected.get("replicas", [])):
        assert np.array_equal(replica["best_pos"][FREE], alone["best_pos"])
        assert np.array_equal(replica["best_pos"][~FREE], X0[~FREE])


@pytest.mark.parametrize(("replicas", "gradient"), [(1, False), (1, True), (3, False)])
def test_minimize_repeats_global_optimize_call_for_call_with_fixed_coordinates(
    replicas, gradient
):
    first, second = BatchRecorder(), BatchRecorder()
    result = anneal.minimize(
        first,
        X0,
        list(zip(LOW, HIGH)),
        first.grad if gradient else None,
        budget=300,
        seed=4,
        replicas=replicas,
    )
    out = anneal.global_optimize(
        second,
        LOW,
        HIGH,
        300,
        seed=4,
        grad_fn=second.grad if gradient else None,
        x0=X0,
        replicas=replicas,
    )
    for mine, theirs in ((first.points, second.points), (first.gradients, second.gradients)):
        assert len(mine) == len(theirs)
        assert all(np.array_equal(a, b) for a, b in zip(mine, theirs))
    assert np.array_equal(result.x, out["best_pos"])
    assert result.fun == out["best_val"]


PRESETS = {"Boltzmann": anneal.Boltzmann, "Fast": anneal.Fast, "Gsa": anneal.Gsa}


def _classical(driver, recorder, low, high, preset, x0, max_evals):
    keywords = {"n_starts": 3} if driver == "run_qmc" else {}
    return getattr(anneal, driver)(
        recorder,
        low,
        high,
        PRESETS[preset](),
        n_epochs=6,
        steps_per_epoch=10,
        seed=3,
        x0=x0,
        max_evals=max_evals,
        **keywords,
    )


@pytest.mark.parametrize("max_evals", [None, 25])
@pytest.mark.parametrize("start", [True, False])
@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("driver", ["run", "run_qmc"])
def test_classical_runs_make_the_calls_of_the_run_on_the_free_coordinates(
    driver, preset, start, max_evals
):
    whole, part = Recorder(), Recorder(free=True)
    history = _classical(
        driver, whole, LOW, HIGH, preset, X0 if start else None, max_evals
    )
    expected = _classical(
        driver, part, LOW[FREE], HIGH[FREE], preset, X0[FREE] if start else None, max_evals
    )

    assert whole.points
    _assert_same_calls(whole, part)
    best = np.asarray(history.best_pos)
    assert best.shape == X0.shape
    assert np.array_equal(best[FREE], expected.best_pos)
    assert np.array_equal(best[~FREE], X0[~FREE])
    assert history.best_val == expected.best_val
    steps = [(line.accepted, line.rejected) for line in history.epochs]
    assert steps == [(line.accepted, line.rejected) for line in expected.epochs]


@pytest.mark.parametrize("start", [True, False])
@pytest.mark.parametrize("driver", ["run", "run_qmc"])
def test_a_classical_run_with_every_coordinate_fixed_is_one_call(driver, start):
    recorder = Recorder()
    history = _classical(
        driver, recorder, ALL_FIXED, ALL_FIXED, "Gsa", ALL_FIXED if start else None, None
    )
    assert len(recorder.points) == 1
    assert np.array_equal(recorder.points[0], ALL_FIXED)
    assert np.array_equal(history.best_pos, ALL_FIXED)
    assert history.best_val == _rosenbrock(ALL_FIXED)
    assert history.epochs == []


@pytest.mark.parametrize(
    "options",
    [{}, {"x0": True}, {"grad": True}, {"noise_sigma": 0.1}, {"replicas": 3}, {"x0": True, "replicas": 3}],
    ids=["values", "values from x0", "gradient", "noise", "persistent replicas", "batch replicas"],
)
def test_global_optimize_with_every_coordinate_fixed_is_one_call(options):
    recorder = BatchRecorder()
    out = _global_optimize(recorder, ALL_FIXED, ALL_FIXED, ALL_FIXED.copy(), options)
    assert len(recorder.points) == 1 and recorder.gradients == []
    assert np.array_equal(recorder.points[0], ALL_FIXED)
    assert np.array_equal(out["best_pos"], ALL_FIXED)
    assert out["best_val"] == _rosenbrock(ALL_FIXED)
    assert (out["n_evals"], out["n_grads"]) == (1, 0)
    assert out["arm_pulls"] == {} and out["arm_successes"] == {}


def test_global_optimize_reads_the_gradient_at_full_length():
    recorder = Recorder()
    with pytest.raises(TypeError, match="grad_fn must return 4 numbers, got 2"):
        anneal.global_optimize(
            recorder, LOW, HIGH, 300, seed=0, grad_fn=lambda x: recorder.grad(x)[FREE], x0=X0
        )


@pytest.mark.parametrize("driver", ["run", "global_optimize"])
def test_low_above_high_is_refused_with_the_rule(driver):
    low = LOW.copy()
    low[1] = 0.8
    arguments = (anneal.Gsa(),) if driver == "run" else (100,)
    with pytest.raises(ValueError, match=r"low\[1\] must be less than or equal to high\[1\]"):
        getattr(anneal, driver)(Recorder(), low, HIGH, *arguments)

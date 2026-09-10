"""Standalone GLE charges callbacks through the installed native objective handle."""

import hashlib
import importlib.machinery
import math
from pathlib import Path
import types

import numpy as np
import pytest

import anneal
import anneal._core as native
from experiments.scripts import run_cutest_benchmarks as benchmarks
from experiments.scripts import run_cutest_full_suite as suite


@pytest.fixture(scope="module", autouse=True)
def native_provenance(record_testsuite_property):
    extension = Path(native.__file__).resolve()
    assert any(
        str(extension).endswith(suffix)
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    )
    digest = hashlib.sha256()
    with extension.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    record_testsuite_property("standalone_gle_native_path", str(extension))
    record_testsuite_property("standalone_gle_native_sha256", digest.hexdigest())
    record_testsuite_property("standalone_gle_version", anneal.__version__)
    record_testsuite_property(
        "standalone_gle_wrapper_path", str(Path(anneal.__file__).resolve())
    )
    record_testsuite_property(
        "standalone_gle_adapter_path", str(Path(benchmarks.__file__).resolve())
    )


class _CountedQuadratic:
    dim = 2
    low = np.array([-1.0, -1.0])
    high = np.array([1.0, 1.0])
    design_low = low
    design_high = high
    minimum = np.array([0.25, -0.4])

    def __init__(self, native_gradient):
        self.objective_points = []
        self.gradient_points = []
        if not native_gradient:
            self.grad = None

    def fn(self, x):
        point = np.asarray(x, dtype=np.float64).copy()
        self.objective_points.append(point)
        offset = point - self.minimum
        return float(np.dot(offset, offset))

    def grad(self, x):
        point = np.asarray(x, dtype=np.float64).copy()
        self.gradient_points.append(point)
        return 2.0 * (point - self.minimum)


@pytest.mark.parametrize("native_gradient", [True, False], ids=["native", "fd"])
def test_standalone_gle_native_handle_charges_real_callbacks(
    native_gradient, record_testsuite_property
):
    problem = _CountedQuadratic(native_gradient)
    args = types.SimpleNamespace(n_epochs=3, k_fixed=21)
    budget = 1 + args.n_epochs * args.k_fixed
    centre = 0.5 * (problem.design_low + problem.design_high)
    offset = centre - problem.minimum
    initial_value = float(np.dot(offset, offset))
    assert initial_value > 0.0

    best_val, reported_work = suite.run_driver(problem, "gle_langevin", 11, args)

    objective_calls = len(problem.objective_points)
    gradient_calls = len(problem.gradient_points)
    actual_work = objective_calls + gradient_calls
    assert reported_work == actual_work
    assert 2 < actual_work <= budget
    assert math.isfinite(best_val)
    assert 0.0 <= best_val <= initial_value

    probe_gradients = 2 * problem.dim
    if native_gradient:
        # Every dynamics point has one value and gradient; the additional
        # gradients are the preconditioner's complete probe pairs.
        assert gradient_calls == objective_calls + probe_gradients
        assert objective_calls > 1
        np.testing.assert_array_equal(problem.objective_points[0], centre)
        np.testing.assert_array_equal(
            problem.gradient_points[probe_gradients:], problem.objective_points
        )
    else:
        assert gradient_calls == 0
        # The handle's first four gradient requests each use the actual
        # forward-difference callback group, before its initial dynamics value.
        for probe in range(probe_gradients):
            start = probe * (problem.dim + 1)
            base = problem.objective_points[start]
            for axis in range(problem.dim):
                expected = base.copy()
                expected[axis] += benchmarks.FINITE_DIFFERENCE_GRAD_STEP
                np.testing.assert_array_equal(
                    problem.objective_points[start + axis + 1], expected
                )
        initial_dynamics_call = probe_gradients * (problem.dim + 1)
        np.testing.assert_array_equal(
            problem.objective_points[initial_dynamics_call], centre
        )
        assert objective_calls > initial_dynamics_call + problem.dim + 2

    kind = "native" if native_gradient else "fd"
    record_testsuite_property(f"standalone_gle_{kind}_objective_calls", objective_calls)
    record_testsuite_property(f"standalone_gle_{kind}_gradient_calls", gradient_calls)
    record_testsuite_property(f"standalone_gle_{kind}_reported_work", reported_work)

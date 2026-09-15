"""ensemble_optimize must return from a Python objective callback."""

import contextlib
import faulthandler
import math
import os
import threading

import numpy as np
import pytest

anneal = pytest.importorskip("anneal")

_HANG_S = 30


@contextlib.contextmanager
def _native_deadline(seconds=_HANG_S):
    """Kill the process if a native search never returns to Python.

    faulthandler.dump_traceback_later uses a watchdog thread and
    os._exit. A daemon thread is the second tripwire when faulthandler
    is compiled out.
    """
    faulthandler.dump_traceback_later(seconds, exit=True)
    stop = threading.Event()

    def _die():
        if not stop.wait(seconds):
            os._exit(1)

    threading.Thread(target=_die, daemon=True).start()
    try:
        yield
    finally:
        stop.set()
        faulthandler.cancel_dump_traceback_later()


def test_ensemble_optimize_python_callback_returns():
    """A Python objective must be callable from the native hop."""

    def fn(x):
        x = np.asarray(x, dtype=float)
        return float(np.dot(x, x))

    def grad(x):
        return 2.0 * np.asarray(x, dtype=float)

    low = np.full(3, -2.0)
    high = np.full(3, 2.0)
    with _native_deadline():
        out = anneal.ensemble_optimize(
            fn, low, high, budget=400, seed=3, grad_fn=grad, replicas=2
        )
    assert math.isfinite(out["best_val"])
    assert out["best_val"] <= 3.0
    assert out["charged"] > 0
    assert out["best_pos"].shape == (3,)
    assert np.all(out["best_pos"] >= low - 1e-8)
    assert np.all(out["best_pos"] <= high + 1e-8)


def test_ensemble_optimize_six_dim_box_returns():
    """Six-coordinate box (e.g. BIGGS6) stays six coordinates."""

    def fn(x):
        x = np.asarray(x, dtype=float)
        return float(np.dot(x, x))

    def grad(x):
        return 2.0 * np.asarray(x, dtype=float)

    low = np.full(6, -2.0)
    high = np.full(6, 2.0)
    with _native_deadline():
        out = anneal.ensemble_optimize(
            fn, low, high, budget=400, seed=3, grad_fn=grad, replicas=2
        )
    assert math.isfinite(out["best_val"])
    assert out["charged"] > 0
    assert out["charged"] <= 400
    assert out["best_pos"].shape == (6,)


def test_ensemble_optimize_charges_every_python_eval():
    """Each Python eval and grad spends one ledger unit; a spent ledger returns."""

    counts = {"n": 0}

    def fn(x):
        counts["n"] += 1
        x = np.asarray(x, dtype=float)
        return float(np.dot(x, x))

    def grad(x):
        counts["n"] += 1
        return 2.0 * np.asarray(x, dtype=float)

    low = np.full(6, -2.0)
    high = np.full(6, 2.0)
    with _native_deadline():
        out = anneal.ensemble_optimize(
            fn, low, high, budget=32, seed=3, grad_fn=grad, replicas=2
        )
    assert counts["n"] > 0
    assert out["charged"] > 0
    assert out["charged"] <= 32
    # Explicit per-callback charge: an uncharged eval or grad makes n larger.
    assert counts["n"] <= out["charged"]

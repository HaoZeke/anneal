"""Pytest suite for the new Python algebra surface (Boltzmann / Fast / Gsa
preset constructors plus run() driver). Replaces the legacy
test_funcs / test_mcsamplers / test_quench suites."""

import numpy as np
import pytest

from anneal import Boltzmann, Fast, Gsa, History, run


def styb_tang_2d(x: np.ndarray) -> float:
    """Styblinski-Tang 2D objective. Global min ~ -78.332 at (-2.9035, -2.9035)."""
    return float(0.5 * np.sum(x**4 - 16 * x**2 + 5 * x))


LOW = np.array([-5.0, -5.0])
HIGH = np.array([5.0, 5.0])
GLOBAL_MIN = -78.33198
N_EPOCHS = 100
STEPS_PER_EPOCH = 200
SEED = 42


def test_boltzmann_finds_global_minimum():
    h = run(
        styb_tang_2d,
        LOW,
        HIGH,
        Boltzmann(t_init=5.0, sigma=0.5),
        n_epochs=N_EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        seed=SEED,
    )
    assert h.best_val == pytest.approx(GLOBAL_MIN, abs=1e-2)


def test_fast_finds_global_minimum():
    h = run(
        styb_tang_2d,
        LOW,
        HIGH,
        Fast(t_init=3.0, gamma=0.5),
        n_epochs=N_EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        seed=SEED,
    )
    assert h.best_val == pytest.approx(GLOBAL_MIN, abs=1e-2)


def test_gsa_finds_global_minimum():
    h = run(
        styb_tang_2d,
        LOW,
        HIGH,
        Gsa(t_init=3.0, q_v=2.62, q_a=1.7),
        n_epochs=N_EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        seed=SEED,
    )
    assert h.best_val == pytest.approx(GLOBAL_MIN, abs=1e-2)


def test_run_returns_history_object():
    h = run(
        styb_tang_2d,
        LOW,
        HIGH,
        Boltzmann(t_init=5.0, sigma=0.5),
        n_epochs=N_EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        seed=SEED,
    )
    assert isinstance(h, History)
    assert len(h.epochs) == N_EPOCHS
    assert h.total_accepted + h.total_rejected == N_EPOCHS * STEPS_PER_EPOCH
    assert h.epochs[0].epoch == 0
    assert h.epochs[-1].epoch == N_EPOCHS - 1
    assert h.epochs[-1].best_val == h.best_val


def test_run_is_deterministic():
    h1 = run(
        styb_tang_2d,
        LOW,
        HIGH,
        Boltzmann(t_init=5.0, sigma=0.5),
        n_epochs=N_EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        seed=SEED,
    )
    h2 = run(
        styb_tang_2d,
        LOW,
        HIGH,
        Boltzmann(t_init=5.0, sigma=0.5),
        n_epochs=N_EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        seed=SEED,
    )
    assert h1.best_val == h2.best_val
    assert h1.best_pos == h2.best_pos
    assert h1.total_accepted == h2.total_accepted
    assert h1.total_rejected == h2.total_rejected


def test_low_high_dimension_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        run(
            styb_tang_2d,
            np.array([-5.0, -5.0]),
            np.array([5.0]),
            Boltzmann(t_init=1.0, sigma=0.5),
            n_epochs=10,
            steps_per_epoch=10,
            seed=SEED,
        )


def test_temperature_is_non_increasing_in_history():
    h = run(
        styb_tang_2d,
        LOW,
        HIGH,
        Boltzmann(t_init=5.0, sigma=0.5),
        n_epochs=N_EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        seed=SEED,
    )
    temps = [e.temp for e in h.epochs]
    for a, b in zip(temps, temps[1:]):
        assert a >= b


def test_best_val_is_non_increasing_in_history():
    h = run(
        styb_tang_2d,
        LOW,
        HIGH,
        Boltzmann(t_init=5.0, sigma=0.5),
        n_epochs=N_EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        seed=SEED,
    )
    bests = [e.best_val for e in h.epochs]
    for a, b in zip(bests, bests[1:]):
        assert a >= b


def test_preset_repr():
    assert "Boltzmann(t_init=1.0, sigma=0.5)" == repr(Boltzmann(t_init=1.0, sigma=0.5))
    assert "Fast(t_init=2.0, gamma=0.3)" == repr(Fast(t_init=2.0, gamma=0.3))
    assert "Gsa(t_init=3.0, q_v=2.5, q_a=1.7)" == repr(
        Gsa(t_init=3.0, q_v=2.5, q_a=1.7)
    )

"""Pytest suite for the new Python algebra surface (Boltzmann / Fast / Gsa
preset constructors plus run() driver). Replaces the legacy
test_funcs / test_mcsamplers / test_quench suites."""

import warnings

import numpy as np
import pytest

from anneal import (
    Boltzmann,
    Bounds,
    Fast,
    Gsa,
    History,
    PyObjective,
    gle_langevin,
    gle_langevin_objective,
    gle_langevin_preconditioned,
    gle_langevin_preconditioned_objective,
    low_discrepancy_points,
    pilot_draws_qmc,
    polish,
    qmc_best1bin_scout,
    qmc_best1bin_scout_objective,
    qmc_gsa_global_search,
    qmc_gsa_global_search_objective,
    qmc_polish,
    qmc_polish_objective,
    qmc_trust_region_poll,
    qmc_trust_region_poll_objective,
    run,
    run_hmc,
    run_qmc,
    shifted_qmc_polish,
)


def styb_tang_2d(x: np.ndarray) -> float:
    """Styblinski-Tang 2D objective. Global min ~ -78.332 at (-2.9035, -2.9035)."""
    return float(0.5 * np.sum(x**4 - 16 * x**2 + 5 * x))


def styb_tang_grad_2d(x: np.ndarray) -> np.ndarray:
    return 0.5 * (4 * x**3 - 32 * x + 5)


def shifted_quadratic(x: np.ndarray) -> float:
    return float((x[0] - 0.25) ** 2 + (x[1] + 0.4) ** 2)


def shifted_quadratic_grad(x: np.ndarray) -> np.ndarray:
    return np.array([2.0 * (x[0] - 0.25), 2.0 * (x[1] + 0.4)])


def anisotropic_quadratic(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    return float(arr[0] * arr[0] + 4.0 * arr[1] * arr[1])


def anisotropic_quadratic_grad(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return np.array([2.0 * arr[0], 8.0 * arr[1]], dtype=np.float64)


def smooth_needle(x: np.ndarray) -> float:
    dx = np.asarray(x, dtype=np.float64) - np.array([0.5, -0.5])
    return float(-np.exp(-40.0 * np.dot(dx, dx)))


def lj_energy(x: np.ndarray) -> float:
    """Lennard-Jones energy of flattened ``(n, 3)`` atomic positions."""
    pos = np.asarray(x, dtype=np.float64).reshape(-1, 3)
    d = pos[:, None, :] - pos[None, :, :]
    r2 = (d**2).sum(-1)[np.triu_indices(len(pos), 1)]
    inv6 = 1.0 / r2**3
    return float(np.sum(4.0 * (inv6 * inv6 - inv6)))


def recording(fn):
    """Wrap ``fn`` so every point it is evaluated at is kept, in call order."""
    seen = []

    def wrapped(x):
        seen.append(np.array(x, copy=True))
        return fn(x)

    return wrapped, seen


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


def test_run_hmc_accepts_initial_position():
    x0 = np.array([-2.903534, -2.903534])
    h = run_hmc(
        styb_tang_2d,
        styb_tang_grad_2d,
        LOW,
        HIGH,
        t_init=5.0,
        epsilon=0.01,
        l_steps=1,
        n_epochs=1,
        steps_per_epoch=1,
        seed=SEED,
        x0=x0,
    )
    assert h.best_pos == pytest.approx(x0)
    assert h.best_val == pytest.approx(GLOBAL_MIN, abs=1e-2)


def test_polish_refines_shifted_quadratic():
    result = polish(
        shifted_quadratic,
        shifted_quadratic_grad,
        np.array([-1.0, -1.0]),
        np.array([1.0, 1.0]),
        np.array([0.9, 0.9]),
        max_fevals=64,
    )

    assert result["best_val"] < 1e-10
    assert result["n_evals"] <= 64
    assert result["best_pos"] == pytest.approx([0.25, -0.4], abs=1e-5)


def test_qmc_polish_refines_best_low_discrepancy_basin():
    def deceptive_basin(x: np.ndarray) -> float:
        shallow = np.sum((x - 0.35) ** 2)
        deep = 0.03 * np.sum((x - np.array([-0.5, 1.0 / 3.0])) ** 2) - 0.75
        return float(min(shallow, deep))

    def deceptive_basin_grad(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        shallow = np.sum((x - 0.35) ** 2)
        deep = 0.03 * np.sum((x - np.array([-0.5, 1.0 / 3.0])) ** 2) - 0.75
        if deep < shallow:
            return 0.06 * (x - np.array([-0.5, 1.0 / 3.0]))
        return 2.0 * (x - 0.35)

    result = qmc_polish(
        deceptive_basin,
        deceptive_basin_grad,
        np.array([-1.0, -1.0]),
        np.array([1.0, 1.0]),
        n_starts=8,
        max_fevals_per_start=32,
        seed=7,
    )

    assert result["best_val"] < -0.74
    assert result["n_polished"] == 8
    assert result["n_evals"] <= 8 * (32 + 1)
    assert result["best_pos"] == pytest.approx([-0.5, 1.0 / 3.0], abs=1e-4)


def test_pyobjective_native_gradient_handle_round_trips():
    bounds = Bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]), 1e-9)
    obj = PyObjective(shifted_quadratic, bounds, grad_fn=shifted_quadratic_grad)

    assert obj.dim == 2
    assert obj.eval(np.array([0.25, -0.4])) == pytest.approx(0.0)
    assert obj.grad(np.array([0.5, -0.25])) == pytest.approx([0.5, 0.3])


def test_gle_langevin_accepts_initial_position():
    x0 = np.array([0.25, -0.4])

    result = gle_langevin(
        shifted_quadratic,
        shifted_quadratic_grad,
        np.array([-1.0, -1.0]),
        np.array([1.0, 1.0]),
        max_fevals=1,
        seed=7,
        x0=x0,
    )

    assert result["n_evals"] == 1
    assert result["best_pos"] == pytest.approx(x0)
    assert result["best_val"] == pytest.approx(0.0)


def test_gle_langevin_accepts_native_objective_handle():
    bounds = Bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]), 1e-9)
    obj = PyObjective(shifted_quadratic, bounds, grad_fn=shifted_quadratic_grad)
    x0 = np.array([0.25, -0.4])

    result = gle_langevin_objective(
        obj,
        max_fevals=1,
        seed=7,
        x0=x0,
    )

    assert result["n_evals"] == 1
    assert result["best_pos"] == pytest.approx(x0)
    assert result["best_val"] == pytest.approx(0.0)


def test_preconditioned_gle_langevin_accepts_initial_position():
    x0 = np.array([0.25, -0.4])

    result = gle_langevin_preconditioned(
        shifted_quadratic,
        shifted_quadratic_grad,
        np.array([-1.0, -1.0]),
        np.array([1.0, 1.0]),
        max_fevals=1,
        seed=7,
        x0=x0,
    )

    assert result["n_evals"] == 1
    assert result["best_pos"] == pytest.approx(x0)
    assert result["best_val"] == pytest.approx(0.0)
    assert result["preconditioner_diag"] == pytest.approx([1.0, 1.0])


def test_preconditioned_gle_langevin_accepts_native_objective_handle():
    bounds = Bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]), 1e-9)
    obj = PyObjective(shifted_quadratic, bounds, grad_fn=shifted_quadratic_grad)
    x0 = np.array([0.25, -0.4])

    result = gle_langevin_preconditioned_objective(
        obj,
        max_fevals=1,
        seed=7,
        x0=x0,
    )

    assert result["n_evals"] == 1
    assert result["best_pos"] == pytest.approx(x0)
    assert result["best_val"] == pytest.approx(0.0)
    assert result["preconditioner_diag"] == pytest.approx([1.0, 1.0])


def test_preconditioned_gle_langevin_honors_probe_count():
    result = gle_langevin_preconditioned(
        anisotropic_quadratic,
        anisotropic_quadratic_grad,
        np.array([-1.0, -1.0]),
        np.array([1.0, 1.0]),
        max_fevals=12,
        seed=7,
        x0=np.array([0.5, 0.5]),
        preconditioner_probes=2,
    )

    assert result["n_preconditioner_grads"] == 4
    assert result["n_evals"] <= 12
    assert result["preconditioner_diag"] == pytest.approx([4.0, 1.0])


def test_qmc_polish_accepts_native_gradient_handle():
    bounds = Bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]), 1e-9)
    obj = PyObjective(shifted_quadratic, bounds, grad_fn=shifted_quadratic_grad)

    result = qmc_polish_objective(
        obj,
        n_starts=8,
        max_fevals_per_start=32,
        seed=7,
        top_k=2,
    )

    assert result["best_val"] < 1e-10
    assert result["n_polished"] == 2
    assert result["best_pos"] == pytest.approx([0.25, -0.4], abs=1e-5)


def test_qmc_best1bin_scout_refines_smooth_basin():
    result = qmc_best1bin_scout(
        smooth_needle,
        np.array([-1.0, -1.0]),
        np.array([1.0, 1.0]),
        max_evals=240,
        seed=0,
        population_size=30,
    )

    assert result["best_val"] < -0.85
    assert result["n_evals"] <= 240
    assert result["n_grads"] == 0
    assert result["n_polished"] == 0
    assert isinstance(result["best_pos"], np.ndarray)


def test_qmc_best1bin_scout_accepts_native_objective_handle():
    bounds = Bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]), 1e-9)
    obj = PyObjective(smooth_needle, bounds)

    result = qmc_best1bin_scout_objective(
        obj,
        max_evals=240,
        seed=0,
        population_size=30,
    )

    assert result["best_val"] < -0.85
    assert result["n_evals"] <= 240
    assert result["n_grads"] == 0
    assert result["n_polished"] == 0
    assert isinstance(result["best_pos"], np.ndarray)


def test_qmc_gsa_global_search_uses_bounded_visiting_distribution():
    result = qmc_gsa_global_search(
        smooth_needle,
        np.array([-1.0, -1.0]),
        np.array([1.0, 1.0]),
        max_evals=240,
        seed=0,
        n_chains=30,
        t_init=1.0,
        q_v=2.62,
        q_a=1.7,
    )

    assert result["best_val"] < -0.85
    assert result["n_evals"] <= 240
    assert result["n_grads"] == 0
    assert result["n_polished"] == 0
    assert isinstance(result["best_pos"], np.ndarray)
    assert np.all(result["best_pos"] >= -1.0)
    assert np.all(result["best_pos"] <= 1.0)


def test_qmc_gsa_global_search_accepts_native_objective_handle():
    bounds = Bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]), 1e-9)
    obj = PyObjective(smooth_needle, bounds)

    result = qmc_gsa_global_search_objective(
        obj,
        max_evals=240,
        seed=0,
        n_chains=30,
        t_init=1.0,
        q_v=2.62,
        q_a=1.7,
    )

    assert result["best_val"] < -0.85
    assert result["n_evals"] <= 240
    assert result["n_grads"] == 0
    assert result["n_polished"] == 0
    assert isinstance(result["best_pos"], np.ndarray)


def test_qmc_trust_region_poll_refines_nearby_basin():
    result = qmc_trust_region_poll(
        shifted_quadratic,
        np.array([-1.0, -1.0]),
        np.array([1.0, 1.0]),
        np.array([0.0, 0.0]),
        max_evals=96,
        seed=7,
        radius_fraction=0.5,
        n_levels=2,
        points_per_level=24,
    )

    assert result["best_val"] < 0.03
    assert result["n_evals"] <= 96
    assert result["n_grads"] == 0
    assert result["n_polished"] == 0
    assert isinstance(result["best_pos"], np.ndarray)


def test_qmc_trust_region_poll_accepts_native_objective_handle():
    bounds = Bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]), 1e-9)
    obj = PyObjective(shifted_quadratic, bounds)

    result = qmc_trust_region_poll_objective(
        obj,
        np.array([0.0, 0.0]),
        max_evals=96,
        seed=7,
        radius_fraction=0.5,
        n_levels=2,
        points_per_level=24,
    )

    assert result["best_val"] < 0.03
    assert result["n_evals"] <= 96
    assert result["n_grads"] == 0
    assert result["n_polished"] == 0


def test_shifted_qmc_polish_exposes_replicated_designs():
    result = shifted_qmc_polish(
        shifted_quadratic,
        shifted_quadratic_grad,
        np.array([-1.0, -1.0]),
        np.array([1.0, 1.0]),
        n_starts=4,
        max_fevals_per_start=32,
        seed=7,
        n_replicates=2,
        top_k=1,
    )

    assert result["best_val"] < 1e-10
    assert result["n_starts"] == 8
    assert result["n_polished"] == 2


def test_low_discrepancy_points_are_bounded_and_deterministic():
    first = low_discrepancy_points(LOW, HIGH, 8)
    second = low_discrepancy_points(LOW, HIGH, 8)

    assert first.shape == (8, 2)
    assert np.all(first >= LOW)
    assert np.all(first <= HIGH)
    assert np.allclose(first, second)


def test_pilot_draws_qmc_are_seeded_and_bounded():
    first = pilot_draws_qmc(8, seed=3)
    second = pilot_draws_qmc(8, seed=3)
    third = pilot_draws_qmc(8, seed=4)

    assert first.shape == (8, 3)
    assert np.all(first[:, 0] > 0.0)
    assert np.all(first[:, 1] > 0.0)
    assert np.all(first[:, 2] > 1.05)
    assert np.all(first[:, 2] < 2.95)
    assert np.allclose(first, second)
    assert not np.allclose(first, third)


def test_run_qmc_sees_deceptive_basin():
    def deceptive_basin(x: np.ndarray) -> float:
        shallow = np.sum((x - 0.35) ** 2)
        deep = 0.03 * np.sum((x - np.array([-0.5, 1.0 / 3.0])) ** 2) - 0.75
        return float(min(shallow, deep))

    h = run_qmc(
        deceptive_basin,
        np.array([-1.0, -1.0]),
        np.array([1.0, 1.0]),
        Gsa(t_init=1.0, q_v=2.2, q_a=1.5),
        n_starts=8,
        n_epochs=2,
        steps_per_epoch=2,
        seed=7,
    )

    assert h.best_val < -0.7


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


DRIVERS = [pytest.param(run, id="run"), pytest.param(run_qmc, id="run_qmc")]
WIDE_PRESETS = [Boltzmann(sigma=10.0), Fast(gamma=10.0), Gsa(t_init=50.0)]


@pytest.mark.parametrize("driver", DRIVERS)
@pytest.mark.parametrize("preset", WIDE_PRESETS, ids=repr)
def test_every_evaluation_stays_in_the_box(driver, preset):
    low = np.array([-3.0, -1.0, 0.5])
    high = np.array([3.0, 2.0, 0.75])
    obj, seen = recording(lambda x: -float(np.sum(x)))

    h = driver(obj, low, high, preset, n_epochs=20, steps_per_epoch=50, seed=3)

    evals = np.array(seen)
    assert np.all((evals >= low) & (evals <= high))
    assert np.all((np.array(h.best_pos) >= low) & (np.array(h.best_pos) <= high))


def test_chemfit_positions_stay_in_the_box():
    n_atoms = 13
    low = np.full(3 * n_atoms, -3.0)
    high = np.full(3 * n_atoms, 3.0)
    init = np.random.default_rng(0).uniform(-1.5, 1.5, size=(n_atoms, 3))
    obj, seen = recording(lj_energy)
    budget = 2000

    h = run(
        obj,
        low,
        high,
        Boltzmann(),
        n_epochs=budget // 100,
        steps_per_epoch=100,
        x0=init.reshape(-1),
    )

    evals = np.array(seen)
    assert evals.shape == (budget + 1, 3 * n_atoms)
    assert np.array_equal(evals[0], init.reshape(-1))
    assert np.all((evals >= low) & (evals <= high))
    assert np.all((np.array(h.best_pos) >= low) & (np.array(h.best_pos) <= high))
    assert h.best_val == min(lj_energy(x) for x in evals)


@pytest.mark.parametrize("driver", DRIVERS)
@pytest.mark.parametrize("preset", WIDE_PRESETS, ids=repr)
def test_x0_is_the_first_evaluation(driver, preset):
    low = np.array([-3.0, -1.0, 0.5])
    high = np.array([3.0, 2.0, 0.75])
    x0 = np.array([0.1, -1.0, 0.75])
    obj, seen = recording(lambda x: -float(np.sum(x)))

    driver(obj, low, high, preset, n_epochs=5, steps_per_epoch=20, seed=3, x0=x0)

    evals = np.array(seen)
    assert np.array_equal(evals[0], x0)
    assert np.all((evals >= low) & (evals <= high))


@pytest.mark.parametrize("driver", DRIVERS)
@pytest.mark.parametrize(
    ("x0", "message"),
    [
        ([0.0], "x0 must have the same length"),
        ([0.0, np.nan], "x0 must be finite"),
        ([-np.inf, 0.0], "x0 must be finite"),
        ([0.0, 1.5], "outside the box"),
    ],
)
def test_invalid_x0_raises_value_error(driver, x0, message):
    with pytest.raises(ValueError, match=message):
        driver(
            styb_tang_2d,
            np.array([-1.0, -1.0]),
            np.array([1.0, 1.0]),
            Boltzmann(),
            n_epochs=1,
            steps_per_epoch=1,
            x0=np.array(x0, dtype=np.float64),
        )


@pytest.mark.parametrize("driver", DRIVERS)
@pytest.mark.parametrize(
    ("low", "high", "message"),
    [
        ([-1.0, -1.0], [1.0], "same length"),
        ([], [], "at least one dimension"),
        ([-1.0, np.nan], [1.0, 1.0], "finite"),
        ([-1.0, -1.0], [1.0, np.inf], "finite"),
        ([-1.0, 2.0], [1.0, 1.0], "must not exceed"),
        ([-1e308], [1e308], "high - low must be finite at dimension 0"),
        (
            [0.0, -np.finfo(float).max],
            [1.0, np.finfo(float).max],
            "high - low must be finite at dimension 1",
        ),
    ],
)
def test_invalid_box_raises_value_error(driver, low, high, message):
    with pytest.raises(ValueError, match=message):
        driver(
            styb_tang_2d,
            np.array(low, dtype=np.float64),
            np.array(high, dtype=np.float64),
            Boltzmann(),
            n_epochs=1,
            steps_per_epoch=1,
        )


ARRAY_LIKES = [
    pytest.param(lambda a: a.tolist(), id="list"),
    pytest.param(lambda a: [int(v) for v in a], id="int-list"),
    pytest.param(lambda a: tuple(a.tolist()), id="tuple"),
    pytest.param(lambda a: a.astype(np.float32), id="float32"),
    pytest.param(lambda a: a.astype(np.int64), id="int64"),
    pytest.param(lambda a: np.repeat(a, 2)[::2], id="strided"),
    pytest.param(lambda a: a[::-1].copy()[::-1], id="reversed"),
    pytest.param(lambda a: np.stack([a, a + 7.0], axis=1)[:, 0], id="column"),
]


@pytest.mark.parametrize("driver", DRIVERS)
@pytest.mark.parametrize("convert", ARRAY_LIKES)
def test_array_like_inputs_match_float64_arrays(driver, convert):
    low = np.array([-2.0, -1.0, 0.0, -3.0])
    high = np.array([2.0, 3.0, 1.0, 3.0])
    x0 = np.array([1.0, -1.0, 0.0, 2.0])
    runs = []
    for lo, hi, start in [(low, high, x0), map(convert, (low, high, x0))]:
        obj, seen = recording(lambda x: float(np.sum(x**2)))
        h = driver(
            obj,
            lo,
            hi,
            Fast(gamma=2.0),
            n_epochs=3,
            steps_per_epoch=10,
            seed=5,
            x0=start,
        )
        runs.append(((h.best_val, h.best_pos, h.total_accepted), np.array(seen)))

    (ref, ref_seen), (got, got_seen) = runs
    assert got == ref
    assert np.array_equal(got_seen, ref_seen)
    assert np.array_equal(got_seen[0], x0)


@pytest.mark.parametrize("driver", DRIVERS)
@pytest.mark.parametrize("order", ["C", "F"])
def test_multidimensional_x0_is_flattened_in_c_order(driver, order):
    n_atoms = 4
    init = np.random.default_rng(0).uniform(-1.5, 1.5, size=(n_atoms, 3))
    obj, seen = recording(lj_energy)

    driver(
        obj,
        np.full(3 * n_atoms, -3.0),
        np.full(3 * n_atoms, 3.0),
        Boltzmann(),
        n_epochs=2,
        steps_per_epoch=10,
        x0=np.asarray(init, order=order),
    )

    assert np.array_equal(seen[0], init.reshape(-1))


@pytest.mark.parametrize("driver", DRIVERS)
@pytest.mark.parametrize(
    ("arg", "value", "message"),
    [
        ("x0", "ab", "x0 must be an array of numbers"),
        ("x0", [0.0, "a"], "x0 must be an array of numbers"),
        ("x0", [[0.0], [0.0, 0.0]], "x0 must be an array of numbers"),
        ("x0", object(), "x0 must be an array of numbers"),
        ("x0", b"ab", "x0 must be an array of numbers"),
        ("x0", np.zeros((2, 2)), "x0 must have the same length"),
        ("low", "ab", "low must be an array of numbers"),
        ("low", [[-1.0, -1.0]], "low must be one-dimensional"),
        ("low", -np.ones((2, 1), dtype=np.float32), "low must be one-dimensional"),
        ("high", 1.0, "high must be one-dimensional"),
        ("high", np.ones((2, 1), dtype=np.int64), "high must be one-dimensional"),
        ("high", [np.ones(1), np.ones(1)], "high must be one-dimensional"),
    ],
)
def test_unreadable_inputs_raise_value_error(driver, arg, value, message):
    args = {"low": np.array([-1.0, -1.0]), "high": np.array([1.0, 1.0]), "x0": None}
    args[arg] = value
    with pytest.raises(ValueError, match=message):
        driver(
            styb_tang_2d,
            args["low"],
            args["high"],
            Boltzmann(),
            n_epochs=1,
            steps_per_epoch=1,
            x0=args["x0"],
        )


@pytest.mark.parametrize("driver", DRIVERS)
def test_the_shape_of_an_input_does_not_depend_on_its_dtype(driver):
    obj, seen = recording(styb_tang_2d)
    budget = {"n_epochs": 1, "steps_per_epoch": 1}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="low must be one-dimensional"):
            driver(obj, -np.ones((2, 1), np.float32), np.ones(2), Boltzmann(), **budget)
        x0 = np.full((2, 1), 0.5, np.float32)
        driver(obj, -np.ones(2), np.ones(2), Boltzmann(), x0=x0, **budget)

    assert [str(w.message) for w in caught] == []
    assert np.array_equal(seen[0], [0.5, 0.5])


@pytest.mark.parametrize("driver", DRIVERS)
@pytest.mark.parametrize("preset", [Boltzmann(), Fast(), Gsa()], ids=repr)
@pytest.mark.parametrize("nan", [np.nan, -np.nan], ids=["nan", "negative-nan"])
def test_a_nan_at_x0_neither_freezes_the_chain_nor_wins(driver, preset, nan):
    x0 = np.array([0.5, 0.5])
    obj, seen = recording(
        lambda x: nan if np.array_equal(x, x0) else float(np.sum(x**2))
    )

    h = driver(
        obj, -np.ones(2), np.ones(2), preset, n_epochs=10, steps_per_epoch=100, x0=x0
    )

    assert np.array_equal(seen[0], x0)
    assert h.total_accepted > 0
    assert h.best_val == min(float(np.sum(x**2)) for x in seen[1:])


def nan_at(x0):
    """NaN at ``x0``; raises elsewhere, which the drivers read as +inf."""

    def f(x):
        if np.array_equal(x, x0):
            return np.nan
        raise RuntimeError("undefined away from x0")

    return f


@pytest.mark.parametrize("driver", DRIVERS)
@pytest.mark.parametrize("preset", [Boltzmann(), Fast(), Gsa()], ids=repr)
def test_an_evaluation_that_raised_beats_a_nan_at_x0(driver, preset):
    x0 = np.array([0.5, 0.5])

    h = driver(
        nan_at(x0),
        -np.ones(2),
        np.ones(2),
        preset,
        n_epochs=2,
        steps_per_epoch=10,
        x0=x0,
    )

    assert h.total_accepted > 0
    assert h.best_val == np.inf


def test_run_qmc_keeps_a_start_that_raised_over_a_nan_at_x0():
    x0 = np.array([0.5, 0.5])

    h = run_qmc(
        nan_at(x0),
        -np.ones(2),
        np.ones(2),
        Boltzmann(),
        n_starts=3,
        n_epochs=1,
        steps_per_epoch=0,
        x0=x0,
    )

    assert h.best_val == np.inf
    assert not np.array_equal(h.best_pos, x0)


@pytest.mark.parametrize("driver", DRIVERS)
@pytest.mark.parametrize("preset", WIDE_PRESETS, ids=repr)
@pytest.mark.parametrize(
    "x0", [None, np.array([1.5, 0.5, -2.0])], ids=["uniform", "x0"]
)
def test_equal_endpoints_fix_the_coordinate(driver, preset, x0):
    low = np.array([-2.0, 0.5, -2.0])
    high = np.array([2.0, 0.5, 2.0])
    obj, seen = recording(lambda x: float(np.sum(x**2)))

    h = driver(obj, low, high, preset, n_epochs=10, steps_per_epoch=50, seed=1, x0=x0)

    evals = np.array(seen)
    assert np.all(evals[:, 1] == 0.5)
    assert h.best_pos[1] == 0.5

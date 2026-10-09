"""Pytest suite for the new Python algebra surface (Boltzmann / Fast / Gsa
preset constructors plus run() driver). Replaces the legacy
test_funcs / test_mcsamplers / test_quench suites."""

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
    global_optimize,
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
    cluster_search,
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


def lj_cluster_energy(x: np.ndarray) -> float:
    """Reduced Lennard-Jones energy of a flattened point set."""
    p = np.asarray(x, dtype=np.float64).reshape(-1, 3)
    r = np.linalg.norm(p[:, None] - p[None], axis=-1)[np.triu_indices(len(p), 1)]
    with np.errstate(divide="ignore", invalid="ignore"):
        energy = float(np.sum(4.0 * (r**-12 - r**-6)))
    return energy if np.isfinite(energy) else float("inf")


class Recorder:
    """Objective wrapper that keeps every point it is asked to evaluate."""

    def __init__(self, fn):
        self.fn = fn
        self.points = []

    def __call__(self, x):
        self.points.append(np.array(x, dtype=np.float64, copy=True))
        return self.fn(x)


PRESETS = [
    Boltzmann(t_init=1.0, sigma=0.5),
    Fast(t_init=1.0, gamma=0.5),
    Gsa(t_init=1.0, q_v=2.62, q_a=1.7),
]


@pytest.mark.parametrize("preset", PRESETS, ids=repr)
def test_run_evaluates_only_inside_the_box(preset):
    # Thirteen atoms in a [-3, 3] box: the ChemFit positions benchmark.
    low, high = np.full(39, -3.0), np.full(39, 3.0)
    objective = Recorder(lj_cluster_energy)
    h = run(objective, low, high, preset, n_epochs=20, steps_per_epoch=100, seed=0)
    points = np.array(objective.points)
    assert len(points) == 1 + 20 * 100
    assert np.all(points >= low) and np.all(points <= high)
    best = np.asarray(h.best_pos)
    assert np.all(best >= low) and np.all(best <= high)


@pytest.mark.parametrize("preset", PRESETS, ids=repr)
def test_run_qmc_evaluates_only_inside_the_box(preset):
    low, high = np.full(6, -1.0), np.full(6, 1.0)
    objective = Recorder(lambda x: float(np.sum(x * x)))
    run_qmc(objective, low, high, preset, n_starts=3, n_epochs=5, steps_per_epoch=20, seed=1)
    points = np.array(objective.points)
    assert len(points) == 3 * (1 + 5 * 20)
    assert np.all(points >= low) and np.all(points <= high)


@pytest.mark.parametrize("preset", PRESETS, ids=repr)
def test_run_starts_from_x0(preset):
    x0 = np.array([-2.903534, -2.903534])
    objective = Recorder(styb_tang_2d)
    h = run(objective, LOW, HIGH, preset, n_epochs=2, steps_per_epoch=5, seed=SEED, x0=x0)
    assert objective.points[0] == pytest.approx(x0)
    assert h.best_val <= styb_tang_2d(x0)


def test_run_qmc_x0_replaces_the_first_start():
    x0 = np.array([0.25, -0.4])
    objective = Recorder(shifted_quadratic)
    h = run_qmc(
        objective,
        np.array([-1.0, -1.0]),
        np.array([1.0, 1.0]),
        Boltzmann(t_init=0.1, sigma=0.1),
        n_starts=4,
        n_epochs=1,
        steps_per_epoch=1,
        seed=7,
        x0=x0,
    )
    assert objective.points[0] == pytest.approx(x0)
    assert h.best_val == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("x0", "message"),
    [
        (np.array([5.5, 0.0]), "outside"),
        (np.array([0.0]), "length"),
        (np.array([np.nan, 0.0]), "not finite"),
    ],
)
def test_run_refuses_a_bad_x0(x0, message):
    with pytest.raises(ValueError, match=message):
        run(styb_tang_2d, LOW, HIGH, Boltzmann(), n_epochs=1, steps_per_epoch=1, x0=x0)


@pytest.mark.parametrize(
    "preset",
    [Boltzmann(sigma=0.0), Fast(t_init=-1.0), Gsa(q_v=3.5)],
    ids=repr,
)
def test_run_refuses_invalid_preset_parameters(preset):
    with pytest.raises(ValueError):
        run(styb_tang_2d, LOW, HIGH, preset, n_epochs=1, steps_per_epoch=1)


def test_global_optimize_starts_from_x0_and_stays_in_the_box():
    low, high = np.full(39, -3.0), np.full(39, 3.0)
    x0 = np.random.default_rng(1).uniform(-1.5, 1.5, 39)
    objective = Recorder(lj_cluster_energy)
    result = global_optimize(objective, low, high, budget=300, seed=0, x0=x0)
    points = np.array(objective.points)
    assert points[0] == pytest.approx(x0)
    assert result["n_evals"] == len(points) <= 300
    assert np.all(points >= low) and np.all(points <= high)
    assert result["best_val"] <= lj_cluster_energy(x0)


def test_global_optimize_refuses_x0_outside_the_box():
    with pytest.raises(ValueError, match="outside"):
        global_optimize(styb_tang_2d, LOW, HIGH, budget=50, x0=np.array([0.0, 7.0]))


def test_run_takes_array_likes_of_any_shape_and_stride():
    # Positions arrive as (n_atoms, 3); bounds as lists; slices are strided.
    n_atoms = 13
    x0 = np.random.default_rng(3).uniform(-1.5, 1.5, (n_atoms, 3))
    wide = np.zeros((n_atoms, 6))
    wide[:, ::2] = x0
    for start in [x0, x0.tolist(), wide[:, ::2], x0.astype(np.float32)]:
        objective = Recorder(lj_cluster_energy)
        h = run(
            objective,
            [-3.0] * (3 * n_atoms),
            np.full((n_atoms, 3), 3.0),
            Boltzmann(),
            n_epochs=2,
            steps_per_epoch=5,
            seed=0,
            x0=start,
        )
        assert objective.points[0] == pytest.approx(np.asarray(start, dtype=np.float64).ravel(), abs=1e-6)
        assert len(h.best_pos) == 3 * n_atoms
    run_qmc(
        lj_cluster_energy,
        np.full((n_atoms, 3), -3.0),
        np.full((n_atoms, 3), 3.0),
        Gsa(),
        n_starts=2,
        n_epochs=1,
        steps_per_epoch=3,
        x0=x0,
    )


@pytest.mark.parametrize("driver", [run, run_qmc])
@pytest.mark.parametrize(
    "convert",
    [
        lambda a: a.tolist(),
        lambda a: tuple(a.tolist()),
        lambda a: a.astype(np.float32),
        lambda a: a.astype(np.int64),
    ],
    ids=["list", "tuple", "float32", "int64"],
)
def test_run_reads_lists_tuples_and_other_dtypes_as_float64(driver, convert):
    # Whole numbers, which every one of these types holds exactly.
    low, high, x0 = np.array([-2.0, -1.0]), np.array([2.0, 3.0]), np.array([1.0, 0.0])

    def evaluated(lo, hi, start):
        objective = Recorder(styb_tang_2d)
        driver(objective, lo, hi, Boltzmann(), n_epochs=2, steps_per_epoch=5, seed=3, x0=start)
        return np.array(objective.points)

    assert np.array_equal(evaluated(convert(low), convert(high), convert(x0)), evaluated(low, high, x0))


@pytest.mark.parametrize("driver", [run, run_qmc])
@pytest.mark.parametrize(
    "view",
    [
        lambda a: np.repeat(a, 2)[::2],
        lambda a: a[::-1].copy()[::-1],
        lambda a: np.stack([a, a + 7.0], axis=1)[:, 0],
    ],
    ids=["strided", "reversed", "column"],
)
def test_run_reads_non_contiguous_views(driver, view):
    # A scipy-style (n, 2) bounds array hands low and high over as column views.
    low, high, x0 = np.array([-2.0, -1.0]), np.array([2.0, 3.0]), np.array([1.0, 0.0])
    assert not any(view(a).flags.c_contiguous for a in (low, high, x0))

    def evaluated(lo, hi, start):
        objective = Recorder(styb_tang_2d)
        driver(objective, lo, hi, Boltzmann(), n_epochs=2, steps_per_epoch=5, seed=3, x0=start)
        return np.array(objective.points)

    assert np.array_equal(evaluated(view(low), view(high), view(x0)), evaluated(low, high, x0))


@pytest.mark.parametrize("budget", [1, 2, 100, 1999, 2000])
def test_run_max_evals_spends_exactly_the_budget(budget):
    objective = Recorder(lj_cluster_energy)
    run(
        objective,
        np.full(39, -3.0),
        np.full(39, 3.0),
        Boltzmann(),
        n_epochs=-(-budget // 100),
        steps_per_epoch=100,
        seed=1,
        max_evals=budget,
    )
    assert len(objective.points) == budget


@pytest.mark.parametrize("budget", [3, 10, 97])
def test_run_qmc_max_evals_spends_exactly_the_budget(budget):
    objective = Recorder(shifted_quadratic)
    run_qmc(
        objective,
        np.array([-1.0, -1.0]),
        np.array([1.0, 1.0]),
        Fast(),
        n_starts=4,
        n_epochs=10,
        steps_per_epoch=10,
        seed=2,
        max_evals=budget,
    )
    assert len(objective.points) == budget


def test_run_refuses_max_evals_of_zero():
    with pytest.raises(ValueError, match="max_evals"):
        run(styb_tang_2d, LOW, HIGH, Boltzmann(), max_evals=0)


def test_run_leaves_a_nan_start():
    def nan_at_origin(x):
        return float("nan") if np.all(x == 0.0) else styb_tang_2d(x)

    h = run(nan_at_origin, LOW, HIGH, Boltzmann(), n_epochs=5, steps_per_epoch=100, seed=4, x0=np.zeros(2))
    assert np.isfinite(h.best_val)
    assert h.total_accepted > 0


def test_keyboard_interrupt_in_the_objective_ends_the_run():
    calls = []

    def interrupted(x):
        calls.append(1)
        if len(calls) == 10:
            raise KeyboardInterrupt
        return styb_tang_2d(x)

    with pytest.raises(KeyboardInterrupt):
        run(interrupted, LOW, HIGH, Boltzmann(), n_epochs=10, steps_per_epoch=100, seed=0)
    assert len(calls) == 10
    calls.clear()
    with pytest.raises(KeyboardInterrupt):
        global_optimize(interrupted, LOW, HIGH, budget=500, seed=0)
    assert len(calls) == 10


def test_a_non_number_from_the_objective_raises_type_error():
    with pytest.raises(TypeError, match="must return a float"):
        run(lambda x: None, LOW, HIGH, Boltzmann(), n_epochs=1, steps_per_epoch=3)


def test_objective_exceptions_are_scored_and_reported_once():
    calls = []

    def flaky(x):
        calls.append(np.array(x, copy=True))
        if len(calls) % 7 == 3:
            raise ValueError("solver did not converge")
        return styb_tang_2d(x)

    with pytest.warns(RuntimeWarning, match=r"raised 72 exception.*solver did not converge"):
        h = run(flaky, LOW, HIGH, Boltzmann(), n_epochs=5, steps_per_epoch=100, seed=0)
    assert len(calls) == 501
    assert np.isfinite(h.best_val)
    assert not any(np.array_equal(calls[i], h.best_pos) for i in range(2, 501, 7))


def test_gsa_near_q_v_three_spends_the_whole_schedule():
    objective = Recorder(lj_cluster_energy)
    run(objective, np.full(39, -3.0), np.full(39, 3.0), Gsa(t_init=1.0, q_v=2.999, q_a=1.7), n_epochs=20, steps_per_epoch=100, seed=0)
    points = np.array(objective.points)
    assert len(points) == 1 + 20 * 100
    assert np.all(np.isfinite(points))


def test_no_evaluation_rounds_past_the_upper_wall():
    low, high = np.array([-3.0, -1.0]), np.array([0.7, 0.3])
    objective = Recorder(styb_tang_2d)
    run(objective, low, high, Gsa(t_init=1.0, q_v=2.99, q_a=1.7), n_epochs=20, steps_per_epoch=100, seed=9, x0=high.copy())
    points = np.array(objective.points)
    assert np.all(points >= low) and np.all(points <= high)
    objective = Recorder(lambda x: float(np.sum((x - 1.0) ** 2)))
    global_optimize(objective, np.full(10, -3.0), np.full(10, 0.7), budget=600, seed=0)
    points = np.array(objective.points)
    assert np.all(points >= -3.0) and np.all(points <= 0.7)


def test_global_optimize_calls_the_gradient_only_inside_the_box():
    # The minimum sits outside the box, so descents push against the wall.
    low, high = np.full(4, -3.0), np.full(4, 3.0)
    objective = Recorder(lambda x: float(np.sum((x - 5.0) ** 2)))
    gradient = Recorder(lambda x: 2.0 * (x - 5.0))
    result = global_optimize(objective, low, high, budget=400, seed=0, grad_fn=gradient)
    for points in (np.array(objective.points), np.array(gradient.points)):
        assert np.all(points >= low) and np.all(points <= high)
    assert result["best_pos"] == pytest.approx(high, abs=1e-6)


def test_global_optimize_takes_bounds_of_any_shape():
    x0 = np.random.default_rng(5).uniform(-1.5, 1.5, (13, 3))
    objective = Recorder(lj_cluster_energy)
    global_optimize(objective, np.full((13, 3), -3.0), np.full((13, 3), 3.0), budget=50, seed=0, x0=x0)
    assert objective.points[0] == pytest.approx(x0.ravel())


@pytest.mark.parametrize("x0", [np.array([0.9, -0.3]), np.array([5.0, 5.0]), np.array([-5.0, 1.0])])
def test_global_optimize_evaluates_x0_exactly(x0):
    objective = Recorder(styb_tang_2d)
    result = global_optimize(objective, LOW, HIGH, budget=1, seed=0, x0=x0)
    assert len(objective.points) == 1
    assert np.array_equal(objective.points[0], x0)
    assert np.array_equal(result["best_pos"], x0)
    assert result["best_val"] == styb_tang_2d(x0)


def test_qmc_gsa_global_search_starts_from_x0():
    x0 = np.array([0.5, -0.5])
    objective = Recorder(smooth_needle)
    result = qmc_gsa_global_search(objective, np.array([-1.0, -1.0]), np.array([1.0, 1.0]), max_evals=120, seed=0, x0=x0)
    assert np.array_equal(objective.points[0], x0)
    assert result["best_val"] <= smooth_needle(x0)


def lj_cluster_gradient(x: np.ndarray) -> np.ndarray:
    p = np.asarray(x, dtype=np.float64).reshape(-1, 3)
    d = p[:, None] - p[None]
    r2 = np.sum(d * d, axis=-1)
    np.fill_diagonal(r2, np.inf)
    inv6 = r2**-3
    coef = 24.0 * (2.0 * inv6 * inv6 - inv6) / r2
    return -np.sum(coef[:, :, None] * d, axis=1).ravel()


def test_cluster_search_ends_on_keyboard_interrupt():
    calls = []

    def interrupted(x):
        calls.append(1)
        if len(calls) == 50:
            raise KeyboardInterrupt
        return lj_cluster_energy(x)

    with pytest.raises(KeyboardInterrupt):
        cluster_search(interrupted, lj_cluster_gradient, 13, 5000, seed=0)
    assert len(calls) == 50


def test_cluster_search_takes_array_like_gradients():
    def as_list(x):
        return lj_cluster_gradient(x).tolist()

    out = cluster_search(lj_cluster_energy, as_list, 13, 3000, seed=0)
    assert out["best_energy"] < -40.0


def test_cluster_search_rejects_a_non_number_energy():
    with pytest.raises(TypeError, match="must return a float"):
        cluster_search(lambda x: None, lj_cluster_gradient, 13, 200, seed=0)


def test_global_optimize_keeps_x0_when_nothing_is_finite():
    x0 = np.random.default_rng(6).uniform(-1.0, 1.0, 39)
    for budget in (1, 2, 50):
        result = global_optimize(lambda x: float("inf"), np.full(39, -3.0), np.full(39, 3.0), budget=budget, seed=0, x0=x0)
        assert np.array_equal(result["best_pos"], x0)


@pytest.mark.parametrize("q_v", [1.0001, 1.001, 1.005])
def test_gsa_near_q_v_one_takes_finite_moving_steps(q_v):
    objective = Recorder(styb_tang_2d)
    h = run(objective, LOW, HIGH, Gsa(t_init=1.0, q_v=q_v, q_a=1.7), n_epochs=5, steps_per_epoch=100, seed=0, x0=np.zeros(2))
    points = np.array(objective.points)
    assert len(points) == 501
    assert len({tuple(p) for p in points}) > 400
    assert h.best_val < styb_tang_2d(np.zeros(2))


def test_qmc_gsa_global_search_refuses_q_v_of_one():
    with pytest.raises(ValueError, match="q_v"):
        qmc_gsa_global_search(smooth_needle, np.array([-1.0, -1.0]), np.array([1.0, 1.0]), max_evals=50, q_v=1.0)


def test_a_walk_started_on_an_infeasible_plateau_finds_the_feasible_region():
    def half_plane(x):
        return float("nan") if x[0] > 1.0 else styb_tang_2d(x)

    for seed in range(10):
        h = run(half_plane, LOW, HIGH, Boltzmann(), n_epochs=10, steps_per_epoch=100, seed=seed, x0=np.array([4.5, 0.0]))
        assert np.isfinite(h.best_val), seed


@pytest.mark.parametrize("bad", [0, -1, 2.5])
def test_run_refuses_a_max_evals_that_is_not_a_positive_whole_number(bad):
    with pytest.raises(ValueError, match="max_evals"):
        run(styb_tang_2d, LOW, HIGH, Boltzmann(), max_evals=bad)


def test_run_hmc_takes_array_like_bounds():
    h = run_hmc(
        styb_tang_2d,
        styb_tang_grad_2d,
        [-5.0, -5.0],
        [5.0, 5.0],
        n_epochs=2,
        steps_per_epoch=5,
        x0=[-2.903534, -2.903534],
    )
    assert h.best_val == pytest.approx(GLOBAL_MIN, abs=1e-2)


def test_global_optimize_never_calls_with_a_non_finite_coordinate():
    def partly_infeasible(x):
        assert np.all(np.isfinite(x)), x
        return float("inf") if x[0] > 0.0 else float(np.sum(x * x))

    for grad_fn in (None, lambda x: 2.0 * x):
        objective = Recorder(partly_infeasible)
        global_optimize(objective, np.full(4, -3.0), np.full(4, 3.0), budget=4000, seed=1, grad_fn=grad_fn)
        assert np.all(np.isfinite(np.array(objective.points)))


def test_qmc_gsa_global_search_never_rounds_past_the_upper_wall():
    low, high = np.full(3, -3.0), np.full(3, 0.7)
    objective = Recorder(lambda x: float(np.sum((x - 1.0) ** 2)))
    qmc_gsa_global_search(objective, low, high, max_evals=2000, seed=0)
    points = np.array(objective.points)
    assert np.all(points >= low) and np.all(points <= high)


@pytest.mark.parametrize(
    ("low", "high"),
    [
        (np.array([1.0, -1.0]), np.array([-1.0, 1.0])),
        (np.array([-np.inf, -1.0]), np.array([1.0, 1.0])),
        (np.array([np.nan, -1.0]), np.array([1.0, 1.0])),
        (np.array([-1.8e308, -1.0]), np.array([1.8e308, 1.0])),
        (np.array([-6e307, -1.0]), np.array([6e307, 1.0])),
        (np.full(5, -2e307), np.full(5, 2e307)),
    ],
)
def test_every_box_driver_refuses_bounds_it_cannot_use(low, high):
    objective = Recorder(lambda x: float(np.sum(np.asarray(x) ** 2)))
    calls = [
        lambda: run(objective, low, high, Boltzmann(), n_epochs=1, steps_per_epoch=2),
        lambda: run_qmc(objective, low, high, Boltzmann(), n_starts=2, n_epochs=1, steps_per_epoch=2),
        lambda: global_optimize(objective, low, high, budget=20),
        lambda: qmc_gsa_global_search(objective, low, high, max_evals=20),
    ]
    for call in calls:
        with pytest.raises(ValueError):
            call()
    assert objective.points == []


@pytest.mark.parametrize("driver", [run, run_qmc])
@pytest.mark.parametrize(
    ("low", "high", "message"),
    [
        ([-1.0, np.nan], [1.0, 1.0], r"low\[1\] = NaN must be finite"),
        ([-1.0, -1.0], [1.0, np.inf], r"high\[1\] = inf must be finite"),
        ([-np.inf, -1.0], [np.nan, 1.0], r"low\[0\] = -inf must be finite"),
    ],
)
def test_run_names_the_bound_that_is_not_finite(driver, low, high, message):
    with pytest.raises(ValueError, match=message):
        driver(styb_tang_2d, low, high, Boltzmann(), n_epochs=1, steps_per_epoch=1)


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [("n_starts", {"n_starts": 0}), ("n_epochs", {"n_epochs": -1}), ("steps_per_epoch", {"steps_per_epoch": 1.5})],
)
def test_run_qmc_refuses_counts_it_cannot_run(name, kwargs):
    with pytest.raises(ValueError, match=name):
        run_qmc(shifted_quadratic, np.array([-1.0, -1.0]), np.array([1.0, 1.0]), Boltzmann(), **kwargs)


def test_cluster_search_probe_follows_the_callback_rules():
    calls = []

    def fails_first(x):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("budget counter tripped")
        return lj_cluster_energy(x)

    with pytest.warns(RuntimeWarning, match="budget counter tripped"):
        out = cluster_search(fails_first, lj_cluster_gradient, 13, 2000, seed=0)
    assert np.isfinite(out["best_energy"])

    with pytest.raises(TypeError, match="grad_fn must return"):
        cluster_search(lj_cluster_energy, lambda x: None, 13, 200, seed=0)

    def infinite_far_out(x):
        p = np.asarray(x).reshape(-1, 3)
        return float("inf") if np.max(np.abs(p)) > 1.2 else lj_cluster_energy(x)

    out = cluster_search(infinite_far_out, lj_cluster_gradient, 13, 2000, seed=1)
    assert isinstance(out["best_energy"], float)

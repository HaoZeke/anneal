"""anneal: simulated annealing components on the eindir typed primitives.

Public API:
  - Boltzmann(t_init, sigma): logarithmic cooling + Gaussian + Metropolis.
  - Fast(t_init, gamma): reciprocal cooling + Cauchy + Metropolis.
  - Gsa(t_init, q_v, q_a): Tsallis cooling + Tsallis visit + Tsallis accept.
  - run(obj_fn, low, high, preset, n_epochs, steps_per_epoch, seed, x0): SA
    loop; every proposal is reflected into [low, high] and the walk starts at
    x0 when one is given.
  - History, EpochLine: returned by `run`.
  - Config.recommended(n) / Config.for_cluster(n), Ledger(budget),
    cluster_search(obj_fn, grad_fn, n, budget, seed, recommended): measured
    cluster-search layer.
  - fit_anneal, fit_chemfit, run_benchmark: gradient-free ChemFit bridges.

The IISE-manuscript composition laws L1-L4 are enforced inside the Rust
SaVariant::checked constructor; preset constructors call it under the hood.
"""

import numpy as np

from anneal._core import (
    BasinBias,
    Boltzmann,
    Bounds,
    Config,
    EpochLine,
    Fast,
    Gsa,
    History,
    Ledger,
    PyObjective,
    __version__,
    cluster_search as _core_cluster_search,
    low_discrepancy_points as _core_low_discrepancy_points,
    pilot_draws_qmc as _core_pilot_draws_qmc,
    polish as _core_polish,
    qmc_best1bin_scout as _core_qmc_best1bin_scout,
    qmc_best1bin_scout_objective as _core_qmc_best1bin_scout_objective,
    qmc_gsa_global_search as _core_qmc_gsa_global_search,
    qmc_gsa_global_search_objective as _core_qmc_gsa_global_search_objective,
    qmc_polish as _core_qmc_polish,
    qmc_polish_objective as _core_qmc_polish_objective,
    qmc_trust_region_poll as _core_qmc_trust_region_poll,
    qmc_trust_region_poll_objective as _core_qmc_trust_region_poll_objective,
    shifted_qmc_polish as _core_shifted_qmc_polish,
    additive_independence as _core_additive_independence,
    estimate_gle_omega0 as _core_estimate_gle_omega0,
    gle_langevin as _core_gle_langevin,
    gle_langevin_objective as _core_gle_langevin_objective,
    gle_langevin_preconditioned as _core_gle_langevin_preconditioned,
    gle_langevin_preconditioned_objective as _core_gle_langevin_preconditioned_objective,
    global_optimize as _core_global_optimize,
    global_optimize_objective as _core_global_optimize_objective,
    dmc_population_optimize as _core_dmc_population_optimize,
    gpmd_optimize as _core_gpmd_optimize,
    amsa_optimize as _core_amsa_optimize,
    bfwt_optimize as _core_bfwt_optimize,
    run as _core_run,
    run_hmc as _core_run_hmc,
    run_qmc as _core_run_qmc,
)
from anneal.device import DeviceHistory, EnsembleHistory, run_device, run_ensemble
from anneal.tvm_ffi import (
    TvmFfiTensorMetadata,
    tvm_ffi_tensor,
    tvm_ffi_tensor_metadata,
    tvm_ffi_tensors_from_history,
)


def _flat(value):
    """A contiguous float64 vector from any array-like, flattened in C order."""
    return np.ascontiguousarray(np.asarray(value, dtype=np.float64).reshape(-1))


def _flat_or_none(value):
    return None if value is None else _flat(value)


def _count(name, value, minimum):
    """A whole number of at least ``minimum``, or a ValueError naming it."""
    count = int(value)
    if count != value or count < minimum:
        msg = f"{name} must be a whole number of at least {minimum}, got {value!r}"
        raise ValueError(msg)
    return count


def _max_evals(value):
    """``None``, or a positive whole number of objective calls."""
    if value is None:
        return None
    count = int(value)
    if count != value or count < 1:
        msg = f"max_evals must be a whole number of calls, at least 1, got {value!r}"
        raise ValueError(msg)
    return count


def run(
    obj_fn,
    low,
    high,
    preset,
    n_epochs: int = 100,
    steps_per_epoch: int = 200,
    seed: int = 42,
    x0=None,
    max_evals: int | None = None,
):
    """Simulated annealing inside the box ``[low, high]``.

    Every proposal is mirror-reflected into the box, so ``obj_fn`` is only
    called inside it and ``best_pos`` lies inside it.

    Args:
      obj_fn: callable ``f(numpy.ndarray) -> float`` on the flat vector.
      low, high: box bounds, any array-like of one shape; flattened in C order.
      preset: ``Boltzmann()``, ``Fast()`` or ``Gsa()``.
      n_epochs, steps_per_epoch: the cooling schedule and its length.
      seed: RNG seed.
      x0: optional start inside the box, same size as ``low``; a uniform draw
        otherwise.
      max_evals: optional cap on calls to ``obj_fn``, the start included. The
        cooling schedule still spans ``n_epochs``; the cap ends the run inside
        the epoch where it falls.

    The start costs one call, so a run makes ``1 + n_epochs * steps_per_epoch``
    calls, or ``max_evals`` when that is smaller. An ordinary exception raised
    by ``obj_fn`` is scored as ``+inf`` and reported once as a
    ``RuntimeWarning``, unless no call returned at all, when the first one is
    re-raised; ``KeyboardInterrupt`` ends the run and is re-raised, as is a
    ``TypeError`` when ``obj_fn`` returns something that is not a number.
    """
    return _core_run(
        obj_fn,
        _flat(low),
        _flat(high),
        preset,
        _count("n_epochs", n_epochs, 0),
        _count("steps_per_epoch", steps_per_epoch, 0),
        int(seed),
        _flat_or_none(x0),
        _max_evals(max_evals),
    )


def run_qmc(
    obj_fn,
    low,
    high,
    preset,
    n_starts: int = 8,
    n_epochs: int = 100,
    steps_per_epoch: int = 200,
    seed: int = 42,
    x0=None,
    max_evals: int | None = None,
):
    """``run`` from a low-discrepancy multistart design inside the box.

    ``x0``, when given, replaces the first design point. ``max_evals``, when
    given, is split as evenly as possible over the starts and caps the total
    number of calls; otherwise a run makes
    ``n_starts * (1 + n_epochs * steps_per_epoch)`` calls.
    """
    return _core_run_qmc(
        obj_fn,
        _flat(low),
        _flat(high),
        preset,
        _count("n_starts", n_starts, 1),
        _count("n_epochs", n_epochs, 0),
        _count("steps_per_epoch", steps_per_epoch, 0),
        int(seed),
        _flat_or_none(x0),
        _max_evals(max_evals),
    )


def run_hmc(
    obj_fn,
    grad_fn,
    low,
    high,
    t_init: float = 5.0,
    epsilon: float = 0.05,
    l_steps: int = 5,
    q: float = 1.0,
    n_epochs: int = 100,
    steps_per_epoch: int = 50,
    seed: int = 42,
    x0=None,
):
    """HMC-driven simulated annealing inside the box ``[low, high]``.

    ``low``, ``high`` and ``x0`` may be any array-like of one shape; they are
    flattened in C order and the callables receive the flat vector.
    """
    return _core_run_hmc(
        obj_fn,
        grad_fn,
        _flat(low),
        _flat(high),
        float(t_init),
        float(epsilon),
        int(l_steps),
        float(q),
        int(n_epochs),
        int(steps_per_epoch),
        int(seed),
        _flat_or_none(x0),
    )


def cluster_search(obj_fn, grad_fn, n: int, budget: int, seed: int = 0, recommended: bool = True):
    """Run the measured cluster-search layer.

    Args:
      obj_fn: callable ``f(numpy.ndarray) -> float``.
      grad_fn: gradient or force callable ``g(x) -> numpy.ndarray``. A charged
        central-difference probe determines its orientation when the budget is
        at least four evaluations.
      n: number of points (state length is ``3 * n``).
      budget: charged objective and gradient evaluations.
      seed: RNG seed.
      recommended: ``Config.recommended(n)`` when true, else
        ``Config.for_cluster(n)``.

    Returns a dict with ``best`` (flat ``3n`` coordinates), ``best_energy``,
    and ``hops``.
    """
    out = _core_cluster_search(
        obj_fn,
        grad_fn,
        int(n),
        int(budget),
        int(seed),
        bool(recommended),
    )
    out["best"] = np.asarray(out["best"], dtype=np.float64)
    return out


def low_discrepancy_points(low, high, n: int, skip: int = 1):
    """Return bounded low-discrepancy points as a NumPy array."""
    low_arr = _flat(low)
    high_arr = _flat(high)
    return np.asarray(
        _core_low_discrepancy_points(low_arr, high_arr, int(n), int(skip)),
        dtype=np.float64,
    )


def pilot_draws_qmc(n: int, seed: int = 42):
    """Return BGSA pilot draws ``(T_0, sigma, q_v)`` as a NumPy array."""
    return np.asarray(_core_pilot_draws_qmc(int(n), int(seed)), dtype=np.float64)


def polish(
    obj_fn,
    grad_fn,
    low,
    high,
    x0,
    max_fevals: int = 200,
    step0: float = 1.0,
    grad_tol: float = 1e-8,
):
    """Refine ``x0`` with bounded projected-gradient polish."""
    out = _core_polish(
        obj_fn,
        grad_fn,
        _flat(low),
        _flat(high),
        _flat(x0),
        int(max_fevals),
        float(step0),
        float(grad_tol),
    )
    out["best_pos"] = np.asarray(out["best_pos"], dtype=np.float64)
    return out


def qmc_polish(
    obj_fn,
    grad_fn,
    low,
    high,
    n_starts: int,
    max_fevals_per_start: int,
    seed: int = 0,
    step0: float = 1.0,
    grad_tol: float = 1e-8,
    top_k: int = 0,
):
    """Refine low-discrepancy starts with bounded projected-gradient polish."""
    out = _core_qmc_polish(
        obj_fn,
        grad_fn,
        _flat(low),
        _flat(high),
        int(n_starts),
        int(max_fevals_per_start),
        int(seed),
        float(step0),
        float(grad_tol),
        int(top_k),
    )
    out["best_pos"] = np.asarray(out["best_pos"], dtype=np.float64)
    return out


def qmc_polish_objective(
    objective,
    n_starts: int,
    max_fevals_per_start: int,
    seed: int = 0,
    step0: float = 1.0,
    grad_tol: float = 1e-8,
    top_k: int = 0,
):
    """Refine QMC starts with a native ``PyObjective`` gradient handle."""
    out = _core_qmc_polish_objective(
        objective,
        int(n_starts),
        int(max_fevals_per_start),
        int(seed),
        float(step0),
        float(grad_tol),
        int(top_k),
    )
    out["best_pos"] = np.asarray(out["best_pos"], dtype=np.float64)
    return out


def qmc_best1bin_scout(
    obj_fn,
    low,
    high,
    max_evals: int,
    seed: int = 0,
    population_size: int = 30,
    weight_min: float = 0.5,
    weight_span: float = 0.5,
    crossover_rate: float = 0.7,
):
    """Run a QMC best/1/bin differential-evolution scout."""
    out = _core_qmc_best1bin_scout(
        obj_fn,
        _flat(low),
        _flat(high),
        int(max_evals),
        int(seed),
        int(population_size),
        float(weight_min),
        float(weight_span),
        float(crossover_rate),
    )
    out["best_pos"] = np.asarray(out["best_pos"], dtype=np.float64)
    return out


def qmc_best1bin_scout_objective(
    objective,
    max_evals: int,
    seed: int = 0,
    population_size: int = 30,
    weight_min: float = 0.5,
    weight_span: float = 0.5,
    crossover_rate: float = 0.7,
):
    """Run a QMC best/1/bin scout with a native ``PyObjective`` handle."""
    out = _core_qmc_best1bin_scout_objective(
        objective,
        int(max_evals),
        int(seed),
        int(population_size),
        float(weight_min),
        float(weight_span),
        float(crossover_rate),
    )
    out["best_pos"] = np.asarray(out["best_pos"], dtype=np.float64)
    return out


def qmc_gsa_global_search(
    obj_fn,
    low,
    high,
    max_evals: int,
    seed: int = 0,
    n_chains: int = 30,
    t_init: float = 1.0,
    q_v: float = 2.62,
    q_a: float = 1.7,
    x0=None,
):
    """Run bounded QMC-initialized generalized simulated annealing.

    ``x0``, when given, replaces the first chain's low-discrepancy start.
    """
    out = _core_qmc_gsa_global_search(
        obj_fn,
        _flat(low),
        _flat(high),
        int(max_evals),
        int(seed),
        int(n_chains),
        float(t_init),
        float(q_v),
        float(q_a),
        _flat_or_none(x0),
    )
    out["best_pos"] = np.asarray(out["best_pos"], dtype=np.float64)
    return out


def qmc_gsa_global_search_objective(
    objective,
    max_evals: int,
    seed: int = 0,
    n_chains: int = 30,
    t_init: float = 1.0,
    q_v: float = 2.62,
    q_a: float = 1.7,
    x0=None,
):
    """Run bounded QMC-initialized GSA with a native objective handle."""
    out = _core_qmc_gsa_global_search_objective(
        objective,
        int(max_evals),
        int(seed),
        int(n_chains),
        float(t_init),
        float(q_v),
        float(q_a),
        _flat_or_none(x0),
    )
    out["best_pos"] = np.asarray(out["best_pos"], dtype=np.float64)
    return out


def qmc_trust_region_poll(
    obj_fn,
    low,
    high,
    center,
    max_evals: int,
    seed: int = 0,
    radius_fraction: float = 0.0,
    n_levels: int = 3,
    points_per_level: int = 0,
):
    """Run a local shifted-QMC trust-region poll."""
    out = _core_qmc_trust_region_poll(
        obj_fn,
        _flat(low),
        _flat(high),
        _flat(center),
        int(max_evals),
        int(seed),
        float(radius_fraction),
        int(n_levels),
        int(points_per_level),
    )
    out["best_pos"] = np.asarray(out["best_pos"], dtype=np.float64)
    return out


def qmc_trust_region_poll_objective(
    objective,
    center,
    max_evals: int,
    seed: int = 0,
    radius_fraction: float = 0.0,
    n_levels: int = 3,
    points_per_level: int = 0,
):
    """Run a local shifted-QMC trust-region poll with a native objective handle."""
    out = _core_qmc_trust_region_poll_objective(
        objective,
        _flat(center),
        int(max_evals),
        int(seed),
        float(radius_fraction),
        int(n_levels),
        int(points_per_level),
    )
    out["best_pos"] = np.asarray(out["best_pos"], dtype=np.float64)
    return out


def shifted_qmc_polish(
    obj_fn,
    grad_fn,
    low,
    high,
    n_starts: int,
    max_fevals_per_start: int,
    seed: int = 0,
    n_replicates: int = 1,
    step0: float = 1.0,
    grad_tol: float = 1e-8,
    top_k: int = 0,
):
    """Refine shifted low-discrepancy replicas with bounded polish."""
    out = _core_shifted_qmc_polish(
        obj_fn,
        grad_fn,
        _flat(low),
        _flat(high),
        int(n_starts),
        int(max_fevals_per_start),
        int(seed),
        int(n_replicates),
        float(step0),
        float(grad_tol),
        int(top_k),
    )
    out["best_pos"] = np.asarray(out["best_pos"], dtype=np.float64)
    return out


def additive_independence(
    obj_fn,
    low,
    high,
    max_fevals: int,
    seed: int = 0,
    degree: int = 8,
    grid_m: int = 65,
    local_frac: float = 0.2,
    n_epochs: int = 40,
    n_pilot: int = 0,
):
    """Rank-1 (mean-field) independence-sampler SA.

    Fits a separable additive surrogate ``c + sum_j g_j(x_j)`` and spends the
    budget on tempered per-coordinate independence proposals accepted by
    Metropolis on the true objective. Values only (no gradient). For a separable
    objective the proposal places every coordinate at its tempered optimum at
    once. Returns ``{best_pos, best_val, n_evals}``.
    """
    out = _core_additive_independence(
        obj_fn,
        _flat(low),
        _flat(high),
        int(max_fevals),
        int(seed),
        int(degree),
        int(grid_m),
        float(local_frac),
        int(n_epochs),
        int(n_pilot),
    )
    out["best_pos"] = np.asarray(out["best_pos"], dtype=np.float64)
    return out


def estimate_gle_omega0(obj_fn, grad_fn, low, high):
    """Estimate the local characteristic frequency for GLE colored noise."""
    return float(
        _core_estimate_gle_omega0(
            obj_fn,
            grad_fn,
            _flat(low),
            _flat(high),
        )
    )


def gle_langevin(
    obj_fn,
    grad_fn,
    low,
    high,
    max_fevals: int,
    seed: int = 0,
    omega0: float | None = None,
    dt: float = 0.2,
    n_epochs: int = 40,
    x0=None,
):
    """GLE-thermostatted Langevin annealing (colored-noise optimal sampling).

    Gradient-driven BAB Langevin dynamics with a generalized-Langevin
    colored-noise thermostat. The fitted optimal-sampling drift, scaled to the
    characteristic frequency ``omega0``, flattens the sampling efficiency across
    ``[omega0, 100*omega0]``. When ``omega0`` is ``None`` the frequency is
    estimated from local gradient curvature over the provided bounds. This
    handles ill-conditioning the way the
    ``1/sqrt(D)`` scale handles dimension. Returns ``{best_pos, best_val,
    n_evals, omega0, dt}``.
    """
    omega_arg = None if omega0 is None else float(omega0)
    out = _core_gle_langevin(
        obj_fn,
        grad_fn,
        _flat(low),
        _flat(high),
        int(max_fevals),
        int(seed),
        omega_arg,
        float(dt),
        int(n_epochs),
        _flat_or_none(x0),
    )
    out["best_pos"] = np.asarray(out["best_pos"], dtype=np.float64)
    out["preconditioner_diag"] = np.asarray(
        out["preconditioner_diag"],
        dtype=np.float64,
    )
    return out


def gle_langevin_objective(
    objective,
    max_fevals: int,
    seed: int = 0,
    omega0: float | None = None,
    dt: float = 0.2,
    n_epochs: int = 40,
    x0=None,
):
    """GLE-Langevin annealing with a native ``PyObjective`` gradient handle."""
    omega_arg = None if omega0 is None else float(omega0)
    out = _core_gle_langevin_objective(
        objective,
        int(max_fevals),
        int(seed),
        omega_arg,
        float(dt),
        int(n_epochs),
        _flat_or_none(x0),
    )
    out["best_pos"] = np.asarray(out["best_pos"], dtype=np.float64)
    out["preconditioner_diag"] = np.asarray(
        out["preconditioner_diag"],
        dtype=np.float64,
    )
    return out


def gle_langevin_preconditioned(
    obj_fn,
    grad_fn,
    low,
    high,
    max_fevals: int,
    seed: int = 0,
    omega0: float | None = None,
    dt: float = 0.2,
    n_epochs: int = 40,
    x0=None,
    preconditioner_probes: int | None = None,
):
    """GLE-Langevin with an adaptive diagonal coordinate preconditioner."""
    omega_arg = None if omega0 is None else float(omega0)
    probe_arg = None if preconditioner_probes is None else int(preconditioner_probes)
    out = _core_gle_langevin_preconditioned(
        obj_fn,
        grad_fn,
        _flat(low),
        _flat(high),
        int(max_fevals),
        int(seed),
        omega_arg,
        float(dt),
        int(n_epochs),
        _flat_or_none(x0),
        probe_arg,
    )
    out["best_pos"] = np.asarray(out["best_pos"], dtype=np.float64)
    out["preconditioner_diag"] = np.asarray(
        out["preconditioner_diag"],
        dtype=np.float64,
    )
    return out


def gle_langevin_preconditioned_objective(
    objective,
    max_fevals: int,
    seed: int = 0,
    omega0: float | None = None,
    dt: float = 0.2,
    n_epochs: int = 40,
    x0=None,
    preconditioner_probes: int | None = None,
):
    """Preconditioned GLE-Langevin with a native ``PyObjective`` gradient handle."""
    omega_arg = None if omega0 is None else float(omega0)
    probe_arg = None if preconditioner_probes is None else int(preconditioner_probes)
    out = _core_gle_langevin_preconditioned_objective(
        objective,
        int(max_fevals),
        int(seed),
        omega_arg,
        float(dt),
        int(n_epochs),
        _flat_or_none(x0),
        probe_arg,
    )
    out["best_pos"] = np.asarray(out["best_pos"], dtype=np.float64)
    out["preconditioner_diag"] = np.asarray(
        out["preconditioner_diag"],
        dtype=np.float64,
    )
    return out


def gpmd_optimize(
    obj_fn,
    low,
    high,
    budget: int,
    seed: int = 0,
    grad_fn=None,
    x0=None,
):
    """Local Metropolis with T = ½ · (f − f_best) / d (D6 packaging).

    Implementation helper / portfolio arm material — not a global SOTA
    solver. See docs/derivations/gpmd_algorithm.org.
    """
    low_arr = _flat(low)
    high_arr = _flat(high)
    x0_arr = _flat_or_none(x0)
    out = _core_gpmd_optimize(
        obj_fn,
        low_arr,
        high_arr,
        int(budget),
        int(seed),
        grad_fn,
        x0_arr,
    )
    return out


def amsa_optimize(
    obj_fn,
    low,
    high,
    budget: int,
    seed: int = 0,
    grad_fn=None,
    x0=None,
):
    """Standalone whitened BFWT annealed descent (AmSa).

    One adaptive Metropolis chain: BFWT temperature (D11), Haario
    covariance whitening, Robbins-Monro scale control toward the design
    acceptance 0.32, online barrier estimate from rejected uphill moves,
    IPOP-style reseeds on stagnation, and a stall-recovering projected
    quasi-Newton polish tail when ``grad_fn`` is supplied.
    """
    low_arr = _flat(low)
    high_arr = _flat(high)
    x0_arr = _flat_or_none(x0)
    return _core_amsa_optimize(
        obj_fn,
        low_arr,
        high_arr,
        int(budget),
        int(seed),
        grad_fn,
        x0_arr,
    )


def bfwt_optimize(
    obj_fn,
    low,
    high,
    budget: int,
    seed: int = 0,
    barrier_hat: float = 0.0,
    grad_fn=None,
    x0=None,
):
    """Local Metropolis with D11 budget-feasible window temperature.

    Clamps design T into the D6∩D7 window. Standalone is not the SOTA
    driver; competitive wins use ``global_optimize`` (portfolio). See
    docs/derivations/bfwt_d11.md.
    """
    low_arr = _flat(low)
    high_arr = _flat(high)
    x0_arr = _flat_or_none(x0)
    out = _core_bfwt_optimize(
        obj_fn,
        low_arr,
        high_arr,
        int(budget),
        int(seed),
        float(barrier_hat),
        grad_fn,
        x0_arr,
    )
    return out


def dmc_population_optimize(
    obj_fn,
    low,
    high,
    budget: int,
    seed: int = 0,
    grad_fn=None,
    target_n: int = 16,
    steps_per_control: int = 4,
    x0=None,
):
    """Population-controlled diffusion search (DMC-inspired; classical objective).

    Parameters
    ----------
    x0 :
        Optional starting point for walker 0 (protocol anchor / incumbent).
    """
    out = _core_dmc_population_optimize(
        obj_fn,
        _flat(low),
        _flat(high),
        int(budget),
        int(seed),
        grad_fn,
        int(target_n),
        int(steps_per_control),
        _flat_or_none(x0),
    )
    out["best_pos"] = np.asarray(out["best_pos"], dtype=np.float64)
    return out


def global_optimize(
    obj_fn,
    low,
    high,
    budget: int,
    seed: int = 0,
    grad_fn=None,
    noise_sigma=None,
    policy: str = "auto",
    x0=None,
):
    """Thompson-allocated portfolio global optimizer.

    One generic driver with a single budget knob. A discounted
    Beta-Bernoulli posterior over the library's building blocks (QMC
    restart descent, adaptive basin hopping, archive-fit
    additive-surrogate independence proposals, best/1/bin differential
    evolution, preconditioned GLE-Langevin, shifted-QMC trust-region
    polls, generalized simulated annealing, the Bayesian-pilot tuned
    classical point, parallel tempering, q-Gaussian HMC, and the
    active-subspace collapse) allocates budget slices by Thompson
    sampling under a decaying uniform floor that preserves the
    restart-measure convergence guarantee. Objective and
    native-gradient evaluations share the budget at one unit each;
    every scheduler quantity derives from the budget, the dimension,
    and the arm count.

    Args:
      obj_fn: callable ``f(numpy.ndarray) -> float``.
      low, high: box bounds.
      budget: combined objective + gradient evaluation budget.
      seed: RNG seed.
      grad_fn: optional gradient callable ``g(x) -> numpy.ndarray``;
        enables the gradient arms and the final polish. Pass
        ``jax.grad(f)``, a torch ``.backward()`` wrapper, or an
        analytic gradient.
      noise_sigma: optional known noise scale of a stochastic
        ``obj_fn``. When ``None`` (the default) acceptance is the exact
        Metropolis rule. When set, the stochastic-evaluation accept
        sites use the Ball, Branke & Meisel (2018) sequential OSA rule,
        which draws repeated noisy evaluations to decide while keeping
        detailed balance under ``Normal(delta, noise_sigma**2)`` cost
        differences. Exact Metropolis under declared noise is refused
        (out of regime).
      policy: ``"auto"`` (default; feature-based regime routing) or
        ``"legacy"`` (flat arm order, uninformative priors; A/B only).
      x0: optional starting point inside the box, same size as ``low``. It is
        the first charged evaluation and the first incumbent.

    ``obj_fn`` and ``grad_fn`` are only called inside ``[low, high]``: a
    point an arm proposes outside is mirror-reflected into the box, and the
    gradient there is reflected with it. Returns a dict with
    ``best_pos``, ``best_val``, ``n_evals``, ``n_grads``, ``arm_pulls``, and
    ``arm_successes``.
    """
    out = _core_global_optimize(
        obj_fn,
        _flat(low),
        _flat(high),
        int(budget),
        int(seed),
        grad_fn,
        noise_sigma if noise_sigma is None else float(noise_sigma),
        str(policy),
        _flat_or_none(x0),
    )
    out["best_pos"] = np.asarray(out["best_pos"], dtype=np.float64)
    return out


def global_optimize_objective(
    objective,
    budget: int,
    seed: int = 0,
    use_gradient: bool = True,
    x0=None,
):
    """Portfolio global optimizer over a native ``PyObjective`` handle."""
    out = _core_global_optimize_objective(
        objective,
        int(budget),
        int(seed),
        bool(use_gradient),
        None,
        _flat_or_none(x0),
    )
    out["best_pos"] = np.asarray(out["best_pos"], dtype=np.float64)
    return out


__all__ = [
    "BasinBias",
    "Boltzmann",
    "Bounds",
    "Config",
    "DeviceHistory",
    "EnsembleHistory",
    "EpochLine",
    "Fast",
    "Gsa",
    "History",
    "Ledger",
    "PyObjective",
    "cluster_search",
    "TvmFfiTensorMetadata",
    "__version__",
    "low_discrepancy_points",
    "pilot_draws_qmc",
    "polish",
    "qmc_best1bin_scout",
    "qmc_best1bin_scout_objective",
    "qmc_gsa_global_search",
    "qmc_gsa_global_search_objective",
    "qmc_polish",
    "qmc_polish_objective",
    "qmc_trust_region_poll",
    "qmc_trust_region_poll_objective",
    "shifted_qmc_polish",
    "additive_independence",
    "estimate_gle_omega0",
    "gle_langevin",
    "gle_langevin_objective",
    "gle_langevin_preconditioned",
    "gle_langevin_preconditioned_objective",
    "dmc_population_optimize",
    "gpmd_optimize",
    "amsa_optimize",
    "bfwt_optimize",
    "global_optimize",
    "global_optimize_objective",
    "ChemFitVector",
    "chemfit_box",
    "fit_anneal",
    "fit_chemfit",
    "flatten_parameters",
    "run_benchmark",
    "run_fitter",
    "unflatten_parameters",
    "run",
    "run_device",
    "run_ensemble",
    "run_hmc",
    "run_qmc",
    "tvm_ffi_tensor",
    "tvm_ffi_tensor_metadata",
    "tvm_ffi_tensors_from_history",
]

from anneal.chemfit import (  # noqa: E402
    ChemFitVector,
    chemfit_box,
    fit_anneal,
    fit_chemfit,
    flatten_parameters,
    run_benchmark,
    run_fitter,
    unflatten_parameters,
)

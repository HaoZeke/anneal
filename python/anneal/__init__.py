"""anneal: simulated annealing on the eindir typed primitives.

Search splits on geometry, then on gradient:

  - cluster_search(obj, grad, n, budget, ...): 3N point set.
    recommended / derived / ras select the hop preset.
  - ensemble_optimize(obj, low, high, budget, grad_fn=None, ...): design box.
    Gradient means hop and quench; no gradient means a values-only search.
  - minimize(fun, x0, bounds, jac=None, budget=..., replicas=4, store=...):
    SciPy/ChemFit shape. replicas is the communicating-chain count.
    store is an existing HDF5 file or a readcon-db campaign directory.
  - box_ensemble_optimize(...): hop primitive. Requires grad_fn.
  - global_optimize(...): portfolio arms when the caller does not want hops.
  - run(obj_fn, low, high, preset, n_epochs, steps_per_epoch, seed, x0): SA
    loop. Every proposal is reflected into [low, high] and the walk starts
    at x0 when one is given.
  - fit_anneal, fit_chemfit, run_benchmark, run_fitter: gradient-free ChemFit
    bridges. They drive evaluate/step or ask/tell and raise the first
    exception the fitter raises.

SA algebra (Cool / Move / Accept) stays on run / run_device / run_ensemble
with Boltzmann, Fast, and Gsa. Last mile is polish / qmc_polish
(bounded L-BFGS). Cluster quench is WarmLbfgs to rgmin::Lbfgs.

The IISE composition laws L1-L4 are enforced in SaVariant::checked.
"""

import json

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
    box_ensemble_optimize as _core_box_ensemble_optimize,
    ensemble_optimize as _core_ensemble_optimize,
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
    ``RuntimeWarning``; ``KeyboardInterrupt`` ends the run and is re-raised, as
    is a ``TypeError`` when ``obj_fn`` returns something that is not a number.
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


def cluster_search(
    obj_fn,
    grad_fn,
    n: int,
    budget: int,
    seed: int = 0,
    recommended: bool = True,
    derived: bool = False,
    communicating: bool = False,
    *,
    ras: bool = False,
    start=None,
    length_scale: float | None = None,
):
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
      derived: ``Config.derived(n)``; overrides ``recommended``.
      communicating: ``Config.communicating(n)``; overrides both.
      ras: residual archive search on the recommended preset.
      start: optional flat ``3n`` start. When set, hops use ``search_from``
        instead of a random compact cluster.
      length_scale: optional physical length. With the recommended preset
        this uses ``with_scales`` so an angstrom start is not crushed into
        the reduced LJ container.

    Returns a dict with ``best`` (flat ``3n`` coordinates), ``best_energy``,
    and ``hops``.
    """
    start_arr = None if start is None else np.asarray(start, dtype=np.float64).reshape(-1)
    args = (
        obj_fn,
        grad_fn,
        int(n),
        int(budget),
        int(seed),
        bool(recommended),
        bool(derived),
        bool(communicating),
    )
    kwargs = {"ras": bool(ras)}
    if start_arr is not None:
        kwargs["start"] = start_arr
    if length_scale is not None:
        kwargs["length_scale"] = float(length_scale)
    out = _core_cluster_search(*args, **kwargs)
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
    """Refine ``x0`` with bounded L-BFGS (two-loop, Armijo backtrack)."""
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
    """Refine low-discrepancy starts with bounded L-BFGS polish."""
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


def box_ensemble_optimize(
    obj_fn,
    low,
    high,
    budget: int,
    seed: int = 0,
    grad_fn=None,
    x0=None,
    replicas: int = 4,
    history: str = "shared",
    membership: str = "accepted",
):
    """Communicating box hops on SharedDesignHistory / MinimumHistory.

    Requires ``grad_fn``. Kick and quench in the box. Without a gradient
    use ``ensemble_optimize`` (portfolio). Not cluster hopping.
    """
    low_arr = np.asarray(low, dtype=np.float64)
    high_arr = np.asarray(high, dtype=np.float64)
    x0_arr = None if x0 is None else np.asarray(x0, dtype=np.float64)
    out = _core_box_ensemble_optimize(
        obj_fn,
        low_arr,
        high_arr,
        int(budget),
        int(seed),
        grad_fn,
        x0_arr,
        int(replicas),
        str(history),
        str(membership),
    )
    out["best_pos"] = np.asarray(out["best_pos"], dtype=np.float64)
    return out


def ensemble_optimize(
    obj_fn,
    low,
    high,
    budget: int,
    seed: int = 0,
    grad_fn=None,
    x0=None,
    replicas: int = 4,
    history: str = "shared",
    membership: str = "accepted",
    *,
    coverage_shared=None,
    coverage_radius=None,
    coverage_neighbors=None,
):
    """Search on a design box.

    With ``grad_fn`` this hops and quenches (the kernel used by
    ``box_ensemble_optimize``). Without it, one replica uses the values-only
    portfolio; multiple replicas use communicating values-only hop chains.
    Equal bounds fix a coordinate without changing callback dimensions.

    The returned dictionary retains ``best_val``, ``best_pos``, separate
    ``n_evals`` / ``n_grads``, their sum ``charged``, ``hops``, minimum-history
    diagnostics, and evaluated-region ``coverage_*`` counters. Coverage is
    not a count of certified minima.

    ``coverage_shared`` overrides sharing independently of ``history``;
    ``coverage_radius`` sets the normalized RMS parameter distance.
    ``coverage_neighbors`` restricts both sample and deposit delivery to direct
    cyclic neighbours: zero is all-to-all, one selects the two adjacent chains.
    Explicit coverage controls select native hop chains, including for one
    replica. With no coverage controls, the one-replica values-only portfolio
    remains the convenience policy. Coverage requires no gradient and does not
    restrict the optional minimum ledger or forward received observations.

    Coordinates are design variables regardless of dimension. Atomic
    symmetry-aware proposals require the explicit ``cluster_search`` adapter.
    """
    low_arr = np.asarray(low, dtype=np.float64)
    high_arr = np.asarray(high, dtype=np.float64)
    x0_arr = None if x0 is None else np.asarray(x0, dtype=np.float64)
    out = _core_ensemble_optimize(
        obj_fn,
        low_arr,
        high_arr,
        int(budget),
        int(seed),
        grad_fn,
        x0_arr,
        int(replicas),
        str(history),
        str(membership),
        coverage_shared=coverage_shared,
        coverage_radius=coverage_radius,
        coverage_neighbors=coverage_neighbors,
    )
    out["best_pos"] = np.asarray(out["best_pos"], dtype=np.float64)
    return out


class MinimizeResult:
    """SciPy-shaped result from :func:`minimize`.

    ``nfev`` counts objective calls, ``njev`` counts gradient calls, and
    ``charged`` is their combined budget cost. ``diagnostics`` retains the
    full engine result, including coverage and optional minimum history.
    ``success`` means a finite feasible candidate was returned; it does not
    certify local convergence or global optimality.
    """

    __slots__ = (
        "x",
        "fun",
        "nfev",
        "njev",
        "charged",
        "success",
        "message",
        "diagnostics",
    )

    def __init__(self, x, fun, nfev, success, message, *, njev=0, diagnostics=None):
        self.x = np.asarray(x, dtype=np.float64)
        self.fun = float(fun)
        self.nfev = int(nfev)
        self.njev = int(njev)
        self.charged = self.nfev + self.njev
        self.success = bool(success)
        self.message = str(message)
        self.diagnostics = {} if diagnostics is None else dict(diagnostics)

    def __repr__(self):
        return (
            f"MinimizeResult(fun={self.fun!r}, nfev={self.nfev}, "
            f"njev={self.njev}, charged={self.charged}, success={self.success})"
        )


class JsonlParameterStore:
    """Parameter archive as JSON lines. Works without h5py."""

    def __init__(self, path):
        from pathlib import Path

        self.path = Path(path)

    def observations(self):
        if not self.path.is_file():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out.append((np.asarray(row["x"], dtype=np.float64), float(row["f"])))
        return out

    def record(self, x, f):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = json.dumps({"x": np.asarray(x, dtype=np.float64).tolist(), "f": float(f)})
        with self.path.open("a") as handle:
            handle.write(row + "\n")


class Hdf5ParameterStore:
    """Parameter archive as an HDF5 table ``x`` / ``f``."""

    def __init__(self, path):
        from pathlib import Path

        self.path = Path(path)

    def observations(self):
        h5py = _require_h5py()
        if not self.path.is_file():
            return []
        with h5py.File(self.path, "r") as h5:
            if "x" not in h5 or "f" not in h5:
                return []
            xs = np.asarray(h5["x"], dtype=np.float64)
            fs = np.asarray(h5["f"], dtype=np.float64).reshape(-1)
        if xs.ndim == 1:
            xs = xs.reshape(1, -1)
        return [(xs[i], float(fs[i])) for i in range(min(len(xs), len(fs)))]

    def record(self, x, f):
        h5py = _require_h5py()
        x = np.asarray(x, dtype=np.float64).reshape(1, -1)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.path, "a") as h5:
            if "x" not in h5:
                h5.create_dataset("x", data=x, maxshape=(None, x.size), chunks=True)
                h5.create_dataset(
                    "f", data=np.array([float(f)]), maxshape=(None,), chunks=True
                )
                return
            xs = h5["x"]
            fs = h5["f"]
            n = fs.shape[0]
            xs.resize((n + 1, x.size))
            fs.resize((n + 1,))
            xs[n] = x
            fs[n] = float(f)


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("HDF5 parameter stores need h5py") from exc
    return h5py


def open_parameter_store(store):
    """Open a ChemFit/anneal parameter archive.

    * object with ``observations`` / ``record``: used as-is
    * ``*.h5`` / ``*.hdf5``: :class:`Hdf5ParameterStore`
    * directory (existing readcon-db campaign root): ``anneal_params.h5``
      if h5py is importable, else ``anneal_params.jsonl`` beside the corpus.
      Parameter vectors are never written as CON frames.
    """
    from pathlib import Path

    if store is None:
        return None
    if hasattr(store, "observations") and hasattr(store, "record"):
        return store
    path = Path(store)
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        return Hdf5ParameterStore(path)
    if suffix == ".jsonl":
        return JsonlParameterStore(path)
    path.mkdir(parents=True, exist_ok=True)
    try:
        _require_h5py()
    except ImportError:
        return JsonlParameterStore(path / "anneal_params.jsonl")
    return Hdf5ParameterStore(path / "anneal_params.h5")


def _scipy_bounds_to_low_high(bounds, dim):
    """Accept SciPy ``Bounds``, a (dim, 2) array, or a sequence of pairs."""
    if hasattr(bounds, "lb") and hasattr(bounds, "ub"):
        low = np.asarray(bounds.lb, dtype=np.float64).reshape(-1)
        high = np.asarray(bounds.ub, dtype=np.float64).reshape(-1)
    else:
        arr = np.asarray(bounds, dtype=np.float64)
        if arr.ndim == 2 and arr.shape == (dim, 2):
            low, high = arr[:, 0].copy(), arr[:, 1].copy()
        elif arr.ndim == 1 and arr.size == dim:
            raise ValueError("bounds must be pairs (low, high) per coordinate")
        else:
            raise ValueError(f"bounds shape {arr.shape} does not match dimension {dim}")
    if low.size != dim or high.size != dim:
        raise ValueError(f"bounds length {low.size} does not match x0 length {dim}")
    return low, high


def minimize(
    fun,
    x0,
    bounds,
    jac=None,
    *,
    budget=None,
    seed=0,
    replicas=4,
    history="shared",
    membership="accepted",
    store=None,
    coverage_shared=None,
    coverage_radius=None,
    coverage_neighbors=None,
):
    """Box search with a SciPy ``minimize`` shape.

    The search-specific controls are:

    - ``replicas``: communicating hop chains. One replica and no
      ``jac`` is the values-only portfolio. Two or more replicas
      without ``jac`` still hop and exchange evaluated-region coverage.
    - ``history``: shared, private, or no minimum ledger. Its default also
      selects shared coverage; coverage itself needs no minimum certificate.
    - ``membership``: which certified observations enter the minimum ledger.
    - ``coverage_shared``, ``coverage_radius`` and ``coverage_neighbors``:
      sharing, normalized distance and direct ring-neighbourhood controls.
      Zero neighbours selects all-to-all; one selects the adjacent chains.
      Use ``history="none", coverage_shared=True`` to communicate
      sampled parameter regions without requesting minimum certificates.
    - ``store``: existing parameter archive. An ``.h5`` / ``.hdf5`` path,
      a campaign directory that already holds a readcon-db corpus (params
      are written beside it, never padded into CON frames), or an object
      with ``observations()`` and ``record(x, f)``.

    If ``store`` already has finite observations, the walk starts from the
    stored best instead of ``x0``. The result is recorded back.

    Bounds are finite closed intervals; equal endpoints fix that coordinate.
    Every callback receives the full design vector. The aggregate ``budget``
    pays for objective and gradient calls, including local improvement and
    validation. The result separates those counts and retains the engine's
    coverage diagnostics. ``success`` is not a global-optimality certificate.

    This uses box geometry. Atomic symmetry-aware proposals are selected
    explicitly through ``cluster_search``, not inferred from vector length.
    """
    x0_arr = np.asarray(x0, dtype=np.float64).reshape(-1)
    dim = int(x0_arr.size)
    if dim < 1:
        raise ValueError("x0 must be nonempty")
    low, high = _scipy_bounds_to_low_high(bounds, dim)
    if budget is None:
        budget = max(32, 80 * dim)
    if replicas < 1:
        raise ValueError("replicas must be positive")
    archive = open_parameter_store(store)
    if archive is not None:
        observed = [
            (np.asarray(x, dtype=np.float64).reshape(-1), float(f))
            for x, f in archive.observations()
            if np.isfinite(f) and np.asarray(x).size == dim
        ]
        if observed:
            x0_arr = min(observed, key=lambda item: item[1])[0]
    out = ensemble_optimize(
        fun,
        low,
        high,
        budget=int(budget),
        seed=int(seed),
        grad_fn=jac,
        x0=x0_arr,
        replicas=int(replicas),
        history=str(history),
        membership=str(membership),
        coverage_shared=coverage_shared,
        coverage_radius=coverage_radius,
        coverage_neighbors=coverage_neighbors,
    )
    fun_v = float(out["best_val"])
    best = np.asarray(out["best_pos"], dtype=np.float64)
    if archive is not None and np.isfinite(fun_v):
        archive.record(best, fun_v)
    return MinimizeResult(
        x=best,
        fun=fun_v,
        nfev=int(out["n_evals"]),
        njev=int(out["n_grads"]),
        success=bool(np.isfinite(fun_v)),
        message=(
            "Finite evaluated candidate returned; global optimality is not certified."
            if np.isfinite(fun_v)
            else "No finite feasible objective value was found."
        ),
        diagnostics=out,
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
    *,
    replicas: int = 1,
    coverage_shared: bool = True,
    coverage_radius: float = 0.05,
    coverage_neighbors: int = 0,
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

    With ``policy="auto"``, no ``grad_fn`` and no ``noise_sigma``, the
    run takes a values-only loop instead, whatever the box: CMA-ES, a
    finite-difference quasi-Newton descent, generalized simulated
    annealing quenched to one-coordinate improvements, differential
    evolution, the additive surrogate, and the QMC restart arm. When the
    budget lets a descent converge, one from the start opens the run
    while it pays; with less, CMA-ES and the descent take one slice each,
    annealing two, and a descent turn settles what annealing found. From
    there annealing and then CMA-ES each keep the turn while they lower
    the incumbent. Arms not yet played then take a turn, while the budget
    holds one beyond the closing descent's reserve, before the
    allocation ranks them, and the same decaying uniform floor follows;
    a turn is one slice, except that the descent keeps it while it pays,
    until it converges. The descent closes the run, going on with its
    current descent, or restarting at the incumbent if another arm has
    lowered it since the descent last played. The floor the guarantee
    needs is asymptotic: the opening runs while it improves by the
    success threshold, which cannot last on a bounded objective, and
    after that every arm keeps a uniform share and is pulled infinitely
    often as the budget grows.

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
      x0: optional starting point. It is clipped onto the box, then charged
        as the first evaluation and the first incumbent. It stays the
        incumbent until a lower value is found (if its value is not finite,
        until any finite value is). The values-only loop starts its
        descents, CMA-ES, one annealing chain and the first evolution member
        there; with ``grad_fn`` or ``noise_sigma``, arms that read the
        incumbent do, such as the trust-region poll and the population arm,
        and with ``grad_fn`` also basin hopping and HMC. Without ``x0`` the
        values-only loop starts from the best of a small seeded design.
      replicas: persistent portfolio controllers sharing the aggregate budget.
        One retains the single-controller behavior. Multiple replicas use
        native worker threads; no caller Jacobian is required.
      coverage_shared: share paid global proposal positions at arm checkpoints.
        False gives independent portfolios with the same starts and work split.
      coverage_radius: interaction radius in normalized RMS box coordinates.
      coverage_neighbors: direct cyclic neighbourhood radius. Zero is all-to-all;
        one admits the two adjacent chain identities without forwarding samples.
        This does not remove the shared arm-checkpoint barrier.

    ``obj_fn`` and ``grad_fn`` are only called inside ``[low, high]``: a
    point an arm proposes outside is mirror-reflected into the box, and the
    gradient there is reflected with it. Peer separation moves global
    candidates, not derivative stencils or raw objective values. Returns a
    dict with ``best_pos``, ``best_val``, ``n_evals``, ``n_grads``,
    ``arm_pulls``, and ``arm_successes``. Multiple replicas also report
    individual results, ``charged`` work and coverage sample counters.
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
        replicas=int(replicas),
        coverage_shared=bool(coverage_shared),
        coverage_radius=float(coverage_radius),
        coverage_neighbors=int(coverage_neighbors),
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
    "Anneal",
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
    "box_ensemble_optimize",
    "ensemble_optimize",
    "minimize",
    "MinimizeResult",
    "open_parameter_store",
    "Hdf5ParameterStore",
    "JsonlParameterStore",
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


def __getattr__(name):
    if name == "Anneal":
        from anneal.ase import Anneal as _Anneal

        return _Anneal
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

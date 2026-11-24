# Copyright (c) anneal developers
"""anneal: simulated annealing on the eindir typed primitives.

Search splits on geometry, then on gradient:

  - cluster_search(obj, grad, n, budget, ...): 3N point set, or 6N when
    rigid=True. species adds a swap move. recommended / derived / ras
    select the hop preset.
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
  - landfold(rows, ...), landfold_book(paths, ...): sketch-map embedding of
    descriptor rows or of minima books. map.project places held-out rows.
  - fit_anneal, fit_chemfit, run_benchmark, run_fitter: gradient-free ChemFit
    bridges. They drive evaluate/step or ask/tell and raise the first
    exception the fitter raises.

SA algebra (Cool / Move / Accept) stays on run / run_device / run_ensemble
with Boltzmann, Fast, and Gsa. Last mile is polish / qmc_polish
(bounded L-BFGS). Cluster quench is WarmLbfgs to rgmin::Lbfgs.

The IISE composition laws L1-L4 are enforced in SaVariant::checked.

A callback that raises BaseException other than Exception, such as
KeyboardInterrupt or a harness stop, ends the search at once. The best
state found is kept, and the exception is re-raised. An ordinary
Exception is scored as the worst value.
"""

import json

import numpy as np

from anneal._core import (
    BasinBias,
    BinaryLennardJones,
    Boltzmann,
    Bounds,
    COPPER_Z,
    Config,
    DEFAULT_BETA0,
    DEFAULT_HIT_TOL,
    DEFAULT_STEPS_PER_CONTROL,
    DEFAULT_TARGET_WALKERS,
    GOLD_Z,
    GminTip4p,
    Gupta,
    IDENTITY_TOL,
    PLATINUM_Z,
    PairPotential,
    SEARCH_HIT_TOL,
    SuttonChen,
    Tip4pCluster,
    EpochLine,
    Fast,
    Gsa,
    History,
    Ledger,
    PyObjective,
    __version__,
    block_from_interleaved_gmin as _core_block_from_interleaved_gmin,
    cluster_search as _core_cluster_search,
    energy_hits,
    energy_ohh as _core_energy_ohh,
    gmin_tip4p_sites,
    interleaved_gmin_from_block as _core_interleaved_gmin_from_block,
    low_discrepancy_points as _core_low_discrepancy_points,
    osa_acceptance_rate,
    pilot_draws_qmc as _core_pilot_draws_qmc,
    polish as _core_polish,
    quench as _core_quench,
    recommend_target_n,
    rigid_from_ohh as _core_rigid_from_ohh,
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
from anneal._landfold import LandfoldMap, landfold, landfold_book
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
    energy_scale: float | None = None,
    species=None,
    sigmas=None,
    rigid: bool = False,
    layout: str = "block",
    target: float | None = None,
    water_sites=None,
    bonds=None,
    groups=None,
    local_step: str | None = None,
    ring_moves: bool | None = None,
    hold_stereocentres: bool | None = None,
    measured_energy_scale: bool | None = None,
    config: Config | None = None,
):
    """Run the measured cluster-search layer.

    Args:
      obj_fn: callable ``f(numpy.ndarray) -> float``, or a native cluster
        potential (``PairPotential``, ``BinaryLennardJones``, ``SuttonChen``,
        ``Gupta``, ``Tip4pCluster``, or ``GminTip4p``).
      grad_fn: gradient or force callable ``g(x) -> numpy.ndarray``, or
        ``None`` when ``obj_fn`` is a native cluster potential. A charged
        central-difference probe determines callable gradient vs force
        orientation when the budget is at least four evaluations.
      n: number of points. The Cartesian state length is ``3 * n``. With
        ``rigid=True`` or ``water_sites``, ``n`` is the number of bodies and
        the state length is ``6 * n``.
      budget: charged objective and gradient evaluations.
      seed: RNG seed.
      recommended: ``Config.recommended(n)`` when true, else
        ``Config.for_cluster(n)``.
      derived: ``Config.derived(n)``; overrides ``recommended``.
      communicating: ``Config.communicating(n)``; overrides both.
      ras: residual archive search on the recommended preset.
      start: optional flat start. Cartesian length is ``3n``. A rigid or
        water start is ``6n``. When set, hops use ``search_from`` instead of
        a random compact cluster.
      length_scale: optional positive finite physical length scale (default
        ``1.0`` when ``energy_scale`` is set). When neither ``config``,
        ``groups``, nor ``water_sites`` is set and either scale is supplied,
        presets are constructed at ``(scale, escale)`` via
        ``Config.recommended_with_scales`` (for ``ras=True``,
        ``recommended=True``, and ``derived=True`` with ``budget_window``,
        ``bayes_screen``, and the cost-asymmetric threshold) or
        ``Config.for_cluster_with_scales`` (for ``recommended=False`` and
        ``communicating=True``, scaling its ``TwoPhase`` surface stiffness as
        ``escale / scale**4`` for ``n < 98``). Stock ``species`` and
        ``rigid`` runs preserve those scales: ``rigid=True`` scales the stock
        GMIN water-site offsets by ``scale``, and ``species`` attaches
        ``with_species(species, sigmas)`` to the scaled preset. Excluded by
        ``water_sites``, ``groups``, and ``config``.
      energy_scale: optional positive finite physical energy scale (default
        ``1.0`` when ``length_scale`` or ``groups`` is set). Scaled into
        cluster presets (including stock ``species`` and ``rigid`` runs) and
        passed to ``recommended_molecular`` / ``for_molecular`` under
        ``groups``. Excluded by ``water_sites`` and ``config``.
      species: optional label per point. Two or more labels add a swap move.
        A wide diameter ratio also tries one downhill unlike-pair exchange
        beside the escape.
      sigmas: optional diameter per point, same length as ``species``.
      rigid: hop in centre and Rodrigues rotation coordinates. A chart of
        more than ten bodies holds the acceptance factor at 4 and scales
        the orientation amplitude toward an acceptance ratio of one half.
        Ten bodies or fewer do not. The quench is the same L-BFGS
        descent, and a rigid record accepts the root-mean-square force.
        The chain keeps the eight lowest distinct minima of its quenched
        start, accepted quenches and lowest recorded minimum. Each stall
        follows 1000 hops with no new lowest minimum.
        At each stall the walker moves to the lowest kept minimum that is
        neither its own nor one of the last four stall destinations. When
        no kept minimum qualifies, the walker stays put. Nothing is drawn,
        the target is not read, and the lowest minimum is kept.
      layout: ``block`` or ``interleaved``. Interleaved is the per-body
        centre and angle-axis order. Both carry the same rotation vectors,
        in the sense GMIN's ``NTIP`` reads them.
      target: stop once a recorded energy is at or below this value.
      water_sites: optional ``(3, 3)`` body-frame O, H, H positions of a
        rigid water, in angstrom, about its rotation centre. ``n`` then counts
        molecules and the state is ``n`` centres followed by ``n``
        exponential-map rotation vectors (active Rodrigues). The presets are
        ``Config.communicating_tip4p``, ``Config.recommended_tip4p`` and, with
        both false, ``Config.for_tip4p``. Requires ``start``. Excludes
        ``ras``, ``derived``, ``length_scale``, ``energy_scale``, ``rigid``,
        ``species``, ``sigmas``, and a non-block layout.
      bonds: optional list of site-index pairs for that custom body. Unset
        bonds are the two O-H contacts, ``[[0, 1], [0, 2]]``. A list with no
        site joined to two others is refused. Requires ``water_sites``.
      groups: optional list of atom-index lists, one per rigid group of a
        molecular search. The presets become ``recommended_molecular`` and,
        with ``recommended=False``, ``for_molecular``, scaled by
        ``energy_scale`` when given. Requires ``species`` and ``start``.
        Excludes ``ras``, ``derived``, ``communicating``, ``length_scale``,
        ``rigid``, ``sigmas``, ``water_sites`` and a non-block layout.
      local_step: ``lbfgs``, ``tric`` or ``auto``, the optimizer of every
        quench of a ``groups`` search. ``tric`` relaxes in translation-rotation
        internal coordinates, in electronvolt and angstrom. ``auto`` times the
        opening Cartesian quench and keeps TRIC only when a potential call
        costs more than a TRIC step. ``None`` keeps the preset's: ``auto`` for
        ``recommended=True``, ``lbfgs`` otherwise. The result's
        ``local_step`` names the quench in force at the end.
      ring_moves: ring crankshafts beside the torsion move of a ``groups``
        search: one arc of a flexible ring turns about the axis through the
        two ring atoms that bound it, which keeps every bond length.
        ``None`` keeps the preset's: on for ``recommended=True``, off
        otherwise.
      hold_stereocentres: refuse a quench step of a ``groups`` search that
        turns a four-bonded centre into its mirror image. ``None`` keeps the
        preset's: on for ``recommended=True``, off otherwise.
      measured_energy_scale: take a ``groups`` search's temperature and
        deposit height from the median quenched energy change of its opening
        hops instead of ``energy_scale``. ``None`` keeps the preset's:
        on for ``recommended=True``, off otherwise.
      config: explicit ``Config`` instance; mutually exclusive with preset
        modifier arguments (mutate ``config`` directly instead).

    A callback raising a ``BaseException`` that is not an ``Exception``, such
    as ``KeyboardInterrupt`` or a harness stop, ends the search at once.
    The best geometry found is kept, and the exception is re-raised here.

    Returns:
      When ``ras=False`` (default), a dict containing:
      - ``best``, ``best_energy``: lowest minimum found (flat coordinates in
        the active layout and its energy).
      - ``final``, ``final_energy``: final chain state (``numpy.ndarray`` or
        ``None``) and energy.
      - ``spent``, ``hops``, ``accepted``, ``basins``, ``screened_out``,
        ``restarts``: evaluation ledger spend and basin-hopping counters.
      - ``visit_counts``, ``escape_scale``, ``escape_threshold``,
        ``merge_radius``, ``mean_step``, ``arms``, ``improvements``,
        ``stop``, ``quenched``, ``energy_trace``, ``nve_unavailable``:
        outcome diagnostics, improvement events ``(hop, charged, basin,
        energy)``, optional stop code, and per-hop traces when enabled on
        ``Config``.
      - ``local_step``, ``quench_seconds``, ``converged``, ``capped``,
        ``screens``, ``screen_charged``, ``full_charged``, ``check_charged``,
        ``overlap_rejected``, ``refused``, ``screen_steps_taken``: quench and
        screening statistics. ``quench_seconds`` is ``None`` unless
        ``local_step="auto"`` timed a potential call and a TRIC step.
      - ``search_seconds``, ``polish_seconds``, ``polish_calls``: wall-clock
        phase timings in seconds and final polish call count.

      When ``ras=True``, a dict from the residual archive search containing
      ``best``, ``best_energy``, ``hops``, ``charged``, ``screens``,
      ``same_floor``, ``artn``, ``best_at``, ``floors``, ``returned``,
      ``events``, and ``nve_unavailable``.
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
    kwargs = {"ras": bool(ras), "rigid": bool(rigid), "layout": str(layout)}
    if start_arr is not None:
        kwargs["start"] = start_arr
    if length_scale is not None:
        kwargs["length_scale"] = float(length_scale)
    if energy_scale is not None:
        kwargs["energy_scale"] = float(energy_scale)
    if species is not None:
        kwargs["species"] = [int(value) for value in species]
    if sigmas is not None:
        kwargs["sigmas"] = [float(value) for value in np.asarray(sigmas, dtype=np.float64).reshape(-1)]
    if target is not None:
        kwargs["target"] = float(target)
    if water_sites is not None:
        kwargs["water_sites"] = np.asarray(water_sites, dtype=np.float64).reshape(-1)
    if bonds is not None:
        kwargs["bonds"] = [[int(a), int(b)] for a, b in bonds]
    if groups is not None:
        kwargs["groups"] = [[int(atom) for atom in group] for group in groups]
    if local_step is not None:
        kwargs["local_step"] = str(local_step)
    if ring_moves is not None:
        kwargs["ring_moves"] = bool(ring_moves)
    if hold_stereocentres is not None:
        kwargs["hold_stereocentres"] = bool(hold_stereocentres)
    if measured_energy_scale is not None:
        kwargs["measured_energy_scale"] = bool(measured_energy_scale)
    if config is not None:
        kwargs["config"] = config
    out = _core_cluster_search(*args, **kwargs)
    out["best"] = np.asarray(out["best"], dtype=np.float64)
    if out.get("final") is not None:
        out["final"] = np.asarray(out["final"], dtype=np.float64)
    return out


_NATIVE_POTENTIALS = (
    PairPotential,
    BinaryLennardJones,
    SuttonChen,
    Gupta,
    Tip4pCluster,
    GminTip4p,
)


def rigid_from_ohh(cartesian):
    """Convert Cartesian ``(O, H, H)`` triples to block Rodrigues coordinates."""
    return np.asarray(_core_rigid_from_ohh(_flat(cartesian)), dtype=np.float64)


def energy_ohh(cartesian) -> float:
    """Evaluate rigid TIP4P energy on Cartesian ``(O, H, H)`` coordinates."""
    return float(_core_energy_ohh(_flat(cartesian)))


def block_from_interleaved_gmin(interleaved):
    """Reorder interleaved GMIN ``(c, omega)`` rows to block ``(centres, omega)``."""
    return np.asarray(_core_block_from_interleaved_gmin(_flat(interleaved)), dtype=np.float64)


def interleaved_gmin_from_block(block):
    """Reorder block ``(centres, omega)`` to interleaved GMIN ``(c, omega)`` rows."""
    return np.asarray(_core_interleaved_gmin_from_block(_flat(block)), dtype=np.float64)


def quench(fg_fn, x0, gtol: float, max_iter: int = 10000, *, method: str = "lbfgs", species=None):
    """Relax ``x0`` with the quench a molecular cluster search runs.

    ``fg_fn`` is either a callable ``fg_fn(x) -> (energy, gradient)`` or a
    native potential instance (``PairPotential``, ``BinaryLennardJones``,
    ``SuttonChen``, ``Gupta``, ``Tip4pCluster``, ``GminTip4p``). The descent
    stops when the largest gradient component is below ``gtol``. ``method`` is
    ``lbfgs``, the search's Cartesian L-BFGS, or ``tric``, translation-rotation
    internal coordinates over the atomic numbers ``species`` (electronvolt and
    angstrom). ``max_iter`` caps iterations for ``lbfgs`` and calls for
    ``tric``. Returns a dict with ``x``, ``energy`` and ``calls``, the number
    of ``fg_fn`` evaluations.
    """
    if isinstance(fg_fn, _NATIVE_POTENTIALS):
        target_fg = fg_fn
    elif hasattr(fg_fn, "value_and_gradient") and not callable(fg_fn):
        def target_fg(x):
            energy, gradient = fg_fn.value_and_gradient(np.asarray(x, dtype=np.float64))
            return float(energy), np.asarray(gradient, dtype=np.float64).reshape(-1).tolist()
    else:
        def target_fg(x):
            energy, gradient = fg_fn(np.asarray(x, dtype=np.float64))
            return float(energy), np.asarray(gradient, dtype=np.float64).reshape(-1).tolist()

    labels = None if species is None else [int(value) for value in species]
    out = _core_quench(
        target_fg,
        np.asarray(x0, dtype=np.float64).reshape(-1),
        float(gtol),
        int(max_iter),
        str(method),
        labels,
    )
    out["x"] = np.asarray(out["x"], dtype=np.float64)
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
    """Refine ``x0`` with bounded L-BFGS (two-loop, Armijo backtrack).

    Returns a dict with ``best_pos``, ``best_val``, ``n_evals``, ``n_grads``,
    ``projected_grad_norm``, ``projected_stationary``, and ``best_grad``.
    """
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
    if out.get("best_grad") is not None:
        out["best_grad"] = np.asarray(out["best_grad"], dtype=np.float64)
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
    n_evals, omega0, dt, preconditioner_diag, n_preconditioner_grads}``.
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
    """GLE-Langevin annealing with a native ``PyObjective`` gradient handle.

    Returns ``{best_pos, best_val, n_evals, omega0, dt, preconditioner_diag,
    n_preconditioner_grads}``.
    """
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
    """GLE-Langevin with an adaptive diagonal coordinate preconditioner.

    Returns ``{best_pos, best_val, n_evals, omega0, dt, preconditioner_diag,
    n_preconditioner_grads}``.
    """
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
    """Preconditioned GLE-Langevin with a native ``PyObjective`` gradient handle.

    Returns ``{best_pos, best_val, n_evals, omega0, dt, preconditioner_diag,
    n_preconditioner_grads}``.
    """
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
    out["best_pos"] = np.asarray(out["best_pos"], dtype=np.float64)
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
    out = _core_amsa_optimize(
        obj_fn,
        low_arr,
        high_arr,
        int(budget),
        int(seed),
        grad_fn,
        x0_arr,
    )
    out["best_pos"] = np.asarray(out["best_pos"], dtype=np.float64)
    return out


def _box_hop_kwargs(
    *,
    identity_tol=None,
    shared_deposits=None,
    shared_visit_policy=None,
    escape=None,
    langevin_steps=None,
    langevin_omega0=None,
    langevin_dt=None,
    langevin_noise=None,
    langevin_friction=None,
    coverage_shared=None,
    coverage_radius=None,
    coverage_neighbors=None,
    coverage_height=None,
    coverage_well_tempering=None,
    coverage_peer_weight=None,
):
    kwargs = {}
    if identity_tol is not None:
        kwargs["identity_tol"] = float(identity_tol)
    if shared_deposits is not None:
        kwargs["shared_deposits"] = int(shared_deposits)
    if shared_visit_policy is not None:
        kwargs["shared_visit_policy"] = str(shared_visit_policy)
    if escape is not None:
        kwargs["escape"] = str(escape)
    if langevin_steps is not None:
        kwargs["langevin_steps"] = int(langevin_steps)
    if langevin_omega0 is not None:
        kwargs["langevin_omega0"] = float(langevin_omega0)
    if langevin_dt is not None:
        kwargs["langevin_dt"] = float(langevin_dt)
    if langevin_noise is not None:
        kwargs["langevin_noise"] = str(langevin_noise)
    if langevin_friction is not None:
        kwargs["langevin_friction"] = float(langevin_friction)
    if coverage_shared is not None:
        kwargs["coverage_shared"] = bool(coverage_shared)
    if coverage_radius is not None:
        kwargs["coverage_radius"] = float(coverage_radius)
    if coverage_neighbors is not None:
        if not isinstance(coverage_neighbors, (int, np.integer)) or isinstance(
            coverage_neighbors, bool
        ):
            raise TypeError("coverage_neighbors must be an integer")
        kwargs["coverage_neighbors"] = int(coverage_neighbors)
    if coverage_height is not None:
        kwargs["coverage_height"] = float(coverage_height)
    if coverage_well_tempering is not None:
        kwargs["coverage_well_tempering"] = float(coverage_well_tempering)
    if coverage_peer_weight is not None:
        kwargs["coverage_peer_weight"] = float(coverage_peer_weight)
    return kwargs


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
    *,
    identity_tol: float | None = None,
    shared_deposits: int | None = None,
    shared_visit_policy: str | None = None,
    escape: str | None = None,
    langevin_steps: int | None = None,
    langevin_omega0: float | None = None,
    langevin_dt: float | None = None,
    langevin_noise: str | None = None,
    langevin_friction: float | None = None,
    coverage_shared: bool | None = None,
    coverage_radius: float | None = None,
    coverage_neighbors: int | None = None,
    coverage_height: float | None = None,
    coverage_well_tempering: float | None = None,
    coverage_peer_weight: float | None = None,
):
    """Communicating box hops on SharedDesignHistory / MinimumHistory.

    Requires ``grad_fn``. Runs ``escape="gaussian"`` (default reflected
    Gaussian kick) or ``escape="langevin"`` (GLE-Langevin escape controlled by
    ``langevin_*``) followed by L-BFGS quench on the box, with ``history`` /
    ``membership`` / ``identity_tol`` / ``shared_visit_policy`` /
    ``shared_deposits`` minimum-history controls and ``coverage_*`` peer
    separation. Without a gradient use ``ensemble_optimize``. Not cluster
    hopping.

    Returns ``best_val``, ``best_pos``, ``charged``, ``n_evals``, ``n_grads``,
    ``hops``, ``history_observations``, ``history_minima``,
    ``history_attempts``, ``history_refusals``, ``history_seconds``,
    ``shared_deposits``, ``coverage_*`` counters, and ``coverage_decisions``.
    """
    low_arr = np.asarray(low, dtype=np.float64)
    high_arr = np.asarray(high, dtype=np.float64)
    x0_arr = None if x0 is None else np.asarray(x0, dtype=np.float64)
    hop_kwargs = _box_hop_kwargs(
        identity_tol=identity_tol,
        shared_deposits=shared_deposits,
        shared_visit_policy=shared_visit_policy,
        escape=escape,
        langevin_steps=langevin_steps,
        langevin_omega0=langevin_omega0,
        langevin_dt=langevin_dt,
        langevin_noise=langevin_noise,
        langevin_friction=langevin_friction,
        coverage_shared=coverage_shared,
        coverage_radius=coverage_radius,
        coverage_neighbors=coverage_neighbors,
        coverage_height=coverage_height,
        coverage_well_tempering=coverage_well_tempering,
        coverage_peer_weight=coverage_peer_weight,
    )
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
        **hop_kwargs,
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
    identity_tol: float | None = None,
    shared_deposits: int | None = None,
    shared_visit_policy: str | None = None,
    escape: str | None = None,
    langevin_steps: int | None = None,
    langevin_omega0: float | None = None,
    langevin_dt: float | None = None,
    langevin_noise: str | None = None,
    langevin_friction: float | None = None,
    coverage_shared: bool | None = None,
    coverage_radius: float | None = None,
    coverage_neighbors: int | None = None,
    coverage_height: float | None = None,
    coverage_well_tempering: float | None = None,
    coverage_peer_weight: float | None = None,
):
    """Search on a design box.

    With ``grad_fn`` this hops and quenches (the kernel used by
    ``box_ensemble_optimize``). Without it, one replica uses the values-only
    portfolio; multiple replicas use communicating values-only hop chains.
    Equal bounds fix a coordinate without changing callback dimensions.

    The returned dictionary retains ``best_val``, ``best_pos``, separate
    ``n_evals`` / ``n_grads``, their sum ``charged``, ``hops``, minimum-history
    diagnostics (``history_observations``, ``history_minima``,
    ``history_attempts``, ``history_refusals``, ``history_seconds``),
    evaluated-region ``coverage_*`` counters, and ``coverage_decisions``.
    Coverage is not a count of certified minima.

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
    hop_kwargs = _box_hop_kwargs(
        identity_tol=identity_tol,
        shared_deposits=shared_deposits,
        shared_visit_policy=shared_visit_policy,
        escape=escape,
        langevin_steps=langevin_steps,
        langevin_omega0=langevin_omega0,
        langevin_dt=langevin_dt,
        langevin_noise=langevin_noise,
        langevin_friction=langevin_friction,
        coverage_shared=coverage_shared,
        coverage_radius=coverage_radius,
        coverage_neighbors=coverage_neighbors,
        coverage_height=coverage_height,
        coverage_well_tempering=coverage_well_tempering,
        coverage_peer_weight=coverage_peer_weight,
    )
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
        **hop_kwargs,
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
    identity_tol: float | None = None,
    shared_deposits: int | None = None,
    shared_visit_policy: str | None = None,
    escape: str | None = None,
    langevin_steps: int | None = None,
    langevin_omega0: float | None = None,
    langevin_dt: float | None = None,
    langevin_noise: str | None = None,
    langevin_friction: float | None = None,
    coverage_shared=None,
    coverage_radius=None,
    coverage_neighbors=None,
    coverage_height: float | None = None,
    coverage_well_tempering: float | None = None,
    coverage_peer_weight: float | None = None,
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
        identity_tol=identity_tol,
        shared_deposits=shared_deposits,
        shared_visit_policy=shared_visit_policy,
        escape=escape,
        langevin_steps=langevin_steps,
        langevin_omega0=langevin_omega0,
        langevin_dt=langevin_dt,
        langevin_noise=langevin_noise,
        langevin_friction=langevin_friction,
        coverage_shared=coverage_shared,
        coverage_radius=coverage_radius,
        coverage_neighbors=coverage_neighbors,
        coverage_height=coverage_height,
        coverage_well_tempering=coverage_well_tempering,
        coverage_peer_weight=coverage_peer_weight,
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
    out["best_pos"] = np.asarray(out["best_pos"], dtype=np.float64)
    return out


def dmc_population_optimize(
    obj_fn,
    low,
    high,
    budget: int,
    seed: int = 0,
    grad_fn=None,
    target_n: int | None = None,
    steps_per_control: int = DEFAULT_STEPS_PER_CONTROL,
    x0=None,
    *,
    beta0: float = DEFAULT_BETA0,
):
    """Population-controlled diffusion search (DMC-inspired; classical objective).

    Parameters
    ----------
    target_n :
        Optional walker population size. ``None`` (default) uses
        ``recommend_target_n(budget, dim)`` matching Rust
        ``dmc_population_optimize``.
    steps_per_control :
        Diffusion steps between population-control events (default ``2``).
    x0 :
        Optional starting point for walker 0 (protocol anchor / incumbent).
    beta0 :
        Initial inverse-temperature scale (default ``1.0``).

    Returns ``{best_val, best_pos, n_evals, n_grads, final_population,
    controls}``.
    """
    out = _core_dmc_population_optimize(
        obj_fn,
        _flat(low),
        _flat(high),
        int(budget),
        int(seed),
        grad_fn,
        None if target_n is None else int(target_n),
        int(steps_per_control),
        _flat_or_none(x0),
        beta0=float(beta0),
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
    coverage_height: float = 0.1,
    coverage_well_tempering: float = 5.0,
    coverage_peer_weight: float = 1.0,
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
    while it pays at the pace of its last few slices; if it stalls short
    of convergence, or stops at the start's value, before it has lowered
    its own value from below that one, CMA-ES takes one slice, once per
    run, and the opening goes on from there. With less, CMA-ES takes one
    slice from the start, the descent keeps the turn until its gain per
    evaluation falls or fewer than three slices are left before the
    closing descent's reserve, and annealing then keeps it while each
    slice gains as fast as the descent would have gone on to, the
    descent taking it back when one falls short. From there annealing
    and then CMA-ES each keep the turn while they lower the incumbent;
    the first annealing phase only while it gains faster per evaluation
    than the descent last did, and a CMA-ES phase that follows it on its
    own likewise from its eighth slice. Once an annealing phase has
    slowed past its fourth slice it lends CMA-ES a slice, which takes the
    turn only if it gains faster than both the slow slice and an eighth
    of the phase's pace so far; a phase CMA-ES took lends annealing a
    slice the same way, and the two alternate for up to eight phases.
    Arms not yet played then take a turn, while the budget holds
    one beyond the closing descent's reserve, before the allocation
    ranks them, and the same decaying uniform floor follows; a turn is
    one slice, except that the descent keeps it while it pays, until it
    converges. A turn pays when it lowers the incumbent by more than
    ``1e-4`` times the incumbent's last gain from below the start's
    value, floored at the resolution of the run's values, so the start's
    value sets no threshold and an added constant moves one only through
    the rounding of the shifted values. The descent closes the run,
    going on with its current descent, or restarting at the incumbent if
    another arm has lowered it since the descent last played. The floor
    the guarantee needs is asymptotic in the budget: the opening, the
    phases and the descent's turns go on only while they pay, which
    cannot last on a bounded objective, and after that every arm keeps a
    uniform share and is pulled infinitely often as the budget grows.

    Measured over 40 paired seeds, these rules trail the ones they
    replaced on some rows, each listed with its numbers, and with what
    the replaced rule costs elsewhere, in the documentation's regime
    auto-selection notes: the short-budget hand-over on shifted
    ellipsoids at 30 dimensions and 1000 evaluations, 50 and 100 at 5000
    and 100 at 20000, on the 100-dimensional rotated ellipsoid,
    Rosenbrock, Levy and shifted Rastrigin rows at 20000 and on rotated
    Styblinski-Tang at 30 and 1000; the opening's bar on the shifted
    ellipsoid and Rosenbrock at 30 dimensions and 5000; the descent's
    resolution test on the six-parameter fit at 1000 and on rotated
    ellipsoids at 30 and 50 dimensions and 20000, and together with the
    bar on the 30-dimensional one at 5000; the threshold on the last
    gain on rotated Rastrigin at 100 dimensions and 20000 and on Levy
    at 30 and 5000 with 100 added to it; lending from an annealing
    phase's fifth slice on Michalewicz at 10 dimensions and 5000; and
    copies and controls of these rows. Rotated ellipsoids short of the
    opening budget trail an earlier form of the loop, and rotated
    Rastrigin and Levy trail 0.10.0.

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
      coverage_height: well-tempered coverage deposit height.
      coverage_well_tempering: well-tempering factor (``> 1``).
      coverage_peer_weight: weight applied to foreign peer samples.

    ``obj_fn`` and ``grad_fn`` are only called inside ``[low, high]``: a
    point an arm proposes outside is mirror-reflected into the box, and the
    gradient there is reflected with it. Peer separation moves global
    candidates, not derivative stencils or raw objective values. Returns a
    dict with ``best_pos``, ``best_val``, ``n_evals``, ``n_grads``,
    ``arm_pulls``, ``arm_successes``, and ``hop_state``. Multiple replicas also
    report ``replicas``, ``charged``, and ``coverage_*`` sample counters
    (``coverage_published_samples``, ``coverage_applied_foreign_samples``,
    ``coverage_sample_peer_checks``, ``coverage_sample_anchor_overlaps``,
    ``coverage_sample_anchor_only_overlaps``, ``coverage_sample_overlaps``,
    ``coverage_repelled_proposals``, ``coverage_constrained_repulsions``).
    """
    # int() truncates a float, so 1.5 would run as 1. A neighbour count is whole.
    if not isinstance(coverage_neighbors, (int, np.integer)):
        raise TypeError("coverage_neighbors must be an integer")
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
        coverage_height=float(coverage_height),
        coverage_well_tempering=float(coverage_well_tempering),
        coverage_peer_weight=float(coverage_peer_weight),
    )
    out["best_pos"] = np.asarray(out["best_pos"], dtype=np.float64)
    if out.get("hop_state") is not None:
        pos, val = out["hop_state"]
        out["hop_state"] = (np.asarray(pos, dtype=np.float64), float(val))
    return out


def global_optimize_objective(
    objective,
    budget: int,
    seed: int = 0,
    use_gradient: bool = True,
    x0=None,
    *,
    noise_sigma: float | None = None,
    policy: str = "auto",
):
    """Portfolio global optimizer over a native ``PyObjective`` handle.

    Returns a dict with ``best_pos``, ``best_val``, ``n_evals``, ``n_grads``,
    ``arm_pulls``, ``arm_successes``, and ``hop_state``.
    """
    out = _core_global_optimize_objective(
        objective,
        int(budget),
        int(seed),
        bool(use_gradient),
        None if noise_sigma is None else float(noise_sigma),
        _flat_or_none(x0),
        policy=str(policy),
    )
    out["best_pos"] = np.asarray(out["best_pos"], dtype=np.float64)
    if out.get("hop_state") is not None:
        pos, val = out["hop_state"]
        out["hop_state"] = (np.asarray(pos, dtype=np.float64), float(val))
    return out


__all__ = [
    "BasinBias",
    "BinaryLennardJones",
    "Boltzmann",
    "Bounds",
    "COPPER_Z",
    "Config",
    "DEFAULT_BETA0",
    "DEFAULT_HIT_TOL",
    "DEFAULT_STEPS_PER_CONTROL",
    "DEFAULT_TARGET_WALKERS",
    "GOLD_Z",
    "GminTip4p",
    "Gupta",
    "IDENTITY_TOL",
    "PLATINUM_Z",
    "PairPotential",
    "PotentialCalculator",
    "SEARCH_HIT_TOL",
    "SuttonChen",
    "Tip4pCluster",
    "DeviceHistory",
    "EnsembleHistory",
    "EpochLine",
    "Fast",
    "Gsa",
    "History",
    "LandfoldMap",
    "Ledger",
    "PyObjective",
    "Anneal",
    "block_from_interleaved_gmin",
    "cluster_search",
    "energy_hits",
    "energy_ohh",
    "gmin_tip4p_sites",
    "interleaved_gmin_from_block",
    "osa_acceptance_rate",
    "recommend_target_n",
    "rigid_from_ohh",
    "TvmFfiTensorMetadata",
    "__version__",
    "low_discrepancy_points",
    "pilot_draws_qmc",
    "polish",
    "quench",
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
    "landfold",
    "landfold_book",
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
    if name == "PotentialCalculator":
        from anneal.ase import PotentialCalculator as _PotentialCalculator

        return _PotentialCalculator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

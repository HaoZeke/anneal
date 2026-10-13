"""anneal: simulated annealing on the eindir typed primitives.

Search splits on geometry, then on gradient:

  - cluster_search(obj, grad, n, budget, ...): 3N point set.
    recommended / derived / ras select the hop preset.
  - ensemble_optimize(obj, low, high, budget, grad_fn=None, ...): design box.
    Gradient means hop and quench; no gradient means a values-only search.
  - minimize(fun, x0, bounds, jac=None, budget=..., replicas=1, store=...):
    SciPy/ChemFit shape over the portfolio from x0; more replicas run as
    batch replicas evaluated together. store is an existing HDF5 file or a
    readcon-db campaign directory.
  - box_ensemble_optimize(...): hop primitive. Requires grad_fn.
  - global_optimize(...): portfolio arms when the caller does not want hops.
  - run(obj_fn, low, high, preset, n_epochs, steps_per_epoch, seed, x0): SA
    loop. Every proposal is reflected into [low, high] and the walk starts
    at x0 when one is given.
  - fit_anneal, fit_chemfit, run_benchmark, run_fitter: gradient-free ChemFit
    bridges. They drive evaluate/step or ask/tell and raise the first exception
    the fitter raises, but run_fitter calls a fitter's own fit_anneal instead.

SA algebra (Cool / Move / Accept) stays on run / run_device / run_ensemble
with Boltzmann, Fast, and Gsa. Last mile is polish / qmc_polish
(bounded L-BFGS). Cluster quench is WarmLbfgs to rgmin::Lbfgs.

The IISE composition laws L1-L4 are enforced in SaVariant::checked.
"""

import json
import operator
import warnings

import numpy as np

from anneal._core import (
    BasinBias,
    Boltzmann,
    Bounds as _CoreBounds,
    Config,
    EpochLine,
    Fast,
    Gsa,
    History,
    Ledger,
    PyObjective as _CorePyObjective,
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
    global_optimize_replicas as _core_global_optimize_replicas,
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


class _ExtensionAlias(type):
    """Metaclass of a Python class standing in for an extension class.

    ``isinstance`` and ``issubclass`` against the class itself, though not
    against its subclasses, also admit the extension class, whose instances
    the extension itself returns.
    """

    def __instancecheck__(cls, instance):
        if type.__instancecheck__(cls, instance):
            return True
        extension = cls.__dict__.get("_extension")
        return extension is not None and isinstance(instance, extension)

    def __subclasscheck__(cls, subclass):
        if type.__subclasscheck__(cls, subclass):
            return True
        extension = cls.__dict__.get("_extension")
        return extension is not None and issubclass(subclass, extension)


def _refuse_a_nan_wall(low, high):
    """Raise ValueError naming the first coordinate whose wall is NaN."""
    nan = np.flatnonzero(np.isnan(low) | np.isnan(high))
    if nan.size:
        i = nan[0]
        msg = f"coordinate {i} has a NaN wall: low {low[i]}, high {high[i]}"
        raise ValueError(msg)


class Bounds(metaclass=_ExtensionAlias):
    """The box ``low <= x <= high``, widened by ``slack`` in :meth:`contains`.

    ``low``, ``high`` and the points :meth:`contains` and :meth:`clip` take
    may be any array-likes of one size, strided views, transposes and the
    columns of a ``(dim, 2)`` array included; each is flattened in C order
    into a float64 copy, as ``global_optimize`` takes its vectors.
    A wall may be infinite but not NaN, and ``low`` may not exceed ``high``:
    either is a ValueError naming the coordinate.
    ``PyObjective.bounds`` returns the extension's own box, which counts as a
    ``Bounds`` too.
    """

    _extension = _CoreBounds
    __slots__ = ("_core", "_high", "_low")

    def __init__(self, low, high, slack=1e-6):
        low = _flat(low).copy()
        high = _flat(high).copy()
        slack = float(slack)
        if low.size != high.size:
            msg = f"low and high must have one size, got {low.size} and {high.size}"
            raise ValueError(msg)
        if not slack >= 0.0:
            raise ValueError(f"slack must be at least 0, got {slack!r}")
        _refuse_a_nan_wall(low, high)
        above = np.flatnonzero(low > high)
        if above.size:
            i = above[0]
            raise ValueError(f"coordinate {i} has low {low[i]} above high {high[i]}")
        low.flags.writeable = False
        high.flags.writeable = False
        self._low = low
        self._high = high
        self._core = _CoreBounds(low, high, slack)

    @property
    def dims(self):
        """Number of coordinates."""
        return self._core.dims

    @property
    def slack(self):
        """How far beyond a face :meth:`contains` still admits a point."""
        return self._core.slack

    @property
    def low(self):
        """The lower faces, as a read-only float64 vector."""
        return self._low

    @property
    def high(self):
        """The upper faces, as a read-only float64 vector."""
        return self._high

    def contains(self, x):
        """True if ``x`` has ``dims`` coordinates, each within ``slack`` of the box."""
        return self._core.contains(_flat(x))

    def clip(self, x):
        """``x`` with each coordinate clipped onto ``[low, high]``."""
        x = _flat(x)
        if x.size != self.dims:
            raise ValueError(f"x must have {self.dims} coordinates, got {x.size}")
        return self._core.clip(x)

    def __repr__(self):
        low, high = self._low.tolist(), self._high.tolist()
        name = type(self).__name__
        return f"{name}(low={low!r}, high={high!r}, slack={self.slack!r})"

    def __reduce__(self):
        return (type(self), (self._low, self._high, self.slack))


class PyObjective(metaclass=_ExtensionAlias):
    """A Python callable as a native objective over ``bounds``.

    ``grad_fn``, when given, is its native gradient: ``qmc_polish_objective``,
    ``gle_langevin_objective``, ``gle_langevin_preconditioned_objective``
    and ``global_optimize_objective`` read their gradients through it, and
    the other ``*_objective`` functions read values only. Given a handle
    built without ``grad_fn``, those four end the run at the first gradient
    they ask for and raise a ``ValueError`` saying so;
    ``global_optimize_objective`` runs on values alone with
    ``use_gradient=False``. ``bounds`` is a :class:`Bounds`. The object made
    is the extension's own ``PyObjective``, the handle the ``*_objective``
    functions take, and counts as a ``PyObjective``.

    ``grad_fn`` may return any array-like of ``bounds.dims`` numbers, as
    ``global_optimize`` takes its gradients: a strided view, a list, a
    column or another dtype is read as a contiguous float64 vector. In the
    ``*_objective`` functions a gradient of another length, or an exception
    from ``grad_fn``, ends the run, and the function raises that exception,
    or a ``TypeError`` giving the length, when it returns.

    ``fn_`` returns a number, anything ``global_optimize`` takes as a value.
    In the ``*_objective`` functions an exception from ``fn_`` ends the run
    and is raised when the function returns, and a value that is not a
    number ends it with the ``TypeError`` ``global_optimize`` raises. A NaN
    value is scored as ``+inf``, as ``global_optimize`` scores it.
    """

    _extension = _CorePyObjective

    def __new__(cls, fn_, bounds, grad_fn=None):
        if isinstance(bounds, Bounds) and not isinstance(bounds, _CoreBounds):
            bounds = bounds._core
        if grad_fn is not None:
            grad_fn = _float_gradient(grad_fn)
        return _CorePyObjective(_float_value(fn_), bounds, grad_fn)


def _float_value(fn_):
    """``fn_`` with what it returns read as a float, as ``global_optimize`` reads it."""

    def value(x):
        return _read_value(fn_(x))

    return value


def _read_value(result):
    """What an objective returned as a float, or ``global_optimize``'s ``TypeError``.

    A number is what float conversion takes without parsing text: a value
    with ``__float__`` or ``__index__``.
    """
    kind = type(result)
    if kind is float:
        return result
    cause = None
    if hasattr(kind, "__float__") or hasattr(kind, "__index__"):
        try:
            return float(result)
        except Exception as err:
            cause = err
    msg = f"obj_fn must return a float, got {kind.__name__}"
    raise TypeError(msg) from cause


def _scored_value(fun, x):
    """``fun(x)`` scored as the drivers score one evaluation.

    An ``Exception`` from ``fun`` is scored as ``+inf`` and reported in the
    warning the drivers give when a run ends; anything else ``fun`` raises
    propagates. The value is read by :func:`_read_value`, and NaN is scored
    as ``+inf``.
    """
    try:
        result = fun(x)
    except Exception as err:
        try:
            text = str(err)
        except Exception:
            text = "<exception str() failed>"
        message = (
            "a callback raised 1 exception(s): an objective call that raised was "
            "scored as the worst value and a gradient call as zero; the first was: "
            f"{type(err).__qualname__}: {text}"
        )
        warnings.warn(message, RuntimeWarning, stacklevel=3)
        return np.inf
    value = _read_value(result)
    return np.inf if np.isnan(value) else value


def _float_gradient(grad_fn):
    """``grad_fn`` with what it returns read as a contiguous float64 vector."""

    def gradient(x):
        value = grad_fn(x)
        try:
            return _flat(value)
        except (TypeError, ValueError) as err:
            msg = f"grad_fn must return numbers, got {type(value).__name__}"
            raise TypeError(msg) from err

    return gradient


def _count(name, value, minimum):
    """A whole number of at least ``minimum``, or a ValueError naming it."""
    count = int(value)
    if count != value or count < minimum:
        msg = f"{name} must be a whole number of at least {minimum}, got {value!r}"
        raise ValueError(msg)
    return count


def _seed(value):
    """``seed`` as a whole number the engine holds.

    The engine reads a ``u64`` and refuses a seed outside it with an
    ``OverflowError``; this refuses it the same way, also on a box with every
    coordinate fixed, where the engine never runs.
    """
    seed = int(value)
    largest = 2**64 - 1
    if not 0 <= seed <= largest:
        raise OverflowError(f"seed must be from 0 to {largest}, got {value!r}")
    return seed


def _coverage_radius(value):
    """``None``, or ``coverage_radius`` as a finite positive float."""
    if value is None:
        return None
    if isinstance(value, (str, bytes, bool, np.bool_)):
        raise TypeError(f"coverage_radius must be a number, got {value!r}")
    radius = float(value)
    if not (np.isfinite(radius) and radius > 0.0):
        raise ValueError(f"coverage_radius must be finite and positive, got {value!r}")
    return radius


def _coverage_neighbors(value):
    """``None``, or ``coverage_neighbors`` as a whole number the engine holds.

    The engine reads a ``usize``; a larger number is refused with the
    ``OverflowError`` the engine raises, before any warning about the keyword.
    """
    if value is None:
        return None
    neighbors = operator.index(value)
    if neighbors < 0:
        raise ValueError(f"coverage_neighbors must be at least 0, got {value!r}")
    largest = int(np.iinfo(np.uintp).max)
    if neighbors > largest:
        msg = f"coverage_neighbors must be at most {largest}, got {value!r}"
        raise OverflowError(msg)
    return neighbors


def _membership(value):
    """``membership`` itself, if it names a policy."""
    if value not in ("accepted", "observed-exclusion"):
        msg = f"membership must be accepted or observed-exclusion, got {value!r}"
        raise ValueError(msg)
    return value


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
        A coordinate whose two ends are equal is fixed: the run makes the
        calls of the same run on the other coordinates, and ``obj_fn``
        receives each point at full length, with the fixed values. With
        every coordinate fixed, the run is one call at ``x0``, or at
        ``low`` without it, and has no epochs.
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
    ``n_starts * (1 + n_epochs * steps_per_epoch)`` calls. Fixed
    coordinates, whose two ends are equal, are held as :func:`run` holds
    them, the design spanning the free ones; with every coordinate fixed
    the run is one call.
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
    start_arr = _flat_or_none(start)
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

    ``low``, ``high`` and ``x0`` may be any array-like of one shape; they are
    flattened in C order and the callables receive the flat vector.
    """
    low_arr = _flat(low)
    high_arr = _flat(high)
    x0_arr = _flat_or_none(x0)
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
        _membership(membership),
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
    ``n_evals`` / ``n_grads``, their sum ``charged``, ``n_answered`` (the
    evaluations one values-only portfolio answers from an earlier call,
    which count against the budget but not in ``charged``; zero for hop
    chains), ``hops``, minimum-history diagnostics, and evaluated-region
    ``coverage_*`` counters. Coverage is not a count of certified minima.

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
    ``low``, ``high`` and ``x0`` may be any array-like of one shape; they are
    flattened in C order and the callables receive the flat vector.
    """
    low_arr = _flat(low)
    high_arr = _flat(high)
    x0_arr = _flat_or_none(x0)
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
        _membership(membership),
        coverage_shared=coverage_shared,
        coverage_radius=_coverage_radius(coverage_radius),
        coverage_neighbors=_coverage_neighbors(coverage_neighbors),
    )
    out["best_pos"] = np.asarray(out["best_pos"], dtype=np.float64)
    return out


class MinimizeResult:
    """SciPy-shaped result from :func:`minimize`.

    ``nfev`` counts objective calls, ``njev`` counts gradient calls, and
    ``charged`` is their sum; an evaluation answered from an earlier call
    counts against the budget and in none of them, but in
    ``diagnostics["n_answered"]``, as does a gradient asked at a point that
    is not finite, which ``jac`` never sees. ``nit`` counts sequential
    evaluation steps: every call with one replica, every batch with more.
    ``diagnostics`` retains the engine's result dict, with ``best_pos`` at
    full length. ``success`` means a finite feasible candidate was
    returned; it does not certify local convergence or global optimality.
    """

    __slots__ = (
        "x",
        "fun",
        "nfev",
        "njev",
        "nit",
        "charged",
        "success",
        "message",
        "diagnostics",
    )

    def __init__(
        self, x, fun, nfev, success, message, *, njev=0, nit=0, diagnostics=None
    ):
        self.x = np.asarray(x, dtype=np.float64)
        self.fun = float(fun)
        self.nfev = int(nfev)
        self.njev = int(njev)
        self.nit = int(nit)
        self.charged = self.nfev + self.njev
        self.success = bool(success)
        self.message = str(message)
        self.diagnostics = {} if diagnostics is None else dict(diagnostics)

    def __repr__(self):
        return (
            f"MinimizeResult(fun={self.fun!r}, nfev={self.nfev}, "
            f"njev={self.njev}, nit={self.nit}, charged={self.charged}, "
            f"success={self.success})"
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
    """``(low, high)`` from SciPy ``Bounds``, a ``(dim, 2)`` array or ``dim`` pairs."""
    if hasattr(bounds, "lb") and hasattr(bounds, "ub"):
        try:
            low = np.broadcast_to(np.asarray(bounds.lb, dtype=np.float64), (dim,))
            high = np.broadcast_to(np.asarray(bounds.ub, dtype=np.float64), (dim,))
        except ValueError as err:
            msg = f"bounds.lb and bounds.ub must have length {dim}, the length of x0"
            raise ValueError(msg) from err
    else:
        try:
            pairs = np.asarray(bounds, dtype=np.float64)
        except (TypeError, ValueError) as err:
            raise ValueError("bounds must be (low, high) pairs of numbers") from err
        if pairs.shape != (dim, 2):
            msg = (
                f"bounds must be {dim} (low, high) pairs, one per coordinate of x0, "
                f"got shape {pairs.shape}"
            )
            raise ValueError(msg)
        low, high = pairs[:, 0], pairs[:, 1]
    low = np.array(low, dtype=np.float64)
    high = np.array(high, dtype=np.float64)
    for i in range(dim):
        if not (np.isfinite(low[i]) and np.isfinite(high[i])):
            raise ValueError(f"bounds must be finite, at coordinate {i}")
        if low[i] > high[i]:
            msg = f"bounds at coordinate {i} have low {low[i]} above high {high[i]}"
            raise ValueError(msg)
    return low, high


class _EndRun(BaseException):
    """Carries ``error`` out of a callback so that it ends the run.

    The drivers score an ``Exception`` from a callback as the worst value or
    a zero gradient and go on; anything else ends the run, and the driver
    raises it when it returns.
    """

    def __init__(self, error):
        super().__init__(error)
        self.error = error


class _FreeCoordinates:
    """``fun`` seen from the free coordinates of a box with fixed ones.

    Each point is completed with the fixed values before ``fun`` sees it, and
    ``eval_batch`` is offered only when ``fun`` has one. The gradient is read
    at full length, as ``global_optimize`` reads it, before its free part is
    taken: one that is not numbers or has another length ends the run with
    ``global_optimize``'s ``TypeError``, carried in an :class:`_EndRun`.
    """

    def __init__(self, fun, template, free):
        self._fun = fun
        self._template = template
        self._free = free
        if callable(getattr(fun, "eval_batch", None)):
            self.eval_batch = self._eval_batch

    def full(self, z):
        x = self._template.copy()
        x[self._free] = z
        return x

    def __call__(self, z):
        return self._fun(self.full(z))

    def _eval_batch(self, rows):
        rows = np.asarray(rows, dtype=np.float64)
        full = np.repeat(self._template[None, :], rows.shape[0], axis=0)
        full[:, self._free] = rows
        return self._fun.eval_batch(full)

    def gradient(self, jac):
        n = self._template.size

        def grad(z):
            value = jac(self.full(z))
            try:
                g = np.asarray(value, dtype=np.float64).reshape(-1)
            except Exception:
                g = None
            if g is None:
                raise _EndRun(TypeError(f"grad_fn must return {n} numbers"))
            if g.size != n:
                raise _EndRun(TypeError(f"grad_fn must return {n} numbers, got {g.size}"))
            return g[self._free]

        return grad


def _stored_start(archive, low, high):
    """The best finite observation of ``archive`` inside the box, or ``None``."""
    best = None
    for x, f in archive.observations():
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        f = float(f)
        if not np.isfinite(f) or x.size != low.size:
            continue
        if not (np.all(x >= low) and np.all(x <= high)):
            continue
        if best is None or f < best[1]:
            best = (x.copy(), f)
    return None if best is None else best[0]


def _replica_archive(history, coverage_shared):
    """``"shared"`` or ``"private"``: whether batch replicas pool their evaluations."""
    if history not in ("shared", "private", "none", "no"):
        raise ValueError(f"history must be none, private, or shared, got {history!r}")
    if coverage_shared is None:
        return "shared" if history == "shared" else "private"
    if not isinstance(coverage_shared, (bool, np.bool_)):
        msg = f"coverage_shared must be True, False or None, got {coverage_shared!r}"
        raise TypeError(msg)
    return "shared" if coverage_shared else "private"


def _ignored_minimize_keywords(membership, coverage_radius, coverage_neighbors):
    """Check the keywords minimize no longer uses, and warn for each one given."""
    ignored = []
    if membership is not None:
        _membership(membership)
        ignored.append(
            "membership has no effect: minimize keeps no ledger of certified "
            "minima for it to select"
        )
    return ignored + _ignored_by_batch_replicas(coverage_radius, coverage_neighbors)


def _ignored_by_batch_replicas(coverage_radius, coverage_neighbors):
    """Check the coverage keywords, and say why batch replicas ignore each one given."""
    ignored = []
    if _coverage_radius(coverage_radius) is not None:
        ignored.append(
            "coverage_radius has no effect: batch replicas share evaluated points "
            "through their archive and do not repel each other's proposals"
        )
    if _coverage_neighbors(coverage_neighbors) is not None:
        ignored.append(
            "coverage_neighbors has no effect: batch replicas pool their "
            "evaluations between all of them or keep them each to itself"
        )
    return ignored


def _future_warnings(reasons):
    """The ``FutureWarning`` message for each keyword that changes nothing yet.

    They are given once every argument has been accepted, by the native
    driver when one runs, so that a refused call raises without one first.
    """
    return [f"{reason}. It will raise in a future release." for reason in reasons]


def minimize(
    fun,
    x0,
    bounds,
    jac=None,
    *,
    budget=None,
    seed=0,
    replicas=1,
    history="shared",
    membership=None,
    store=None,
    coverage_shared=None,
    coverage_radius=None,
    coverage_neighbors=None,
):
    """Minimize ``fun`` over a box from the start point ``x0``.

    A SciPy-shaped front end over the portfolio of :func:`global_optimize`.

    Args:
      fun: callable ``fun(x) -> float`` on vectors of the length of ``x0``.
        When it has a method ``eval_batch(X)`` that takes a 2-D array, one
        point per row, and returns their values, batch replicas evaluate
        each step's points through it.
      x0: start point, finite and inside ``bounds``. It is the run's first
        evaluation, made once whatever the number of replicas: the run
        answers an evaluation at a point ``fun`` has already been called
        at, ``x0`` or any other, from that call, and one of ``jac``
        likewise.
      bounds: the box, as SciPy ``Bounds`` (anything with ``lb`` and ``ub``;
        scalars broadcast), an ``(n, 2)`` array, or ``n`` pairs
        ``(low, high)``, all finite. A coordinate whose two ends are equal
        is fixed: the search runs in the free coordinates, and ``fun`` and
        ``jac`` still receive full-length vectors. With every coordinate
        fixed, the run is one evaluation of ``fun`` at the start, and
        ``jac`` is not called. Without ``jac`` the run is that evaluation
        too when no free side is wide enough for a difference step to
        change its coordinate at the start, a unit or two in the last place.
      jac: optional gradient callable ``jac(x) -> array`` of the length of
        ``x0``. It takes the gradient route of :func:`global_optimize`
        (``grad_fn``), which runs one replica; its calls count in ``njev``.
      budget: total evaluations, objective and gradient together, over all
        replicas, those answered from an earlier call included: ``nfev +
        njev <= budget``. The default is
        ``max(1000, 10 * n * (n + 1))`` for ``n`` free coordinates, twice
        the budget from which the portfolio opens with a descent from ``x0``
        that can converge, and at least 1000.
      seed: RNG seed, from 0 to ``2**64 - 1``; one outside that range
        raises ``OverflowError``. The same seed and budget repeat a run.
      replicas: portfolio instances run together. With one, the default,
        ``minimize`` runs ``global_optimize(fun, low, high, budget,
        seed=seed, grad_fn=jac, x0=x0)`` on the free coordinates and repeats
        it call for call. With ``R > 1`` and no ``jac``, ``R`` values-only
        portfolios start from ``x0`` and split the rest of the budget as
        evenly as it goes; at each step every replica still running
        proposes one point, and the points are evaluated as one batch of at
        most ``R``, through ``fun.eval_batch`` when ``fun`` has it and one
        by one otherwise. Replica 0 keeps ``seed`` and the others mix their
        index into it. Replica ``r`` also differences its descent's
        gradients from coordinate ``r * n / R`` on, which leaves each
        gradient unchanged but has replicas that descend from one point ask
        for different stencil points. These are the replicas
        ``global_optimize(fun, low, high, budget, seed=seed, x0=x0,
        replicas=R)`` runs, its ``coverage_shared`` standing for
        ``history`` and its ``coverage_radius`` and ``coverage_neighbors``
        warning as they do here, and ``minimize`` repeats it call for call
        on the free coordinates. With ``jac``, ``replicas`` above one gives a
        ``FutureWarning`` and one replica runs.
      history: with replicas, ``"shared"`` (the default) keeps one archive
        and one incumbent for all of them: arms that start at the incumbent
        start at the best point any replica has found, and the additive
        surrogate fits every replica's evaluations. A point another replica
        has already evaluated is answered without a charge, once per replica,
        and replicas that propose one point at the same step share one row
        of the batch, charged to the one with the most of its share left,
        so a descent they follow together costs the budget once and takes
        about ``n / R`` batches per gradient. ``"private"`` gives each
        replica its own archive, as independent runs batched together would
        have, each paying for every point it asks for while ``fun`` is
        called once at a point, and so does ``"none"``: ``minimize`` keeps
        no ledger of certified minima for it to switch off.
      membership: accepted and checked, with a ``FutureWarning``: it chose
        which certified minima entered that ledger.
      store: parameter archive: an ``.h5`` / ``.hdf5`` or ``.jsonl`` path, a
        campaign directory that already holds a readcon-db corpus (params
        are written beside it, never padded into CON frames), or an object
        with ``observations()`` and ``record(x, f)``. When it holds a
        finite observation inside ``bounds``, the run starts from the best
        such one instead of ``x0``; the result is recorded back.
      coverage_shared: whether batch replicas pool their evaluations; when
        given it overrides ``history``, so ``history="none",
        coverage_shared=True`` keeps one archive.
      coverage_radius, coverage_neighbors: accepted and checked, with a
        ``FutureWarning``: batch replicas pool their evaluations between
        all of them or none, and do not repel each other's proposals within
        a radius. This warning, like those for ``membership`` and for
        ``replicas`` with ``jac``, comes once every argument has been
        accepted, so a call refused for any argument raises without one.

    Returns:
      A :class:`MinimizeResult` with ``x`` (full length), ``fun``, ``nfev``,
      ``njev``, ``charged`` (``nfev + njev``), ``nit`` (sequential
      evaluation steps: every call with one replica, every batch with
      more), ``success`` (``fun`` is finite; not a certificate of a global
      minimum), ``message`` and ``diagnostics``, the engine's result dict
      with ``best_pos`` at full length.

    ``fun`` and ``jac`` are only called inside the box: a point an arm
    proposes outside it is mirror-reflected in. As in :func:`global_optimize`,
    an exception ``fun`` raises is scored as the worst value and reported in
    one warning when the run ends, and ``KeyboardInterrupt`` ends the run.
    Atomic symmetry-aware proposals are selected explicitly through
    ``cluster_search``, not inferred from vector length.
    """
    start = _flat(x0)
    n = start.size
    if n == 0:
        raise ValueError("x0 must hold at least one coordinate")
    low, high = _scipy_bounds_to_low_high(bounds, n)
    for i, value in enumerate(start):
        if not np.isfinite(value):
            raise ValueError(f"x0[{i}] is not finite")
        if value < low[i] or value > high[i]:
            raise ValueError(f"x0[{i}] = {value} lies outside [{low[i]}, {high[i]}]")
    replicas = _count("replicas", replicas, 1)
    archive_mode = _replica_archive(history, coverage_shared)
    ignored = _ignored_minimize_keywords(membership, coverage_radius, coverage_neighbors)
    if jac is not None and not callable(jac):
        raise TypeError("jac must be a callable returning the gradient of fun, or None")
    free = low < high
    n_free = int(np.count_nonzero(free))
    if budget is None:
        budget = max(1000, 10 * n_free * (n_free + 1))
    budget = _count("budget", budget, 1)
    seed = _seed(seed)
    if jac is not None and replicas > 1:
        ignored.append(
            f"replicas={replicas} has no effect with jac: the gradient route runs "
            "one replica; pass replicas=1 with jac"
        )
        replicas = 1
    deferred = _future_warnings(ignored)
    archive = open_parameter_store(store)
    if archive is not None:
        stored = _stored_start(archive, low, high)
        if stored is not None:
            start = stored

    if n_free == 0:
        for message in deferred:
            warnings.warn(message, FutureWarning, stacklevel=2)
        value = _scored_value(fun, start.copy())
        out = {
            "best_pos": np.empty(0),
            "best_val": value,
            "n_evals": 1,
            "n_grads": 0,
            "n_answered": 0,
        }
        nit = 1
    else:
        if n_free == n:
            objective, gradient = fun, jac
        else:
            objective = _FreeCoordinates(fun, start, free)
            gradient = None if jac is None else objective.gradient(jac)
        if replicas == 1:
            try:
                out = _core_global_optimize(
                    objective,
                    low[free],
                    high[free],
                    budget,
                    seed,
                    gradient,
                    x0=start[free],
                    future_warnings=deferred,
                )
            except _EndRun as stop:
                raise stop.error from None
            nit = out["n_evals"] + out["n_grads"]
        else:
            out = _core_global_optimize_replicas(
                objective,
                low[free],
                high[free],
                budget,
                start[free],
                seed,
                replicas,
                archive_mode,
                future_warnings=deferred,
            )
            nit = out["n_batches"]

    x = start.copy()
    x[free] = np.asarray(out["best_pos"], dtype=np.float64)
    out["best_pos"] = x.copy()
    value = float(out["best_val"])
    if archive is not None and np.isfinite(value):
        archive.record(x, value)
    return MinimizeResult(
        x=x,
        fun=value,
        nfev=int(out["n_evals"]),
        njev=int(out["n_grads"]),
        nit=int(nit),
        success=bool(np.isfinite(value)),
        message=(
            "Finite evaluated candidate returned; global optimality is not certified."
            if np.isfinite(value)
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
    coverage_radius=None,
    coverage_neighbors=None,
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
    evolution, the additive surrogate, and the QMC restart arm. The
    descent also models the logarithm of its values while they stay
    positive. Each curvature pair counts for the logarithm when the
    trapezoid rule along its step misses the change of the values more
    than four times as far as the change of their logarithm, and for the
    values when it misses the logarithm's that much further; from two
    pairs per coordinate on, once more of the pairs since the descent
    last started afresh have counted for the logarithm than for the
    values, as they do for a squared error far from its fit, it steps on
    that model, and from then until it next starts afresh each accepted
    step puts the next line search on whichever model predicted the
    step's value more closely. Multiplying the objective by a
    power of two leaves a run unchanged, while adding a constant to a
    positive objective can change the descent's steps. When the budget lets
    a descent converge, one from the start opens the run while it pays
    at the pace of its last few slices, whose gains after the first
    from below the start read as log ratios while every value is
    positive; if it stalls short of
    convergence, or stops at the start's value, before it has lowered
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

    Measured over 40 paired seeds, the loop trails its form before these
    rules on some rows, each listed with its numbers and what the
    replaced rule costs elsewhere, in the documentation's regime
    auto-selection notes: the short-budget hand-over on shifted
    ellipsoids at 30 dimensions and 1000 evaluations, 50 and 100 at 5000
    and 100 at 20000, on the 100-dimensional rotated ellipsoid,
    Rosenbrock, Levy and shifted Rastrigin rows at 20000 and on rotated
    Styblinski-Tang at 30 and 1000; the descent's resolution test and
    its floor on Griewank at 10 dimensions and 5000 without ``x0``;
    the threshold on the last gain on rotated Rastrigin at 100
    dimensions and 20000 and on Levy at 30 and 5000 with 100 added to
    it; lending from an annealing phase's fifth slice on Michalewicz at
    10 dimensions and 5000; and copies and controls of these rows.
    Rotated ellipsoids short of the opening budget trail an earlier
    form of the loop, and rotated Rastrigin and Levy trail 0.10.0.

    Args:
      obj_fn: callable ``f(numpy.ndarray) -> float``.
      low, high: box bounds. A coordinate whose two ends are equal is
        fixed: as in :func:`minimize`, the run makes the calls of the same
        run on the other coordinates, and ``obj_fn`` and ``grad_fn``
        receive each point at full length, with the fixed values. With
        every coordinate fixed, the run is one call of ``obj_fn`` at
        ``x0``, or at ``low`` without it, whatever the number of replicas;
        ``grad_fn`` is not called, and the dict holds the keys of one
        controller, its arm counts empty. In the values-only loop, a box
        with no free side wide enough for a difference step to change its
        coordinate at the start, a unit or two in the last place, ends the
        run there: it evaluates ``x0`` once, or without it the box's centre
        once per replica, up to the budget.
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
      x0: optional starting point, finite and inside the box, faces
        included; one with a coordinate outside it raises ``ValueError``
        before any call, as one of the wrong length or not finite does. It
        is charged as the first evaluation and the first incumbent. It stays
        the incumbent until a lower value is found (if its value is not
        finite, until any finite value is). The values-only loop starts its
        descents, CMA-ES, one annealing chain and the first evolution member
        there; with ``grad_fn`` or ``noise_sigma``, arms that read the
        incumbent do, such as the trust-region poll and the population arm,
        and with ``grad_fn`` also basin hopping and HMC. Without ``x0`` the
        values-only loop starts from the best of a small seeded design.
        With ``replicas`` above one, ``x0`` is still the first evaluation,
        made once whatever their number. Without ``noise_sigma`` the run
        answers an evaluation at a point ``obj_fn`` has already been called
        at, ``x0`` or any other, from that call, and one of ``grad_fn``
        likewise: it costs its unit of ``budget`` but no call, and
        ``n_evals`` and ``n_grads`` count calls. Persistent replicas are
        given ``x0``'s value and otherwise answer only the points each has
        called itself, so a point two of them reach is called by each, and
        the gradient at ``x0`` by each that asks for it. With ``noise_sigma``
        every evaluation is a call, at ``x0`` too, since a noisy value is a
        draw.
      replicas: portfolio instances sharing the aggregate budget. One, the
        default, is the single controller. With ``x0``, no ``grad_fn``, no
        ``noise_sigma`` and ``policy="auto"``, ``R`` replicas are the batch
        replicas of :func:`minimize`: ``R`` values-only loops start from
        ``x0``, evaluated once and shared, and step together, each step's
        points that no replica has evaluated before going to
        ``obj_fn.eval_batch`` when it has one, so
        ``minimize(obj_fn, x0, bounds, budget=budget, seed=seed,
        replicas=R)`` on the same box repeats the run call for call. As in
        :func:`minimize`, ``coverage_radius`` and ``coverage_neighbors``
        have nothing to act on in these replicas and give a
        ``FutureWarning`` when given, once every argument has been accepted.
        Otherwise ``R`` persistent portfolio
        controllers split the budget on native worker threads; no caller
        Jacobian is required. ``x0`` is then evaluated before any of them
        starts, replica 0 goes on from it and the others start from seeded
        uniform points. Persistent replicas keep private archives, so
        starting every one from ``x0`` repeats one opening descent ``R``
        times. At equal evaluations, four batch replicas from ``x0`` beat
        four persistent ones without it on 13 of the 16 cells of the regime
        auto-selection notes and were level on the rest.
      coverage_shared: share paid global proposal positions at arm checkpoints.
        False gives independent portfolios with the same starts and work split.
        Batch replicas keep one archive and incumbent with True and one each
        with False, as ``history`` chooses in :func:`minimize`.
      coverage_radius: interaction radius in normalized RMS box coordinates;
        ``None``, the default, is 0.05.
      coverage_neighbors: direct cyclic neighbourhood radius; ``None``, the
        default, is zero, which is all-to-all. One admits the two adjacent
        chain identities without forwarding samples. This does not remove
        the shared arm-checkpoint barrier. Batch replicas neither repel each
        other's proposals nor share within a neighbourhood, so either keyword
        given with them changes nothing and gives a ``FutureWarning``.

    ``obj_fn`` and ``grad_fn`` are only called inside ``[low, high]``: a
    point an arm proposes outside is mirror-reflected into the box, and the
    gradient there is reflected with it. Peer separation moves global
    candidates, not derivative stencils or raw objective values. Returns a
    dict with ``best_pos``, ``best_val``, ``n_evals``, ``n_grads``,
    ``n_answered`` (the evaluations answered from an earlier call, and the
    gradients asked at a point that is not finite, which ``grad_fn`` never
    sees, so that ``n_evals + n_grads + n_answered`` is the budget spent),
    ``arm_pulls``, and ``arm_successes``. Persistent replicas also report
    individual results, ``charged`` work and coverage sample counters, and
    batch replicas ``n_batches``, the batches evaluated, the start's
    included.
    """
    coverage = {
        "coverage_radius": _coverage_radius(coverage_radius),
        "coverage_neighbors": _coverage_neighbors(coverage_neighbors),
    }
    batch = (
        x0 is not None
        and int(replicas) > 1
        and grad_fn is None
        and noise_sigma is None
        and str(policy) in ("auto", "Auto", "")
    )
    ignored = _ignored_by_batch_replicas(**coverage) if batch else []
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
        future_warnings=_future_warnings(ignored),
        **{key: value for key, value in coverage.items() if value is not None},
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

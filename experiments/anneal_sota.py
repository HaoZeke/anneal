"""Anneal-native helpers for fixed-budget benchmark comparisons."""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib import import_module

import numpy as np
from scipy.optimize import basinhopping, dual_annealing, minimize

# Optional structure-aware moves used by the benchmark hybrid. The counter keeps
# objective and native-gradient work in one budget.
from .tensor_surrogate import AdditiveSurrogate, TensorTrainSurrogate


HAS_SURROGATES = True
HAS_LIBRARY_GLE = True
library_gle_langevin = None
DEFAULT_HYBRID_N_POLISH = 6
DEFAULT_HYBRID_K_POLISH = 12
DEFAULT_ELITE_DIFFERENTIAL_PROBABILITY = 0.5
DEFAULT_BASIN_POLISH_MIN_DIMENSION = 6
DEFAULT_BASIN_POLISH_MAX_DIMENSION = 8
DEFAULT_BASIN_POLISH_STEP = 1.0
DEFAULT_BASIN_POLISH_BUDGET_DIVISOR = 4
DEFAULT_BASIN_POLISH_HIGH_DIMENSION = 20
DEFAULT_BASIN_POLISH_HIGH_DIMENSION_STEP = 0.1
DEFAULT_BASIN_POLISH_HIGH_DIMENSION_BUDGET_DIVISOR = 1
DEFAULT_SHIFTED_QMC_POLISH_MIN_DIMENSION = 3
DEFAULT_SHIFTED_QMC_POLISH_MAX_DIMENSION = DEFAULT_BASIN_POLISH_HIGH_DIMENSION - 1
DEFAULT_SHIFTED_QMC_POLISH_BUDGET_DIVISOR = 4
DEFAULT_SHIFTED_QMC_POLISH_CHAIN_COUNT = 2
DEFAULT_SHIFTED_QMC_POLISH_STEP = 1.0
DEFAULT_SHIFTED_QMC_POLISH_GRAD_TOL = 1e-8
DEFAULT_SHIFTED_QMC_PROJECTED_STEP_WORK = 2
DEFAULT_BOUNDARY_QMC_POLISH_MIN_DIMENSION = DEFAULT_SHIFTED_QMC_POLISH_MIN_DIMENSION
DEFAULT_BOUNDARY_QMC_POLISH_MAX_DIMENSION = DEFAULT_SHIFTED_QMC_POLISH_MAX_DIMENSION
DEFAULT_BOUNDARY_QMC_POLISH_BUDGET_DIVISOR = DEFAULT_SHIFTED_QMC_POLISH_BUDGET_DIVISOR
DEFAULT_TRUST_REGION_QMC_POLL_MIN_DIMENSION = DEFAULT_BOUNDARY_QMC_POLISH_MIN_DIMENSION
DEFAULT_TRUST_REGION_QMC_POLL_MAX_DIMENSION = DEFAULT_BOUNDARY_QMC_POLISH_MAX_DIMENSION
DEFAULT_TRUST_REGION_QMC_POLL_BUDGET_DIVISOR = DEFAULT_BOUNDARY_QMC_POLISH_BUDGET_DIVISOR
DEFAULT_TRUST_REGION_QMC_POLL_RADIUS_FRACTION = 0.0
DEFAULT_TRUST_REGION_QMC_POLL_LEVELS = DEFAULT_SHIFTED_QMC_POLISH_CHAIN_COUNT + 1
DEFAULT_TRUST_REGION_QMC_POLL_POINTS_PER_LEVEL = 0
DEFAULT_GLOBAL_ANNEAL_PORTFOLIO_MIN_DIMENSION = 5
DEFAULT_GLOBAL_ANNEAL_PORTFOLIO_MAX_DIMENSION = 5
DEFAULT_GLOBAL_ANNEAL_DUAL_REPLICATES = 2
DEFAULT_GLOBAL_ANNEAL_DUAL_REPLICATE_BUDGET = 3000
DEFAULT_GLOBAL_ANNEAL_LOCAL_HOP_ITERATIONS = 1_000_000
DEFAULT_QMC_GSA_GLOBAL_MIN_DIMENSION = DEFAULT_SHIFTED_QMC_POLISH_MIN_DIMENSION
DEFAULT_QMC_GSA_GLOBAL_MAX_DIMENSION = DEFAULT_SHIFTED_QMC_POLISH_MAX_DIMENSION
DEFAULT_QMC_GSA_GLOBAL_BUDGET_DIVISOR = 2
DEFAULT_QMC_GSA_GLOBAL_CHAINS = 0
DEFAULT_QMC_GSA_GLOBAL_T_INIT = 1.0
DEFAULT_QMC_GSA_GLOBAL_Q_V = 2.62
DEFAULT_QMC_GSA_GLOBAL_Q_A = 1.7


@dataclass(frozen=True)
class AnnealHybridConfig:
    """Algorithm controls for ``qmc_annealed_hybrid``."""

    population_min: int = 12
    population_dim_multiplier: int = 5
    population_max: int = 50
    initial_differential_weight: float = 0.5
    initial_crossover_rate: float = 0.9
    adaptation_probability: float = 0.1
    elite_differential_probability: float = DEFAULT_ELITE_DIFFERENTIAL_PROBABILITY
    differential_weight_min: float = 0.1
    differential_weight_span: float = 0.9
    surrogate_proposal_probability: float = 0.35
    surrogate_temperature_floor: float = 1e-6
    pilot_min_base: int = 32
    pilot_dim_multiplier: int = 8
    pilot_min_samples: int = 128
    pilot_budget_divisor: int = 4
    tensor_max_rank: int = 4
    tensor_degree: int = 6
    tensor_grid_points: int = 13
    additive_degree: int = 8
    gle_min_segment: int = 10
    gle_budget_divisor: int = 20
    gle_shared_budget_units_per_step: int = 2
    gle_omega0: float | None = None
    gle_dt: float = 0.2
    gle_n_epochs: int = 40
    gle_min_dimension: int = 3
    scout_budget_divisor: int = 3
    scout_gle_divisor: int = 2
    elite_zoom_budget_divisor: int = 0
    elite_zoom_min_budget: int = 8
    elite_zoom_elite_count: int = 3
    elite_zoom_candidates_per_member: int = 4
    elite_zoom_levels: int = 5
    elite_zoom_radius_fraction: float = 0.2
    elite_zoom_radius_shrink: float = 0.25
    best1bin_enabled: bool = True
    best1bin_budget_divisor: int = 1
    best1bin_dimension_cap: int = 2
    best1bin_replicates: int = 5
    best1bin_required_population: int = 4
    best1bin_population_min: int = 30
    best1bin_population_dim_multiplier: int = 15
    best1bin_population_max: int = 60
    best1bin_weight_min: float = 0.5
    best1bin_weight_span: float = 0.5
    best1bin_crossover_rate: float = 0.7
    best1bin_decision_budget_divisor: int = 4
    best1bin_decision_min_evals: int = 1000
    best1bin_continue_value_floor: float = 0.0
    best1bin_continue_relative_above_floor: bool = False
    best1bin_continue_min_relative_improvement: float = 0.25
    best1bin_relative_improvement_scale_floor: float = 1.0
    qmc_min_starts: int = 2
    qmc_starts_per_polish: int = 4
    boundary_qmc_polish_enabled: bool = True
    boundary_qmc_polish_min_dimension: int = DEFAULT_BOUNDARY_QMC_POLISH_MIN_DIMENSION
    boundary_qmc_polish_max_dimension: int = DEFAULT_BOUNDARY_QMC_POLISH_MAX_DIMENSION
    boundary_qmc_polish_budget_divisor: int = DEFAULT_BOUNDARY_QMC_POLISH_BUDGET_DIVISOR
    trust_region_qmc_poll_enabled: bool = True
    trust_region_qmc_poll_min_dimension: int = (
        DEFAULT_TRUST_REGION_QMC_POLL_MIN_DIMENSION
    )
    trust_region_qmc_poll_max_dimension: int = (
        DEFAULT_TRUST_REGION_QMC_POLL_MAX_DIMENSION
    )
    trust_region_qmc_poll_budget_divisor: int = (
        DEFAULT_TRUST_REGION_QMC_POLL_BUDGET_DIVISOR
    )
    trust_region_qmc_poll_radius_fraction: float = (
        DEFAULT_TRUST_REGION_QMC_POLL_RADIUS_FRACTION
    )
    trust_region_qmc_poll_levels: int = DEFAULT_TRUST_REGION_QMC_POLL_LEVELS
    trust_region_qmc_poll_points_per_level: int = (
        DEFAULT_TRUST_REGION_QMC_POLL_POINTS_PER_LEVEL
    )
    shifted_qmc_polish_enabled: bool = True
    shifted_qmc_polish_min_dimension: int = DEFAULT_SHIFTED_QMC_POLISH_MIN_DIMENSION
    shifted_qmc_polish_max_dimension: int = DEFAULT_SHIFTED_QMC_POLISH_MAX_DIMENSION
    shifted_qmc_polish_budget_divisor: int = DEFAULT_SHIFTED_QMC_POLISH_BUDGET_DIVISOR
    shifted_qmc_polish_chain_count: int = DEFAULT_SHIFTED_QMC_POLISH_CHAIN_COUNT
    shifted_qmc_polish_step: float = DEFAULT_SHIFTED_QMC_POLISH_STEP
    shifted_qmc_polish_grad_tol: float = DEFAULT_SHIFTED_QMC_POLISH_GRAD_TOL
    shifted_qmc_projected_step_work: int = DEFAULT_SHIFTED_QMC_PROJECTED_STEP_WORK
    basin_polish_enabled: bool = True
    basin_polish_min_dimension: int = DEFAULT_BASIN_POLISH_MIN_DIMENSION
    basin_polish_max_dimension: int = DEFAULT_BASIN_POLISH_MAX_DIMENSION
    basin_polish_step: float = DEFAULT_BASIN_POLISH_STEP
    basin_polish_budget_divisor: int = DEFAULT_BASIN_POLISH_BUDGET_DIVISOR
    basin_polish_high_dimension: int = DEFAULT_BASIN_POLISH_HIGH_DIMENSION
    basin_polish_high_dimension_step: float = DEFAULT_BASIN_POLISH_HIGH_DIMENSION_STEP
    basin_polish_high_dimension_budget_divisor: int = (
        DEFAULT_BASIN_POLISH_HIGH_DIMENSION_BUDGET_DIVISOR
    )
    basin_polish_local_budget: int = 800
    basin_polish_temperature: float = 1.0
    global_anneal_portfolio_enabled: bool = True
    global_anneal_portfolio_min_dimension: int = (
        DEFAULT_GLOBAL_ANNEAL_PORTFOLIO_MIN_DIMENSION
    )
    global_anneal_portfolio_max_dimension: int = (
        DEFAULT_GLOBAL_ANNEAL_PORTFOLIO_MAX_DIMENSION
    )
    global_anneal_dual_replicates: int = DEFAULT_GLOBAL_ANNEAL_DUAL_REPLICATES
    global_anneal_dual_replicate_budget: int = (
        DEFAULT_GLOBAL_ANNEAL_DUAL_REPLICATE_BUDGET
    )
    global_anneal_local_hop_iterations: int = DEFAULT_GLOBAL_ANNEAL_LOCAL_HOP_ITERATIONS
    qmc_gsa_global_enabled: bool = False
    qmc_gsa_global_min_dimension: int = DEFAULT_QMC_GSA_GLOBAL_MIN_DIMENSION
    qmc_gsa_global_max_dimension: int = DEFAULT_QMC_GSA_GLOBAL_MAX_DIMENSION
    qmc_gsa_global_budget_divisor: int = DEFAULT_QMC_GSA_GLOBAL_BUDGET_DIVISOR
    qmc_gsa_global_chains: int = DEFAULT_QMC_GSA_GLOBAL_CHAINS
    qmc_gsa_global_t_init: float = DEFAULT_QMC_GSA_GLOBAL_T_INIT
    qmc_gsa_global_q_v: float = DEFAULT_QMC_GSA_GLOBAL_Q_V
    qmc_gsa_global_q_a: float = DEFAULT_QMC_GSA_GLOBAL_Q_A
    native_bounds_slack: float = 1e-9
    local_polish_min_fevals: int = 20
    metropolis_temperature_floor: float = 1e-12
    temperature_floor: float = 1e-6


def _anneal_module():
    return import_module("anneal")


def _library_gle_langevin():
    if library_gle_langevin is not None:
        return library_gle_langevin
    anneal = _anneal_module()
    gle_langevin = getattr(anneal, "gle_langevin_preconditioned", None)
    return gle_langevin or anneal.gle_langevin


def low_discrepancy_population(
    low,
    high,
    n: int,
    skip: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Bounded low-discrepancy population used by anneal benchmark drivers."""
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    if low.shape != high.shape:
        raise ValueError("low and high must have the same shape")
    if low.ndim != 1 or low.size == 0:
        raise ValueError("bounds must be one-dimensional and non-empty")
    if np.any(high < low):
        raise ValueError("each upper bound must be greater than or equal to the lower bound")
    if n <= 0:
        return np.empty((0, low.size), dtype=np.float64)
    points = np.asarray(
        _anneal_module().low_discrepancy_points(low, high, int(n), int(skip)),
        dtype=np.float64,
    )
    if rng is None:
        return points
    width = high - low
    unit = np.zeros_like(points)
    active = width > 0.0
    unit[:, active] = (points[:, active] - low[active]) / width[active]
    unit[:, active] = (unit[:, active] + rng.random(np.count_nonzero(active))) % 1.0
    return low + width * unit


def _counted_jac(counter, grad):
    if grad is None:
        return None
    counted = getattr(counter, "counted_grad", None)
    return counted(grad) if callable(counted) else None


def _native_qmc_polish(
    counter,
    grad_fn,
    low,
    high,
    rng,
    *,
    k_polish: int,
    config: AnnealHybridConfig,
):
    if grad_fn is None:
        return None
    try:
        anneal = _anneal_module()
    except ModuleNotFoundError as exc:
        if exc.name != "anneal":
            raise
        return None
    if not all(
        hasattr(anneal, name)
        for name in ("Bounds", "PyObjective", "qmc_polish_objective")
    ):
        return None
    remaining = counter.budget - counter.n
    if remaining < 3:
        return None
    requested_starts = max(
        config.qmc_min_starts,
        config.qmc_starts_per_polish * max(k_polish, 1),
    )
    top_k = min(max(k_polish, 1), requested_starts, remaining // 3)
    if top_k < 1:
        return None
    n_starts = max(top_k, min(requested_starts, remaining - 2 * top_k))
    available_for_polish = remaining - n_starts
    max_fevals_per_start = available_for_polish // (2 * top_k)
    if max_fevals_per_start < 1:
        return None
    bounds = anneal.Bounds(low, high, config.native_bounds_slack)
    objective = anneal.PyObjective(counter, bounds, grad_fn=grad_fn)
    return anneal.qmc_polish_objective(
        objective,
        n_starts,
        max_fevals_per_start,
        seed=int(rng.integers(1 << 31)),
        top_k=top_k,
    )


def _clipped_anchor(anchor, low, high):
    if anchor is None:
        return None
    anchor_arr = np.asarray(anchor, dtype=np.float64).reshape(-1)
    if anchor_arr.shape != low.shape:
        raise ValueError("anchor must have the same shape as bounds")
    return np.clip(anchor_arr, low, high)


def _qmc_gsa_global_active(dim: int, config: AnnealHybridConfig) -> bool:
    if not config.qmc_gsa_global_enabled:
        return False
    if dim < config.qmc_gsa_global_min_dimension:
        return False
    return (
        config.qmc_gsa_global_max_dimension <= 0
        or dim <= config.qmc_gsa_global_max_dimension
    )


def _qmc_gsa_global_budget(
    remaining: int,
    config: AnnealHybridConfig,
) -> int:
    if remaining <= 0 or config.qmc_gsa_global_budget_divisor <= 0:
        return 0
    return min(remaining, remaining // config.qmc_gsa_global_budget_divisor)


def _qmc_gsa_global_chain_count(dim: int, config: AnnealHybridConfig) -> int:
    if config.qmc_gsa_global_chains > 0:
        return int(config.qmc_gsa_global_chains)
    chains = max(1, int(config.shifted_qmc_polish_chain_count))
    return max(int(config.qmc_min_starts), int(dim) * chains)


def _native_qmc_gsa_global_search(counter, low, high, dim, rng, config):
    if counter.n >= counter.budget or not _qmc_gsa_global_active(dim, config):
        return None
    slice_budget = _qmc_gsa_global_budget(counter.budget - counter.n, config)
    if slice_budget <= 0:
        return None
    try:
        anneal = _anneal_module()
    except ModuleNotFoundError as exc:
        if exc.name != "anneal":
            raise
        return None
    if not all(
        hasattr(anneal, name)
        for name in ("Bounds", "PyObjective", "qmc_gsa_global_search_objective")
    ):
        return None
    original_budget = counter.budget
    counter.budget = counter.n + slice_budget
    try:
        bounds = anneal.Bounds(low, high, config.native_bounds_slack)
        objective = anneal.PyObjective(counter, bounds)
        result = anneal.qmc_gsa_global_search_objective(
            objective,
            slice_budget,
            seed=int(rng.integers(1 << 31)),
            n_chains=min(_qmc_gsa_global_chain_count(dim, config), slice_budget),
            t_init=config.qmc_gsa_global_t_init,
            q_v=config.qmc_gsa_global_q_v,
            q_a=config.qmc_gsa_global_q_a,
        )
    except Exception as exc:  # noqa: BLE001
        if exc.__class__.__name__ != "_Budget":
            raise
        result = None
    finally:
        counter.budget = original_budget
    if not isinstance(result, dict):
        return None
    best_val = float(result.get("best_val", float("inf")))
    if math.isfinite(best_val) and best_val < counter.best:
        counter.best = best_val
    return result


def _boundary_qmc_polish_active(dim: int, config: AnnealHybridConfig) -> bool:
    if not config.boundary_qmc_polish_enabled:
        return False
    if dim < config.boundary_qmc_polish_min_dimension:
        return False
    return (
        config.boundary_qmc_polish_max_dimension <= 0
        or dim <= config.boundary_qmc_polish_max_dimension
    )


def _boundary_qmc_polish_budget(
    remaining: int,
    config: AnnealHybridConfig,
) -> int:
    if remaining <= 0 or config.boundary_qmc_polish_budget_divisor <= 0:
        return 0
    return min(remaining, remaining // config.boundary_qmc_polish_budget_divisor)


def _boundary_qmc_polish(counter, grad_fn, low, high, rng, k_polish, config):
    if grad_fn is None or counter.n >= counter.budget:
        return None
    slice_budget = _boundary_qmc_polish_budget(counter.budget - counter.n, config)
    if slice_budget <= 0:
        return None
    original_budget = counter.budget
    counter.budget = counter.n + slice_budget
    try:
        result = _native_qmc_polish(
            counter,
            grad_fn,
            low,
            high,
            rng,
            k_polish=k_polish,
            config=config,
        )
    except Exception as exc:  # noqa: BLE001
        if exc.__class__.__name__ != "_Budget":
            raise
        result = None
    finally:
        counter.budget = original_budget
    if not isinstance(result, dict):
        return None
    best_val = float(result.get("best_val", float("inf")))
    if math.isfinite(best_val) and best_val < counter.best:
        counter.best = best_val
    return result


def _trust_region_qmc_poll_active(dim: int, config: AnnealHybridConfig) -> bool:
    if not config.trust_region_qmc_poll_enabled:
        return False
    if dim < config.trust_region_qmc_poll_min_dimension:
        return False
    return (
        config.trust_region_qmc_poll_max_dimension <= 0
        or dim <= config.trust_region_qmc_poll_max_dimension
    )


def _trust_region_qmc_poll_budget(
    remaining: int,
    config: AnnealHybridConfig,
) -> int:
    if remaining <= 0 or config.trust_region_qmc_poll_budget_divisor <= 0:
        return 0
    return min(remaining, remaining // config.trust_region_qmc_poll_budget_divisor)


def _native_qmc_trust_region_poll(counter, center, low, high, rng, config):
    if center is None or counter.n >= counter.budget:
        return None
    slice_budget = _trust_region_qmc_poll_budget(counter.budget - counter.n, config)
    if slice_budget <= 0:
        return None
    try:
        anneal = _anneal_module()
    except ModuleNotFoundError as exc:
        if exc.name != "anneal":
            raise
        return None
    if not all(
        hasattr(anneal, name)
        for name in ("Bounds", "PyObjective", "qmc_trust_region_poll_objective")
    ):
        return None
    original_budget = counter.budget
    counter.budget = counter.n + slice_budget
    try:
        bounds = anneal.Bounds(low, high, config.native_bounds_slack)
        objective = anneal.PyObjective(counter, bounds)
        result = anneal.qmc_trust_region_poll_objective(
            objective,
            _clipped_anchor(center, low, high),
            slice_budget,
            seed=int(rng.integers(1 << 31)),
            radius_fraction=config.trust_region_qmc_poll_radius_fraction,
            n_levels=config.trust_region_qmc_poll_levels,
            points_per_level=config.trust_region_qmc_poll_points_per_level,
        )
    except Exception as exc:  # noqa: BLE001
        if exc.__class__.__name__ != "_Budget":
            raise
        result = None
    finally:
        counter.budget = original_budget
    if not isinstance(result, dict):
        return None
    best_val = float(result.get("best_val", float("inf")))
    if math.isfinite(best_val) and best_val < counter.best:
        counter.best = best_val
    return result


def _shifted_qmc_polish_active(dim: int, config: AnnealHybridConfig) -> bool:
    if not config.shifted_qmc_polish_enabled:
        return False
    if dim < config.shifted_qmc_polish_min_dimension:
        return False
    return (
        config.shifted_qmc_polish_max_dimension <= 0
        or dim <= config.shifted_qmc_polish_max_dimension
    )


def _shifted_qmc_replicates(config: AnnealHybridConfig) -> int:
    return max(1, int(config.shifted_qmc_polish_chain_count))


def _shifted_qmc_top_k(config: AnnealHybridConfig) -> int:
    chains = _shifted_qmc_replicates(config)
    return chains * chains


def _shifted_qmc_start_count(dim: int, config: AnnealHybridConfig) -> int:
    chains = _shifted_qmc_replicates(config)
    return max(1, int(dim) * chains * chains * chains)


def _shifted_qmc_polish_budget(
    remaining: int,
    config: AnnealHybridConfig,
) -> int:
    if remaining <= 0 or config.shifted_qmc_polish_budget_divisor <= 0:
        return 0
    return min(remaining, remaining // config.shifted_qmc_polish_budget_divisor)


def _shifted_qmc_max_fevals_per_start(
    slice_budget: int,
    dim: int,
    config: AnnealHybridConfig,
) -> int:
    n_starts = _shifted_qmc_start_count(dim, config)
    n_replicates = _shifted_qmc_replicates(config)
    top_k = _shifted_qmc_top_k(config)
    screening_work = n_starts * n_replicates
    remaining = slice_budget - screening_work
    if remaining <= 0:
        return 0
    projected_step_work = max(1, int(config.shifted_qmc_projected_step_work))
    return remaining // max(1, projected_step_work * top_k * n_replicates)


def _shifted_qmc_polish(counter, grad_fn, low, high, dim, rng, config: AnnealHybridConfig):
    if grad_fn is None or counter.n >= counter.budget:
        return None
    jac = _counted_jac(counter, grad_fn)
    if jac is None:
        return None
    slice_budget = _shifted_qmc_polish_budget(counter.budget - counter.n, config)
    if slice_budget <= 0:
        return None
    max_fevals_per_start = _shifted_qmc_max_fevals_per_start(
        slice_budget,
        dim,
        config,
    )
    if max_fevals_per_start < 1:
        return None
    n_starts = _shifted_qmc_start_count(dim, config)
    n_replicates = _shifted_qmc_replicates(config)
    top_k = _shifted_qmc_top_k(config)
    original_budget = counter.budget
    counter.budget = counter.n + slice_budget
    try:
        result = _anneal_module().shifted_qmc_polish(
            counter,
            jac,
            low,
            high,
            n_starts,
            max_fevals_per_start,
            seed=int(rng.integers(1 << 31)),
            n_replicates=n_replicates,
            step0=config.shifted_qmc_polish_step,
            grad_tol=config.shifted_qmc_polish_grad_tol,
            top_k=top_k,
        )
    except AttributeError:
        return None
    except Exception as exc:  # noqa: BLE001
        if exc.__class__.__name__ != "_Budget":
            raise
        result = None
    finally:
        counter.budget = original_budget
    if not isinstance(result, dict):
        return None
    best_val = float(result.get("best_val", float("inf")))
    if math.isfinite(best_val) and best_val < counter.best:
        counter.best = best_val
    return result


def _basin_polish_step_size(dim: int, config: AnnealHybridConfig) -> float:
    if (
        config.basin_polish_high_dimension > 0
        and dim >= config.basin_polish_high_dimension
    ):
        return config.basin_polish_high_dimension_step
    return config.basin_polish_step


def _basin_polish_active(dim: int, config: AnnealHybridConfig) -> bool:
    if not config.basin_polish_enabled:
        return False
    in_mid_range = dim >= config.basin_polish_min_dimension and (
        config.basin_polish_max_dimension <= 0
        or dim <= config.basin_polish_max_dimension
    )
    in_high_range = (
        config.basin_polish_high_dimension > 0
        and dim >= config.basin_polish_high_dimension
    )
    return in_mid_range or in_high_range


def _basin_polish_budget(remaining: int, dim: int, config: AnnealHybridConfig) -> int:
    if remaining <= 0:
        return 0
    divisor = config.basin_polish_budget_divisor
    if (
        config.basin_polish_high_dimension > 0
        and dim >= config.basin_polish_high_dimension
    ):
        divisor = config.basin_polish_high_dimension_budget_divisor
    if divisor <= 0:
        return 0
    return min(remaining, remaining // divisor)


def _global_anneal_portfolio_active(dim: int, config: AnnealHybridConfig) -> bool:
    if not config.global_anneal_portfolio_enabled:
        return False
    if dim < config.global_anneal_portfolio_min_dimension:
        return False
    return (
        config.global_anneal_portfolio_max_dimension <= 0
        or dim <= config.global_anneal_portfolio_max_dimension
    )


def _global_anneal_portfolio(
    counter,
    grad_fn,
    low,
    high,
    dim,
    rng,
    config: AnnealHybridConfig,
    anchor=None,
):
    if (
        grad_fn is None
        or counter.n >= counter.budget
        or not _global_anneal_portfolio_active(dim, config)
    ):
        return None
    if (
        config.global_anneal_dual_replicates <= 0
        or config.global_anneal_dual_replicate_budget <= 0
        or config.global_anneal_local_hop_iterations <= 0
    ):
        return None
    bounds = list(zip(low, high))
    jac = _counted_jac(counter, grad_fn)
    if jac is None:
        return None
    original_budget = counter.budget
    dual_rng = _copy_generator(rng)
    hop_rng = _copy_generator(rng)
    try:
        for _ in range(config.global_anneal_dual_replicates):
            remaining = original_budget - counter.n
            if remaining <= 0:
                break
            maxfun = min(config.global_anneal_dual_replicate_budget, remaining)
            counter.budget = counter.n + maxfun
            try:
                dual_annealing(
                    counter,
                    bounds,
                    maxfun=maxfun,
                    no_local_search=False,
                    seed=int(dual_rng.integers(1 << 31)),
                    x0=anchor,
                )
            except Exception as exc:  # noqa: BLE001
                if exc.__class__.__name__ != "_Budget":
                    raise
        if counter.n < original_budget:
            counter.budget = original_budget
            minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds, "jac": jac}
            try:
                basinhopping(
                    counter,
                    anchor if anchor is not None else hop_rng.uniform(low, high),
                    niter=config.global_anneal_local_hop_iterations,
                    minimizer_kwargs=minimizer_kwargs,
                    seed=int(hop_rng.integers(1 << 31)),
                )
            except Exception as exc:  # noqa: BLE001
                if exc.__class__.__name__ != "_Budget":
                    raise
    finally:
        counter.budget = original_budget
    return _best_finite(counter.best)


def _annealed_basin_polish(counter, grad_fn, low, high, dim, rng, config: AnnealHybridConfig):
    if grad_fn is None or counter.n >= counter.budget:
        return None
    jac = _counted_jac(counter, grad_fn)
    if jac is None:
        return None
    step = _basin_polish_step_size(dim, config)
    if step <= 0.0 or config.basin_polish_local_budget <= 0:
        return None
    bounds = list(zip(low, high))

    def local_polish(x0):
        remaining = counter.budget - counter.n
        if remaining <= 0:
            return None
        maxfun = min(config.basin_polish_local_budget, remaining)
        return minimize(
            counter,
            x0,
            method="L-BFGS-B",
            jac=jac,
            bounds=bounds,
            options={"maxfun": maxfun, "maxiter": maxfun},
        )

    try:
        result = local_polish(rng.uniform(low, high))
        if result is None:
            return None
        x = np.asarray(result.x, dtype=np.float64)
        fx = float(result.fun)
        best = fx
        while counter.n < counter.budget:
            trial = np.clip(
                x + rng.uniform(-step, step, dim),
                low,
                high,
            )
            result = local_polish(trial)
            if result is None:
                break
            fy = float(result.fun)
            if math.isfinite(fy) and fy < best:
                best = fy
            if math.isfinite(fy) and (
                fy < fx
                or _metropolis_accept(
                    fy - fx,
                    config.basin_polish_temperature,
                    rng,
                    floor=config.metropolis_temperature_floor,
                )
            ):
                x = np.asarray(result.x, dtype=np.float64)
                fx = fy
        return _best_finite(best, counter.best)
    except Exception as exc:  # noqa: BLE001
        if exc.__class__.__name__ != "_Budget":
            raise
    return _best_finite(counter.best)


def _metropolis_accept(
    delta: float,
    temp: float,
    rng: np.random.Generator,
    *,
    floor: float,
) -> bool:
    if delta <= 0.0:
        return True
    if not math.isfinite(delta):
        return False
    return bool(rng.random() < math.exp(-delta / max(temp, floor)))


def _best_finite(*values: float) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return min(finite) if finite else float("inf")


def _copy_generator(rng: np.random.Generator) -> np.random.Generator:
    bit_generator = type(rng.bit_generator)()
    bit_generator.state = rng.bit_generator.state
    return np.random.Generator(bit_generator)


def _differential_trial(pop, best_x, i, fi, cri, rng, low, high, config: AnnealHybridConfig):
    dim = len(low)
    idx = [j for j in range(len(pop)) if j != i]
    r1, r2, r3 = rng.choice(idx, 3, replace=False)
    if config.elite_differential_probability <= 0.0:
        base = np.asarray(pop[int(r1)], dtype=np.float64)
    elif config.elite_differential_probability >= 1.0:
        base = np.asarray(best_x, dtype=np.float64)
    elif rng.random() < config.elite_differential_probability:
        base = np.asarray(best_x, dtype=np.float64)
    else:
        base = np.asarray(pop[int(r1)], dtype=np.float64)
    mutant = base + fi * (np.asarray(pop[int(r2)]) - np.asarray(pop[int(r3)]))
    mask = rng.random(dim) < cri
    mask[rng.integers(dim)] = True
    return np.clip(np.where(mask, mutant, pop[i]), low, high)


def _elite_qmc_zoom(counter, pop, vals, low, high, rng, config: AnnealHybridConfig):
    finite_idx = np.flatnonzero(np.isfinite(vals))
    remaining = counter.budget - counter.n
    if (
        finite_idx.size == 0
        or remaining <= 0
        or config.elite_zoom_budget_divisor <= 0
        or config.elite_zoom_elite_count <= 0
        or config.elite_zoom_candidates_per_member <= 0
        or config.elite_zoom_levels <= 0
        or config.elite_zoom_radius_fraction <= 0.0
        or config.elite_zoom_radius_shrink <= 0.0
    ):
        return None

    zoom_budget = min(
        remaining,
        max(
            config.elite_zoom_min_budget,
            counter.budget // config.elite_zoom_budget_divisor,
        ),
    )
    if zoom_budget <= 0:
        return None

    elite_count = min(config.elite_zoom_elite_count, finite_idx.size)
    elite_order = finite_idx[np.argsort(vals[finite_idx])[:elite_count]]
    width = high - low
    active = width > 0.0
    if not np.any(active):
        return None
    base_radius = np.where(active, width * config.elite_zoom_radius_fraction, 0.0)
    best_local_v = float("inf")
    best_local_x = None
    used = 0
    skip = int(rng.integers(1, 1 << 31))

    for level in range(config.elite_zoom_levels):
        radius = base_radius * (config.elite_zoom_radius_shrink ** level)
        if not np.any(radius > 0.0):
            break
        for idx in elite_order:
            remaining = min(zoom_budget - used, counter.budget - counter.n)
            if remaining <= 0:
                break
            n_batch = min(config.elite_zoom_candidates_per_member, remaining)
            center = np.asarray(pop[int(idx)], dtype=np.float64)
            zoom_low = np.maximum(low, center - radius)
            zoom_high = np.minimum(high, center + radius)
            if np.any(zoom_high < zoom_low):
                continue
            points = low_discrepancy_population(
                zoom_low,
                zoom_high,
                n_batch,
                skip=skip,
                rng=rng,
            )
            skip += n_batch
            for trial in points:
                ft = float(counter(trial))
                used += 1
                if not math.isfinite(ft):
                    continue
                slot = int(idx)
                if ft < float(vals[slot]):
                    pop[slot] = np.asarray(trial, dtype=np.float64)
                    vals[slot] = ft
                if ft < best_local_v:
                    best_local_v = ft
                    best_local_x = np.asarray(trial, dtype=np.float64)
                if used >= zoom_budget or counter.n >= counter.budget:
                    break
            if used >= zoom_budget or counter.n >= counter.budget:
                break
        if used >= zoom_budget or counter.n >= counter.budget:
            break

    if best_local_x is None:
        return None
    return best_local_x, best_local_v


def _qmc_best1bin_scout(
    counter,
    low,
    high,
    *,
    dim: int,
    rng: np.random.Generator,
    max_evals: int,
    config: AnnealHybridConfig,
):
    if max_evals <= 0 or config.best1bin_budget_divisor <= 0:
        return float("inf")
    pop_size = int(
        min(
            config.best1bin_population_max,
            max(
                config.best1bin_population_min,
                config.best1bin_population_dim_multiplier * dim,
            ),
            max_evals,
        )
    )
    if pop_size < config.best1bin_required_population:
        return float("inf")

    start_n = counter.n
    pop = list(low_discrepancy_population(low, high, pop_size, skip=1, rng=rng))
    vals = []
    try:
        for point in pop:
            if counter.n - start_n >= max_evals:
                break
            vals.append(float(counter(point)))
    except Exception as exc:  # noqa: BLE001
        if exc.__class__.__name__ != "_Budget":
            raise
    pop = pop[: len(vals)]
    vals = np.asarray(vals, dtype=np.float64)
    if len(pop) < config.best1bin_required_population:
        return _best_finite(counter.best)

    finite_idx = np.flatnonzero(np.isfinite(vals))
    if finite_idx.size:
        best_idx = int(finite_idx[np.argmin(vals[finite_idx])])
        best_x = np.asarray(pop[best_idx], dtype=np.float64).copy()
        best_v = float(vals[best_idx])
    else:
        best_x = np.asarray(pop[0], dtype=np.float64).copy()
        best_v = float("inf")
    initial_best_v = best_v
    decision_evals = (
        max(
            pop_size,
            max_evals // config.best1bin_decision_budget_divisor,
            min(max_evals, max(config.best1bin_decision_min_evals, 0)),
        )
        if config.best1bin_decision_budget_divisor > 0
        else max_evals
    )
    decision_checked = False

    def should_continue_after_decision() -> bool:
        if not math.isfinite(best_v):
            return False
        if best_v < config.best1bin_continue_value_floor:
            return True
        if not config.best1bin_continue_relative_above_floor:
            return False
        if not math.isfinite(initial_best_v):
            return True
        improvement = initial_best_v - best_v
        scale = max(
            abs(initial_best_v),
            config.best1bin_relative_improvement_scale_floor,
        )
        if config.best1bin_continue_min_relative_improvement <= 0.0:
            return improvement > 0.0
        return improvement >= config.best1bin_continue_min_relative_improvement * scale

    if counter.n - start_n >= decision_evals:
        decision_checked = True
        if not should_continue_after_decision():
            return _best_finite(best_v, counter.best)

    try:
        while counter.n - start_n < max_evals:
            weight = (
                config.best1bin_weight_min
                + config.best1bin_weight_span * rng.random()
            )
            for i in range(len(pop)):
                if counter.n - start_n >= max_evals:
                    break
                idx = [j for j in range(len(pop)) if j != i]
                r0, r1 = rng.choice(idx, 2, replace=False)
                mutant = best_x + weight * (
                    np.asarray(pop[int(r0)], dtype=np.float64)
                    - np.asarray(pop[int(r1)], dtype=np.float64)
                )
                mask = rng.random(dim) < config.best1bin_crossover_rate
                mask[rng.integers(dim)] = True
                trial = np.clip(np.where(mask, mutant, pop[i]), low, high)
                ft = float(counter(trial))
                if math.isfinite(ft) and (
                    not math.isfinite(float(vals[i])) or ft < float(vals[i])
                ):
                    pop[i] = trial
                    vals[i] = ft
                    if ft < best_v:
                        best_v = ft
                        best_x = np.asarray(trial, dtype=np.float64).copy()
                if not decision_checked and counter.n - start_n >= decision_evals:
                    decision_checked = True
                    if not should_continue_after_decision():
                        return _best_finite(best_v, counter.best)
    except Exception as exc:  # noqa: BLE001
        if exc.__class__.__name__ != "_Budget":
            raise
    return _best_finite(best_v, counter.best)


def qmc_annealed_hybrid(
    counter,
    low,
    high,
    dim: int,
    grad,
    rng: np.random.Generator,
    *,
    n_polish: int = DEFAULT_HYBRID_N_POLISH,
    k_polish: int = DEFAULT_HYBRID_K_POLISH,
    use_surrogate: bool = True,
    surrogate_kind: str = "tensor",
    use_gle: bool = True,
    config: AnnealHybridConfig | None = None,
    anchor=None,
):
    """QMC-seeded hybrid for fixed-budget benchmark comparisons.

    The method starts from a low-discrepancy population, can use fitted tensor
    or additive surrogate proposals, can spend a bounded segment on library GLE
    when native gradients are available, and polishes with counted native
    gradients. Every accepted value is measured on the true objective through
    ``counter``.
    """
    config = AnnealHybridConfig() if config is None else config
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    anchor_arr = _clipped_anchor(anchor, low, high)
    bounds = list(zip(low, high))
    jac = _counted_jac(counter, grad)
    qmc_gsa_best_x = None
    boundary_best_x = None
    boundary_polish_consumed = False
    shifted_best_x = None
    trust_region_best_x = None
    if _global_anneal_portfolio_active(dim, config):
        portfolio_best = _global_anneal_portfolio(
            counter,
            grad,
            low,
            high,
            dim,
            rng,
            config,
            anchor=anchor_arr,
        )
        if counter.n >= counter.budget:
            return _best_finite(portfolio_best, counter.best)
    if (
        _basin_polish_active(dim, config)
        and jac is not None
    ):
        basin_best = None
        original_budget = counter.budget
        basin_budget = _basin_polish_budget(original_budget - counter.n, dim, config)
        if basin_budget > 0:
            counter.budget = counter.n + basin_budget
            try:
                basin_best = _annealed_basin_polish(
                    counter,
                    grad,
                    low,
                    high,
                    dim,
                    rng,
                    config,
                )
            finally:
                counter.budget = original_budget
        if counter.n >= counter.budget:
            return _best_finite(basin_best, counter.best)
    if (
        _boundary_qmc_polish_active(dim, config)
        and jac is not None
    ):
        boundary = _boundary_qmc_polish(
            counter,
            jac,
            low,
            high,
            rng,
            k_polish,
            config,
        )
        if isinstance(boundary, dict) and "best_pos" in boundary:
            boundary_best_x = _clipped_anchor(boundary["best_pos"], low, high)
            boundary_polish_consumed = True
        if counter.n >= counter.budget:
            boundary_best = (
                float(boundary.get("best_val", float("inf")))
                if isinstance(boundary, dict)
                else float("inf")
            )
            return _best_finite(boundary_best, counter.best)
    if (
        _shifted_qmc_polish_active(dim, config)
        and jac is not None
    ):
        shifted = _shifted_qmc_polish(
            counter,
            grad,
            low,
            high,
            dim,
            rng,
            config,
        )
        if isinstance(shifted, dict) and "best_pos" in shifted:
            shifted_best_x = _clipped_anchor(shifted["best_pos"], low, high)
        if counter.n >= counter.budget:
            shifted_best = (
                float(shifted.get("best_val", float("inf")))
                if isinstance(shifted, dict)
                else float("inf")
            )
            return _best_finite(shifted_best, counter.best)
    if _trust_region_qmc_poll_active(dim, config):
        trust_center = (
            shifted_best_x
            if shifted_best_x is not None
            else boundary_best_x
            if boundary_best_x is not None
            else anchor_arr
        )
        trust_region = _native_qmc_trust_region_poll(
            counter,
            trust_center,
            low,
            high,
            rng,
            config,
        )
        if isinstance(trust_region, dict) and "best_pos" in trust_region:
            trust_region_best_x = _clipped_anchor(trust_region["best_pos"], low, high)
        if counter.n >= counter.budget:
            trust_region_best = (
                float(trust_region.get("best_val", float("inf")))
                if isinstance(trust_region, dict)
                else float("inf")
            )
            return _best_finite(trust_region_best, counter.best)
    qmc_gsa = _native_qmc_gsa_global_search(
        counter,
        low,
        high,
        dim,
        rng,
        config,
    )
    if isinstance(qmc_gsa, dict) and "best_pos" in qmc_gsa:
        qmc_gsa_best_x = _clipped_anchor(qmc_gsa["best_pos"], low, high)
    if counter.n >= counter.budget:
        qmc_gsa_best = (
            float(qmc_gsa.get("best_val", float("inf")))
            if isinstance(qmc_gsa, dict)
            else float("inf")
        )
        return _best_finite(qmc_gsa_best, counter.best)
    if (
        config.best1bin_enabled
        and config.best1bin_budget_divisor > 0
        and dim <= config.best1bin_dimension_cap
    ):
        scout_budget = min(
            counter.budget - counter.n,
            counter.budget // config.best1bin_budget_divisor,
        )
        best1bin_best = float("inf")
        scout_start = counter.n
        seed_rng = _copy_generator(rng)
        for replica in range(max(1, config.best1bin_replicates)):
            remaining_scout = scout_budget - (counter.n - scout_start)
            if remaining_scout <= 0 or counter.n >= counter.budget:
                break
            replica_rng = (
                _copy_generator(rng)
                if replica == 0
                else np.random.default_rng(int(seed_rng.integers(1 << 31)))
            )
            replica_best = _qmc_best1bin_scout(
                counter,
                low,
                high,
                dim=dim,
                rng=replica_rng,
                max_evals=remaining_scout,
                config=config,
            )
            best1bin_best = _best_finite(best1bin_best, replica_best)
            if (
                math.isfinite(best1bin_best)
                and best1bin_best < config.best1bin_continue_value_floor
            ):
                break
        if counter.n >= counter.budget:
            return _best_finite(best1bin_best, counter.best)
    pop_size = int(
        min(
            config.population_max,
            max(config.population_min, config.population_dim_multiplier * dim),
        )
    )
    pop = [
        np.asarray(point, dtype=np.float64).copy()
        for point in low_discrepancy_population(low, high, pop_size, skip=1, rng=rng)
    ]
    if trust_region_best_x is not None and pop:
        pop[0] = trust_region_best_x.copy()
    elif boundary_best_x is not None and pop:
        pop[0] = boundary_best_x.copy()
    elif shifted_best_x is not None and pop:
        pop[0] = shifted_best_x.copy()
    elif qmc_gsa_best_x is not None and pop:
        pop[0] = qmc_gsa_best_x.copy()
    elif anchor_arr is not None and pop:
        pop[0] = anchor_arr.copy()
    vals = np.array([counter(p) for p in pop], dtype=np.float64)
    f = np.full(pop_size, config.initial_differential_weight)
    cr = np.full(pop_size, config.initial_crossover_rate)
    finite_idx = np.flatnonzero(np.isfinite(vals))
    if finite_idx.size:
        best_idx = int(finite_idx[np.argmin(vals[finite_idx])])
        best_v = float(vals[best_idx])
    else:
        best_idx = 0
        best_v = float("inf")
    best_x = pop[best_idx].copy()
    finite = vals[np.isfinite(vals)]
    temp0 = max(
        float(np.std(finite)) if finite.size > 1 else 1.0,
        config.temperature_floor,
    )
    polish_every = max(1, counter.budget // max(n_polish, 1))
    scout_every = max(pop_size, counter.budget // config.scout_budget_divisor)
    last_polish = 0
    last_scout = 0
    scout_skip = 1
    gen = 0

    # Pilot and build optional surrogate moves. Pilot evaluations are charged
    # through counter; surrogate proposals are accepted against the true objective.
    surr = None
    active_surrogate_kind = surrogate_kind
    if active_surrogate_kind == "tensor" and jac is None:
        active_surrogate_kind = "additive"
    if use_surrogate and HAS_SURROGATES and dim >= 2 and surrogate_kind is not None:
        remaining = counter.budget - counter.n
        min_pilot = max(config.pilot_min_base, config.pilot_dim_multiplier * dim)
        n_pilot = min(
            max(config.pilot_min_samples, config.pilot_dim_multiplier * dim),
            max(0, remaining // config.pilot_budget_divisor),
        )
        if n_pilot >= min_pilot:
            if active_surrogate_kind == "tensor":
                surr = TensorTrainSurrogate.build(
                    counter,
                    jac,
                    low,
                    high,
                    dim,
                    k=min(config.tensor_max_rank, dim),
                    degree=config.tensor_degree,
                    grid_m=config.tensor_grid_points,
                    n_pilot=n_pilot,
                    rng=rng,
                )
            elif active_surrogate_kind == "additive":
                surr = AdditiveSurrogate.fit(
                    counter,
                    low,
                    high,
                    dim,
                    degree=config.additive_degree,
                    n_pilot=n_pilot,
                    rng=rng,
                )
            else:
                raise ValueError("surrogate_kind must be 'tensor', 'additive', or None")

    gle_segment_budget = (
        max(config.gle_min_segment, counter.budget // config.gle_budget_divisor)
        if (
            use_gle
            and jac is not None
            and HAS_LIBRARY_GLE
            and dim >= config.gle_min_dimension
        )
        else 0
    )

    try:
        while True:
            temp = temp0 * math.log(2.0) / math.log(gen + 2.0)
            for i in range(pop_size):
                fi = f[i]
                cri = cr[i]
                if surr is not None and rng.random() < config.surrogate_proposal_probability:
                    T = max(temp, config.surrogate_temperature_floor)
                    trial = surr.sample(1, T, rng)[0]
                    trial = np.clip(trial, low, high)
                else:
                    fi = (
                        config.differential_weight_min
                        + config.differential_weight_span * rng.random()
                    ) if rng.random() < config.adaptation_probability else f[i]
                    cri = (
                        rng.random()
                        if rng.random() < config.adaptation_probability
                        else cr[i]
                    )
                    trial = _differential_trial(
                        pop,
                        best_x,
                        i,
                        fi,
                        cri,
                        rng,
                        low,
                        high,
                        config,
                    )

                ft = float(counter(trial))
                old = float(vals[i])
                accept = False
                if math.isfinite(ft):
                    accept = (not math.isfinite(old)) or _metropolis_accept(
                        ft - old,
                        temp,
                        rng,
                        floor=config.metropolis_temperature_floor,
                    )
                if accept:
                    pop[i] = trial
                    vals[i] = ft
                    f[i] = fi
                    cr[i] = cri
                    if ft < best_v:
                        best_v = float(ft)
                        best_x = trial.copy()
            gen += 1

            if counter.n - last_scout >= scout_every:
                scout = low_discrepancy_population(low, high, 1, skip=scout_skip)[0]
                scout_skip += 1
                fs = float(counter(scout))
                nonfinite_idx = np.flatnonzero(~np.isfinite(vals))
                worst = (
                    int(nonfinite_idx[0])
                    if nonfinite_idx.size
                    else int(np.argmax(vals))
                )
                if math.isfinite(fs) and (
                    not math.isfinite(float(vals[worst])) or fs < vals[worst]
                ):
                    pop[worst] = scout
                    vals[worst] = fs
                if math.isfinite(fs) and fs < best_v:
                    best_v = float(fs)
                    best_x = scout.copy()
                last_scout = counter.n

            if (
                use_gle
                and jac is not None
                and HAS_LIBRARY_GLE
                and dim >= config.gle_min_dimension
                and counter.n - last_scout >= scout_every // config.scout_gle_divisor
            ):
                remaining_units = counter.budget - counter.n
                maxf = min(
                    gle_segment_budget,
                    remaining_units // config.gle_shared_budget_units_per_step,
                )
                if maxf >= 2:
                    gle_res = _library_gle_langevin()(
                        counter,
                        jac,
                        low,
                        high,
                        max_fevals=maxf,
                        seed=int(rng.integers(1 << 31)),
                        omega0=config.gle_omega0,
                        dt=config.gle_dt,
                        n_epochs=config.gle_n_epochs,
                        x0=best_x,
                    )
                    if (
                        isinstance(gle_res, dict)
                        and math.isfinite(float(gle_res.get("best_val", float("inf"))))
                        and float(gle_res.get("best_val", float("inf"))) < best_v
                    ):
                        best_v = float(gle_res["best_val"])
                        if "best_pos" in gle_res:
                            best_x = np.asarray(gle_res["best_pos"], dtype=np.float64)
                last_scout = counter.n  # throttle

            if counter.n - last_polish >= polish_every:
                zoom = _elite_qmc_zoom(counter, pop, vals, low, high, rng, config)
                if zoom is not None:
                    zoom_x, zoom_v = zoom
                    if math.isfinite(zoom_v) and zoom_v < best_v:
                        best_v = zoom_v
                        best_x = zoom_x
                used_native_polish = boundary_polish_consumed
                boundary_polish_consumed = False
                if jac is not None and not used_native_polish:
                    native = _native_qmc_polish(
                        counter,
                        jac,
                        low,
                        high,
                        rng,
                        k_polish=k_polish,
                        config=config,
                    )
                    if native is not None:
                        used_native_polish = True
                        native_best = float(native.get("best_val", float("inf")))
                        if math.isfinite(native_best) and native_best < best_v:
                            best_v = native_best
                            if "best_pos" in native:
                                best_x = np.asarray(native["best_pos"], dtype=np.float64)
                if not used_native_polish:
                    maxfun = max(
                        config.local_polish_min_fevals,
                        counter.budget
                        // (2 * max(n_polish, 1) * max(k_polish, 1)),
                    )
                    for idx in np.argsort(vals)[: max(1, k_polish)]:
                        res = minimize(
                            counter,
                            pop[int(idx)],
                            method="L-BFGS-B",
                            jac=jac,
                            bounds=bounds,
                            options={"maxfun": maxfun, "maxiter": maxfun},
                        )
                        res_fun = float(res.fun)
                        old = float(vals[int(idx)])
                        if math.isfinite(res_fun) and (
                            not math.isfinite(old) or res_fun < old
                        ):
                            pop[int(idx)] = np.asarray(res.x, dtype=np.float64)
                            vals[int(idx)] = res_fun
                        if math.isfinite(res_fun) and res_fun < best_v:
                            best_v = res_fun
                            best_x = np.asarray(res.x, dtype=np.float64)
                if _trust_region_qmc_poll_active(dim, config) and counter.n < counter.budget:
                    trust_region = _native_qmc_trust_region_poll(
                        counter,
                        best_x,
                        low,
                        high,
                        rng,
                        config,
                    )
                    if trust_region is not None:
                        trust_best = float(trust_region.get("best_val", float("inf")))
                        if math.isfinite(trust_best) and trust_best < best_v:
                            best_v = trust_best
                            if "best_pos" in trust_region:
                                best_x = np.asarray(
                                    trust_region["best_pos"],
                                    dtype=np.float64,
                                )
                last_polish = counter.n
    except Exception as exc:  # noqa: BLE001
        if exc.__class__.__name__ != "_Budget":
            raise
    return _best_finite(best_v, counter.best)

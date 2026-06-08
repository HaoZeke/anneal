"""Anneal-native SOTA helpers for fixed-budget benchmark comparisons."""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib import import_module

import numpy as np
from scipy.optimize import minimize

# Optional structure-aware moves used by the benchmark hybrid. The counter keeps
# objective and native-gradient work in one budget.
from .tensor_surrogate import AdditiveSurrogate, TensorTrainSurrogate


HAS_SURROGATES = True
HAS_LIBRARY_GLE = True
library_gle_langevin = None
DEFAULT_HYBRID_N_POLISH = 6
DEFAULT_HYBRID_K_POLISH = 12
DEFAULT_ELITE_DIFFERENTIAL_PROBABILITY = 0.5


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
    gle_omega0: float | None = 0.2
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
    basin_polish_enabled: bool = True
    basin_polish_min_dimension: int = 20
    basin_polish_step: float = 0.1
    basin_polish_local_budget: int = 800
    basin_polish_temperature: float = 1.0
    native_bounds_slack: float = 1e-9
    local_polish_min_fevals: int = 20
    metropolis_temperature_floor: float = 1e-12
    temperature_floor: float = 1e-6


def _anneal_module():
    return import_module("anneal")


def _library_gle_langevin():
    return library_gle_langevin or _anneal_module().gle_langevin


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
    anneal = _anneal_module()
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


def _annealed_basin_polish(counter, grad_fn, low, high, dim, rng, config: AnnealHybridConfig):
    if grad_fn is None or counter.n >= counter.budget:
        return None
    jac = _counted_jac(counter, grad_fn)
    if jac is None:
        return None
    if config.basin_polish_step <= 0.0 or config.basin_polish_local_budget <= 0:
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
                x + rng.uniform(-config.basin_polish_step, config.basin_polish_step, dim),
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
    bounds = list(zip(low, high))
    jac = _counted_jac(counter, grad)
    if (
        config.basin_polish_enabled
        and jac is not None
        and dim >= config.basin_polish_min_dimension
    ):
        basin_best = _annealed_basin_polish(counter, grad, low, high, dim, rng, config)
        if counter.n >= counter.budget:
            return _best_finite(basin_best, counter.best)
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
    pop = [rng.uniform(low, high) for _ in range(pop_size)]
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
                used_native_polish = False
                if jac is not None:
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
                last_polish = counter.n
    except Exception as exc:  # noqa: BLE001
        if exc.__class__.__name__ != "_Budget":
            raise
    return _best_finite(best_v, counter.best)

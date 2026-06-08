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


@dataclass(frozen=True)
class AnnealHybridConfig:
    """Algorithm controls for ``qmc_annealed_hybrid``."""

    population_min: int = 12
    population_dim_multiplier: int = 5
    population_max: int = 50
    initial_differential_weight: float = 0.5
    initial_crossover_rate: float = 0.9
    adaptation_probability: float = 0.1
    differential_weight_min: float = 0.1
    differential_weight_span: float = 0.9
    surrogate_proposal_probability: float = 0.35
    surrogate_temperature_floor: float = 1e-6
    random_fallback_scale: float = 0.1
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
    scout_budget_divisor: int = 3
    scout_gle_divisor: int = 2
    qmc_min_starts: int = 2
    qmc_starts_per_polish: int = 4
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
    if remaining <= 0:
        return None
    n_starts = max(
        config.qmc_min_starts,
        config.qmc_starts_per_polish * max(k_polish, 1),
    )
    top_k = max(1, min(k_polish, n_starts))
    max_fevals_per_start = max(1, remaining // (n_starts + top_k))
    bounds = anneal.Bounds(low, high, config.native_bounds_slack)
    objective = anneal.PyObjective(counter, bounds, grad_fn=grad_fn)
    return anneal.qmc_polish_objective(
        objective,
        n_starts,
        max_fevals_per_start,
        seed=int(rng.integers(1 << 31)),
        top_k=top_k,
    )


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


def qmc_annealed_hybrid(
    counter,
    low,
    high,
    dim: int,
    grad,
    rng: np.random.Generator,
    *,
    n_polish: int = 6,
    k_polish: int = 3,
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
    best_idx = int(np.argmin(vals))
    best_x = pop[best_idx].copy()
    best_v = float(vals[best_idx])
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
        if (use_gle and jac is not None and HAS_LIBRARY_GLE)
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
                    if hasattr(surr, "sample"):
                        trial = surr.sample(1, T, rng)[0]
                    else:
                        trial = pop[i] + rng.normal(
                            0,
                            config.random_fallback_scale,
                            dim,
                        )
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
                    idx = [j for j in range(pop_size) if j != i]
                    r1, r2, r3 = rng.choice(idx, 3, replace=False)
                    mutant = pop[r1] + fi * (pop[r2] - pop[r3])
                    mask = rng.random(dim) < cri
                    mask[rng.integers(dim)] = True
                    trial = np.clip(np.where(mask, mutant, pop[i]), low, high)

                ft = counter(trial)
                if _metropolis_accept(
                    float(ft - vals[i]),
                    temp,
                    rng,
                    floor=config.metropolis_temperature_floor,
                ):
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
                fs = counter(scout)
                worst = int(np.argmax(vals))
                if fs < vals[worst]:
                    pop[worst] = scout
                    vals[worst] = fs
                if fs < best_v:
                    best_v = float(fs)
                    best_x = scout.copy()
                last_scout = counter.n

            if (
                use_gle
                and jac is not None
                and HAS_LIBRARY_GLE
                and counter.n - last_scout >= scout_every // config.scout_gle_divisor
            ):
                maxf = min(gle_segment_budget, counter.budget - counter.n)
                if maxf > 0:
                    gle_res = _library_gle_langevin()(
                        counter,
                        jac,
                        low,
                        high,
                        max_fevals=maxf,
                        seed=int(rng.integers(1 << 31)),
                    )
                    if (
                        isinstance(gle_res, dict)
                        and gle_res.get("best_val", float("inf")) < best_v
                    ):
                        best_v = float(gle_res["best_val"])
                        if "best_pos" in gle_res:
                            best_x = np.asarray(gle_res["best_pos"], dtype=np.float64)
                last_scout = counter.n  # throttle

            if counter.n - last_polish >= polish_every:
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
                        if math.isfinite(float(res.fun)) and res.fun < vals[int(idx)]:
                            pop[int(idx)] = np.asarray(res.x, dtype=np.float64)
                            vals[int(idx)] = float(res.fun)
                        if math.isfinite(float(res.fun)) and res.fun < best_v:
                            best_v = float(res.fun)
                            best_x = np.asarray(res.x, dtype=np.float64)
                last_polish = counter.n
    except Exception as exc:  # noqa: BLE001
        if exc.__class__.__name__ != "_Budget":
            raise
    return min(best_v, counter.best)

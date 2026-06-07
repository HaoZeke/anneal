"""Anneal-native SOTA helpers for fixed-budget benchmark comparisons."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize


def _radical_inverse(index: int, base: int) -> float:
    inv_base = 1.0 / float(base)
    fraction = inv_base
    value = 0.0
    while index > 0:
        value += (index % base) * fraction
        index //= base
        fraction *= inv_base
    return value


def _first_primes(n: int) -> list[int]:
    primes: list[int] = []
    candidate = 2
    while len(primes) < n:
        root = int(math.sqrt(candidate))
        if all(candidate % p for p in primes if p <= root):
            primes.append(candidate)
        candidate += 1 if candidate == 2 else 2
    return primes


def _halton_population(low: np.ndarray, high: np.ndarray, n: int, skip: int) -> np.ndarray:
    primes = _first_primes(low.size)
    points = np.empty((n, low.size), dtype=np.float64)
    for row in range(n):
        index = skip + row
        for axis, base in enumerate(primes):
            points[row, axis] = _radical_inverse(index, base)
    return low + (high - low) * points


def low_discrepancy_population(
    low,
    high,
    n: int,
    skip: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Bounded low-discrepancy population used by anneal SOTA drivers."""
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
    try:
        from anneal import low_discrepancy_points as core_low_discrepancy_points

        points = np.asarray(
            core_low_discrepancy_points(low, high, int(n), int(skip)),
            dtype=np.float64,
        )
    except Exception:  # noqa: BLE001
        points = _halton_population(low, high, int(n), int(skip))
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


def _metropolis_accept(delta: float, temp: float, rng: np.random.Generator) -> bool:
    if delta <= 0.0:
        return True
    if not math.isfinite(delta):
        return False
    return bool(rng.random() < math.exp(-delta / max(temp, 1e-12)))


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
):
    """QMC-seeded jDE/SA hybrid with budgeted local polish."""
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    bounds = list(zip(low, high))
    jac = _counted_jac(counter, grad)
    pop_size = int(min(50, max(12, 5 * dim)))
    n_qmc = max(1, pop_size // 3)
    qmc_pop = low_discrepancy_population(low, high, n_qmc, skip=1)
    random_pop = rng.uniform(low, high, size=(pop_size - n_qmc, dim))
    pop = [p.copy() for p in np.vstack([qmc_pop, random_pop])]
    vals = np.array([counter(p) for p in pop], dtype=np.float64)
    f = np.full(pop_size, 0.5)
    cr = np.full(pop_size, 0.9)
    best_idx = int(np.argmin(vals))
    best_x = pop[best_idx].copy()
    best_v = float(vals[best_idx])
    finite = vals[np.isfinite(vals)]
    temp0 = max(float(np.std(finite)) if finite.size > 1 else 1.0, 1e-6)
    polish_every = max(1, counter.budget // max(n_polish, 1))
    last_polish = 0
    gen = 0
    try:
        while True:
            temp = temp0 * math.log(2.0) / math.log(gen + 2.0)
            for i in range(pop_size):
                fi = (0.1 + 0.9 * rng.random()) if rng.random() < 0.1 else f[i]
                cri = rng.random() if rng.random() < 0.1 else cr[i]
                idx = [j for j in range(pop_size) if j != i]
                r1, r2, r3 = rng.choice(idx, 3, replace=False)
                mutant = pop[r1] + fi * (pop[r2] - pop[r3])
                mask = rng.random(dim) < cri
                mask[rng.integers(dim)] = True
                trial = np.clip(np.where(mask, mutant, pop[i]), low, high)
                ft = counter(trial)
                if _metropolis_accept(float(ft - vals[i]), temp, rng):
                    pop[i] = trial
                    vals[i] = ft
                    f[i] = fi
                    cr[i] = cri
                    if ft < best_v:
                        best_v = float(ft)
                        best_x = trial.copy()
            gen += 1

            if counter.n - last_polish >= polish_every:
                maxfun = max(20, counter.budget // (2 * max(n_polish, 1) * max(k_polish, 1)))
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

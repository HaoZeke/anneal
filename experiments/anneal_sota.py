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


def low_discrepancy_population(low, high, n: int, skip: int = 1) -> np.ndarray:
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

        return np.asarray(
            core_low_discrepancy_points(low, high, int(n), int(skip)),
            dtype=np.float64,
        )
    except Exception:  # noqa: BLE001
        return _halton_population(low, high, int(n), int(skip))


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
    """QMC-seeded jDE/GSA hybrid with rethermalising scouts and local polish."""
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    bounds = list(zip(low, high))
    jac = _counted_jac(counter, grad)
    pop_size = int(min(60, max(16, 6 * dim)))
    pop = [p.copy() for p in low_discrepancy_population(low, high, pop_size, skip=1)]
    vals = np.array([counter(p) for p in pop], dtype=np.float64)
    f = np.full(pop_size, 0.5, dtype=np.float64)
    cr = np.full(pop_size, 0.9, dtype=np.float64)
    best_idx = int(np.nanargmin(vals))
    best_x = pop[best_idx].copy()
    best_v = float(vals[best_idx])
    finite = vals[np.isfinite(vals)]
    temp0 = max(float(np.std(finite)) if finite.size > 1 else 1.0, 1e-6)
    sigma = float(np.clip(0.20 * np.linalg.norm(high - low) / max(dim, 1), 1e-9, np.inf))
    polish_every = max(1, counter.budget // max(n_polish, 1))
    scout_every = max(pop_size, counter.budget // max(2 * max(n_polish, 1), 1))
    last_polish = 0
    last_scout = 0
    scout_skip = 1 + pop_size
    stagnant = 0
    gen = 0
    try:
        while True:
            temp = temp0 * math.log(2.0) / math.log(gen + 2.0)
            order = np.argsort(vals)
            pbest_count = max(2, int(math.ceil(0.25 * pop_size)))
            pbest_pool = order[:pbest_count]
            gen_improved = False
            for i in range(pop_size):
                fi = (0.1 + 0.9 * rng.random()) if rng.random() < 0.1 else f[i]
                cri = rng.random() if rng.random() < 0.1 else cr[i]
                pbest = pop[int(rng.choice(pbest_pool))]
                idx = [j for j in range(pop_size) if j != i]
                r1, r2 = rng.choice(idx, 2, replace=False)
                mutant = pop[i] + fi * (pbest - pop[i]) + fi * (pop[r1] - pop[r2])
                if rng.random() < 0.25:
                    tail = np.clip(rng.standard_cauchy(dim), -25.0, 25.0)
                    mutant = mutant + sigma * max(temp / temp0, 0.05) * tail
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
                        gen_improved = True
            gen += 1
            stagnant = 0 if gen_improved else stagnant + 1

            if stagnant >= 2 or counter.n - last_scout >= scout_every:
                n_scout = max(1, pop_size // 8)
                for _ in range(n_scout):
                    if rng.random() < 0.5:
                        scout = low_discrepancy_population(low, high, 1, skip=scout_skip)[0]
                        scout_skip += 1
                    else:
                        tail = np.clip(rng.standard_cauchy(dim), -25.0, 25.0)
                        scout = np.clip(best_x + sigma * tail, low, high)
                    fs = counter(scout)
                    worst = int(np.nanargmax(vals))
                    if _metropolis_accept(float(fs - vals[worst]), max(temp, temp0), rng):
                        pop[worst] = scout
                        vals[worst] = fs
                        if fs < best_v:
                            best_v = float(fs)
                            best_x = scout.copy()
                last_scout = counter.n
                stagnant = 0

            if counter.n - last_polish >= polish_every:
                maxfun = max(12, counter.budget // (2 * max(n_polish, 1) * max(k_polish, 1)))
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

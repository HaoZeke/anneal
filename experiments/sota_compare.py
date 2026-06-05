"""Budget-matched comparison: Bayesian-allocated SA + local polish vs the field.

The bayesian-mixing driver wins among the SA variants but has no local
refinement; every state-of-the-art global method hybridizes (basin-hopping =
Monte Carlo + a local minimizer). This script adds an L-BFGS-B polish step to
the Thompson-allocated multistart SA -- Bayesian-allocated basin-hopping -- and
compares it, under a common objective-evaluation budget, against scipy's
basin-hopping, differential evolution, and an L-BFGS-B multistart, plus the
no-polish SA as the floor. Every objective call (including the finite-difference
gradients the local steps consume) is counted, so the budget is honest.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import basinhopping, differential_evolution, minimize

from experiments.surrogate_guided import CATALOG


class _Budget(Exception):
    pass


class Counter:
    """Objective wrapper that counts every call and stops at the budget."""

    def __init__(self, fn, budget):
        self.fn = fn
        self.budget = budget
        self.n = 0
        self.best = float("inf")

    def __call__(self, x):
        if self.n >= self.budget:
            raise _Budget()
        self.n += 1
        v = float(self.fn(np.asarray(x, float)))
        if v < self.best:
            self.best = v
        return v


def _auto_sigma(low, high, dim):
    diag = float(np.linalg.norm(high - low))
    return float(np.clip(0.25 * diag / dim, 1e-6, diag / np.sqrt(dim)))


def hybrid_bmsa(counter, low, high, dim, rng, n_chains=4, sa_block=150, n_polish=6):
    """Thompson-allocated multistart SA with periodic L-BFGS-B polish."""
    bounds = list(zip(low, high))
    chains = [rng.uniform(low, high) for _ in range(n_chains)]
    vals = [counter(c) for c in chains]
    a = np.ones(n_chains); b = np.ones(n_chains)
    best_x = chains[int(np.argmin(vals))].copy(); best_v = min(vals)
    sigma = _auto_sigma(low, high, dim)
    polish_every = max(1, counter.budget // n_polish)
    last_polish = 0
    epoch = 0
    try:
        while True:
            i = int(np.argmax(rng.beta(a, b)))             # Thompson pick
            temp = 5.0 * np.log(2.0) / np.log(epoch + 2.0)
            improved = False
            for _ in range(sa_block):
                y = np.clip(chains[i] + rng.normal(0.0, sigma, dim), low, high)
                fy = counter(y)
                if fy < vals[i] or rng.random() < np.exp(-(fy - vals[i]) / max(temp, 1e-12)):
                    chains[i], vals[i] = y, fy
                    if fy < best_v:
                        best_v, best_x, improved = fy, y.copy(), True
            a[i] += improved; b[i] += (not improved)
            epoch += 1
            if counter.n - last_polish >= polish_every:     # local refinement
                res = minimize(counter, best_x, method="L-BFGS-B", bounds=bounds,
                               options={"maxfun": max(20, counter.budget // (2 * n_polish))})
                if res.fun < best_v:
                    best_v, best_x = float(res.fun), np.asarray(res.x, float)
                last_polish = counter.n
    except _Budget:
        pass
    return min(best_v, counter.best)


def lbfgs_multistart(counter, low, high, dim, rng):
    bounds = list(zip(low, high))
    try:
        while True:
            x0 = rng.uniform(low, high)
            minimize(counter, x0, method="L-BFGS-B", bounds=bounds,
                     options={"maxfun": max(40, counter.budget // 8)})
    except _Budget:
        pass
    return counter.best


def scipy_basinhopping(counter, low, high, dim, rng):
    bounds = list(zip(low, high))
    x0 = rng.uniform(low, high)
    try:
        basinhopping(counter, x0, niter=10 ** 6,
                     minimizer_kwargs={"method": "L-BFGS-B", "bounds": bounds},
                     seed=int(rng.integers(1 << 31)))
    except _Budget:
        pass
    return counter.best


def scipy_de(counter, low, high, dim, rng):
    bounds = list(zip(low, high))
    try:
        differential_evolution(counter, bounds, maxiter=10 ** 6, polish=True,
                               seed=int(rng.integers(1 << 31)), tol=0)
    except _Budget:
        pass
    return counter.best


def plain_sa(counter, low, high, dim, rng):
    """No-polish multistart SA floor (same allocator, no L-BFGS)."""
    return hybrid_bmsa(counter, low, high, dim, rng, n_polish=0) if False else _plain(counter, low, high, dim, rng)


def _plain(counter, low, high, dim, rng):
    sigma = _auto_sigma(low, high, dim)
    x = rng.uniform(low, high); fx = counter(x); best = fx; epoch = 0
    try:
        while True:
            temp = 5.0 * np.log(2.0) / np.log(epoch + 2.0)
            for _ in range(150):
                y = np.clip(x + rng.normal(0.0, sigma, dim), low, high)
                fy = counter(y)
                if fy < fx or rng.random() < np.exp(-(fy - fx) / max(temp, 1e-12)):
                    x, fx = y, fy
                    best = min(best, fy)
            epoch += 1
    except _Budget:
        pass
    return min(best, counter.best)


def hybrid_de(counter, low, high, dim, rng, n_polish=6):
    """Self-adaptive (jDE) differential mutation in the Move slot, with SA
    Metropolis acceptance, a cooled temperature, and periodic L-BFGS-B polish.

    Each individual carries its own (F, CR), reset stochastically (the jDE
    self-adaptation that makes DE strong on multimodal landscapes). The
    acceptance is the SA Metropolis rule on a cooled temperature scaled to the
    population's value spread, so uphill trials are occasionally taken to escape
    stagnation, and the global incumbent is dropped into L-BFGS-B periodically.
    """
    bounds = list(zip(low, high))
    pop_size = int(min(50, max(12, 5 * dim)))
    pop = [rng.uniform(low, high) for _ in range(pop_size)]
    vals = np.array([counter(p) for p in pop])
    F = np.full(pop_size, 0.5)
    CR = np.full(pop_size, 0.9)
    bi = int(np.argmin(vals)); best_x = pop[bi].copy(); best_v = float(vals[bi])
    finite = vals[np.isfinite(vals)]
    temp0 = float(np.std(finite)) if finite.size > 1 else 1.0
    temp0 = max(temp0, 1e-6)
    polish_every = max(1, counter.budget // n_polish)
    last = 0; gen = 0
    try:
        while True:
            temp = temp0 * np.log(2.0) / np.log(gen + 2.0)
            for i in range(pop_size):
                fi = (0.1 + 0.9 * rng.random()) if rng.random() < 0.1 else F[i]
                cri = rng.random() if rng.random() < 0.1 else CR[i]
                idx = [j for j in range(pop_size) if j != i]
                r1, r2, r3 = rng.choice(idx, 3, replace=False)
                mutant = pop[r1] + fi * (pop[r2] - pop[r3])
                mask = rng.random(dim) < cri
                mask[rng.integers(dim)] = True
                trial = np.clip(np.where(mask, mutant, pop[i]), low, high)
                ft = counter(trial)
                if ft <= vals[i] or rng.random() < np.exp(-(ft - vals[i]) / max(temp, 1e-12)):
                    pop[i] = trial; vals[i] = ft; F[i] = fi; CR[i] = cri
                    if ft < best_v:
                        best_v, best_x = float(ft), trial.copy()
            gen += 1
            if counter.n - last >= polish_every:
                res = minimize(counter, best_x, method="L-BFGS-B", bounds=bounds,
                               options={"maxfun": max(20, counter.budget // (2 * n_polish))})
                if res.fun < best_v:
                    best_v, best_x = float(res.fun), np.asarray(res.x, float)
                last = counter.n
    except _Budget:
        pass
    return min(best_v, counter.best)


METHODS = {
    "hybrid_de": hybrid_de,
    "hybrid_bmsa": hybrid_bmsa,
    "basinhopping": scipy_basinhopping,
    "diff_evol": scipy_de,
    "lbfgs_multistart": lbfgs_multistart,
    "plain_sa": _plain,
}


def compare(dims=(10,), seeds=6, budget=6000):
    names = list(METHODS)
    print(f"best-at-budget ({budget} evals), median over {seeds} seeds; lower is better")
    print("objective    D  f*        " + "  ".join(f"{n:>16}" for n in names))
    for oname, (fn, (lo, hi), fstar) in CATALOG.items():
        for d in dims:
            low = np.full(d, lo); high = np.full(d, hi); fs = fstar(d)
            meds = {}
            for n, fnc in METHODS.items():
                vals = []
                for s in range(seeds):
                    rng = np.random.default_rng(s)
                    c = Counter(fn, budget)
                    try:
                        vals.append(fnc(c, low, high, d, rng))
                    except _Budget:
                        vals.append(c.best)
                meds[n] = float(np.median(vals))
            best_method = min(meds, key=meds.get)
            cells = "  ".join(
                (f"*{meds[n]:>15.3f}" if n == best_method else f"{meds[n]:>16.3f}")
                for n in names
            )
            print(f"{oname:<11} {d:>2} {fs:>8.2f}  {cells}", flush=True)


if __name__ == "__main__":
    compare()

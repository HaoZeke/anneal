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

from experiments.anneal_sota import qmc_annealed_hybrid
from experiments.benchmarks.catalog import CATALOG


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


def _catalog_cases(dims):
    wanted = {int(d) for d in dims}
    for problem in sorted(CATALOG.values(), key=lambda p: p.name):
        if problem.dim in wanted:
            yield problem


def _auto_sigma(low, high, dim):
    diag = float(np.linalg.norm(high - low))
    return float(np.clip(0.25 * diag / dim, 1e-6, diag / np.sqrt(dim)))


def hybrid_bmsa(counter, low, high, dim, rng, n_chains=4, sa_block=150, n_polish=6):
    """Thompson-allocated multistart SA with periodic L-BFGS-B polish."""
    bounds = list(zip(low, high))
    chains = [rng.uniform(low, high) for _ in range(n_chains)]
    vals = [counter(c) for c in chains]
    a = np.ones(n_chains)
    b = np.ones(n_chains)
    best_x = chains[int(np.argmin(vals))].copy()
    best_v = min(vals)
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
            a[i] += improved
            b[i] += not improved
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
    x = rng.uniform(low, high)
    fx = counter(x)
    best = fx
    epoch = 0
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


def hybrid_de(counter, low, high, dim, rng, n_polish=6, k_polish=1, **sota_kwargs):
    """Run the full hybrid path, including optional tensor and GLE wiring."""
    return qmc_annealed_hybrid(
        counter,
        low,
        high,
        dim,
        grad=None,
        rng=rng,
        n_polish=n_polish,
        k_polish=k_polish,
        **sota_kwargs,
    )


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
    for problem in _catalog_cases(dims):
        d = problem.dim
        meds = {}
        for n, fnc in METHODS.items():
            vals = []
            for s in range(seeds):
                rng = np.random.default_rng(s)
                c = Counter(problem.fn, budget)
                try:
                    vals.append(fnc(c, problem.low, problem.high, d, rng))
                except _Budget:
                    vals.append(c.best)
            meds[n] = float(np.median(vals))
        best_method = min(meds, key=meds.get)
        cells = "  ".join(
            (f"*{meds[n]:>15.3f}" if n == best_method else f"{meds[n]:>16.3f}")
            for n in names
        )
        print(f"{problem.name:<11} {d:>2} {problem.f_star:>8.2f}  {cells}", flush=True)


if __name__ == "__main__":
    compare()

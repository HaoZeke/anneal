"""Budget-matched SOTA comparison on the CUTEst set.

The ``hybrid_de`` entry is the anneal comparison point. It uses QMC seeding,
optional tensor/additive surrogate proposals, optional library GLE segments when
native gradients are available, and L-BFGS-B polish with counted gradients.
The SciPy baselines use the same objective/gradient budget accounting through
``Counter``.

Run on a host with pycutest bootstrapped (see bootstrap_cutest.sh):
  python -m experiments.scripts.sota_cutest --dim-cap 30 \
      --max-problems 60 --budget 8000 --seeds 3 --out results/sota_cutest.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import numpy as np
from scipy.optimize import basinhopping, differential_evolution, minimize

from experiments.anneal_sota import (
    DEFAULT_HYBRID_K_POLISH,
    DEFAULT_HYBRID_N_POLISH,
    qmc_annealed_hybrid,
)
from experiments.benchmarks.cutest_runner import configured_pycutest, default_cutest_config, load
from experiments.scripts.run_cutest_full_suite import list_target_problems


class _Budget(Exception):
    pass


class Counter:
    def __init__(self, fn, budget):
        self.fn = fn
        self.budget = budget
        self.n = 0
        self.objective_evals = 0
        self.grad_evals = 0
        self.best = float("inf")

    def _consume(self):
        if self.n >= self.budget:
            raise _Budget()
        self.n += 1

    def __call__(self, x):
        self._consume()
        self.objective_evals += 1
        v = float(self.fn(np.asarray(x, float).reshape(-1)))
        if math.isfinite(v) and v < self.best:
            self.best = v
        return v

    def counted_grad(self, grad):
        def jac(x):
            self._consume()
            self.grad_evals += 1
            return np.asarray(grad(np.asarray(x, float).reshape(-1)), float)

        return jac


def _counted_jac(counter, grad):
    return counter.counted_grad(grad) if grad else None


def _auto_sigma(low, high, dim):
    diag = float(np.linalg.norm(high - low))
    return float(np.clip(0.25 * diag / dim, 1e-6, diag / np.sqrt(dim)))


def classical(counter, low, high, dim, grad, rng):
    sigma = _auto_sigma(low, high, dim)
    x = rng.uniform(low, high); fx = counter(x); epoch = 0
    try:
        while True:
            temp = 5.0 * np.log(2.0) / np.log(epoch + 2.0)
            for _ in range(150):
                y = np.clip(x + rng.normal(0.0, sigma, dim), low, high)
                fy = counter(y)
                if fy < fx or rng.random() < np.exp(-(fy - fx) / max(temp, 1e-12)):
                    x, fx = y, fy
            epoch += 1
    except _Budget:
        pass
    return counter.best


def hybrid_de(
    counter,
    low,
    high,
    dim,
    grad,
    rng,
    n_polish=DEFAULT_HYBRID_N_POLISH,
    k_polish=DEFAULT_HYBRID_K_POLISH,
    config=None,
):
    """Anneal benchmark entry backed by ``qmc_annealed_hybrid``."""
    return qmc_annealed_hybrid(
        counter,
        low,
        high,
        dim,
        grad,
        rng,
        n_polish=n_polish,
        k_polish=k_polish,
        config=config,
    )


def sci_basinhopping(counter, low, high, dim, grad, rng):
    bounds = list(zip(low, high)); x0 = rng.uniform(low, high)
    jac = _counted_jac(counter, grad)
    mk = {"method": "L-BFGS-B", "bounds": bounds}
    if jac is not None:
        mk["jac"] = jac
    try:
        basinhopping(counter, x0, niter=10 ** 6, minimizer_kwargs=mk,
                     seed=int(rng.integers(1 << 31)))
    except _Budget:
        pass
    return counter.best


def sci_de(counter, low, high, dim, grad, rng):
    bounds = list(zip(low, high))
    try:
        differential_evolution(counter, bounds, maxiter=10 ** 6, polish=True,
                               seed=int(rng.integers(1 << 31)), tol=0)
    except _Budget:
        pass
    return counter.best


METHODS = {"hybrid_de": hybrid_de, "basinhopping": sci_basinhopping,
           "diff_evol": sci_de, "classical": classical}
FIELDNAMES = ["problem", "dim", "method", "seed", "best", "evals"]


def _write_sota_row(writer, stream, row):
    writer.writerow(row)
    stream.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/sota_cutest.csv")
    p.add_argument("--dim-cap", type=int, default=30)
    p.add_argument("--max-problems", type=int, default=60)
    p.add_argument("--budget", type=int, default=8000)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--bench-root", default=None,
                   help="Project root containing .bench/ with CUTEst, SIFDecode, and sif.")
    p.add_argument("--pycutest-cache", default=None,
                   help="Explicit PyCUTEst cache directory; defaults to .bench/cache.")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    config = default_cutest_config(args.bench_root, cache_dir=args.pycutest_cache)

    targets = list_target_problems(args.dim_cap, config=config)[: args.max_problems]
    print(f"{len(targets)} CUTEst problems, dim <= {args.dim_cap}, budget {args.budget}", flush=True)
    rows = []
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        f.flush()
        for t in targets:
            try:
                prob = load(t.name, sif_params=None, config=config)
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {t.name}: {type(exc).__name__}", flush=True)
                continue
            low = np.asarray(prob.low, float); high = np.asarray(prob.high, float)
            dim = prob.dim
            for s in range(args.seeds):
                for name, fnc in METHODS.items():
                    rng = np.random.default_rng(s)
                    c = Counter(prob.fn, args.budget)
                    try:
                        best = fnc(c, low, high, dim, prob.grad, rng)
                    except (_Budget, Exception):  # noqa: BLE001
                        best = c.best
                    row = dict(problem=t.name, dim=dim, method=name, seed=s,
                               best=best, evals=c.n)
                    rows.append(row)
                    _write_sota_row(w, f, row)
            try:
                pycutest = configured_pycutest(config)
                pycutest.clear_cache(t.name)
            except Exception:
                pass
            print(f"  {t.name} (dim {dim}) done", flush=True)

    # win-rate summary: per (problem, seed), which method reached the lowest best
    methods = list(METHODS)
    wins = {m: 0 for m in methods}
    cells = {}
    for r in rows:
        cells.setdefault((r["problem"], r["seed"]), []).append((r["method"], r["best"]))
    for cand in cells.values():
        finite = [(m, v) for m, v in cand if math.isfinite(v)]
        if finite:
            bv = min(v for _, v in finite)
            for m, v in finite:
                if v <= bv + 1e-9:
                    wins[m] += 1
    print(f"\nWin counts over {len(cells)} cells:")
    for m in sorted(methods, key=lambda k: -wins[k]):
        print(f"  {m:>14} {wins[m]:5d}")
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    sys.exit(main())

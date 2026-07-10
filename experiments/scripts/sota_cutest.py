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
from scipy.optimize import basinhopping, differential_evolution, dual_annealing, minimize

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


def _comparison_box(prob):
    low = np.asarray(getattr(prob, "design_low", prob.low), dtype=np.float64)
    high = np.asarray(getattr(prob, "design_high", prob.high), dtype=np.float64)
    anchor = getattr(prob, "x0", None)
    if anchor is None:
        anchor = 0.5 * (low + high)
    anchor = np.asarray(anchor, dtype=np.float64).reshape(-1)
    if anchor.shape != low.shape:
        anchor = 0.5 * (low + high)
    return low, high, np.clip(anchor, low, high)


def classical(counter, low, high, dim, grad, rng, anchor=None):
    sigma = _auto_sigma(low, high, dim)
    x = (
        np.asarray(anchor, dtype=np.float64).copy()
        if anchor is not None
        else rng.uniform(low, high)
    )
    fx = counter(x); epoch = 0
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
    anchor=None,
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
        anchor=anchor,
    )


def sci_basinhopping(counter, low, high, dim, grad, rng, anchor=None):
    bounds = list(zip(low, high))
    x0 = (
        np.asarray(anchor, dtype=np.float64).copy()
        if anchor is not None
        else rng.uniform(low, high)
    )
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


def portfolio(counter, low, high, dim, grad, rng, anchor=None, policy="auto"):
    """Native Thompson-allocated portfolio over anneal building blocks."""
    del anchor
    import anneal

    remaining = counter.budget - counter.n
    if remaining <= 0:
        return counter.best
    jac = counter.counted_grad(grad) if grad is not None else None
    try:
        out = anneal.global_optimize(
            counter,
            low,
            high,
            budget=remaining,
            seed=int(rng.integers(1 << 31)),
            grad_fn=jac,
            policy=policy,
        )
        best = float(out.get("best_val", float("inf")))
        pos = np.asarray(out.get("best_pos", []), dtype=float).reshape(-1)
        # Feasibility gate: OOB bests are non-solutions (score as +inf).
        if pos.size == low.size:
            if np.any(pos < low - 1e-8) or np.any(pos > high + 1e-8):
                best = float("inf")
        if math.isfinite(best) and best < counter.best:
            counter.best = best
    except _Budget:
        pass
    return counter.best


def portfolio_legacy(counter, low, high, dim, grad, rng, anchor=None):
    """Pre-regime portfolio (flat order, Beta(1,1)) for same-protocol A/B."""
    return portfolio(
        counter, low, high, dim, grad, rng, anchor=anchor, policy="legacy"
    )


def sci_dual_annealing(counter, low, high, dim, grad, rng, anchor=None):
    bounds = list(zip(low, high))
    x0 = (
        np.asarray(anchor, dtype=np.float64).copy()
        if anchor is not None
        else None
    )
    try:
        dual_annealing(counter, bounds, maxfun=10 ** 9, maxiter=10 ** 9,
                       seed=int(rng.integers(1 << 31)), x0=x0)
    except _Budget:
        pass
    return counter.best


def sci_de(counter, low, high, dim, grad, rng, anchor=None):
    del anchor
    bounds = list(zip(low, high))
    try:
        differential_evolution(counter, bounds, maxiter=10 ** 6, polish=True,
                               seed=int(rng.integers(1 << 31)), tol=0)
    except _Budget:
        pass
    return counter.best


def cma_es(counter, low, high, dim, grad, rng, anchor=None):
    """CMA-ES restarts (pycma) under the shared budget counter."""
    import cma

    width = np.where(high > low, high - low, 1.0)
    try:
        while counter.n < counter.budget:
            x0 = (
                np.asarray(anchor, dtype=np.float64).copy()
                if anchor is not None and counter.n == 0
                else rng.uniform(low, high)
            )
            es = cma.CMAEvolutionStrategy(
                x0,
                0.25 * float(np.mean(width)),
                {
                    "bounds": [list(low), list(high)],
                    "verbose": -9,
                    "seed": int(rng.integers(1 << 31)),
                    "maxfevals": counter.budget - counter.n,
                },
            )
            while not es.stop() and counter.n < counter.budget:
                xs = es.ask()
                es.tell(xs, [counter(x) for x in xs])
    except _Budget:
        pass
    return counter.best


def ngopt(counter, low, high, dim, grad, rng, anchor=None):
    """Nevergrad NGOpt wizard, restarted until the shared budget is spent.

    NGOpt can terminate before its declared budget; restart-until-budget
    matches the other baselines. Wall-clock warning: NGOpt's per-ask
    overhead dwarfs cheap CUTEst objectives, so full-matrix runs are
    disclosed with their wall cost in the campaign notes.
    """
    import nevergrad as ng

    del grad, anchor
    try:
        while counter.n < counter.budget:
            before = counter.n
            param = ng.p.Array(
                init=rng.uniform(low, high).astype(np.float64)
            ).set_bounds(list(low), list(high))
            param.random_state = np.random.RandomState(int(rng.integers(1 << 31)))
            opt = ng.optimizers.NGOpt(
                parametrization=param, budget=counter.budget - counter.n
            )
            opt.minimize(lambda x: counter(np.asarray(x, dtype=np.float64)))
            if counter.n == before:
                break
    except _Budget:
        pass
    return counter.best


def bobyqa(counter, low, high, dim, grad, rng, anchor=None):
    """Py-BOBYQA multistart-restarts under the shared budget counter."""
    import pybobyqa

    del grad
    try:
        while counter.n < counter.budget:
            x0 = (
                np.asarray(anchor, dtype=np.float64).copy()
                if anchor is not None and counter.n == 0
                else rng.uniform(low, high)
            )
            remaining = counter.budget - counter.n
            if remaining < 2 * dim + 2:
                break
            pybobyqa.solve(
                counter,
                x0,
                bounds=(np.asarray(low), np.asarray(high)),
                maxfun=remaining,
                seek_global_minimum=True,
                scaling_within_bounds=True,
                do_logging=False,
            )
    except _Budget:
        pass
    return counter.best


def cma_es_ipop(counter, low, high, dim, grad, rng, anchor=None):
    """IPOP-style CMA-ES: restart with growing population (stronger baseline)."""
    import cma

    del grad
    width = np.where(high > low, high - low, 1.0)
    mean_w = float(np.mean(width))
    pop0 = int(np.clip(4 + 3 * np.log(max(dim, 1)), 8, 40))
    pop = pop0
    try:
        while counter.n < counter.budget:
            x0 = (
                np.asarray(anchor, dtype=np.float64).copy()
                if anchor is not None and counter.n == 0
                else rng.uniform(low, high)
            )
            remaining = counter.budget - counter.n
            es = cma.CMAEvolutionStrategy(
                x0,
                0.3 * mean_w,
                {
                    "bounds": [list(low), list(high)],
                    "verbose": -9,
                    "seed": int(rng.integers(1 << 31)),
                    "maxfevals": remaining,
                    "popsize": pop,
                },
            )
            while not es.stop() and counter.n < counter.budget:
                xs = es.ask()
                es.tell(xs, [counter(x) for x in xs])
            # IPOP: grow population after each restart until budget ends.
            pop = min(pop * 2, 200)
    except _Budget:
        pass
    return counter.best


METHODS = {
    "portfolio": portfolio,
    "hybrid_de": hybrid_de,
    "basinhopping": sci_basinhopping,
    "dual_annealing": sci_dual_annealing,
    "diff_evol": sci_de,
    "cma_es": cma_es,
    "cma_es_ipop": cma_es_ipop,
    "ngopt": ngopt,
    "bobyqa": bobyqa,
    "classical": classical,
}
FIELDNAMES = ["problem", "dim", "method", "seed", "best", "evals"]


def _write_sota_row(writer, stream, row):
    writer.writerow(row)
    stream.flush()


def _shard_targets(targets, shard_index: int, shard_count: int):
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    return [
        target
        for index, target in enumerate(targets)
        if index % shard_count == shard_index
    ]


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
    p.add_argument("--shard-index", type=int, default=0,
                   help="Stable shard index for distributed CUTEst sweeps.")
    p.add_argument("--shard-count", type=int, default=1,
                   help="Number of stable shards in the distributed CUTEst sweep.")
    p.add_argument("--methods", default=None,
                   help="Comma-separated subset of methods to run.")
    p.add_argument(
        "--problems-file",
        default=None,
        help="Optional newline-separated problem names (paper list). "
        "When set, only these names are run (still filtered by dim-cap).",
    )
    args = p.parse_args()
    if args.methods:
        requested = [m.strip() for m in args.methods.split(",") if m.strip()]
        unknown = sorted(set(requested) - set(METHODS))
        if unknown:
            p.error(f"unknown methods: {unknown}")
        methods = {m: METHODS[m] for m in requested}
    else:
        methods = dict(METHODS)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    config = default_cutest_config(args.bench_root, cache_dir=args.pycutest_cache)

    if args.problems_file:
        from experiments.scripts.run_cutest_full_suite import TargetProblem

        names = [
            line.strip()
            for line in open(args.problems_file, encoding="utf-8")
            if line.strip() and not line.strip().startswith("#")
        ]
        # Prefer paper order; load will skip failures later.
        # kind/dim filled after load; placeholder for listing only.
        all_targets = [
            TargetProblem(name=n, kind="unconstrained", dim=0) for n in names
        ]
        all_targets = all_targets[: args.max_problems]
    else:
        all_targets = list_target_problems(args.dim_cap, config=config)[
            : args.max_problems
        ]
    targets = _shard_targets(all_targets, args.shard_index, args.shard_count)
    print(
        f"{len(targets)} CUTEst problems, dim <= {args.dim_cap}, "
        f"budget {args.budget}, shard {args.shard_index}/{args.shard_count}",
        flush=True,
    )
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
            low, high, anchor = _comparison_box(prob)
            dim = prob.dim
            for s in range(args.seeds):
                for name, fnc in methods.items():
                    rng = np.random.default_rng(s)
                    c = Counter(prob.fn, args.budget)
                    try:
                        best = fnc(c, low, high, dim, prob.grad, rng, anchor=anchor)
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

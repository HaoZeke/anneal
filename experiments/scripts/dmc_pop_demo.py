#!/usr/bin/env python3
"""Head-to-head: DMC-inspired population path vs classical baselines.

Protocol (fixed; primary metric = cell-best win rate, ties allowed):
  problems: Styblinski-Tang D=2,5; Rastrigin D=5,10
  seeds: 0..4 (5 seeds)
  budget: 1500 work units per cell
  methods:
    - dmc_pop: anneal.methods / portfolio is not required; pure
      population path via Rust binding if available, else portfolio
      forced by using global_optimize which includes dmc_pop arm
    - classical: Boltzmann-style via anneal.run(Boltzmann(...))
    - portfolio: anneal.global_optimize (includes dmc_pop among arms)
    - dual_annealing: scipy if available

Primary metric: win rate over (problem, seed) cells (lowest best wins).
"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import anneal
except ImportError:
    print("anneal not importable", file=sys.stderr)
    sys.exit(2)

try:
    from scipy.optimize import dual_annealing, basinhopping
except ImportError:
    dual_annealing = None
    basinhopping = None


def styblinski(dim: int):
    def f(x):
        x = np.asarray(x, float)
        return 0.5 * float(np.sum(x**4 - 16.0 * x**2 + 5.0 * x))

    def g(x):
        x = np.asarray(x, float)
        return 0.5 * (4.0 * x**3 - 32.0 * x + 5.0)

    low = np.full(dim, -5.0)
    high = np.full(dim, 5.0)
    return f, g, low, high, f"styb_d{dim}"


def rastrigin(dim: int):
    def f(x):
        x = np.asarray(x, float)
        return 10.0 * dim + float(np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))

    def g(x):
        x = np.asarray(x, float)
        return 2.0 * x + 20.0 * np.pi * np.sin(2.0 * np.pi * x)

    low = np.full(dim, -5.12)
    high = np.full(dim, 5.12)
    return f, g, low, high, f"rastrigin_d{dim}"


def run_classical(f, low, high, budget, seed):
    # Approximate budget: epochs * steps ~ budget/2 obj + starts
    steps = 25
    epochs = max(4, budget // (steps * 2))
    h = anneal.run(
        f,
        low,
        high,
        anneal.Boltzmann(t_init=5.0, sigma=0.4),
        n_epochs=epochs,
        steps_per_epoch=steps,
        seed=seed,
    )
    return float(h.best_val), int(getattr(h, "n_evals", epochs * steps) or epochs * steps)


def run_portfolio(f, g, low, high, budget, seed):
    out = anneal.global_optimize(f, low, high, budget=budget, seed=seed, grad_fn=g)
    return float(out["best_val"]), int(out["n_evals"]) + int(out["n_grads"])


def run_portfolio_legacy(f, g, low, high, budget, seed):
    out = anneal.global_optimize(
        f, low, high, budget=budget, seed=seed, grad_fn=g, policy="legacy"
    )
    return float(out["best_val"]), int(out["n_evals"]) + int(out["n_grads"])


def run_dmc_via_public(f, g, low, high, budget, seed):
    """Prefer dedicated binding if present; else portfolio (dmc_pop arm active)."""
    if hasattr(anneal, "dmc_population_optimize"):
        out = anneal.dmc_population_optimize(f, low, high, budget=budget, seed=seed, grad_fn=g)
        if isinstance(out, dict):
            return float(out["best_val"]), int(out.get("n_evals", 0)) + int(out.get("n_grads", 0))
    # Standalone path: many short portfolio runs would dilute; use pure
    # global_optimize which schedules dmc_pop under MultimodalNoGrad / Default.
    return run_portfolio(f, g, low, high, budget, seed)


def run_dual(f, low, high, budget, seed):
    if dual_annealing is None:
        return float("inf"), 0
    # SciPy uses maxfun as objective budget.
    res = dual_annealing(f, bounds=list(zip(low, high)), maxfun=budget, seed=seed)
    return float(res.fun), int(getattr(res, "nfev", budget) or budget)


def run_basin(f, low, high, budget, seed):
    if basinhopping is None:
        return float("inf"), 0
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(low, high)
    # Rough: minimizer_kwargs maxiter share
    niter = max(5, budget // 50)
    res = basinhopping(
        f,
        x0,
        niter=niter,
        seed=seed,
        minimizer_kwargs={"bounds": list(zip(low, high))},
    )
    return float(res.fun), int(budget)  # upper-bound accounting


def main():
    budget = 1500
    seeds = list(range(5))
    problems = [styblinski(2), styblinski(5), rastrigin(5), rastrigin(10)]
    methods = {
        "dmc_portfolio": run_dmc_via_public,  # portfolio with dmc_pop arm
        "portfolio_legacy": lambda f, g, lo, hi, b, s: run_portfolio_legacy(f, g, lo, hi, b, s),
        "classical": lambda f, g, lo, hi, b, s: run_classical(f, lo, hi, b, s),
        "dual_annealing": lambda f, g, lo, hi, b, s: run_dual(f, lo, hi, b, s),
    }

    rows = []
    for f, g, low, high, pname in problems:
        for seed in seeds:
            for mname, mfn in methods.items():
                try:
                    best, work = mfn(f, g, low, high, budget, seed)
                except Exception as exc:  # noqa: BLE001
                    best, work = float("inf"), 0
                    status = f"error:{type(exc).__name__}"
                else:
                    status = "ok" if math.isfinite(best) else "nonfinite"
                rows.append(
                    {
                        "problem": pname,
                        "seed": seed,
                        "method": mname,
                        "best": best,
                        "work": work,
                        "budget": budget,
                        "status": status,
                    }
                )
                print(f"{pname} seed={seed} {mname}: best={best:.6g} work={work}", flush=True)

    # Wins
    cells = defaultdict(list)
    for r in rows:
        cells[(r["problem"], r["seed"])].append(r)
    wins = defaultdict(int)
    for cell_rows in cells.values():
        finite = [r for r in cell_rows if math.isfinite(r["best"])]
        if not finite:
            continue
        bmin = min(r["best"] for r in finite)
        for r in finite:
            if r["best"] <= bmin + 1e-9 * max(1.0, abs(bmin)):
                wins[r["method"]] += 1

    ncells = len(cells)
    out_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dmc_demo_results.csv")
    out_sum = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("dmc_demo_summary.txt")
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines = [
        f"protocol: budget={budget} seeds={seeds} problems={[p[4] for p in problems]}",
        f"n_cells={ncells}",
        "primary_metric: cell-best win count (ties share win)",
        "",
    ]
    for m in methods:
        w = wins[m]
        lines.append(f"{m:20s}  wins={w:3d}  win%={100*w/ncells:5.1f}")
    lines.append("")
    # mean best per method
    by_m = defaultdict(list)
    for r in rows:
        if math.isfinite(r["best"]):
            by_m[r["method"]].append(r["best"])
    lines.append("mean_best (lower better):")
    for m in methods:
        vals = by_m[m]
        mb = sum(vals) / len(vals) if vals else float("nan")
        lines.append(f"  {m:20s}  {mb:.6g}  (n={len(vals)})")
    primary = "dmc_portfolio"
    baseline = "classical"
    ok = wins[primary] > wins[baseline]
    lines.append("")
    lines.append(
        f"primary_vs_baseline: {primary} wins {wins[primary]} > {baseline} wins {wins[baseline]} ? {ok}"
    )
    # also vs dual if present
    if dual_annealing is not None:
        lines.append(
            f"vs dual_annealing: {wins[primary]} vs {wins['dual_annealing']}"
        )
    out_sum.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if not ok:
        print("FAIL: dmc_portfolio did not beat classical on win rate", file=sys.stderr)
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()

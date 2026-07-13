#!/usr/bin/env python3
"""Head-to-head: pure dmc_pop vs classical SA, portfolio_legacy, dual_annealing.

Protocol (fixed; primary metric = cell-best win rate over cells):
  problems: Styblinski-Tang D=2,5; Rastrigin D=5
  seeds: 0..4
  budget: 1200 work units
  methods:
    dmc_pop          — anneal.dmc_population_optimize (standalone population path)
    classical        — anneal.run(Boltzmann(...)) with epochs*steps ≈ budget
    portfolio_legacy — global_optimize(..., policy=\"legacy\")  # no Auto dmc bias
    dual_annealing   — scipy.optimize.dual_annealing(maxfun=budget)

Primary metric: wins over (problem, seed) cells (lowest best wins; ties share).
Success: dmc_pop wins > classical wins (strict).
"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import anneal

try:
    from scipy.optimize import dual_annealing
except ImportError:
    dual_annealing = None


def styblinski(dim: int):
    def f(x):
        x = np.asarray(x, float)
        return 0.5 * float(np.sum(x**4 - 16.0 * x**2 + 5.0 * x))

    def g(x):
        x = np.asarray(x, float)
        return 0.5 * (4.0 * x**3 - 32.0 * x + 5.0)

    return f, g, np.full(dim, -5.0), np.full(dim, 5.0), f"styb_d{dim}"


def rastrigin(dim: int):
    def f(x):
        x = np.asarray(x, float)
        return 10.0 * dim + float(np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))

    def g(x):
        x = np.asarray(x, float)
        return 2.0 * x + 20.0 * np.pi * np.sin(2.0 * np.pi * x)

    return f, g, np.full(dim, -5.12), np.full(dim, 5.12), f"rastrigin_d{dim}"


def run_dmc(f, g, low, high, budget, seed):
    out = anneal.dmc_population_optimize(
        f, low, high, budget=budget, seed=seed, grad_fn=g, target_n=16, steps_per_control=4
    )
    return float(out["best_val"]), int(out["n_evals"]) + int(out["n_grads"]), "ok"


def run_classical(f, g, low, high, budget, seed):
    steps = 30
    epochs = max(5, budget // steps)
    # Cap so obj evals ~ budget
    epochs = min(epochs, max(5, budget // steps))
    h = anneal.run(
        f,
        low,
        high,
        anneal.Boltzmann(t_init=8.0, sigma=0.5),
        n_epochs=epochs,
        steps_per_epoch=steps,
        seed=seed,
    )
    work = epochs * steps  # each SA step evaluates once
    return float(h.best_val), work, "ok"


def run_legacy(f, g, low, high, budget, seed):
    out = anneal.global_optimize(
        f, low, high, budget=budget, seed=seed, grad_fn=g, policy="legacy"
    )
    return float(out["best_val"]), int(out["n_evals"]) + int(out["n_grads"]), "ok"


def run_dual(f, g, low, high, budget, seed):
    if dual_annealing is None:
        return float("inf"), 0, "skip"
    res = dual_annealing(f, bounds=list(zip(low, high)), maxfun=budget, seed=seed)
    return float(res.fun), int(getattr(res, "nfev", budget) or budget), "ok"


def main():
    budget = 1200
    seeds = list(range(5))
    problems = [styblinski(2), styblinski(5), rastrigin(5)]
    methods = {
        "dmc_pop": run_dmc,
        "classical": run_classical,
        "portfolio_legacy": run_legacy,
        "dual_annealing": run_dual,
    }

    rows = []
    for f, g, low, high, pname in problems:
        for seed in seeds:
            for mname, mfn in methods.items():
                try:
                    best, work, status = mfn(f, g, low, high, budget, seed)
                except Exception as exc:  # noqa: BLE001
                    best, work, status = float("inf"), 0, f"error:{type(exc).__name__}"
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
                print(
                    f"{pname} seed={seed} {mname}: best={best:.6g} work={work} {status}",
                    flush=True,
                )

    cells = defaultdict(list)
    for r in rows:
        if r["status"] == "skip":
            continue
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
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines = [
        f"protocol: budget={budget} seeds={seeds} problems={[p[4] for p in problems]}",
        f"n_cells={ncells}",
        "primary_metric: cell-best win count (ties share win)",
        "methods: dmc_pop (standalone population), classical (Boltzmann SA), "
        "portfolio_legacy, dual_annealing",
        "",
    ]
    for m in methods:
        wcount = wins[m]
        lines.append(f"{m:20s}  wins={wcount:3d}  win%={100 * wcount / ncells:5.1f}")
    by_m = defaultdict(list)
    for r in rows:
        if math.isfinite(r["best"]) and r["status"] == "ok":
            by_m[r["method"]].append(r["best"])
    lines.append("")
    lines.append("mean_best (lower better):")
    for m in methods:
        vals = by_m[m]
        mb = sum(vals) / len(vals) if vals else float("nan")
        lines.append(f"  {m:20s}  {mb:.6g}  (n={len(vals)})")
    primary, baseline = "dmc_pop", "classical"
    ok = wins[primary] > wins[baseline]
    lines.append("")
    lines.append(
        f"primary_vs_baseline: {primary} wins {wins[primary]} > "
        f"{baseline} wins {wins[baseline]} ? {ok}"
    )
    if dual_annealing is not None:
        lines.append(
            f"vs dual_annealing: {wins[primary]} vs {wins.get('dual_annealing', 0)}"
        )
    lines.append(
        f"vs portfolio_legacy: {wins[primary]} vs {wins.get('portfolio_legacy', 0)}"
    )
    out_sum.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if not ok:
        print("FAIL: dmc_pop did not beat classical on win rate", file=sys.stderr)
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()

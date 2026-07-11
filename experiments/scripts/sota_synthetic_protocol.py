"""Regenerable work-unit protocol (CUTEst-spirit) without Fortran CUTEst.

Same accounting as sota_cutest.py: shared budget, objective+grad units,
listed methods including portfolio and cma_es_ipop. Runs on pure-Python
multimodal boxes so remote builders without pycutest still measure ranking.

Protocol defaults:
  - 12 problems x 3 seeds = 36 cells
  - budget 4000 work units
  - methods: portfolio, basinhopping, dual_annealing, diff_evol, cma_es,
    cma_es_ipop, classical
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import numpy as np

# Reuse counters / method wrappers from sota_cutest
from experiments.scripts.sota_cutest import (
    FIELDNAMES,
    Counter,
    classical,
    cma_es,
    cma_es_ipop,
    portfolio,
    portfolio_legacy,
    run_method_cell,
    sci_basinhopping,
    sci_de,
    sci_dual_annealing,
)


def _rastrigin(d):
    def f(x):
        x = np.asarray(x, float).reshape(-1)
        return 10.0 * d + np.sum(x * x - 10.0 * np.cos(2.0 * np.pi * x))

    def g(x):
        x = np.asarray(x, float).reshape(-1)
        return 2.0 * x + 20.0 * np.pi * np.sin(2.0 * np.pi * x)

    return f, g, np.full(d, -5.12), np.full(d, 5.12)


def _ackley(d):
    def f(x):
        x = np.asarray(x, float).reshape(-1)
        a, b, c = 20.0, 0.2, 2.0 * np.pi
        s1 = np.mean(x * x)
        s2 = np.mean(np.cos(c * x))
        return -a * np.exp(-b * np.sqrt(s1)) - np.exp(s2) + a + np.e

    def g(x):
        # finite-diff free analytic gradient of Ackley
        x = np.asarray(x, float).reshape(-1)
        a, b, c = 20.0, 0.2, 2.0 * np.pi
        n = len(x)
        s1 = np.mean(x * x)
        s2 = np.mean(np.cos(c * x))
        r = np.sqrt(s1) + 1e-16
        g1 = a * b * np.exp(-b * r) * (x / (n * r))
        g2 = np.exp(s2) * (c * np.sin(c * x) / n)
        return g1 + g2

    return f, g, np.full(d, -5.0), np.full(d, 5.0)


def _styblinski(d):
    def f(x):
        x = np.asarray(x, float).reshape(-1)
        return 0.5 * np.sum(x**4 - 16.0 * x**2 + 5.0 * x)

    def g(x):
        x = np.asarray(x, float).reshape(-1)
        return 0.5 * (4.0 * x**3 - 32.0 * x + 5.0)

    return f, g, np.full(d, -5.0), np.full(d, 5.0)


def _rosenbrock(d):
    def f(x):
        x = np.asarray(x, float).reshape(-1)
        return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))

    def g(x):
        x = np.asarray(x, float).reshape(-1)
        g = np.zeros_like(x)
        # standard Rosenbrock gradient
        for i in range(d - 1):
            g[i] += -400.0 * x[i] * (x[i + 1] - x[i] ** 2) - 2.0 * (1 - x[i])
            g[i + 1] += 200.0 * (x[i + 1] - x[i] ** 2)
        return g

    return f, g, np.full(d, -2.0), np.full(d, 2.0)


def _schwefel(d):
    def f(x):
        x = np.asarray(x, float).reshape(-1)
        return float(418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x)))))

    def g(x):
        x = np.asarray(x, float).reshape(-1)
        # subgradient-like analytic form (approx)
        ax = np.abs(x) + 1e-12
        return -(
            np.sin(np.sqrt(ax)) + 0.5 * np.sqrt(ax) * np.cos(np.sqrt(ax)) * np.sign(x)
        )

    return f, g, np.full(d, -500.0), np.full(d, 500.0)


def _griewank(d):
    def f(x):
        x = np.asarray(x, float).reshape(-1)
        s = np.sum(x * x) / 4000.0
        p = np.prod(np.cos(x / np.sqrt(np.arange(1, d + 1))))
        return float(s - p + 1.0)

    def g(x):
        x = np.asarray(x, float).reshape(-1)
        idx = np.arange(1, d + 1, dtype=float)
        # finite-diff free: product rule for cos product
        c = np.cos(x / np.sqrt(idx))
        prod = np.prod(c)
        g = x / 2000.0
        for i in range(d):
            # d/dx_i of -prod cos
            if abs(c[i]) < 1e-15:
                continue
            g[i] += prod / c[i] * np.sin(x[i] / np.sqrt(idx[i])) / np.sqrt(idx[i])
        return g

    return f, g, np.full(d, -600.0), np.full(d, 600.0)


def problem_catalog():
    out = []
    for d in (2, 5, 10, 20, 30):
        for name, maker in (
            ("rastrigin", _rastrigin),
            ("ackley", _ackley),
            ("styblinski", _styblinski),
            ("schwefel", _schwefel),
            ("griewank", _griewank),
        ):
            f, g, lo, hi = maker(d)
            out.append((f"{name}_d{d}", d, f, g, lo, hi, True))
            if name in ("rastrigin", "schwefel") and d in (10, 20, 30):
                out.append((f"{name}_nograd_d{d}", d, f, g, lo, hi, False))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/sota_synthetic.csv")
    p.add_argument("--budget", type=int, default=4000)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument(
        "--methods",
        default=(
            "portfolio,portfolio_legacy,basinhopping,dual_annealing,"
            "diff_evol,cma_es,cma_es_ipop,classical"
        ),
    )
    args = p.parse_args()
    method_names = [m.strip() for m in args.methods.split(",") if m.strip()]
    runners = {
        "portfolio": portfolio,
        "portfolio_legacy": portfolio_legacy,
        "basinhopping": sci_basinhopping,
        "dual_annealing": sci_dual_annealing,
        "diff_evol": sci_de,
        "cma_es": cma_es,
        "cma_es_ipop": cma_es_ipop,
        "classical": classical,
    }
    for m in method_names:
        if m not in runners:
            p.error(f"unknown method {m}")

    problems = problem_catalog()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = []
    oob_counts = {m: 0 for m in method_names}
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        w.writeheader()
        for row in problems:
            if len(row) == 6:
                pname, dim, fn, grad, low, high = row
                use_grad = True
            else:
                pname, dim, fn, grad, low, high, use_grad = row
            if not use_grad:
                grad = None
            for s in range(args.seeds):
                for m in method_names:
                    c = Counter(fn, args.budget)
                    out_row = run_method_cell(
                        method_name=m,
                        method=runners[m],
                        problem=pname,
                        dim=dim,
                        seed=s,
                        counter=c,
                        low=low,
                        high=high,
                        grad=grad,
                        anchor=None,
                    )
                    rows.append(out_row)
                    w.writerow(out_row)
                    fh.flush()
            print(f"  {pname} done", flush=True)

    # Domain-feasibility sanitize for known analytic boxes (Schwefel min ~0 on box).
    for r in rows:
        if r["problem"].startswith("schwefel") and math.isfinite(float(r["best"])):
            if float(r["best"]) < -1.0:
                oob_counts[r["method"]] = oob_counts.get(r["method"], 0) + 1
                r["best"] = float("inf")
    oob_total = sum(oob_counts.values())
    print(
        f"OOB/domain-impossible bests (sanitized to +inf): {oob_counts} total={oob_total}"
    )
    if any(
        m.startswith("portfolio") and oob_counts.get(m, 0) > 0 for m in method_names
    ):
        raise SystemExit(
            f"FAIL: portfolio methods produced domain-impossible bests: {oob_counts}"
        )

    # wins + ranks (average ranks on ties — no list-order artifact)
    wins = {m: 0 for m in method_names}
    near = {m: 0 for m in method_names}
    ranks = {m: [] for m in method_names}
    cells = {}
    for r in rows:
        cells.setdefault((r["problem"], r["seed"]), []).append(r)
    for group in cells.values():
        finite = [r for r in group if math.isfinite(float(r["best"]))]
        if not finite:
            continue
        best = min(float(r["best"]) for r in finite)
        # Average rank for ties: sort by value, assign mean rank to equal plateaus.
        ordered = sorted(finite, key=lambda r: float(r["best"]))
        i = 0
        while i < len(ordered):
            j = i
            v = float(ordered[i]["best"])
            while j < len(ordered) and abs(
                float(ordered[j]["best"]) - v
            ) <= 1e-15 * max(1.0, abs(v)):
                j += 1
            # ranks i+1 .. j inclusive -> average
            avg_rank = 0.5 * ((i + 1) + j)
            for r in ordered[i:j]:
                ranks[r["method"]].append(avg_rank)
            i = j
        denom = max(abs(best), 1.0)
        for r in finite:
            gap = abs(float(r["best"]) - best) / denom
            if gap <= 1e-9:
                wins[r["method"]] += 1
            if gap <= 1e-3:
                near[r["method"]] += 1
    n = len(cells)
    print(f"\nProtocol: {n} cells, budget={args.budget}")
    print(f"{'method':16s} {'wins':>5} {'win%':>7} {'near%':>7} {'mean_rank':>10}")
    for m in method_names:
        mr = sum(ranks[m]) / len(ranks[m]) if ranks[m] else float("nan")
        print(
            f"{m:16s} {wins[m]:5d} {100 * wins[m] / n:7.1f} "
            f"{100 * near[m] / n:7.1f} {mr:10.2f}"
        )
    # Pairwise Auto vs Legacy if both present (strict best, not ties)
    if "portfolio" in method_names and "portfolio_legacy" in method_names:
        a_win = l_win = ties = 0
        for group in cells.values():
            by_m = {
                r["method"]: float(r["best"])
                for r in group
                if math.isfinite(float(r["best"]))
            }
            if "portfolio" not in by_m or "portfolio_legacy" not in by_m:
                continue
            pa, pl = by_m["portfolio"], by_m["portfolio_legacy"]
            scale = max(abs(pa), abs(pl), 1.0)
            if abs(pa - pl) / scale <= 1e-9:
                ties += 1
            elif pa < pl:
                a_win += 1
            else:
                l_win += 1
        print(
            f"Pairwise portfolio vs portfolio_legacy: "
            f"auto_strict={a_win} legacy_strict={l_win} equal={ties}"
        )
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    sys.exit(main() or 0)

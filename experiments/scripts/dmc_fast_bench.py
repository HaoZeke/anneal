#!/usr/bin/env python3
"""Wall-clock and best-value comparison for anneal.dmc_population_optimize.

Equal evaluation budget, shared random non-optimal starts, vs SciPy
dual_annealing and classical anneal Boltzmann SA. Writes CSV + text summary.
Exit code 0 if dmc_pop is faster on mean wall-clock for every problem and
mean best is within 5% (relative) of dual's mean best.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import numpy as np

import anneal
from scipy.optimize import dual_annealing


def rastrigin(dim: int):
    def f(x):
        x = np.asarray(x, float)
        return 10 * dim + float(np.sum(x * x - 10 * np.cos(2 * np.pi * x)))

    def g(x):
        x = np.asarray(x, float)
        return 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)

    return f, g, np.full(dim, -5.12), np.full(dim, 5.12)


def styb(dim: int):
    def f(x):
        x = np.asarray(x, float)
        return 0.5 * float(np.sum(x**4 - 16 * x**2 + 5 * x))

    def g(x):
        x = np.asarray(x, float)
        return 0.5 * (4 * x**3 - 32 * x + 5)

    return f, g, np.full(dim, -5.0), np.full(dim, 5.0)


def ackley(dim: int):
    def f(x):
        x = np.asarray(x, float)
        a, b, c = 20.0, 0.2, 2 * np.pi
        return float(
            -a * np.exp(-b * np.sqrt(np.mean(x * x)))
            - np.exp(np.mean(np.cos(c * x)))
            + a
            + np.e
        )

    def g(x):
        return np.zeros_like(x)

    return f, g, np.full(dim, -32.768), np.full(dim, 32.768)


def start_x(lo, hi, seed: int):
    rng = np.random.default_rng(seed + 999)
    return lo + 0.85 * (hi - lo) * rng.random(len(lo))


def quality_tol(mean_base: float) -> float:
    return max(1e-6, 0.05 * abs(mean_base))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    problems = [
        ("rastrigin_d5", *rastrigin(5)),
        ("rastrigin_d10", *rastrigin(10)),
        ("styb_d5", *styb(5)),
        ("ackley_d5", *ackley(5)),
    ]

    rows = []
    lines = []
    lines.append(f"method=dmc_population_optimize")
    lines.append(f"budget={args.budget} seeds={args.seeds} start=random_nonoptimal")
    lines.append(f"anneal={getattr(anneal, '__file__', '?')}")
    lines.append("baselines: scipy.optimize.dual_annealing; anneal.run Boltzmann")
    lines.append("")

    overall_time_wins = 0
    overall_quality_ok = 0
    overall_problems = 0
    overall_t_dmc = 0.0
    overall_t_dual = 0.0

    for pname, f, g, lo, hi in problems:
        use_g = g if "ackley" not in pname else None
        target_n = max(8, min(40, int(args.budget**0.5 * 0.9)))
        dmc_v, dual_v, cl_v = [], [], []
        dmc_t, dual_t, cl_t = [], [], []
        for seed in seeds:
            x0 = start_x(lo, hi, seed)

            t0 = time.perf_counter()
            out = anneal.dmc_population_optimize(
                f,
                lo,
                hi,
                budget=args.budget,
                seed=seed,
                grad_fn=use_g,
                target_n=target_n,
                steps_per_control=3,
                x0=x0,
            )
            dmc_t.append(time.perf_counter() - t0)
            dmc_v.append(float(out["best_val"]))

            t0 = time.perf_counter()
            r = dual_annealing(
                f, bounds=list(zip(lo, hi)), maxfun=args.budget, seed=seed, x0=x0
            )
            dual_t.append(time.perf_counter() - t0)
            dual_v.append(float(r.fun))

            steps, epochs = 40, max(5, args.budget // 40)
            t0 = time.perf_counter()
            h = anneal.run(
                f,
                lo,
                hi,
                anneal.Boltzmann(t_init=8.0, sigma=0.5),
                n_epochs=epochs,
                steps_per_epoch=steps,
                seed=seed,
            )
            cl_t.append(time.perf_counter() - t0)
            cl_v.append(float(h.best_val))

            for method, val, wall in (
                ("dmc_pop", dmc_v[-1], dmc_t[-1]),
                ("dual_annealing", dual_v[-1], dual_t[-1]),
                ("classical_boltzmann", cl_v[-1], cl_t[-1]),
            ):
                rows.append(
                    {
                        "problem": pname,
                        "seed": seed,
                        "budget": args.budget,
                        "method": method,
                        "best_val": val,
                        "wall_s": wall,
                    }
                )

        md, mu = statistics.mean(dmc_v), statistics.mean(dual_v)
        mtd, mtu = statistics.mean(dmc_t), statistics.mean(dual_t)
        mcl, mtcl = statistics.mean(cl_v), statistics.mean(cl_t)
        tol = quality_tol(mu)
        time_win = mtd < mtu
        quality_ok = md <= mu + tol
        pairwise_q = sum(1 for a, b in zip(dmc_v, dual_v) if a <= b + 1e-12)
        lines.append(pname)
        lines.append(
            f"  dmc_pop         mean_best={md:.6g} mean_wall={mtd:.6f}s"
        )
        lines.append(
            f"  dual_annealing  mean_best={mu:.6g} mean_wall={mtu:.6f}s"
        )
        lines.append(
            f"  classical       mean_best={mcl:.6g} mean_wall={mtcl:.6f}s"
        )
        lines.append(
            f"  time_win_vs_dual={time_win} quality_ok={quality_ok} "
            f"tol={tol:.3g} pairwise_q_dmc_le_dual={pairwise_q}/{len(seeds)}"
        )
        lines.append(
            f"  speedup_vs_dual={mtu / max(mtd, 1e-12):.3f}x"
        )
        overall_problems += 1
        if time_win:
            overall_time_wins += 1
        if quality_ok:
            overall_quality_ok += 1
        overall_t_dmc += mtd
        overall_t_dual += mtu

    lines.append("")
    lines.append(
        f"OVERALL problems with time_win={overall_time_wins}/{overall_problems} "
        f"quality_ok={overall_quality_ok}/{overall_problems}"
    )
    lines.append(
        f"OVERALL mean_wall sum dmc={overall_t_dmc:.6f}s dual={overall_t_dual:.6f}s "
        f"speedup={overall_t_dual / max(overall_t_dmc, 1e-12):.3f}x"
    )
    ok = (
        overall_time_wins == overall_problems
        and overall_quality_ok == overall_problems
        and overall_t_dmc < overall_t_dual
    )
    lines.append(f"PASS={ok}")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["problem", "seed", "budget", "method", "best_val", "wall_s"],
        )
        w.writeheader()
        w.writerows(rows)
    text = "\n".join(lines) + "\n"
    args.summary.write_text(text)
    print(text, end="")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

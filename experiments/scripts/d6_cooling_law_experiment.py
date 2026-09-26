#!/usr/bin/env python3
"""Head-to-head test of the D6 gap-proportional cooling law.

One Metropolis chain per (schedule, problem, dimension, seed): identical
isotropic Gaussian proposals with Robbins-Monro step-size targeting of
the D6 acceptance alpha* = 0.32, identical budgets, identical seeds.
Only the temperature policy differs:

  geometric   T_k = T0 * 0.95^k          (folklore default)
  logarithmic T_k = T0 / ln(k + e)       (classical guarantee schedule)
  d6          T   = 0.5 (f(x) - f_best)/d  (derived, constant-free)

T0 is set per problem from the sampled initial-energy spread (a common
adaptive recipe), so the classical schedules are not straw men. The D6
law needs no constant at all. Reported: median final gap to the known
optimum over seeds.

Usage: python d6_cooling_law_experiment.py [--budget 4000] [--seeds 16]
"""

from __future__ import annotations

import argparse
import math

import numpy as np


def sphere(x):
    return float(np.sum(x * x))


def rosenbrock(x):
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def rastrigin(x):
    return float(10.0 * len(x) + np.sum(x * x - 10.0 * np.cos(2.0 * np.pi * x)))


def ackley(x):
    d = len(x)
    return float(
        -20.0 * math.exp(-0.2 * math.sqrt(np.sum(x * x) / d))
        - math.exp(np.sum(np.cos(2.0 * np.pi * x)) / d)
        + 20.0
        + math.e
    )


def styblinski(x):
    return float(0.5 * np.sum(x**4 - 16.0 * x**2 + 5.0 * x) + 39.16617 * len(x))


PROBLEMS = {
    "sphere": (sphere, 5.12, 0.0),
    "rosenbrock": (rosenbrock, 5.0, 0.0),
    "rastrigin": (rastrigin, 5.12, 0.0),
    "ackley": (ackley, 32.0, 0.0),
    "styblinski": (styblinski, 5.0, 0.0),
}

ALPHA_TARGET = 0.32
THETA_TILDE = 0.5


def run_chain(f, width, dim, budget, seed, schedule):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-width, width, dim)
    fx = f(x)
    best = fx
    # Adaptive T0 from the initial-energy spread (both classical
    # schedules get this; the D6 law ignores it).
    probes = [f(rng.uniform(-width, width, dim)) for _ in range(8)]
    t0 = max(np.std(probes), 1e-12)
    log_sigma = math.log(0.1 * width)
    n = 8  # probes charged
    k = 0
    rm = 0
    while n < budget:
        if schedule == "geometric":
            temp = t0 * (0.95 ** (k / max(1, budget // (100 * 1))))
            # ~100 cooling stages over the budget
        elif schedule == "log":
            temp = t0 / math.log(k + math.e)
        else:  # d6
            temp = THETA_TILDE * max(fx - best, 0.0) / dim
        y = x + math.exp(log_sigma) * rng.standard_normal(dim)
        y = np.clip(y, -width, width)
        fy = f(y)
        n += 1
        k += 1
        delta = fy - fx
        accept = delta <= 0.0 or (temp > 0.0 and rng.random() < math.exp(-delta / temp))
        rm += 1
        log_sigma += (float(accept) - ALPHA_TARGET) / math.sqrt(rm)
        log_sigma = min(max(log_sigma, math.log(1e-9 * width)), math.log(2.0 * width))
        if accept:
            x, fx = y, fy
            best = min(best, fx)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 5, 10])
    args = ap.parse_args()

    schedules = ["geometric", "log", "d6"]
    print(f"{'problem':<12} {'D':>3} " + " ".join(f"{s:>12}" for s in schedules))
    wins = {s: 0 for s in schedules}
    for name, (f, width, fstar) in PROBLEMS.items():
        for dim in args.dims:
            medians = {}
            for s in schedules:
                gaps = [
                    run_chain(f, width, dim, args.budget, seed, s) - fstar
                    for seed in range(args.seeds)
                ]
                medians[s] = float(np.median(gaps))
            best_s = min(medians, key=medians.get)
            wins[best_s] += 1
            print(
                f"{name:<12} {dim:>3} "
                + " ".join(f"{medians[s]:>12.4g}" for s in schedules)
                + f"   <- {best_s}"
            )
    print("\ncell wins:", wins)


if __name__ == "__main__":
    main()

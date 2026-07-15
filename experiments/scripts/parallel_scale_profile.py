#!/usr/bin/env python3
"""Profile multi-start / population paths under RAYON_NUM_THREADS.

Runs anneal.dmc_population_optimize and qmc_polish (when available) at fixed
budget while varying OMP/Rayon thread counts. Prints wall times.
"""
from __future__ import annotations

import os
import statistics
import subprocess
import sys
import time

import numpy as np


def rastrigin(dim):
    def f(x):
        x = np.asarray(x, float)
        return 10 * dim + float(np.sum(x * x - 10 * np.cos(2 * np.pi * x)))

    def g(x):
        x = np.asarray(x, float)
        return 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)

    return f, g, np.full(dim, -5.12), np.full(dim, 5.12)


def one_run(threads: int, budget: int = 4000, seeds: int = 5):
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = str(threads)
    # child process so rayon pool size is fixed at import
    code = f"""
import time, numpy as np, anneal
def f(x):
    x=np.asarray(x,float); d=len(x)
    return 10*d+float(np.sum(x*x-10*np.cos(2*np.pi*x)))
def g(x):
    x=np.asarray(x,float)
    return 2*x+20*np.pi*np.sin(2*np.pi*x)
lo=np.full(10,-5.12); hi=np.full(10,5.12)
ts=[]
for seed in range({seeds}):
    t0=time.perf_counter()
    out=anneal.dmc_population_optimize(f,lo,hi,budget={budget},seed=seed,grad_fn=g,target_n=32,steps_per_control=2)
    ts.append(time.perf_counter()-t0)
print(sum(ts)/len(ts), min(ts), max(ts), out['best_val'])
"""
    r = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    parts = r.stdout.strip().split()
    return float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])


def main():
    print("dmc_population_optimize Rastrigin d=10 budget=4000 target_n=32")
    print("threads mean_s min_s max_s last_best")
    for t in (1, 2, 4, 8):
        mean, mn, mx, best = one_run(t)
        print(f"{t:7d} {mean:.6f} {mn:.6f} {mx:.6f} {best:.6g}")


if __name__ == "__main__":
    main()

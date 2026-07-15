#!/usr/bin/env python3
import math, time
import numpy as np
import anneal
from scipy.optimize import dual_annealing

def rastrigin(dim):
    def f(x):
        x = np.asarray(x, float)
        return 10 * dim + float(np.sum(x * x - 10 * np.cos(2 * np.pi * x)))
    def g(x):
        x = np.asarray(x, float)
        return 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)
    return f, g, np.full(dim, -5.12), np.full(dim, 5.12)

def styb(dim):
    def f(x):
        x = np.asarray(x, float)
        return 0.5 * float(np.sum(x**4 - 16 * x**2 + 5 * x))
    def g(x):
        x = np.asarray(x, float)
        return 0.5 * (4 * x**3 - 32 * x + 5)
    return f, g, np.full(dim, -5.0), np.full(dim, 5.0)

def ackley(dim):
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

def start_x(lo, hi, seed):
    rng = np.random.default_rng(seed + 999)
    return lo + 0.85 * (hi - lo) * rng.random(len(lo))

def main():
    budget = 2000
    seeds = list(range(8))
    problems = [
        ("rastrigin_d5", *rastrigin(5)),
        ("rastrigin_d10", *rastrigin(10)),
        ("styb_d5", *styb(5)),
        ("ackley_d5", *ackley(5)),
    ]
    print(f"budget={budget} seeds={len(seeds)} start=random_nonoptimal")
    overall_pd = overall_pc = 0
    overall_dd = overall_dual = 0
    for pname, f, g, lo, hi in problems:
        dmc, cl, dual, port = [], [], [], []
        td, tc, tdu, tp = [], [], [], []
        use_g = g if "ackley" not in pname else None
        for seed in seeds:
            x0 = start_x(lo, hi, seed)
            t0 = time.perf_counter()
            out = anneal.dmc_population_optimize(
                f, lo, hi, budget=budget, seed=seed, grad_fn=use_g,
                target_n=max(8, min(40, int(budget ** 0.5 * 0.9))),
                steps_per_control=3, x0=x0,
            )
            dmc.append(float(out["best_val"]))
            td.append(time.perf_counter() - t0)
            steps, epochs = 40, max(5, budget // 40)
            t0 = time.perf_counter()
            h = anneal.run(
                f, lo, hi, anneal.Boltzmann(t_init=8.0, sigma=0.5),
                n_epochs=epochs, steps_per_epoch=steps, seed=seed,
            )
            cl.append(float(h.best_val))
            tc.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            out = anneal.global_optimize(f, lo, hi, budget=budget, seed=seed, grad_fn=use_g)
            port.append(float(out["best_val"]))
            tp.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            r = dual_annealing(f, bounds=list(zip(lo, hi)), maxfun=budget, seed=seed, x0=x0)
            dual.append(float(r.fun))
            tdu.append(time.perf_counter() - t0)
        pd = pc = 0
        for a, b in zip(dmc, cl):
            if a < b:
                pd += 1
            elif b < a:
                pc += 1
        dd = du = 0
        for a, b in zip(dmc, dual):
            if a < b:
                dd += 1
            elif b < a:
                du += 1
        overall_pd += pd
        overall_pc += pc
        overall_dd += dd
        overall_dual += du
        print(pname)
        for name, vals, ts in [
            ("dmc_pop", dmc, td),
            ("classical", cl, tc),
            ("portfolio", port, tp),
            ("dual", dual, tdu),
        ]:
            m = sum(vals) / len(vals)
            med = float(np.median(vals))
            print(f"  {name:12s} mean={m:12.5g} med={med:12.5g} t={sum(ts)/len(ts):.3f}s")
        print(f"  pairwise dmc vs classical: {pd}-{pc}  vs dual: {dd}-{du}")
    print(f"OVERALL dmc vs classical: {overall_pd}-{overall_pc}")
    print(f"OVERALL dmc vs dual: {overall_dd}-{overall_dual}")
    print("DONE")

if __name__ == "__main__":
    main()

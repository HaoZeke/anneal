"""Thirteen Lennard-Jones atoms optimised through a ChemFit 4 Fitter.

Every driver gets the same evaluation budget, the box [-3, 3]^39 and the same
initial positions, and evaluates through ``Fitter.evaluate``. The score is the
best energy ChemFit recorded. Usage::

    python positions_benchmark.py BUDGET SEEDS [NAME_FILTER] > rows.jsonl

With NAME_FILTER only the drivers whose name contains it run.

Needs chemfit (4.x), nevergrad, scipy and anneal.
"""

import json
import math
import statistics
import sys
from collections import defaultdict

import anneal
import nevergrad as ng
import numpy as np
from chemfit.fitter import Fitter
from scipy.optimize import minimize

N_ATOMS = 13
LOW, HIGH = -3.0, 3.0


def lj_energy(positions):
    p = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    r = np.linalg.norm(p[:, None] - p[None], axis=-1)[np.triu_indices(len(p), 1)]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        e = float(np.sum(4.0 * (r**-12 - r**-6)))
    return e if np.isfinite(e) else 1e10


def objective(params):
    return lj_energy(params["positions"])


class Budgeted:
    """The fitter's objective with an exact budget and a one-point cache."""

    def __init__(self, fitter, budget):
        self.fitter, self.budget, self.calls, self.last = fitter, budget, 0, None

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.last is not None and np.array_equal(self.last[0], x):
            return self.last[1]
        if self.calls >= self.budget:
            return float("inf")
        self.calls += 1
        loss = self.fitter.evaluate({"positions": x.reshape(N_ATOMS, 3)})
        self.fitter.step()
        self.last = (x.copy(), loss)
        return loss

    def forward_difference(self, x):
        x = np.asarray(x, dtype=np.float64)
        f0 = self(x)
        grad = np.zeros_like(x)
        for i in range(x.size):
            h = 1e-6 * max(1.0, abs(x[i]))
            xi = x.copy()
            xi[i] = x[i] + h if x[i] + h <= HIGH else x[i] - h
            grad[i] = (self(xi) - f0) / (xi[i] - x[i])
        self.last = (x.copy(), f0)
        return grad


def nevergrad_driver(name):
    def drive(fitter, budget, seed, initial):
        param = ng.p.Array(init=initial.ravel()).set_bounds(LOW, HIGH)
        param.random_state = np.random.RandomState(seed)
        opt = ng.optimizers.registry[name](parametrization=param, budget=budget)
        for _ in range(budget):
            cand = opt.ask()
            loss = fitter.evaluate({"positions": cand.value.reshape(N_ATOMS, 3)})
            opt.tell(cand, loss)
            fitter.step()

    return drive


def anneal_driver(name):
    def drive(fitter, budget, seed, initial):
        f = Budgeted(fitter, budget)
        low, high = np.full((N_ATOMS, 3), LOW), np.full((N_ATOMS, 3), HIGH)
        if name in ("boltzmann", "fast", "gsa"):
            preset = {
                "boltzmann": anneal.Boltzmann(t_init=1.0, sigma=0.1),
                "fast": anneal.Fast(t_init=1.0, gamma=0.05),
                "gsa": anneal.Gsa(t_init=1.0),
            }[name]
            anneal.run(f, low, high, preset, n_epochs=math.ceil(budget / 100), steps_per_epoch=100,
                       seed=seed, x0=initial, max_evals=budget)
        elif name == "portfolio":
            anneal.global_optimize(f, low, high, budget=budget, seed=seed, x0=initial)
        elif name == "gpmd":
            anneal.gpmd_optimize(f, low, high, budget=budget, seed=seed, x0=initial)
        elif name == "dmc":
            anneal.dmc_population_optimize(f, low, high, budget=budget, seed=seed, x0=initial)
        elif name == "cluster_fd":
            anneal.cluster_search(f, f.forward_difference, N_ATOMS, budget, seed=seed)

    return drive


def fit_anneal_driver(method):
    def drive(fitter, budget, seed, initial):
        return fitter.fit_anneal(budget=budget, method=method, seed=seed)

    return drive


def scipy_fd(fitter, budget, seed, initial):
    f = Budgeted(fitter, budget)
    bounds = [(LOW, HIGH)] * (3 * N_ATOMS)
    minimize(f, initial.ravel(), method="L-BFGS-B", bounds=bounds, options={"maxfun": budget})


DRIVERS = {
    "nevergrad CMA": nevergrad_driver("CMA"),
    "nevergrad NgIohTuned": nevergrad_driver("NgIohTuned"),
    "anneal cluster_search, difference gradients": anneal_driver("cluster_fd"),
    "scipy L-BFGS-B, difference gradients": scipy_fd,
    "anneal dmc_population_optimize": anneal_driver("dmc"),
    "anneal Gsa": anneal_driver("gsa"),
    "anneal gpmd_optimize": anneal_driver("gpmd"),
    "anneal Boltzmann sigma=0.1": anneal_driver("boltzmann"),
    "anneal global_optimize, values only": anneal_driver("portfolio"),
    "Fitter.fit_anneal portfolio": fit_anneal_driver("portfolio"),
    "Fitter.fit_anneal boltzmann": fit_anneal_driver("boltzmann"),
    "Fitter.fit_anneal fast": fit_anneal_driver("fast"),
    "Fitter.fit_anneal gsa": fit_anneal_driver("gsa"),
    "Fitter.fit_anneal gpmd": fit_anneal_driver("gpmd"),
}


def main(budget, seeds, name_filter=""):
    best = defaultdict(list)
    for seed in range(seeds):
        initial = np.random.default_rng(1000 + seed).uniform(-1.6, 1.6, (N_ATOMS, 3))
        for name, drive in DRIVERS.items():
            if name_filter not in name:
                continue
            fitter = Fitter(objective, initial_params={"positions": initial.copy()},
                            bounds={"positions": (LOW, HIGH)}, value_bad_params=1e10)
            if name.startswith("Fitter.fit_anneal"):
                result = drive(fitter, budget, seed, initial.copy())
            else:
                fitter.init()
                drive(fitter, budget, seed, initial.copy())
                result = fitter.finish()
            evals = sum(ctx.n_evals for ctx in fitter.contexts)
            energy = lj_energy(result["positions"])
            best[name].append(energy)
            print(json.dumps({"budget": budget, "seed": seed, "driver": name, "best": energy,
                              "evals": evals, "inside": bool(np.all(np.abs(result["positions"]) <= HIGH))}),
                  flush=True)
    for name, values in best.items():
        print(f"# {budget} {name}: median {statistics.median(values):.2f} over {len(values)} seeds", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), *sys.argv[3:4])

"""Every anneal entry point a ChemFit 4 user can reach, on one Fitter.

Thirteen Lennard-Jones atoms in [-3, 3]^39, started from uniform positions in
[-1.6, 1.6]. For each path the script records the evaluations ChemFit
counted, how many evaluated geometries left the box, whether the first one
was the initial geometry, and the best energy ChemFit kept. Usage::

    python bridge_check.py BUDGET > bridge_check.jsonl

Needs chemfit (4.x, ``evaluate`` / ``step``) and anneal.
"""

import json
import math
import sys

import anneal
import numpy as np
from anneal.chemfit import fit_anneal, fit_chemfit, run_benchmark
from chemfit.fitter import Fitter

N_ATOMS = 13
LOW, HIGH = -3.0, 3.0


def lj_energy(positions):
    p = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    r = np.linalg.norm(p[:, None] - p[None], axis=-1)[np.triu_indices(len(p), 1)]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        e = float(np.sum(4.0 * (r**-12 - r**-6)))
    return e if np.isfinite(e) else 1e10


def reported_loop(context):
    """The bug report's code, unchanged apart from the imports."""
    fitter = context["fitter"]
    budget = context["budget"]
    n_atoms = len(context["initial_params"]["positions"])

    def obj(positions):
        loss = fitter.evaluate({"positions": positions.reshape(n_atoms, 3)})
        fitter.step()
        positions.reshape((n_atoms * 3))
        return loss

    low = -3.0 * np.ones(3 * n_atoms)
    high = 3.0 * np.ones(3 * n_atoms)

    fitter.init()
    anneal.run(obj, low, high, anneal.Boltzmann(), n_epochs=int(budget / 100), steps_per_epoch=100)
    return fitter.finish()


def corrected_loop(context):
    fitter = context["fitter"]
    budget = context["budget"]
    initial = np.asarray(context["initial_params"]["positions"])
    n_atoms = len(initial)

    def obj(positions):
        loss = fitter.evaluate({"positions": positions.reshape(n_atoms, 3)})
        fitter.step()
        return loss

    fitter.init()
    anneal.run(
        obj,
        np.full((n_atoms, 3), LOW),
        np.full((n_atoms, 3), HIGH),
        anneal.Boltzmann(t_init=1.0, sigma=0.1),
        n_epochs=math.ceil(budget / 100),
        steps_per_epoch=100,
        x0=initial,
        max_evals=budget,
    )
    return fitter.finish()


PATHS = {
    "bug report loop": reported_loop,
    "corrected loop": corrected_loop,
    "run_benchmark boltzmann": lambda c: run_benchmark(
        c, method="boltzmann", preset=anneal.Boltzmann(t_init=1.0, sigma=0.1), low=LOW, high=HIGH
    ),
    "run_benchmark portfolio": lambda c: run_benchmark(c, method="portfolio", low=LOW, high=HIGH),
    "fit_anneal boltzmann": lambda c: fit_anneal(
        c["fitter"], c["budget"], driver="boltzmann", preset_kwargs={"t_init": 1.0, "sigma": 0.1}
    ),
    "fit_anneal portfolio": lambda c: fit_anneal(c["fitter"], c["budget"]),
    "fit_chemfit boltzmann": lambda c: fit_chemfit(
        c["fitter"], c["budget"], method="boltzmann", t_init=1.0, sigma=0.1
    ),
    "fit_chemfit portfolio": lambda c: fit_chemfit(c["fitter"], c["budget"]),
    "Fitter.fit_anneal boltzmann": lambda c: c["fitter"].fit_anneal(budget=c["budget"], method="boltzmann"),
    "Fitter.fit_anneal portfolio": lambda c: c["fitter"].fit_anneal(budget=c["budget"]),
}


def main(budget):
    initial = np.random.default_rng(1000).uniform(-1.6, 1.6, (N_ATOMS, 3))
    for name, path in PATHS.items():
        seen = []

        def objective(params, seen=seen):
            seen.append(np.array(params["positions"], dtype=np.float64))
            return lj_energy(params["positions"])

        fitter = Fitter(
            objective,
            initial_params={"positions": initial.copy()},
            bounds={"positions": (LOW, HIGH)},
            value_bad_params=1e10,
        )
        context = {"fitter": fitter, "budget": budget, "initial_params": {"positions": initial.copy()}}
        result = path(context)
        points = np.array(seen)
        outside = int(np.sum(np.any((points < LOW) | (points > HIGH), axis=(1, 2))))
        print(
            json.dumps(
                {
                    "budget": budget,
                    "path": name,
                    "evals": sum(ctx.n_evals for ctx in fitter.contexts),
                    "outside": outside,
                    "starts_at_initial": bool(np.allclose(points[0], initial)),
                    "best": lj_energy(result["positions"]),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main(int(sys.argv[1]))

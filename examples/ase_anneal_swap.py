#!/usr/bin/env python3
"""Compare local ASE opts to an explicit anneal hop on EMT Cu7.

Default ``Anneal(atoms)`` is ``mode="local"``: a quench, same call site
as ``BFGS``. This script uses ``mode="search"`` so hops start from the
wire (``search_from``), not a random compact cluster. Local methods stay
in the chain basin; hops may leave it. Unequal budgets: ASE ``steps=200``
versus hop ``budget=4000``.

Run the comparison on the build host (not the laptop)::

    python examples/ase_anneal_swap.py
"""

from __future__ import annotations

import json
import time

import numpy as np


def _count_calc(inner):
    from ase.calculators.calculator import Calculator, all_changes

    class Counted(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self):
            super().__init__()
            self.inner = inner
            self.calls = 0

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            Calculator.calculate(self, atoms, properties, system_changes)
            self.calls += 1
            self.results["energy"] = float(self.inner.get_potential_energy(atoms))
            self.results["forces"] = np.asarray(self.inner.get_forces(atoms), dtype=float)

    return Counted()


def stretched_cu7():
    """A line of Cu atoms. Local ASE opts stay on the chain; hops collapse it."""
    from ase import Atoms
    from ase.calculators.emt import EMT

    pos = np.zeros((7, 3))
    pos[:, 0] = np.linspace(0.0, 14.0, 7)
    atoms = Atoms("Cu7", positions=pos)
    atoms.center(vacuum=8.0)
    atoms.calc = EMT()
    return atoms


def run_ase(name, cls, atoms, fmax, steps):
    from ase.calculators.emt import EMT

    atoms = atoms.copy()
    counted = _count_calc(EMT())
    atoms.calc = counted
    t0 = time.perf_counter()
    opt = cls(atoms, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    elapsed = time.perf_counter() - t0
    forces = atoms.get_forces()
    fmax_now = float(np.sqrt((forces**2).sum(axis=1)).max())
    return {
        "name": name,
        "energy": float(atoms.get_potential_energy()),
        "fmax": fmax_now,
        "calls": int(counted.calls),
        "seconds": elapsed,
        "converged": fmax_now < fmax,
    }


def run_anneal(atoms, fmax, steps, seed):
    from ase.calculators.emt import EMT

    from anneal.ase import Anneal

    atoms = atoms.copy()
    counted = _count_calc(EMT())
    atoms.calc = counted
    t0 = time.perf_counter()
    opt = Anneal(atoms, logfile=None, seed=seed, mode="search", budget=4000)
    opt.run(fmax=fmax, steps=steps)
    elapsed = time.perf_counter() - t0
    forces = atoms.get_forces()
    fmax_now = float(np.sqrt((forces**2).sum(axis=1)).max())
    return {
        "name": "Anneal",
        "energy": float(atoms.get_potential_energy()),
        "fmax": fmax_now,
        "calls": int(counted.calls),
        "seconds": elapsed,
        "converged": fmax_now < fmax,
        "nsteps": opt.get_number_of_steps(),
    }


def main():
    from ase.optimize import BFGS, FIRE, LBFGS

    fmax = 0.05
    steps = 200
    seed = 0
    start = stretched_cu7()
    start_e = float(start.get_potential_energy())
    rows = [
        run_ase("BFGS", BFGS, start, fmax, steps),
        run_ase("LBFGS", LBFGS, start, fmax, steps),
        run_ase("FIRE", FIRE, start, fmax, steps),
        run_anneal(start, fmax, steps, seed),
    ]
    report = {"start_energy": start_e, "fmax": fmax, "steps": steps, "rows": rows}
    print(json.dumps(report, indent=2))
    best = min(rows, key=lambda r: r["energy"])
    print(f"lowest energy: {best['name']} {best['energy']:.6f} eV")
    return report


if __name__ == "__main__":
    main()

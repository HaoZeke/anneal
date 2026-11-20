#!/usr/bin/env python3
# Copyright (c) anneal developers
"""One traced ASE minima hopping seed for the G3 funnel panel.

Usage: run_mh.py SYSTEM SEED CAP_S WORKDIR TRACE

The search is the committed ASE harness unchanged: the start from its
``place``, its metered calculator (``LjCalculator`` for Lennard-Jones,
``make_morse_calculator`` for Morse), ``np.random.seed(seed)`` and
``MinimaHopping(atoms)(totalsteps=10**9)`` with the ASE 3.29 defaults, stopped
by the budget, a quenched hit or the wall cap. The accepted minima are the
ones minima hopping itself appends to ``minima.traj``. Each is matched to
the first force call, after the previous minimum's, that returned its energy
bit for bit, and written to TRACE in the format ``classify.py`` reads with
move ``start`` for the first and ``md`` for the rest. The row goes to stdout
as ``key=value`` pairs.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from array import array
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
for path in (SCRIPTS, SCRIPTS / "ase_gofee_h2h", SCRIPTS / "ase_panel"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from potential import BudgetReached, CapReached, HitReached, LjCalculator, lj_value_and_gradient, warmup  # noqa: E402
from target_tol import DEFAULT_HIT_TOL  # noqa: E402


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ase_panel = load("g3_ase_panel", SCRIPTS / "ase_panel" / "run_one.py")
ase_nonlj = load("g3_ase_nonlj", SCRIPTS / "ase_nonlj" / "run_one.py")
gmin_panel = load("g3_gmin_panel", SCRIPTS / "gmin_panel" / "run_panel.py")

FMAX = ase_panel.FMAX["ase_mh"]


def log_energies(calc) -> array:
    energies = array("d")
    base = type(calc)

    class Logged(base):
        def calculate(self, atoms=None, properties=None, system_changes=None):
            before = self.calls
            try:
                base.calculate(self, atoms, properties, system_changes)
            finally:
                if self.calls > before:
                    energies.append(float(self.results["energy"]))

    calc.__class__ = Logged
    return energies


def uncharged_energy(kind: str, rho: float):
    if kind == "lj":
        return lambda positions: float(lj_value_and_gradient(np.ascontiguousarray(positions.ravel()))[0])
    from ase import Atoms
    from ase.calculators.morse import MorsePotential

    def energy(positions):
        atoms = Atoms(f"Ar{len(positions)}", positions=positions, pbc=False)
        atoms.calc = MorsePotential(epsilon=1.0, r0=1.0, rho0=rho, rcut1=1.0e6, rcut2=2.0e6)
        return float(atoms.get_potential_energy())

    return energy


def write_trace(minima_traj: Path, energies: array, energy_of, trace: Path) -> tuple[int, int]:
    from ase.io import Trajectory

    frames = []
    if minima_traj.exists() and minima_traj.stat().st_size > 0:
        try:
            with Trajectory(str(minima_traj)) as traj:
                for atoms in traj:
                    frames.append(atoms)
        except Exception:
            pass
    unmatched = 0
    cursor = 0
    with trace.open("w") as out:
        for index, atoms in enumerate(frames):
            positions = atoms.get_positions()
            try:
                energy = float(atoms.get_potential_energy())
            except Exception:
                energy = energy_of(positions)
            call = ""
            for k in range(cursor, len(energies)):
                if energies[k] == energy:
                    call = str(k + 1)
                    cursor = k + 1
                    break
            else:
                unmatched += 1
            move = "start" if index == 0 else "md"
            out.write(f"{len(positions)}\ncall={call} hop={index} move={move} energy={energy:.10f}\n")
            for x, y, z in positions:
                out.write(f"X {x:.6f} {y:.6f} {z:.6f}\n")
    return len(frames), unmatched


def main() -> int:
    if len(sys.argv) != 6:
        sys.exit(__doc__)
    system, seed_text, cap_text, work_text, trace_text = sys.argv[1:]
    seed = int(seed_text)
    cap_s = float(cap_text)
    work = Path(work_text).resolve()
    trace = Path(trace_text).resolve()
    import ase
    from ase import Atoms
    from ase.optimize.minimahopping import MinimaHopping

    if ase.__version__ != ase_panel.ASE_VERSION:
        raise RuntimeError(f"ASE {ase_panel.ASE_VERSION} required, found {ase.__version__}")
    _name, kind, n, reference, rho, _driver, _binary = gmin_panel.spec(system)
    budget = gmin_panel.PAPER_BUDGETS[system]
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)
    if kind == "lj":
        positions = ase_panel.place(n, seed, 1.0)
    elif kind == "morse":
        positions = ase_nonlj.place(n, seed, ase_nonlj.MORSE_LENGTH_SCALE)
    else:
        raise RuntimeError(f"{system}: no ASE harness")
    atoms = Atoms(f"Ar{n}", positions=positions, pbc=False)
    started = time.perf_counter()
    if kind == "lj":
        warmup()
        calc = LjCalculator(budget, reference, DEFAULT_HIT_TOL, FMAX, deadline=started + cap_s)
    else:
        calc = ase_nonlj.make_morse_calculator(rho, budget, reference, FMAX, started + cap_s)
    energies = log_energies(calc)
    atoms.calc = calc
    np.random.seed(seed)
    stop = "budget"
    try:
        MinimaHopping(atoms)(totalsteps=10**9)
    except CapReached:
        stop = "wall"
    except (BudgetReached, HitReached):
        stop = "budget"
    wall = time.perf_counter() - started
    best = calc.best_quenched if calc.best_quenched < 1.0e299 else calc.best
    hit = calc.hit_calls is not None
    minima, unmatched = write_trace(work / "minima.traj", energies, uncharged_energy(kind, rho), trace)
    row = {
        "system": system,
        "seed": seed,
        "budget": budget,
        "hit": int(hit),
        "evals_to_hit": calc.hit_calls if hit else "",
        "charged": calc.calls,
        "best": f"{float(best):.10f}",
        "wall_s": f"{wall:.6f}",
        "pot_s": f"{calc.inside_s:.6f}",
        "stop": "hit" if hit else stop,
        "cap_s": f"{cap_s:.6f}",
        "minima": minima,
        "unmatched": unmatched,
    }
    print("g3_row " + " ".join(f"{key}={value}" for key, value in row.items()), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

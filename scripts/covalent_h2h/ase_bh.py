#!/usr/bin/env python3
# Copyright (c) anneal developers
"""One seeded ASE basin hopping run on a Stillinger-Weber or Tersoff cluster.

GMIN has no Tersoff potential, so the carbon rows compare with
``ase.optimize.basin.BasinHopping``. The calculator is the same
``librgpot_covalent.so`` the anneal example loads, in the same units. The
start is anneal's start for that seed. Each quench is ASE's L-BFGS to a
largest force of 1e-3, the anneal record tolerance. The step and the
temperature are the GMIN file's ``STEP 0.40`` and ``TEMPERATURE 0.2`` on
the dimer scale ``h2h_covalent scales`` prints. ASE has no container. One calculator call is one energy and
gradient. The run stops at the shared call budget, or when a finished
quench lies within 0.001 of the reference.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import LBFGS
from ase.optimize.basin import BasinHopping

HIT_TOL = 0.001
FMAX = 1.0e-3
STEP = 0.40
TEMPERATURE = 0.2


class BudgetSpent(RuntimeError):
    """The shared call budget was reached inside a quench."""


class Covalent:
    """The two rgpot kernels in the units of the anneal example."""

    def __init__(self, library: Path, system: str):
        self.lib = ctypes.CDLL(str(library))
        self.eval = self.lib.rgpot_covalent_energy_gradient
        pointer = ctypes.POINTER(ctypes.c_double)
        self.eval.argtypes = [ctypes.c_int, ctypes.c_long, pointer, pointer, pointer]
        self.eval.restype = ctypes.c_int
        if system.startswith("sw"):
            self.kind = 0
            sigma, epsilon = ctypes.c_double(), ctypes.c_double()
            self.lib.rgpot_covalent_sw_units(ctypes.byref(sigma), ctypes.byref(epsilon))
            self.length, self.energy = sigma.value, epsilon.value
        elif system.startswith("tc"):
            self.kind = 1
            self.length, self.energy = 1.0, 1.0
        else:
            raise ValueError(f"unknown system {system}")

    def __call__(self, xyz: np.ndarray) -> tuple[float, np.ndarray]:
        positions = np.ascontiguousarray(xyz, dtype=float).ravel() * self.length
        gradient = np.zeros_like(positions)
        energy = ctypes.c_double()
        pointer = ctypes.POINTER(ctypes.c_double)
        status = self.eval(
            self.kind,
            len(positions) // 3,
            positions.ctypes.data_as(pointer),
            ctypes.byref(energy),
            gradient.ctypes.data_as(pointer),
        )
        if status != 0:
            return float("inf"), np.zeros_like(xyz)
        scale = self.length / self.energy
        return energy.value / self.energy, (gradient * scale).reshape(xyz.shape)


class BudgetCalculator(Calculator):
    """Counts calls and keeps the time inside the library."""

    implemented_properties = ("energy", "forces")

    def __init__(self, potential: Covalent, budget: int, level: float):
        super().__init__()
        self.potential = potential
        self.budget = budget
        self.level = level
        self.calls = 0
        self.pot_s = 0.0
        self.best = float("inf")
        self.calls_at_hit: int | None = None

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        if self.calls >= self.budget:
            raise BudgetSpent
        Calculator.calculate(self, atoms, properties, system_changes)
        started = time.perf_counter()
        energy, gradient = self.potential(self.atoms.get_positions())
        self.pot_s += time.perf_counter() - started
        self.calls += 1
        self.results = {"energy": energy, "forces": -gradient}

    def note_quench(self, energy: float, forces: np.ndarray) -> None:
        if np.abs(forces).max() > FMAX:
            return
        if energy < self.best:
            self.best = energy
        if energy <= self.level and self.calls_at_hit is None:
            self.calls_at_hit = self.calls


class StopAtLevel(RuntimeError):
    """A finished quench reached the reference."""


class Hopping(BasinHopping):
    """``BasinHopping`` that reports every finished quench to the budget."""

    def __init__(self, *args, calc: BudgetCalculator, **kwargs):
        self.budget_calc = calc
        super().__init__(*args, **kwargs)

    def get_energy(self, positions):
        energy = super().get_energy(positions)
        forces = self.optimizable.get_forces()
        self.budget_calc.note_quench(energy, forces)
        if self.budget_calc.calls_at_hit is not None:
            raise StopAtLevel
        return energy


def binary_lines(binary: Path, command: str, system: str) -> list[str]:
    text = subprocess.run(
        [str(binary), command, system], check=True, capture_output=True, text=True
    ).stdout
    return text.split()


def run(args: argparse.Namespace) -> dict:
    budget_text, reference_text, count_text = binary_lines(args.binary, "budget", args.system)
    length, energy_scale, _container = (
        float(v) for v in binary_lines(args.binary, "scales", args.system)
    )
    reference = float(os.environ.get("H2H_TARGET", reference_text))
    budget = args.budget if args.budget > 0 else int(budget_text)
    count = int(count_text)
    with tempfile.TemporaryDirectory() as scratch:
        start_path = Path(scratch) / "start.xyz"
        subprocess.run(
            [str(args.binary), "coords", args.system, str(args.seed), str(start_path)], check=True
        )
        start = np.loadtxt(start_path).reshape(count, 3)
    potential = Covalent(args.library, args.system)
    calc = BudgetCalculator(potential, budget, reference + HIT_TOL)
    atoms = Atoms(numbers=[14 if args.system.startswith("sw") else 6] * count, positions=start)
    atoms.calc = calc
    np.random.seed(args.seed)
    started = time.perf_counter()
    try:
        hopping = Hopping(
            atoms,
            calc=calc,
            temperature=TEMPERATURE * energy_scale,
            optimizer=LBFGS,
            fmax=FMAX,
            dr=STEP * length,
            logfile=None,
            trajectory=None,
            optimizer_logfile=None,
            local_minima_trajectory=None,
        )
        hopping.run(10**9)
    except (BudgetSpent, StopAtLevel):
        pass
    wall = time.perf_counter() - started
    hit = calc.calls_at_hit is not None
    outside = max(0.0, wall - calc.pot_s)
    calls = max(calc.calls, 1)
    return {
        "system": args.system,
        "seed": args.seed,
        "budget": budget,
        "hit": str(hit).lower(),
        "evals": calc.calls_at_hit if hit else budget,
        "charged": calc.calls,
        "wall": wall,
        "best": calc.best if np.isfinite(calc.best) else 0.0,
        "pot_s": calc.pot_s,
        "outside": outside,
        "pot_per": calc.pot_s / calls,
        "outside_per": outside / calls,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", type=int, default=0)
    args = parser.parse_args(argv[1:])
    row = run(args)
    print(
        "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,pot_s,outside_s,pot_per_eval_s,outside_per_eval_s"
    )
    print(
        "{system},ase,{seed},{budget},{hit},{evals},{charged},{wall:.6f},{best:.8f},{pot_s:.6f},{outside:.6f},{pot_per:.9f},{outside_per:.9f}".format(
            **row
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

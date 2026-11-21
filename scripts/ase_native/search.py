# Copyright (c) anneal developers
"""One seed of the native-unit head-to-head.

Anneal runs the existing release examples with the for_cluster preset and
no wall cap. ASE runs basin hopping or minima hopping against a calculator
that counts every energy-and-force evaluation and stops at the paper
budget. The calculator's numbers come from the anneal kernel. The
agreement test checks that kernel against the eV/angstrom formulas.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from controls import run_basin_hopping, run_minima_hopping
from eval_client import EvalServer, read_scales
from mapping import (
    DENSITY,
    HIT_TOL,
    MIN_SEP,
    SYSTEMS,
    energy_hits,
    fmax_from_infinity,
    record_gradient,
)

ROOT = Path(__file__).resolve().parents[2]
PANEL = Path(os.environ.get("ASE_PANEL", ROOT.parent / "tmp" / "anneal-target" / "release" / "examples" / "ase_panel"))
NONLJ = Path(os.environ.get("ASE_NONLJ", ROOT.parent / "tmp" / "anneal-target" / "release" / "examples" / "ase_nonlj"))


class BudgetReached(BaseException):
    """The evaluation budget is spent. A BaseException so optimizers do not catch it."""


class HitReached(BaseException):
    """A recorded quench is inside the hit window."""


def place(n: int, seed: int, length: float) -> np.ndarray:
    """Same sphere placement the previous ASE panel used."""
    rng = np.random.default_rng(seed)
    radius = length * (3.0 * n / (4.0 * np.pi * DENSITY)) ** (1.0 / 3.0)
    gap = MIN_SEP * length
    points: list[np.ndarray] = []
    draws = 0
    while len(points) < n:
        draws += 1
        if draws > 1_000_000:
            raise RuntimeError("could not place the cluster")
        direction = rng.normal(size=3)
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            continue
        point = direction * (radius * (float(rng.random()) ** (1.0 / 3.0)) / norm)
        if all(float(np.linalg.norm(point - other)) >= gap for other in points):
            points.append(point)
    return np.vstack(points)


class BudgetCalculator(Calculator):
    """One charge per geometry. Stops on the budget or on a recorded hit."""

    implemented_properties = ["energy", "forces", "free_energy"]

    def check_state(self, atoms, tol=1e-15):
        # Same tolerance as ase.calculators.calculator.compare_atoms. Numbers,
        # cell and pbc are fixed for these clusters, so a position match is a
        # cache hit and is not a new evaluation.
        if self.atoms is None:
            return ["positions"]
        if atoms.positions.shape != self.atoms.positions.shape:
            return ["positions"]
        if np.array_equal(atoms.positions, self.atoms.positions):
            return []
        if np.allclose(atoms.positions, self.atoms.positions, rtol=0.0, atol=tol):
            return []
        return ["positions"]

    def __init__(self, server: EvalServer, budget: int, reference: float, record: float):
        super().__init__()
        self.server = server
        self.budget = budget
        self.reference = reference
        self.record = record
        self.calls = 0
        self.hit_calls: int | None = None
        self.best = float("inf")
        self.best_quenched = float("inf")
        self.best_infinity = float("inf")
        self.best_ase = float("inf")

    def calculate(self, atoms=None, properties=None, system_changes=None):
        if self.calls >= self.budget:
            raise BudgetReached()
        # Record the geometry first so a later request for another property
        # of the same atoms hits the calculator cache instead of a second charge.
        super().calculate(atoms, properties, system_changes)
        energy, gradient = self.server.evaluate(atoms.positions)
        self.calls += 1
        forces = -gradient
        self.results = {"energy": energy, "forces": forces, "free_energy": energy}
        infinity = float(np.max(np.abs(gradient))) if gradient.size else 0.0
        per_atom = np.linalg.norm(forces, axis=1)
        ase_norm = float(per_atom.max()) if per_atom.size else 0.0
        if energy < self.best:
            self.best = energy
        if infinity < self.record and energy < self.best_quenched:
            self.best_quenched = energy
            self.best_infinity = infinity
            self.best_ase = ase_norm
        if (
            self.hit_calls is None
            and infinity < self.record
            and energy_hits(energy, self.reference, HIT_TOL)
        ):
            self.hit_calls = self.calls
            raise HitReached()


def _parse_anneal(text: str) -> dict[str, str]:
    for line in text.splitlines():
        if line.startswith("system="):
            return dict(part.split("=", 1) for part in line.split())
    raise RuntimeError("anneal stdout has no row")


def run_anneal(system: str, seed: int, budget: int, kind: str) -> dict[str, float | int | str]:
    if kind == "morse":
        binary = NONLJ
        command = [str(binary), "run", system, str(budget), str(seed), "for_cluster"]
    else:
        binary = PANEL
        command = [str(binary), system, str(budget), str(seed), "for_cluster"]
    if not binary.is_file():
        raise RuntimeError(f"missing anneal binary {binary}")
    started = time.perf_counter()
    proc = subprocess.run(command, check=False, capture_output=True, text=True)
    wall = time.perf_counter() - started
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-2000:])
    fields = _parse_anneal(proc.stdout)
    if fields.get("system") != system or fields.get("seed") != str(seed) or fields.get("arm") != "for_cluster":
        raise RuntimeError("anneal row does not match the request")
    if fields.get("stop") == "wall":
        raise RuntimeError("anneal stopped on a wall cap")
    hit = fields["hit"] == "1"
    first = fields["first_calls"]
    return {
        "hit": int(hit),
        "evals": int(fields["charged"]),
        "evals_to_hit": "" if first == "-" else int(first),
        "best_energy": float(fields["best"]),
        "best_infinity": "",
        "best_ase": "",
        "wall_s": wall,
        "stop": "hit" if hit else "budget",
    }


def run_ase(
    system: str,
    arm: str,
    seed: int,
    budget: int,
    kind: str,
    reference: float,
    work: Path,
) -> dict[str, float | int | str]:
    scales = read_scales(system)
    n_atoms = int(scales["n"])
    length_scale = float(scales["length_scale"])
    energy_scale = float(scales["energy_scale"])
    if int(scales["budget"]) != SYSTEMS[system]["budget"]:
        raise RuntimeError("evaluator budget does not match the paper budget")
    if abs(float(scales["reference"]) - SYSTEMS[system]["reference"]) > 1.0e-9:
        raise RuntimeError("evaluator reference does not match the paper target")
    if abs(float(reference) - SYSTEMS[system]["reference"]) > 1.0e-9:
        raise RuntimeError("requested reference does not match the paper target")
    record = record_gradient(energy_scale, length_scale)
    fmax = fmax_from_infinity()
    work.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    server = EvalServer(system, n_atoms)
    os.chdir(work)
    started = time.perf_counter()
    try:
        calc = BudgetCalculator(server, budget, reference, record)
        symbol = "Au" if kind == "gupta" else "Ar"
        atoms = Atoms(f"{symbol}{n_atoms}", positions=place(n_atoms, seed, length_scale), pbc=False)
        atoms.set_masses(np.ones(n_atoms))
        atoms.calc = calc
        stop = "budget"
        try:
            if arm == "ase_bh":
                run_basin_hopping(atoms, seed, length_scale, energy_scale, fmax)
            elif arm == "ase_mh":
                run_minima_hopping(atoms, seed, length_scale, energy_scale, fmax)
            else:
                raise RuntimeError(f"unknown ASE arm {arm}")
        except HitReached:
            stop = "hit"
        except BudgetReached:
            stop = "budget"
        wall = time.perf_counter() - started
        quenched = calc.best_quenched
        best = quenched if np.isfinite(quenched) else float("inf")
        hit = calc.hit_calls is not None
        return {
            "hit": int(hit),
            "evals": calc.calls,
            "evals_to_hit": "" if calc.hit_calls is None else calc.hit_calls,
            "best_energy": best,
            "best_infinity": "" if not np.isfinite(quenched) else calc.best_infinity,
            "best_ase": "" if not np.isfinite(quenched) else calc.best_ase,
            "wall_s": wall,
            "stop": "hit" if hit else stop,
        }
    finally:
        os.chdir(previous)
        server.close()

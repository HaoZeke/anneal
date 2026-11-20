# Copyright (c) anneal developers
"""Finite-difference sign checks for the Python force adapters.

The step and the bound match ``tests/rgpot_force_fd``. An analytic energy
uses one unit in the last place. A missing module or library is named and
skipped. A variable that names a missing file fails the check.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
GEOMETRY = np.array(
    [0.20, 0.15, 0.05, 1.12, 0.28, -0.10, -0.18, 1.02, 0.22],
    dtype=np.float64,
)


def ulp(value: float) -> float:
    magnitude = abs(value)
    if not math.isfinite(magnitude) or magnitude == 0.0:
        return sys.float_info.min
    return abs(math.nextafter(magnitude, math.inf) - magnitude)


def summed_precision(terms: int, absolute_sum: float, returned: float) -> float:
    """Forward error of a sum, never finer than one ulp of the result."""
    place = ulp(returned)
    count = max(terms, 1)
    machine = sys.float_info.epsilon
    if count * machine >= 1.0 or not math.isfinite(absolute_sum):
        return place
    forward = count * machine / (1.0 - count * machine) * abs(absolute_sum)
    if not math.isfinite(forward):
        return place
    return max(forward, place)


def model_step(precision: float, length: float, slope: float) -> float:
    return (3.0 * precision * length * length / abs(slope)) ** (1.0 / 3.0)


def shortest_pair(coords: np.ndarray) -> float:
    points = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    shortest = math.inf
    for left in range(len(points)):
        for right in range(left + 1, len(points)):
            delta = points[left] - points[right]
            shortest = min(shortest, float(np.dot(delta, delta) ** 0.5))
    if not math.isfinite(shortest) or shortest <= 0.0:
        raise AssertionError("the geometry has no positive pair distance")
    return shortest


def compare(
    name: str,
    geometry: np.ndarray,
    slope: np.ndarray,
    energy: float,
    energy_at,
    precision: float | None = None,
):
    if precision is None:
        precision = ulp(energy)
    length = shortest_pair(geometry)
    slope = np.asarray(slope, dtype=np.float64).reshape(-1)
    assert slope.shape == geometry.shape
    largest = int(np.argmax(np.abs(slope)))
    resolved = 0
    for component, component_slope in enumerate(slope):
        magnitude = abs(float(component_slope))
        step = model_step(precision, length, magnitude)
        if (
            not math.isfinite(step)
            or step <= ulp(float(geometry[component]))
            or step >= length
        ):
            if component == largest:
                raise AssertionError(
                    f"{name} component {component} step {step} does not fit inside {length}"
                )
            continue
        plus = geometry.copy()
        minus = geometry.copy()
        plus[component] += step
        minus[component] -= step
        coarse = (energy_at(plus) - energy_at(minus)) / (2.0 * step)
        plus[component] = geometry[component] + 0.5 * step
        minus[component] = geometry[component] - 0.5 * step
        # The finer step is h/2, so its denominator is h.
        fine = (energy_at(plus) - energy_at(minus)) / step
        bound = abs(coarse - fine) / 3.0 + 2.0 * precision / step
        miss = abs(float(component_slope) - fine)
        if not bound < magnitude:
            if component == largest:
                raise AssertionError(
                    f"{name} largest component {component} is not resolved "
                    f"(slope {component_slope}, bound {bound})"
                )
            continue
        if miss > bound:
            raise AssertionError(
                f"{name} component {component} slope {component_slope} "
                f"fine {fine} miss {miss} bound {bound}"
            )
        resolved += 1
    if resolved < 3:
        raise AssertionError(f"{name} resolved {resolved} components")
    print(
        f"{name} energy {energy:.12} precision {precision:.6e} "
        f"resolved {resolved}/{slope.size}"
    )


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ase_callbacks_match_the_energy():
    try:
        import anneal  # noqa: F401
    except Exception:
        stub = type(sys)("anneal")
        stub.cluster_search = None
        stub.polish = None
        sys.modules["anneal"] = stub
    ase_adapter = load_module("anneal_ase_local", ROOT / "python" / "anneal" / "ase.py")

    class Triangle:
        def __init__(self):
            self.positions = GEOMETRY.reshape(-1, 3).copy()
            self.calc = object()

        def __len__(self):
            return 3

        def get_positions(self):
            return self.positions.copy()

        def set_positions(self, positions):
            self.positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)

        def get_potential_energy(self):
            return float(spring_energy(self.positions))

        def get_forces(self):
            return -spring_gradient(self.positions)

    atoms = Triangle()
    energy_fn, grad_fn = ase_adapter._callbacks(atoms)
    geometry = GEOMETRY.copy()
    energy = float(energy_fn(geometry))
    slope = np.asarray(grad_fn(geometry), dtype=np.float64)
    compare(
        "ase-callbacks",
        geometry,
        slope,
        energy,
        energy_fn,
        precision=spring_precision(atoms.positions, energy),
    )


def spring_energy(positions: np.ndarray) -> float:
    points = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    energy = 0.0
    for left in range(len(points)):
        for right in range(left + 1, len(points)):
            distance = float(np.linalg.norm(points[left] - points[right]))
            gap = distance - 1.2
            energy += gap * gap
    return energy


def spring_precision(positions: np.ndarray, energy: float) -> float:
    points = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    terms = 0
    absolute = 0.0
    for left in range(len(points)):
        for right in range(left + 1, len(points)):
            distance = float(np.linalg.norm(points[left] - points[right]))
            gap = distance - 1.2
            terms += 1
            absolute += abs(gap * gap)
    return summed_precision(terms, absolute, energy)


def spring_gradient(positions: np.ndarray) -> np.ndarray:
    points = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    gradient = np.zeros_like(points)
    for left in range(len(points)):
        for right in range(left + 1, len(points)):
            delta = points[left] - points[right]
            distance = float(np.linalg.norm(delta))
            if distance == 0.0:
                continue
            scale = 2.0 * (distance - 1.2) / distance
            gradient[left] += scale * delta
            gradient[right] -= scale * delta
    return gradient


def library_file(variable: str, file_name: str) -> Path | None:
    raw = os.environ.get(variable, "")
    if raw:
        path = Path(raw)
        if path.is_file():
            return path
        local = Path(file_name)
        if local.is_file():
            return local
        pytest.fail(f"{variable} names {path} which is not a file, and {file_name} is absent")
    local = Path(file_name)
    if local.is_file():
        return local
    return None


def test_gupta_calculator_forces_match_the_energy():
    pytest.importorskip("ase")
    from ase import Atoms

    calculator_module = load_module(
        "gupta_calc_local", ROOT / "scripts" / "metal_h2h" / "gupta_calc.py"
    )
    numbers = [30, 30, 30]
    atoms = Atoms(numbers=numbers, positions=GEOMETRY.reshape(-1, 3))
    atoms.calc = calculator_module.GuptaCalculator()

    def energy_at(flat):
        atoms.set_positions(np.asarray(flat, dtype=np.float64).reshape(-1, 3))
        return float(atoms.get_potential_energy())

    energy = energy_at(GEOMETRY)
    forces = np.asarray(atoms.get_forces(), dtype=np.float64).reshape(-1)
    compare("gupta-ase", GEOMETRY.copy(), -forces, energy, energy_at)


def test_gofee_calculator_forces_match_the_energy():
    pytest.importorskip("ase")
    pytest.importorskip("numba")
    from ase import Atoms

    potential = load_module(
        "gofee_potential_local", ROOT / "scripts" / "ase_gofee_h2h" / "potential.py"
    )

    class Server:
        def eval(self, native):
            point = np.asarray(native, dtype=np.float64)
            energy = float(np.dot(point, point))
            return energy, 0.0, 2.0 * point

    scale = 1.7
    calculator = potential.AnnealCalculator(Server(), 500, -1.0e6, 1.0e-3, 1.0, scale)
    atoms = Atoms(numbers=[18, 18, 18], positions=GEOMETRY.reshape(-1, 3) * scale)
    atoms.calc = calculator

    def energy_at(flat):
        atoms.set_positions(np.asarray(flat, dtype=np.float64).reshape(-1, 3))
        calculator.calculate(atoms)
        return float(calculator.results["energy"])

    positions = np.asarray(atoms.get_positions(), dtype=np.float64).reshape(-1)
    energy = energy_at(positions)
    forces = np.asarray(calculator.results["forces"], dtype=np.float64).reshape(-1)
    compare("gofee-ase", positions, -forces, energy, energy_at)


def test_covalent_ase_forces_match_the_energy():
    pytest.importorskip("ase")
    path = library_file("RGPOT_COVALENT_LIB", "librgpot_covalent.so")
    if path is None:
        fallback = Path("/work/covalent-h2h/lib/librgpot_covalent.so")
        if fallback.is_file():
            path = fallback
    if path is None:
        pytest.skip(
            "covalent-ase: skipped, librgpot_covalent.so is absent "
            "(RGPOT_COVALENT_LIB or ./librgpot_covalent.so)"
        )
    from ase import Atoms

    adapter = load_module("covalent_ase_local", ROOT / "scripts" / "covalent_h2h" / "ase_bh.py")
    covalent = adapter.Covalent(path, "sw")
    calculator = adapter.BudgetCalculator(covalent, 100, 0.0)
    atoms = Atoms(numbers=[14, 14, 14], positions=GEOMETRY.reshape(-1, 3))
    atoms.calc = calculator

    def energy_at(flat):
        atoms.set_positions(np.asarray(flat, dtype=np.float64).reshape(-1, 3))
        calculator.calculate(atoms)
        return float(calculator.results["energy"])

    energy = energy_at(GEOMETRY)
    forces = np.asarray(calculator.results["forces"], dtype=np.float64).reshape(-1)
    compare("covalent-ase", GEOMETRY.copy(), -forces, energy, energy_at)


def test_cluster_search_orients_a_force_callback():
    """The public search must descend when ``grad_fn`` returns forces.

    ``cluster_search`` does not return the oriented gradient, so this check
    is the descent of that entry. A second negation leaves the start as the
    best energy. The component-wise bound is the ASE callback test.
    """
    try:
        import anneal
        from anneal import cluster_search
    except Exception as error:
        pytest.skip(f"cluster-search: skipped, anneal._core is not importable ({error})")
    here = (ROOT / "python" / "anneal").resolve()
    if Path(anneal.__file__).resolve().parent != here:
        pytest.skip(
            f"cluster-search: skipped, imported anneal is {anneal.__file__}, not {here}"
        )

    def energy(point):
        values = np.asarray(point, dtype=np.float64)
        return float(np.dot(values, values))

    def force(point):
        return -2.0 * np.asarray(point, dtype=np.float64)

    start = GEOMETRY.copy()
    start_energy = energy(start)
    found = cluster_search(
        energy,
        force,
        3,
        budget=40,
        seed=0,
        start=start,
        recommended=False,
    )
    if not found["best_energy"] < start_energy:
        raise AssertionError(
            f"cluster-search best {found['best_energy']} did not descend from {start_energy}"
        )

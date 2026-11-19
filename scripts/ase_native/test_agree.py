# Copyright (c) anneal developers
"""Agreement between the anneal kernel and the eV/angstrom formulas."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np

from eval_client import EvalServer, read_scales
from mapping import SYSTEMS, morse_length_scale
from potentials import (
    GOLD,
    gradient_relative_error,
    gupta_gold,
    lennard_jones,
    morse,
    relative_error,
)

REL = 1.0e-10


def _bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def _check(energy_a: float, grad_a: np.ndarray, energy_b: float, grad_b: np.ndarray, label: str) -> None:
    energy_error = relative_error(energy_a, energy_b)
    force_error = gradient_relative_error(grad_a, grad_b)
    if energy_error > REL or (isinstance(force_error, float) and force_error > REL and force_error > 1.0e-10):
        raise AssertionError(f"{label}: energy {energy_error:.3e} force {force_error:.3e}")
    if force_error > REL:
        raise AssertionError(f"{label}: force {force_error:.3e}")


def _geometries() -> list[tuple[str, np.ndarray]]:
    dimer_lj = np.zeros((2, 3))
    dimer_lj[1, 0] = 2.0 ** (1.0 / 6.0)
    shifted_lj = dimer_lj.copy()
    shifted_lj[1, 0] = 1.4
    triangle = np.array(
        [[0.0, 0.0, 0.0], [1.3, 0.1, 0.0], [0.4, 1.2, 0.2], [1.1, 1.0, 0.9]],
        dtype=np.float64,
    )
    dimer_morse = np.zeros((2, 3))
    dimer_morse[1, 0] = 1.0
    stretched = dimer_morse.copy()
    stretched[1, 0] = 1.3
    dimer_au = np.zeros((2, 3))
    dimer_au[1, 0] = GOLD["r0"]
    gold_triangle = np.array(
        [[0.0, 0.0, 0.0], [2.7, 0.1, 0.0], [1.2, 2.5, 0.2]],
        dtype=np.float64,
    )
    return [
        ("lj-dimer", dimer_lj),
        ("lj-shifted", shifted_lj),
        ("lj-cluster", triangle),
        ("morse-dimer", dimer_morse),
        ("morse-stretched", stretched),
        ("au-dimer", dimer_au),
        ("au-triangle", gold_triangle),
    ]


def _python(system: str, positions: np.ndarray):
    kind = SYSTEMS[system]["kind"]
    if kind == "lj":
        return lennard_jones(positions, epsilon=1.0, sigma=1.0)
    if kind == "morse":
        return morse(positions, rho=SYSTEMS[system]["rho"], epsilon=1.0, r0=1.0)
    return gupta_gold(positions)


def check_scales() -> None:
    for system, spec in SYSTEMS.items():
        scales = read_scales(system)
        assert int(scales["n"]) == spec["n"]
        assert int(scales["budget"]) == spec["budget"]
        reference = struct.unpack("<d", struct.pack("<Q", int(scales["reference_bits"])))[0]
        assert reference == spec["reference"]
        assert float(scales["epsilon"]) == 1.0
        if spec["kind"] == "lj":
            assert float(scales["length_scale"]) == 1.0
            assert float(scales["energy_scale"]) == 1.0
            assert scales["kind"] == "lj"
        elif spec["kind"] == "morse":
            assert abs(float(scales["length_scale"]) - morse_length_scale()) < 1.0e-15
            assert float(scales["energy_scale"]) == 1.0
            assert abs(float(scales["rho"]) - spec["rho"]) < 1.0e-15
        else:
            assert scales["kind"] == "gupta"
            assert float(scales["length_scale"]) > 1.0
            assert float(scales["energy_scale"]) > 0.0


def check_agreement() -> None:
    check_scales()
    samples = {
        "lj38": [name for name, _ in _geometries() if name.startswith("lj")],
        "m38r6": [name for name, _ in _geometries() if name.startswith("morse")],
        "m55r14": [name for name, _ in _geometries() if name.startswith("morse")],
        "au55": [name for name, _ in _geometries() if name.startswith("au")],
    }
    catalog = dict(_geometries())
    for system, names in samples.items():
        server = EvalServer(system, SYSTEMS[system]["n"])
        try:
            for name in names:
                positions = catalog[name]
                # Pad to the system size with a distant copy of the fragment
                # so the extra atoms do not change the fragment forces much
                # is unnecessary: evaluate the fragment by embedding it in a
                # system-sized cluster whose extra atoms sit far away, and
                # compare the whole cluster. Both evaluators see that cluster.
                full = _embed(positions, SYSTEMS[system]["n"], scale=8.0 if system != "au55" else 20.0)
                kernel_energy, kernel_gradient = server.evaluate(full)
                formula_energy, formula_gradient = _python(system, full)
                _check(kernel_energy, kernel_gradient, formula_energy, formula_gradient, f"{system}:{name}")
        finally:
            server.close()


def _embed(fragment: np.ndarray, n_atoms: int, scale: float) -> np.ndarray:
    full = np.zeros((n_atoms, 3), dtype=np.float64)
    take = min(len(fragment), n_atoms)
    full[:take] = fragment[:take]
    for index in range(take, n_atoms):
        full[index] = (index * 1.7, (index % 5) * 1.3, scale + index)
    return full


def check_budget_stops() -> None:
    """A short search must stop on the evaluation cap, not on a wall clock."""
    import tempfile

    from search import run_anneal, run_ase

    with tempfile.TemporaryDirectory() as tmp:
        ase = run_ase("lj38", "ase_bh", 1999, 40, "lj", SYSTEMS["lj38"]["reference"], Path(tmp) / "bh")
        hopping = run_ase("lj38", "ase_mh", 1999, 80, "lj", SYSTEMS["lj38"]["reference"], Path(tmp) / "mh")
    assert ase["evals"] <= 40
    assert hopping["evals"] <= 80
    assert ase["stop"] in ("budget", "hit")
    assert hopping["stop"] in ("budget", "hit")
    anneal = run_anneal("lj38", 1999, 250, "lj")
    assert anneal["stop"] in ("budget", "hit")
    assert int(anneal["evals"]) <= 250


def check_compare_matches_ase() -> None:
    """The fast identity test must return ASE's distance, not a new metric."""
    from ase import Atoms
    from ase.optimize.minimahopping import ComparePositions

    from controls import greedy_dmax, install_fast_compare

    rng = np.random.default_rng(7)
    original = ComparePositions._indistinguishable_compare
    cases = []

    def add_pair(pos1: np.ndarray, pos2: np.ndarray) -> None:
        n_atoms = len(pos1)
        left = Atoms(f"Ar{n_atoms}", positions=pos1, pbc=False)
        right = Atoms(f"Ar{n_atoms}", positions=pos2, pbc=False)
        slow = original(ComparePositions(translate=False), left, right)
        fast = greedy_dmax(pos1, pos2)
        if slow != fast:
            raise AssertionError(f"greedy dmax {fast} != {slow}")
        whole = ComparePositions(translate=True)(left, right)
        cases.append((left, right, whole))

    for n_atoms in (4, 9, 15):
        for _ in range(4):
            pos1 = rng.normal(size=(n_atoms, 3))
            add_pair(pos1, pos1 + rng.normal(scale=0.05, size=pos1.shape))
            add_pair(pos1, rng.normal(size=(n_atoms, 3)))
            order = rng.permutation(n_atoms)
            add_pair(pos1, pos1[order] + np.array([1.25, -0.4, 0.7]))
    install_fast_compare()
    for left, right, whole in cases:
        sped = ComparePositions(translate=True)(left, right)
        if sped != whole:
            raise AssertionError(f"translated dmax {sped} != {whole}")
    from eval_client import DmaxServer

    # One persistent process, several sizes. The scalar walk has to return
    # ASE's distance. With a threshold, the value may be a bound, and the
    # comparison with that threshold has to match ASE outside the band.
    by_size: dict[int, list[tuple[np.ndarray, np.ndarray, float]]] = {}
    for left, right, whole in cases:
        by_size.setdefault(len(left), []).append((left.positions.copy(), right.positions.copy(), float(whole)))
    thresholds = (0.05, 0.2, 0.7, 1.4, 3.0)
    for n_atoms, samples in by_size.items():
        server = DmaxServer(n_atoms)
        decided = {threshold: DmaxServer(n_atoms, threshold=threshold) for threshold in thresholds}
        try:
            for pos1, pos2, whole in samples:
                distance = server.distance(pos1, pos2)
                if abs(distance - whole) > 1.0e-12:
                    raise AssertionError(f"scalar dmax {distance} != {whole}")
                for threshold, checker in decided.items():
                    got = checker.distance(pos1, pos2)
                    if abs(whole - threshold) <= 1.0e-9:
                        if abs(got - whole) > 1.0e-12:
                            raise AssertionError(f"band dmax {got} != {whole}")
                    elif (got < threshold) != (whole < threshold):
                        raise AssertionError(f"threshold decision changed at {threshold}")
        finally:
            server.close()
            for checker in decided.values():
                checker.close()


def main() -> int:
    check_compare_matches_ase()
    check_agreement()
    check_budget_stops()
    print("agreement tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

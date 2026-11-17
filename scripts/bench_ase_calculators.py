#!/usr/bin/env python3
"""Time RgpotCalculator against ASE's own calculators for the same model.

Pairs: rgpot LJ with ``ase.calculators.lj.LennardJones`` (argon), rgpot
Morse with ``ase.calculators.morse.MorsePotential`` (copper), and rgpot
EAM-Al with ``ase.calculators.eam.EAM`` driven by the closed-form EAM-Al
functions of ``tests/python/eam_al_reference.py`` (aluminium). Each pair
describes the same energy surface; the script checks that before timing.

One timed call is ``atoms.get_forces()`` on positions displaced from the
fcc lattice by fresh Gaussian noise, so no call sees coordinates an
earlier call saw and no calculator answers from its cache. Each round
times both calculators once, in a random order. The table reports the
median over the rounds.

rgpot runs single-threaded unless OMP_NUM_THREADS is set.

    python scripts/bench_ase_calculators.py --rounds 21 --sizes 64 1024 8192
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.calculators.morse import MorsePotential

import rgpot
from rgpot.ase import RgpotCalculator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "python"))
from eam_al_reference import ase_eam_al  # noqa: E402

REPEATS = {64: (2, 2, 4), 1024: (4, 8, 8), 8192: (8, 16, 16)}
AR = dict(u0=0.0104, psi=3.40, cutoff=8.5)
CU = dict(De=0.3429, a=1.3588, re=2.866)
RCUT1, RCUT2 = 1.9, 2.7


def pairs():
    yield (
        "LJ (Ar)",
        ("Ar", 5.26),
        lambda: RgpotCalculator("lj", **AR),
        lambda: LennardJones(epsilon=AR["u0"], sigma=AR["psi"], rc=AR["cutoff"]),
        "LennardJones",
    )
    yield (
        "Morse (Cu)",
        ("Cu", 3.6),
        lambda: RgpotCalculator(
            "morse",
            **CU,
            cutoff=RCUT2 * CU["re"],
            switch_width=(RCUT2 - RCUT1) * CU["re"],
        ),
        lambda: MorsePotential(
            epsilon=CU["De"], rho0=CU["a"] * CU["re"], r0=CU["re"], rcut1=RCUT1, rcut2=RCUT2
        ),
        "MorsePotential",
    )
    if rgpot.has_fortran_pots:
        yield (
            "EAM-Al",
            ("Al", 4.05),
            lambda: RgpotCalculator("eam_al"),
            ase_eam_al,
            "EAM (analytic)",
        )


def crystal(symbol: str, a: float, n_atoms: int):
    atoms = bulk(symbol, "fcc", a=a, cubic=True).repeat(REPEATS[n_atoms])
    assert len(atoms) == n_atoms
    return atoms


def check_agreement(base, make_ours, make_theirs, rng) -> float:
    """Largest relative energy or force difference at one displaced geometry."""
    positions = base.positions + rng.normal(scale=0.02, size=base.positions.shape)
    ours, theirs = base.copy(), base.copy()
    ours.positions = positions
    theirs.positions = positions.copy()
    ours.calc = make_ours()
    theirs.calc = make_theirs()
    e_ours, e_theirs = ours.get_potential_energy(), theirs.get_potential_energy()
    f_ours, f_theirs = ours.get_forces(), theirs.get_forces()
    de = abs(e_ours - e_theirs) / abs(e_theirs)
    df = float(np.max(np.abs(f_ours - f_theirs)) / np.max(np.abs(f_theirs)))
    return max(de, df)


def time_pair(base, make_ours, make_theirs, rounds: int, noise: float, rng):
    runners = {"rgpot": base.copy(), "ase": base.copy()}
    runners["rgpot"].calc = make_ours()
    runners["ase"].calc = make_theirs()
    times: dict[str, list[float]] = {"rgpot": [], "ase": []}
    # One untimed call each builds neighbour lists and workspaces.
    for atoms in runners.values():
        atoms.positions = base.positions + rng.normal(scale=noise, size=base.positions.shape)
        atoms.get_forces()
    for _ in range(rounds):
        order = list(runners)
        rng.shuffle(order)
        for name in order:
            atoms = runners[name]
            atoms.positions = base.positions + rng.normal(scale=noise, size=base.positions.shape)
            start = time.perf_counter()
            atoms.get_forces()
            times[name].append(time.perf_counter() - start)
    return times


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=21)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 1024, 8192])
    parser.add_argument("--noise", type=float, default=0.02, help="displacement sigma, Angstrom")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"# python {platform.python_version()}, numpy {np.__version__}, rgpot {rgpot.__version__}")
    print(f"# {platform.processor() or platform.machine()}, {os.cpu_count()} CPUs, "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}")
    print(f"# median of {args.rounds} interleaved rounds, fresh displacements every call")
    header = (
        f"{'atoms':>6} | {'potential':<11} | {'reference':<15} | {'rgpot ms':>10} | "
        f"{'ASE ms':>10} | {'ASE/rgpot':>9} | {'max rel diff':>12}"
    )
    print(header)
    print("-" * len(header))
    for n_atoms in args.sizes:
        for label, (symbol, a), make_ours, make_theirs, ref in pairs():
            base = crystal(symbol, a, n_atoms)
            diff = check_agreement(base, make_ours, make_theirs, rng)
            times = time_pair(base, make_ours, make_theirs, args.rounds, args.noise, rng)
            ours = statistics.median(times["rgpot"]) * 1e3
            theirs = statistics.median(times["ase"]) * 1e3
            print(
                f"{n_atoms:>6} | {label:<11} | {ref:<15} | {ours:>10.3f} | "
                f"{theirs:>10.3f} | {theirs / ours:>9.1f} | {diff:>12.2e}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

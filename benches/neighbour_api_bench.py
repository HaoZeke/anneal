#!/usr/bin/env python3
"""Time rgpot's neighbour query against vesin, ASE and matscipy.

FCC copper uses a = 3.615 Angstrom and a strict cutoff of 5 Angstrom.
30 cubed cells are 108000 atoms and 64 cubed cells are 1048576 atoms.
The triclinic system is a 4x4x4 grid in a sheared cell of 64 atoms.
Skin is zero. Each library is a fresh process. The reported time is the
median of seven calls after two discarded calls, and the resident set is
VmHWM of that process.

rgpot is the C++ query in neighbour_api_rgpot.cc. vesin uses
NeighborList.compute with the same cutoff and an explicit thread count.
ASE's neighbor_list and matscipy's neighbour_list return a full list and
have no half-list switch and no thread argument. ASE's NeighborList class
with bothways=False is the half list, timed as a fresh object per call
because an unchanged geometry would skip the build.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

CUTOFF = 5.0
LATTICE = 3.615
REPEATS = 7
ROOT = Path(__file__).resolve().parents[1]
BINARY = Path("/tmp/rgpot-prof/neighbour_api_rgpot")


def fcc(n: int) -> tuple[np.ndarray, np.ndarray]:
    basis = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    idx = np.stack(np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij"), axis=-1)
    idx = idx.reshape(-1, 3)
    pos = ((idx[:, None, :] + basis[None, :, :]) * LATTICE).reshape(-1, 3)
    length = n * LATTICE
    cell = np.diag([length, length, length])
    return np.ascontiguousarray(pos, dtype=np.float64), cell


def triclinic() -> tuple[np.ndarray, np.ndarray]:
    cell = np.array([[6.0, 0.0, 0.0], [1.2, 5.4, 0.0], [0.3, 0.7, 5.8]], dtype=np.float64)
    frac = np.array(
        [[(a + 0.2) / 4.0, (b + 0.2) / 4.0, (c + 0.2) / 4.0] for a in range(4) for b in range(4) for c in range(4)],
        dtype=np.float64,
    )
    return np.ascontiguousarray(frac @ cell), cell


def geometry(system: str) -> tuple[np.ndarray, np.ndarray]:
    if system == "tri":
        return triclinic()
    if system == "cu1m":
        return fcc(64)
    return fcc(30)


def peak_rss_kb() -> int:
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    return 0


def median(samples: list[float]) -> float:
    ordered = sorted(samples)
    return ordered[len(ordered) // 2]


def time_calls(fn) -> tuple[float, int, int]:
    for _ in range(2):
        out = fn()
    samples = []
    pairs = 0
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        out = fn()
        samples.append(time.perf_counter() - t0)
        pairs = int(np.shape(out[0])[0]) if isinstance(out, tuple) else int(out)
    return median(samples), pairs, peak_rss_kb()


def run_vesin(system: str, fmt: str, threads: int) -> None:
    import vesin

    pos, cell = geometry(system)
    calc = vesin.NeighborList(
        cutoff=CUTOFF, full_list=(fmt == "full"), skin=0.0, n_threads=threads
    )

    def once():
        return calc.compute(pos, cell, True, quantities="ijSDd", copy=False)

    seconds, pairs, rss = time_calls(once)
    print(
        f"vesin {system} {fmt} T={threads} n={len(pos)} pairs={pairs} "
        f"median {seconds:.6f} rss_kb {rss}",
        flush=True,
    )


def run_ase(system: str, fmt: str, threads: int) -> None:
    from ase import Atoms
    from ase.neighborlist import NeighborList, neighbor_list

    os.environ["OMP_NUM_THREADS"] = str(threads)
    pos, cell = geometry(system)
    atoms = Atoms(positions=pos, cell=cell, pbc=True)
    if fmt == "full":

        def once():
            return neighbor_list("ijSD", atoms, CUTOFF)

        seconds, pairs, rss = time_calls(once)
    else:

        def once():
            nlist = NeighborList(
                [CUTOFF / 2.0] * len(atoms),
                skin=0.0,
                sorted=False,
                self_interaction=False,
                bothways=False,
            )
            nlist.update(atoms)
            return sum(indices.size for indices in nlist.nl.neighbors)

        seconds, pairs, rss = time_calls(once)
    print(
        f"ase {system} {fmt} T={threads} n={len(pos)} pairs={pairs} "
        f"median {seconds:.6f} rss_kb {rss}",
        flush=True,
    )


def run_matscipy(system: str, fmt: str, threads: int) -> None:
    if fmt != "full":
        print(f"matscipy {system} {fmt} T={threads} unsupported", flush=True)
        return
    from ase import Atoms
    from matscipy.neighbours import neighbour_list

    os.environ["OMP_NUM_THREADS"] = str(threads)
    pos, cell = geometry(system)
    atoms = Atoms(positions=pos, cell=cell, pbc=True)

    def once():
        return neighbour_list("ijSD", atoms, CUTOFF)

    seconds, pairs, rss = time_calls(once)
    print(
        f"matscipy {system} {fmt} T={threads} n={len(pos)} pairs={pairs} "
        f"median {seconds:.6f} rss_kb {rss}",
        flush=True,
    )


def compile_rgpot() -> None:
    BINARY.parent.mkdir(parents=True, exist_ok=True)
    source = ROOT / "benches" / "neighbour_api_rgpot.cc"
    cmd = [
        "g++",
        "-std=c++20",
        "-O3",
        "-DNDEBUG",
        "-fopenmp",
        "-I",
        str(ROOT / "CppCore"),
        str(source),
        str(ROOT / "CppCore/rgpot/Slices.cc"),
        str(ROOT / "CppCore/rgpot/nlist/NeighbourList.cc"),
        str(ROOT / "CppCore/rgpot/nlist/Neighbours.cc"),
        str(ROOT / "CppCore/rgpot/nlist/MinimumImage.cpp"),
        str(ROOT / "CppCore/rgpot/simd/exp.cc"),
        str(ROOT / "CppCore/rgpot/simd/exact_fma_hw.cc"),
        "-o",
        str(BINARY),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    if len(sys.argv) == 5 and sys.argv[1] in {"vesin", "ase", "matscipy"}:
        lib, system, fmt, threads = sys.argv[1:]
        {"vesin": run_vesin, "ase": run_ase, "matscipy": run_matscipy}[lib](
            system, fmt, int(threads)
        )
        return 0
    compile_rgpot()
    systems = ["tri", "cu108", "cu1m"]
    formats = ["half", "full"]
    threads = [1, 8]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    for system in systems:
        for fmt in formats:
            for nthreads in threads:
                print(
                    subprocess.run(
                        [str(BINARY), system, fmt, str(nthreads), str(REPEATS)],
                        check=True,
                        env={**env, "OMP_NUM_THREADS": str(nthreads)},
                        stdout=subprocess.PIPE,
                        text=True,
                    ).stdout,
                    end="",
                    flush=True,
                )
                for lib in ("vesin", "ase", "matscipy"):
                    subprocess.run(
                        [sys.executable, str(Path(__file__).resolve()), lib, system, fmt, str(nthreads)],
                        check=False,
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

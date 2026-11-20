#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Classify the final structure of every G3 miss.

Usage: misses.py ROWS.csv MISSES_DIR OUT_DIR

Reads ``<system>-<method>-<seed>.con`` for every row that is not a hit and
applies ``bench/g3-funnel/misses-rules.md``: point group from pymatgen's
``PointGroupAnalyzer``, CNA class from ``classify.py``, structure identity by
sorted interatomic distances. Writes OUT_DIR/misses.csv (one row per miss)
and OUT_DIR/traps.csv (one row per system and miss class, with the dominant
trap marked).
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TOLERANCE = 0.05
EIGEN_TOLERANCE = 0.01
MATRIX_TOLERANCE = 0.1
SAME_STRUCTURE = 0.01
METHODS = ("anneal", "ase_mh", "gmin_1", "gmin_csm")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


classify = load("g3_classify", HERE / "classify.py")


def point_group(xyz: np.ndarray) -> str:
    from pymatgen.core import Molecule
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        molecule = Molecule(["Ar"] * len(xyz), xyz - xyz.mean(axis=0))
        return PointGroupAnalyzer(
            molecule,
            tolerance=TOLERANCE,
            eigen_tolerance=EIGEN_TOLERANCE,
            matrix_tolerance=MATRIX_TOLERANCE,
        ).sch_symbol


def distances(xyz: np.ndarray) -> np.ndarray:
    diff = xyz[:, None, :] - xyz[None, :, :]
    upper = np.triu_indices(len(xyz), 1)
    return np.sort(np.sqrt((diff**2).sum(-1))[upper])


def main() -> int:
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    from ase.io import read

    rows_path, misses_dir, out_dir = map(Path, sys.argv[1:])
    with rows_path.open() as handle:
        rows = [row for row in csv.DictReader(handle) if row["hit"] != "1"]
    found = []
    for row in rows:
        stem = f"{row['system']}-{row['method']}-{row['seed']}"
        path = misses_dir / f"{stem}.con"
        atoms = read(str(path), format="eon")
        xyz = atoms.get_positions()
        cna = classify.Cna(classify.cutoff(row["system"]))
        f421, f422, f555, fother = cna.shares(xyz)
        found.append(
            {
                "system": row["system"],
                "method": row["method"],
                "seed": row["seed"],
                "status": row["status"],
                "best": row["best"],
                "con": path.name,
                "con_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "point_group": point_group(xyz),
                "cna_class": classify.funnel(f421, f422, f555),
                "f421": f"{f421:.4f}",
                "f422": f"{f422:.4f}",
                "f555": f"{f555:.4f}",
                "fother": f"{fother:.4f}",
                "energy_key": float(row["best"]) if row["best"] else float("inf"),
                "distances": distances(xyz),
            }
        )
    by_system = defaultdict(list)
    for miss in found:
        by_system[miss["system"]].append(miss)
    for misses in by_system.values():
        misses.sort(key=lambda m: m["energy_key"])
        representatives: list[np.ndarray] = []
        for miss in misses:
            for index, rep in enumerate(representatives):
                if rep.shape == miss["distances"].shape and np.max(np.abs(rep - miss["distances"])) <= SAME_STRUCTURE:
                    miss["structure"] = f"S{index + 1}"
                    break
            else:
                representatives.append(miss["distances"])
                miss["structure"] = f"S{len(representatives)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["system", "method", "seed", "status", "best", "point_group", "cna_class", "miss_class",
              "structure", "f421", "f422", "f555", "fother", "con", "con_sha256"]
    with (out_dir / "misses.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for miss in sorted(found, key=lambda m: (m["system"], m["method"], int(m["seed"]))):
            miss["miss_class"] = f"{miss['cna_class']} {miss['point_group']}"
            writer.writerow(miss)
    with (out_dir / "traps.csv").open("w") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["system", "miss_class", "misses"] + [f"misses_{m}" for m in METHODS]
                        + ["structures", "lowest_best", "highest_best", "dominant"])
        for system, misses in sorted(by_system.items()):
            classes = defaultdict(list)
            for miss in misses:
                classes[miss["miss_class"]].append(miss)
            ranked = sorted(classes.items(), key=lambda kv: (-len(kv[1]), min(m["energy_key"] for m in kv[1])))
            for rank, (name, members) in enumerate(ranked):
                count = Counter(m["method"] for m in members)
                energies = [m["energy_key"] for m in members]
                writer.writerow(
                    [system, name, len(members)] + [count.get(m, 0) for m in METHODS]
                    + [len({m["structure"] for m in members}), f"{min(energies):.8f}", f"{max(energies):.8f}",
                       int(rank == 0)]
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())

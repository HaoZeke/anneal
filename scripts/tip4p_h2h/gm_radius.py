#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Container radius for the GMIN TIP4P data files.

The radius is the largest oxygen distance from the oxygen centroid of the
Wales-Hodges global minimum, plus 3 angstrom, the same rule at every size.
GMIN tests molecular centres of mass against the origin (takestep.f90), so
the centre-of-mass extent is printed beside it.

``--gmin DIR`` also writes each minimum as the NEWTIP centre and angle-axis
block, ``DIR/tip4p<n>.gm.gmin``, so GMIN can quench its own reference.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MINIMA = {
    4: REPO / "bench" / "tip4p-h2h" / "radius" / "TIP4P-4.xyz",
    8: REPO / "bench" / "tip4p-h2h" / "radius" / "TIP4P-8.xyz",
    15: REPO / "tests" / "fixtures" / "tip4p-15.xyz",
    20: REPO / "tests" / "fixtures" / "tip4p-20.xyz",
}
MARGIN = 3.0
R_OH = 0.9572
HOH_DEG = 104.52
MASSES = np.array([16.0, 1.0, 1.0])


def read_minimum(path: Path) -> tuple[float, np.ndarray]:
    lines = path.read_text().splitlines()
    energy = float(lines[1].split("=")[1].split()[0])
    sites = [[float(v) for v in line.split()[1:4]] for line in lines[2:] if line.strip()]
    return energy, np.array(sites).reshape(-1, 3, 3)


def body_sites() -> np.ndarray:
    half = math.radians(HOH_DEG / 2.0)
    sites = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, math.sin(half) * R_OH, math.cos(half) * R_OH],
            [0.0, -math.sin(half) * R_OH, math.cos(half) * R_OH],
        ]
    )
    return sites - MASSES @ sites / MASSES.sum()


def rotation_vector(rot: np.ndarray) -> np.ndarray:
    angle = math.acos(max(-1.0, min(1.0, (np.trace(rot) - 1.0) / 2.0)))
    if angle < 1e-12:
        return np.zeros(3)
    if math.pi - angle < 1e-6:
        w, v = np.linalg.eigh((rot + np.eye(3)) / 2.0)
        return v[:, np.argmax(w)] * angle
    axis = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])
    return axis / (2.0 * math.sin(angle)) * angle


def gmin_block(molecules: np.ndarray) -> tuple[str, float]:
    body = body_sites()
    centres, axes, worst = [], [], 0.0
    for mol in molecules:
        com = MASSES @ mol / MASSES.sum()
        u, _, vt = np.linalg.svd(body.T @ (mol - com))
        d = np.sign(np.linalg.det(vt.T @ u.T))
        rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
        worst = max(worst, float(np.abs(body @ rot.T + com - mol).max()))
        centres.append(com)
        # NEWTIP turns the opposite way to the Rodrigues map.
        axes.append(-rotation_vector(rot))
    text = "".join(f"{x:.10f} {y:.10f} {z:.10f}\n" for x, y, z in centres + axes)
    return text, worst


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gmin", type=Path)
    args = parser.parse_args()
    print("n,reference_kj,o_extent,com_extent,radius,lj_default")
    for n, path in MINIMA.items():
        energy, molecules = read_minimum(path)
        oxygens = molecules[:, 0, :]
        o_extent = float(np.linalg.norm(oxygens - oxygens.mean(axis=0), axis=1).max())
        coms = molecules.transpose(0, 2, 1) @ MASSES / MASSES.sum()
        com_extent = float(np.linalg.norm(coms - coms.mean(axis=0), axis=1).max())
        # io1.f90 with RADIUS unset; NATOMS counts centre and angle-axis rows.
        lj_default = (2.0 + (3.0 * 2 * n / 17.77153175) ** (1.0 / 3.0)) * 2.0 ** (1.0 / 6.0)
        radius = o_extent + MARGIN
        print(f"{n},{energy:.10f},{o_extent:.4f},{com_extent:.4f},{radius:.4f},{lj_default:.4f}")
        if args.gmin:
            args.gmin.mkdir(parents=True, exist_ok=True)
            centred = molecules - oxygens.mean(axis=0)
            text, worst = gmin_block(centred)
            if worst > 1e-4:
                raise SystemExit(f"n={n} rigid fit off by {worst:.2e} A")
            (args.gmin / f"tip4p{n}.gm.gmin").write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

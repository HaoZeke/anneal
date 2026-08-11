#!/usr/bin/env python3
"""Write the rgpot FCC Cu slab + H2 as an eOn .con.

Geometry is the CuH2 bench in rgpot CppCore/tests/PotBench.cc:
4 x 4 conventional cells in xy, 2 along z (128 Cu, four (100) planes),
H2 2.3 Angstrom above the top layer, vacuum 30 Angstrom. Cu is frozen;
H is free. The 2-Cu fixture in examples/fixtures/cuh2_tiny.con is the
CuH2PotTest pin, not a surface.
"""

from __future__ import annotations

import argparse
from pathlib import Path

LATTICE = 3.615
CELLS_XY = 4
LAYERS = 2
VACUUM = 30.0
H_HEIGHT = 2.3
H_BOND = 0.74
BASIS = (
    (0.0, 0.0, 0.0),
    (0.0, 0.5, 0.5),
    (0.5, 0.0, 0.5),
    (0.5, 0.5, 0.0),
)


def build() -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]], float]:
    cu: list[tuple[float, float, float]] = []
    for ix in range(CELLS_XY):
        for iy in range(CELLS_XY):
            for iz in range(LAYERS):
                for bx, by, bz in BASIS:
                    cu.append(
                        (
                            (ix + bx) * LATTICE,
                            (iy + by) * LATTICE,
                            (iz + bz) * LATTICE,
                        )
                    )
    side = LATTICE * CELLS_XY
    top = LATTICE * LAYERS
    zh = top + H_HEIGHT
    xc = 0.5 * side
    yc = 0.5 * side
    half = 0.5 * H_BOND
    h = [(xc - half, yc, zh), (xc + half, yc, zh)]
    return cu, h, side


def write_con(path: Path) -> None:
    cu, h, side = build()
    ncu, nh = len(cu), len(h)
    if ncu != 128 or nh != 2:
        raise SystemExit(f"expected 128 Cu + 2 H, got {ncu} Cu + {nh} H")
    lines = [
        "FCC Cu(100) 4x4x2 conventional cells, H2 above the top layer",
        "0.0000 TIME",
        f"{side:.10f}    {side:.10f}    {VACUUM:.10f}",
        "90.0000000000   90.0000000000   90.0000000000",
        "0 0",
        "0 0 0",
        "2",
        f"{ncu} {nh}",
        "63.546 1.008",
        "Cu",
        "Coordinates of Component 1",
    ]
    idx = 0
    for x, y, z in cu:
        lines.append(f"   {x:.16f}    {y:.16f}    {z:.16f} 1    {idx}")
        idx += 1
    lines.append("H")
    lines.append("Coordinates of Component 2")
    for x, y, z in h:
        lines.append(f"   {x:.16f}    {y:.16f}    {z:.16f} 0    {idx}")
        idx += 1
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "examples"
        / "fixtures"
        / "cuh2_fcc_slab.con",
    )
    args = p.parse_args()
    write_con(args.output)
    print(f"wrote {args.output} (128 Cu frozen, 2 H free)")


if __name__ == "__main__":
    main()

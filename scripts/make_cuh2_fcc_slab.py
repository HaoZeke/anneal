#!/usr/bin/env python3
"""Write an rgpot FCC Cu slab plus hydrogen as an eOn .con.

The default path is the CuH2 bench in rgpot CppCore/tests/PotBench.cc:
4 x 4 conventional cells in xy, 2 along z (128 Cu, four (100) planes),
H2 2.3 Angstrom above the top layer, vacuum 30 Angstrom. All Cu is frozen;
H is free. The 2-Cu fixture in examples/fixtures/cuh2_tiny.con is the
CuH2PotTest pin, not a surface.

``--surface 111`` (or ``--hydrogens`` other than 2) writes an orthogonal
Cu(111) slab of at least 4 x 4 primitive surface cells and 4 layers.
The bottom two layers are frozen. N hydrogen atoms (default 6) sit at
random in-plane positions above the top layer.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

LATTICE = 3.615
CELLS_XY = 4
LAYERS = 2
VACUUM = 30.0
H_HEIGHT = 2.3
H_BOND = 0.74
H_MIN_SEP = 1.5
BASIS = (
    (0.0, 0.0, 0.0),
    (0.0, 0.5, 0.5),
    (0.5, 0.0, 0.5),
    (0.5, 0.5, 0.0),
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_100 = ROOT / "examples" / "fixtures" / "cuh2_fcc_slab.con"
DEFAULT_111 = ROOT / "examples" / "fixtures" / "cuh2_fcc_slab_h6.con"


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


def _place_hydrogens(
    rng: random.Random,
    n_h: int,
    lx: float,
    ly: float,
    z: float,
) -> list[tuple[float, float, float]]:
    placed: list[tuple[float, float, float]] = []
    for _ in range(n_h):
        for _attempt in range(10_000):
            x = rng.random() * lx
            y = rng.random() * ly
            if all((x - px) ** 2 + (y - py) ** 2 >= H_MIN_SEP**2 for px, py, _ in placed):
                placed.append((x, y, z))
                break
        else:
            raise SystemExit(f"could not place {n_h} H with min separation {H_MIN_SEP}")
    return placed


def build_111(
    cells: int,
    layers: int,
    n_h: int,
    seed: int,
    freeze_bottom: int,
) -> tuple[
    list[tuple[float, float, float, int]],
    list[tuple[float, float, float]],
    float,
    float,
]:
    if cells < 4:
        raise SystemExit("Cu(111) needs at least 4 surface cells")
    if layers < 4:
        raise SystemExit("Cu(111) needs at least 4 layers")
    if freeze_bottom < 1 or freeze_bottom >= layers:
        raise SystemExit("freeze-bottom must be in [1, layers)")
    if n_h < 1:
        raise SystemExit("need at least one hydrogen")
    # Orthogonal supercell of the hexagonal (111) mesh: a1 and 2*a2-a1.
    # nx = cells repeats of a1, ny = cells/2 repeats of the orthogonal
    # vector, so the surface has cells*cells primitive sites.
    if cells % 2:
        raise SystemExit("orthogonal Cu(111) cells must be even")
    a_nn = LATTICE / math.sqrt(2.0)
    d_layer = LATTICE / math.sqrt(3.0)
    nx = cells
    ny = cells // 2
    lx = nx * a_nn
    ly = ny * a_nn * math.sqrt(3.0)
    shifts = (
        (0.0, 0.0),
        (a_nn / 2.0, a_nn * math.sqrt(3.0) / 6.0),
        (a_nn, a_nn * math.sqrt(3.0) / 3.0),
    )
    cu: list[tuple[float, float, float, int]] = []
    for iz in range(layers):
        sx, sy = shifts[iz % 3]
        frozen = 1 if iz < freeze_bottom else 0
        for ix in range(nx):
            for iy in range(ny):
                x0 = ix * a_nn + sx
                y0 = iy * a_nn * math.sqrt(3.0) + sy
                cu.append((x0 % lx, y0 % ly, iz * d_layer, frozen))
                cu.append(
                    (
                        (x0 + a_nn / 2.0) % lx,
                        (y0 + a_nn * math.sqrt(3.0) / 2.0) % ly,
                        iz * d_layer,
                        frozen,
                    )
                )
    n_per_layer = 2 * nx * ny
    if n_per_layer != cells * cells:
        raise SystemExit(f"expected {cells * cells} Cu per layer, got {n_per_layer}")
    if len(cu) != n_per_layer * layers:
        raise SystemExit(f"expected {n_per_layer * layers} Cu, got {len(cu)}")
    z_top = max(atom[2] for atom in cu)
    rng = random.Random(seed)
    h = _place_hydrogens(rng, n_h, lx, ly, z_top + H_HEIGHT)
    return cu, h, lx, ly


def _con_lines(
    comment: str,
    box: tuple[float, float, float],
    cu: list[tuple[float, float, float, int]],
    h: list[tuple[float, float, float]],
) -> list[str]:
    ncu, nh = len(cu), len(h)
    lines = [
        comment,
        "0.0000 TIME",
        f"{box[0]:.10f}    {box[1]:.10f}    {box[2]:.10f}",
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
    for x, y, z, frozen in cu:
        lines.append(f"   {x:.16f}    {y:.16f}    {z:.16f} {frozen}    {idx}")
        idx += 1
    lines.append("H")
    lines.append("Coordinates of Component 2")
    for x, y, z in h:
        lines.append(f"   {x:.16f}    {y:.16f}    {z:.16f} 0    {idx}")
        idx += 1
    return lines


def write_con(path: Path) -> None:
    cu, h, side = build()
    ncu, nh = len(cu), len(h)
    if ncu != 128 or nh != 2:
        raise SystemExit(f"expected 128 Cu + 2 H, got {ncu} Cu + {nh} H")
    lines = _con_lines(
        "FCC Cu(100) 4x4x2 conventional cells, H2 above the top layer",
        (side, side, VACUUM),
        [(x, y, z, 1) for x, y, z in cu],
        h,
    )
    path.write_text("\n".join(lines) + "\n")


def write_111(
    path: Path,
    cells: int,
    layers: int,
    n_h: int,
    seed: int,
    freeze_bottom: int,
) -> None:
    cu, h, lx, ly = build_111(cells, layers, n_h, seed, freeze_bottom)
    n_frozen = sum(1 for *_, frozen in cu if frozen)
    comment = (
        f"Cu(111) {cells}x{cells} primitive, {layers} layers, "
        f"bottom {freeze_bottom} frozen, {n_h} H random seed {seed}"
    )
    lines = _con_lines(comment, (lx, ly, VACUUM), cu, h)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    n_free_cu = len(cu) - n_frozen
    print(
        f"wrote {path} ({len(cu)} Cu, {n_frozen} frozen / {n_free_cu} free, {len(h)} H free)"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output .con (default: cuh2_fcc_slab.con, or cuh2_fcc_slab_h6.con for 111)",
    )
    p.add_argument(
        "--surface",
        choices=("100", "111"),
        default=None,
        help="surface orientation (default 100; 111 when --hydrogens is set)",
    )
    p.add_argument(
        "--cells",
        type=int,
        default=4,
        help="primitive surface cells along each in-plane axis (111, min 4)",
    )
    p.add_argument(
        "--layers",
        type=int,
        default=None,
        help="(111) layers (default 4, min 4)",
    )
    p.add_argument(
        "--hydrogens",
        type=int,
        default=None,
        help="number of H atoms on (111); implies --surface 111 (default 6)",
    )
    p.add_argument("--seed", type=int, default=1, help="RNG seed for (111) H placement")
    p.add_argument(
        "--freeze-bottom",
        type=int,
        default=2,
        help="(111) Cu layers frozen from the bottom (default 2)",
    )
    args = p.parse_args()
    use_111 = args.surface == "111" or args.hydrogens is not None or args.layers is not None
    if not use_111:
        out = args.output or DEFAULT_100
        write_con(out)
        print(f"wrote {out} (128 Cu frozen, 2 H free)")
        return
    n_h = 6 if args.hydrogens is None else args.hydrogens
    layers = 4 if args.layers is None else args.layers
    out = args.output or DEFAULT_111
    write_111(out, args.cells, layers, n_h, args.seed, args.freeze_bottom)


if __name__ == "__main__":
    main()

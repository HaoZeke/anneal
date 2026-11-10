#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Draw fig:si-water: the TIP4P minima of tbl:tip4p and the archived hexamers of each class.

(a) The TIP4P global minimum of each (H2O)n, n = 6 to 21, of the Cambridge Cluster
Database, at the published energy. (b, c) The lowest archived minimum of each hexamer
class at GFN2-xTB and at PBE/6-31G*. Each is read by system_visuals.py, which holds its
files to their digests, and drawn by xyzrender.

Usage: fig_si_water.py [--data DIR] out.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figstyle
import system_visuals

ROW = 1.35
COLUMNS = 4


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=system_visuals.DATA)
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    structures = system_visuals.structures(args.data)
    tip4p = [s for s in structures if s.cls == "TIP4P water"]
    by_key = {s.key: s for s in structures}
    blocks = [
        ("a", [tip4p[i : i + COLUMNS] for i in range(0, len(tip4p), COLUMNS)]),
        ("b", [[by_key[f"w6_gfn2_{cls}"] for cls in system_visuals.HEXAMER_CLASSES]]),
        ("c", [[by_key[f"w6_pbe_{cls}"] for cls in system_visuals.HEXAMER_CLASSES]]),
    ]
    rows = sum(len(block) for _, block in blocks)
    height = rows * ROW + 0.05
    figstyle.setup()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(figstyle.DOUBLE, height))
    column = figstyle.DOUBLE / COLUMNS
    panels = []
    r = 0
    for letter, block in blocks:
        for n, row in enumerate(block):
            bottom = height - (r + 1) * ROW
            if n == 0:
                fig.text(0.0, (bottom + ROW - 0.18) / height, f"({letter})", ha="left", va="bottom",
                         fontsize=figstyle.BASE)
            for c, structure in enumerate(row):
                ax = fig.add_axes(((c * column + 0.12) / figstyle.DOUBLE, (bottom + 0.25) / height,
                                   (column - 0.24) / figstyle.DOUBLE, (ROW - 0.42) / height))
                ax.set_axis_off()
                panels.append((ax, system_visuals.render(structure)))
                fig.text((c + 0.5) * column / figstyle.DOUBLE, (bottom + 0.12) / height,
                         system_visuals.caption(structure), ha="center", va="center", fontsize=figstyle.SMALLEST)
            r += 1
    system_visuals.place(fig, panels, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

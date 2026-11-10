#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Draw fig:si-structures: the Lennard-Jones and Sutton--Chen reference minima of the benchmark.

(a) The global minimum of each Lennard-Jones size of tbl:lj, LJ38, LJ75, LJ98 and LJ104
from the tests of anneal and the others from the Cambridge Cluster Database. (b) The
Sutton--Chen 10-8 minima of Pt38 and Pt75 from the same database. Each is read by
system_visuals.py, which holds its files to their digests and its energy to the published
one, and drawn by xyzrender.

Usage: fig_si_structures.py [--data DIR] out.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figstyle
import system_visuals

ROW = 1.55
COLUMNS = 4


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=system_visuals.DATA)
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    drawn = [s for s in system_visuals.structures(args.data) if s.cls in ("Lennard-Jones", "Sutton--Chen")]
    rows = -(-len(drawn) // COLUMNS)
    height = rows * ROW + 0.05
    figstyle.setup()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(figstyle.DOUBLE, height))
    column = figstyle.DOUBLE / COLUMNS
    panels = []
    for n, structure in enumerate(drawn):
        r, c = divmod(n, COLUMNS)
        bottom = height - (r + 1) * ROW
        if n == 0 or structure.cls != drawn[n - 1].cls:
            fig.text(c * column / figstyle.DOUBLE, (bottom + ROW - 0.18) / height,
                     "(a)" if structure.cls == "Lennard-Jones" else "(b)", ha="left", va="bottom",
                     fontsize=figstyle.BASE)
        ax = fig.add_axes(((c * column + 0.12) / figstyle.DOUBLE, (bottom + 0.25) / height,
                           (column - 0.24) / figstyle.DOUBLE, (ROW - 0.45) / height))
        ax.set_axis_off()
        panels.append((ax, system_visuals.render(structure)))
        fig.text((c + 0.5) * column / figstyle.DOUBLE, (bottom + 0.12) / height, system_visuals.caption(structure),
                 ha="center", va="center", fontsize=figstyle.SMALLEST)
    system_visuals.place(fig, panels, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

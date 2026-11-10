#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Draw fig:systems: the reference structures of the benchmark.

(a) The Lennard-Jones minima LJ38, LJ75, LJ98 and LJ104 that the tests of anneal hold to
the Cambridge Cluster Database, in the style of system_visuals.py. (b) The TIP4P minima of
10, 15 and 20 molecules, and the lowest archived GFN2-xTB minimum of each hexamer class,
drawn by xyzrender with their hydrogen bonds dotted. Every structure is read by
system_visuals.py, which holds its files to their digests.

Usage: fig_systems.py [--data DIR] out.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figstyle
import system_visuals

HEIGHT = 3.2


def inches(fig, x, y, w, h):
    return fig.add_axes((x / figstyle.DOUBLE, y / HEIGHT, w / figstyle.DOUBLE, h / HEIGHT))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=system_visuals.DATA)
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    structures = {s.key: s for s in system_visuals.structures(args.data)}
    figstyle.setup()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(figstyle.DOUBLE, HEIGHT))
    panels = []
    column = figstyle.DOUBLE / 4
    for i, key in enumerate(system_visuals.LJ_FILES):
        ax = inches(fig, i * column + 0.08, 1.72, column - 0.16, 1.28)
        ax.set_axis_off()
        panels.append((ax, system_visuals.render(structures[key])))
        fig.text((i + 0.5) * column / figstyle.DOUBLE, 1.6 / HEIGHT, system_visuals.caption(structures[key]),
                 ha="center", va="center", fontsize=figstyle.SMALLEST)
    water = [*system_visuals.TIP4P_FILES, *(f"w6_gfn2_{cls}" for cls in system_visuals.HEXAMER_CLASSES)]
    width = figstyle.DOUBLE / len(water)
    for i, key_name in enumerate(water):
        ax = inches(fig, i * width + 0.04, 0.3, width - 0.08, 0.9)
        ax.set_axis_off()
        panels.append((ax, system_visuals.render(structures[key_name])))
        structure = structures[key_name]
        text = structure.label if structure.cls != "Water hexamer" else structure.label.split(",")[0]
        fig.text((i + 0.5) * width / figstyle.DOUBLE, 0.16 / HEIGHT, text, ha="center", va="center",
                 fontsize=figstyle.SMALLEST)
    fig.text(1.5 * width / figstyle.DOUBLE, 0.03 / HEIGHT, "TIP4P", ha="center", va="center",
             fontsize=figstyle.SMALLEST)
    fig.text((3 + 2) * width / figstyle.DOUBLE, 0.03 / HEIGHT, "hexamer, GFN2-xTB", ha="center", va="center",
             fontsize=figstyle.SMALLEST)
    for letter, y in (("a", 3.02), ("b", 1.27)):
        fig.text(0.0, y / HEIGHT, f"({letter})", ha="left", va="bottom", fontsize=figstyle.BASE)
    system_visuals.place(fig, panels, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

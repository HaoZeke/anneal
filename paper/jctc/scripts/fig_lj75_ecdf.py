#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Draw the share of LJ75 runs that have hit against the force calls they spent, per arm.

Reads the runs through lj75_gmin_table.load, which holds the CSVs of data/lj75_gmin to
their digests and to the sealed verdict of the audit that reran them, so the figure
draws only runs that audit confirmed. Each arm is a step ECDF over its 200 runs of the
force calls to the first hit, a miss never stepping; a dotted line marks each arm's call
PAR-2, and the lower right, clear of the lines, prints it in the arm's colour as tbl:lj75
does. Wall times are not drawn: the per-run walls of the
CSVs are of another run than the audit's.

Usage: fig_lj75_ecdf.py [--data DIR] OUT.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figstyle
import lj75_gmin_table

STYLE = {
    "anneal, recommended hop": ("recommended hop", figstyle.SKY, "-"),
    "anneal, derived hop": ("derived hop", figstyle.MAGENTA, "--"),
    "GMIN, core orbits": ("GMIN, core orbits", figstyle.CORAL, "-."),
}


def ecdf(arm) -> tuple[list[float], list[float]]:
    """The force calls of the hits in order, and the share of all runs hit by each."""
    calls = sorted(run.calls for run in arm.runs if run.hit)
    n = len(arm.runs)
    return [float(c) for c in calls], [(i + 1) / n for i in range(len(calls))]


def draw(data: Path, out: Path) -> None:
    import matplotlib.pyplot as plt

    figstyle.setup()
    _, arms = lj75_gmin_table.load(data)
    fig, ax = plt.subplots(figsize=(figstyle.SINGLE, 2.35), layout="constrained")
    for arm in arms:
        name, colour, dash = STYLE[arm.label]
        x, y = ecdf(arm)
        ax.step([x[0], *x], [0.0, *y], where="post", color=colour, linestyle=dash, label=name)
        ax.axvline(arm.par2, color=colour, linestyle=":", linewidth=0.9)
        ax.text(
            0.98, 0.26 - 0.08 * list(STYLE).index(arm.label), lj75_gmin_table.cells(arm)["par2"],
            transform=ax.transAxes, color=colour, fontsize=figstyle.SMALLEST, ha="right", va="center",
        )
    ax.text(0.98, 0.34, "call PAR-2", transform=ax.transAxes, fontsize=figstyle.SMALLEST, ha="right", va="center")
    ax.set_xscale("log")
    figstyle.plain_log(ax.xaxis)
    ax.set_xlabel("Force calls to the first hit")
    ax.set_ylabel("Share of the runs that hit")
    ax.set_ylim(0, 1.02)
    ax.plot([], [], color=figstyle.GREY, linestyle=":", linewidth=0.9, label="call PAR-2")
    ax.legend(loc="upper left", handlelength=2.2)
    figstyle.save(fig, out)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=lj75_gmin_table.DATA)
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    draw(args.data, args.out)


if __name__ == "__main__":
    main(sys.argv[1:])

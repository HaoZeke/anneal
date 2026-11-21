#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Draw the water hexamer against CREST from the 192-seed rerun receipts.

Reads the two arms through water_rerun.load, which holds every receipt to its checksum
and takes every statistic from scripts/paired_stats.py, so the figure draws what the
Supporting Information prints and nothing else.

  (a) the hits of each arm out of the 192 seeds, with the Wilson interval;
  (b) the hit rate of this code less that of CREST, paired by seed, with the Newcombe
      interval and the exact McNemar p.

Usage: fig_water_crest.py [--data DIR] OUT.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figstyle
import water_rerun

COLOURS = {"anneal": figstyle.SKY, "crest": figstyle.CORAL}
SHORT = {"anneal": "this code", "crest": "CREST"}


def draw(data: Path, out: Path) -> None:
    import matplotlib.pyplot as plt

    figstyle.setup()
    rerun = water_rerun.load(data)
    runs = len(water_rerun.SEEDS)
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(figstyle.DOUBLE, 1.7), layout="constrained", gridspec_kw={"width_ratios": [1.15, 1]}
    )
    for y, arm in enumerate(water_rerun.ARMS):
        share = rerun.hits[arm] / runs
        low, high = rerun.wilson[arm]
        a.barh(y, share, color=COLOURS[arm], height=0.62, edgecolor=figstyle.BLACK, linewidth=0.4)
        a.errorbar(share, y, xerr=[[share - low], [high - share]], fmt="none", ecolor=figstyle.BLACK,
                   elinewidth=0.8, capsize=2.5)
        a.text(high + 0.02, y, f"{rerun.hits[arm]}/{runs}", va="center", ha="left", fontsize=figstyle.SMALLEST)
    a.set_yticks(range(len(water_rerun.ARMS)), [SHORT[arm] for arm in water_rerun.ARMS])
    a.invert_yaxis()
    a.set_xlim(0, 1.0)
    a.set_xlabel(f"Share of the {runs} seeds that hit (fraction)")

    low, high = rerun.newcombe
    difference = (rerun.table[1] - rerun.table[2]) / runs
    b.errorbar(difference, 0, xerr=[[difference - low], [high - difference]], fmt="o",
               color=COLOURS["anneal"], ecolor=figstyle.BLACK, markeredgecolor=figstyle.BLACK,
               markeredgewidth=0.5, markersize=5, elinewidth=0.9, capsize=3)
    b.text(high + 0.015, 0, f"$p={rerun.p:.3g}$", va="center", ha="left", fontsize=figstyle.SMALLEST)
    b.axvline(0, color=figstyle.GREY, linewidth=0.7, linestyle=":")
    b.set_yticks([0], ["this code\nless CREST"])
    b.set_ylim(0.5, -0.5)
    b.set_xlim(-0.1, 0.45)
    b.set_xlabel("Paired difference in hit rate (fraction)")
    for ax, letter in zip((a, b), "ab"):
        figstyle.label(ax, letter)
    figstyle.save(fig, out)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=water_rerun.DATA)
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    draw(args.data, args.out)


if __name__ == "__main__":
    main(sys.argv[1:])

#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Draw the water hexamer against CREST from the water_v45b per-seed table.

Reads the four arms through water_results_table.load, which recomputes Wilson, McNemar and
Newcombe method 10 from data/water_v45b/water-v45b-perseed.csv, so the figure draws what
tbl:water prints and nothing else. The CSV carries no run-by-run force-call trace beyond
the sealed totals, so nothing is drawn against force calls to the first hit.

  (a) the hits of each arm out of the 192 seeds, with the Wilson interval;
  (b) for each arm of the hop scored against CREST, the hit rate of the hop less that of
      CREST, with the Newcombe interval and the exact McNemar p.

Usage: fig_water_crest.py [--data DIR] OUT.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figstyle
import water_results_table

COLOURS = {
    "anneal": figstyle.SKY,
    "polish_fair": figstyle.TEAL,
    water_results_table.CREST_FOLDER: figstyle.CORAL,
    "crest_fair": figstyle.SUNSHINE,
}
# The label of each arm on the axis of the figure, broken to fit it.
SHORT = {
    "anneal": "anneal,\ndefault hop",
    "polish_fair": "anneal, budget-fair\npolish, opt-in",
    water_results_table.CREST_FOLDER: "CREST",
    "crest_fair": "CREST plus the\npolish, budget-fair",
}


def draw(data: Path, out: Path) -> None:
    import matplotlib.pyplot as plt

    figstyle.setup()
    _, arms, _ = water_results_table.load(data)
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(figstyle.DOUBLE, 2.3), layout="constrained", gridspec_kw={"width_ratios": [1.15, 1]}
    )
    for y, arm in enumerate(arms):
        share = arm.summary["hits"] / arm.summary["n"]
        low, high = (float(bound) for bound in arm.wilson)
        a.barh(y, share, color=COLOURS[arm.folder], height=0.62, edgecolor=figstyle.BLACK, linewidth=0.4)
        a.errorbar(share, y, xerr=[[share - low], [high - share]], fmt="none", ecolor=figstyle.BLACK,
                   elinewidth=0.8, capsize=2.5)
        a.text(high + 0.02, y, arm.hits, va="center", ha="left", fontsize=figstyle.SMALLEST)
    a.set_yticks(range(len(arms)), [SHORT[arm.folder] for arm in arms])
    a.invert_yaxis()
    a.set_xlim(0, 1.0)
    a.set_xlabel(f"Share of the {water_results_table.RUNS} seeds that hit")

    scored = [arm for arm in arms if arm.contrast]
    for y, arm in enumerate(scored):
        c = arm.contrast
        low, high = c["newcombe_95_6"]
        difference = c["hit_rate_difference"]
        b.errorbar(difference, y, xerr=[[difference - low], [high - difference]], fmt="o",
                   color=COLOURS[arm.folder], ecolor=figstyle.BLACK, markeredgecolor=figstyle.BLACK,
                   markeredgewidth=0.5, markersize=5, elinewidth=0.9, capsize=3)
        b.text(high + 0.015, y, f"p {water_results_table.cells(arm)['p']}", va="center", ha="left",
               fontsize=figstyle.SMALLEST)
    b.axvline(0, color=figstyle.GREY, linewidth=0.7, linestyle=":")
    b.set_yticks(range(len(scored)), [SHORT[arm.folder] for arm in scored])
    b.set_ylim(len(scored) - 0.5, -0.5)
    b.set_xlim(-0.1, 0.75)
    b.set_xlabel("Hit rate of the hop less CREST's")
    for ax, letter in zip((a, b), "ab"):
        figstyle.label(ax, letter)
    figstyle.save(fig, out)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=water_results_table.DATA)
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    draw(args.data, args.out)


if __name__ == "__main__":
    main(sys.argv[1:])

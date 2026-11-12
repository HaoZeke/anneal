#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Draw the six-seed CREST counter chart as figs/water_crest.pdf.

The bars are the old counter and CREST's energy-and-gradient total for seeds
0 through 5. Both come from logs/water/crest-hexamer-seedN.log through
hexamer_numbers.load. This script does not read water-v45b-perseed.csv and
does not draw a hit rate, a PAR-2, or a McNemar value.

Usage: fig_water_crest.py OUT.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figstyle
from hexamer_numbers import load


def draw(out: Path) -> None:
    import matplotlib.pyplot as plt

    figstyle.setup()
    rows = load()
    seeds = list(range(6))
    old = [rows[seed]["old"] for seed in seeds]
    engrad = [rows[seed]["engrad"] for seed in seeds]
    figure, axis = plt.subplots(figsize=(figstyle.DOUBLE, 2.6), layout="constrained")
    positions = list(range(6))
    width = 0.36
    axis.bar(
        [item - width / 2 for item in positions],
        old,
        width=width,
        color=figstyle.SKY,
        edgecolor=figstyle.BLACK,
        linewidth=0.4,
        label="Old counter",
    )
    axis.bar(
        [item + width / 2 for item in positions],
        engrad,
        width=width,
        color=figstyle.SUNSHINE,
        edgecolor=figstyle.BLACK,
        linewidth=0.4,
        label="Energy and gradient",
    )
    axis.set_xticks(positions, [str(seed) for seed in seeds])
    axis.set_xlabel("Seed")
    axis.set_ylabel("CREST calls")
    axis.set_ylim(0, 250000)
    axis.legend()
    figstyle.save(figure, out)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    draw(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

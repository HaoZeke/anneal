#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Draw the single-chain search used for the reported atomic-cluster runs.

The top band is the state held by one chain. The middle band is one hop: the
Thompson draw over the arms from the Normal-Gamma posterior of eq:thompson,
the proposal and quench, the acceptance test at T = 0.8, and the depth of
eq:depth that updates the posterior. The bottom band is rgpot, through which
every potential of tbl:systems reaches the hop. No run enters the figure.

Usage: fig_algorithm_stack.py OUT.pdf
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figstyle

POTENTIALS = (
    "Lennard-Jones", "Morse", "Sutton-Chen", "EMT", "EAM", "Stillinger-Weber",
    "Tersoff", "TIP4P", "GFN2-xTB", "GFN-FF", "PBE/6-31G*", "PET-MAD",
)
WIDTH = figstyle.DOUBLE
HEIGHT = 3.35
BANDS = (
    ("State of one chain", 2.36, 0.94, figstyle.SKY),
    ("One hop", 0.98, 1.3, figstyle.CORAL),
    ("Engine layer", 0.08, 0.82, figstyle.TEAL),
)


def draw(out: Path) -> None:
    import matplotlib.pyplot as plt

    figstyle.setup()
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    ax = figstyle.canvas(fig)
    for name, y, h, colour in BANDS:
        figstyle.box(ax, 0.05, y, WIDTH - 0.1, h, fill=colour + "22", edge=colour)
        ax.text(WIDTH - 0.12, y + h - 0.05, name, ha="right", va="top", fontsize=figstyle.BASE, weight="bold")

    state = (
        (0.15, 1.35, "current quenched\nminimum"),
        (1.65, 1.35, "recorded basin\nentry"),
        (3.15, 1.35, "eight-entry\ntabu list"),
        (4.65, 1.65, "one Normal-Gamma\nposterior per move"),
    )
    for x, width, text in state:
        figstyle.box(ax, x, 2.52, width, 0.38, text)
    ax.text(
        WIDTH / 2, 2.42,
        "one chain; no population resampling or inter-chain communication",
        ha="center", va="top", fontsize=figstyle.SMALLEST,
    )

    row, tall = 1.6, 0.4
    flow = [
        (0.15, 1.1, "Thompson draw\nover the arms"),
        (1.45, 1.0, "propose the\narm's move"),
        (2.65, 1.0, "quench: screen,\nthen finish"),
        (3.85, 1.0, "accept at\ntemperature $T$"),
        (5.05, 1.25, "depth $r$ updates the\nNormal-Gamma posterior"),
    ]
    boxes = [figstyle.box(ax, x, row, w, tall, text) for x, w, text in flow]
    for (x0, _, w0, _), (x1, _, _, _) in itertools.pairwise(boxes):
        figstyle.arrow(ax, (x0 + w0, row + tall / 2), (x1, row + tall / 2))
    last, first = boxes[-1], boxes[0]
    figstyle.arrow(
        ax, (last[0] + 0.35, row + tall), (first[0] + first[2] / 2, row + tall), colour=figstyle.MAGENTA, rad=0.09,
    )
    temperature = figstyle.box(
        ax, 2.45, 1.06, 2.05, 0.4,
        "$T$ held at 0.8\n(the runs' setting)",
    )
    quench, accept = boxes[2], boxes[3]
    figstyle.arrow(
        ax, (accept[0] + accept[2] / 2, temperature[1] + temperature[3]), (accept[0] + accept[2] / 2, row),
    )

    engine = figstyle.box(ax, 0.15, 0.18, 1.2, 0.52, "rgpot: energy and\nforces through\none interface")
    chip, gap, left = 0.78, 0.05, 1.45
    for i, name in enumerate(POTENTIALS):
        column, line = i % 6, i // 6
        figstyle.box(ax, left + column * (chip + gap), 0.45 - line * 0.27, chip, 0.22, name, size=figstyle.SMALLEST)
    centre = quench[0] + quench[2] / 2
    figstyle.arrow(ax, (engine[0] + engine[2], 0.44), (left, 0.44), both=True)
    figstyle.arrow(ax, (centre, 0.74), (centre, row), both=True)
    ax.text(centre - 0.06, 1.26, "energies,\nforces", ha="right", va="center", fontsize=figstyle.SMALLEST)
    figstyle.save(fig, out)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    draw(parser.parse_args(argv).out)


if __name__ == "__main__":
    main(sys.argv[1:])

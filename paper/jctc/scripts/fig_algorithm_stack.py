#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Draw the stack of the search: the communicating chains, one hop, and the engine layer.

The top band is the bank of Section sec:chains, whose chains share minima under the
radial key of eq:radial and are resampled on the within-epoch ranks of eq:weight. The
middle band is one hop: the Thompson draw over the arms from the Normal-Gamma posterior
of eq:ng, the quench, the acceptance test at the held temperature T = 0.8 E_0, the depth
of eq:reward that updates the posterior, and the Good-Turing census of eq:unseen whose
saturation retires a packing by eq:retire. The bottom band is rgpot, through which every
potential of tbl:systems reaches the hop. No run enters the figure.

Usage: fig_algorithm_stack.py OUT.pdf
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figstyle

CHAINS = ("chain 1", "chain 2", "chain 3", "\N{HORIZONTAL ELLIPSIS}", "chain 40")
POTENTIALS = (
    "Lennard-Jones", "Morse", "Sutton-Chen", "EMT", "EAM", "Stillinger-Weber",
    "Tersoff", "TIP4P", "GFN2-xTB", "GFN-FF", "PBE/6-31G*", "PET-MAD",
)
WIDTH = figstyle.DOUBLE
HEIGHT = 3.35
BANDS = (
    ("Communicating chains", 2.36, 0.94, figstyle.SKY),
    ("One hop of each chain", 0.98, 1.3, figstyle.CORAL),
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

    for i, name in enumerate(CHAINS):
        gap = name == CHAINS[3]
        figstyle.box(ax, 0.15 + i * 0.63, 2.5, 0.55, 0.32, name, fill="none" if gap else "white", edge="none" if gap else figstyle.BLACK)
    bank = figstyle.box(
        ax, 3.72, 2.42, 2.6, 0.68,
        "annealed-flow bank, one slot per chain:\nresampled on within-epoch ranks;\n"
        "minima keyed by radial distances and\nmerged inside a held distance; every\nforce call charged to its chain",
    )
    figstyle.arrow(ax, (3.27, 2.66), (bank[0], 2.66), both=True)
    ax.text(3.495, 2.72, "share,\nadopt", ha="center", va="bottom", fontsize=figstyle.SMALLEST)

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
    census = figstyle.box(
        ax, 0.15, 1.06, 2.3, 0.4,
        "Good-Turing census: unseen mass $n$\N{SUBSCRIPT ONE}/$n$;\na saturated packing retires",
        fill="white",
    )
    temperature = figstyle.box(
        ax, 3.4, 1.06, 2.05, 0.4,
        "$T$ held at 0.8$E$\N{SUBSCRIPT ZERO}\n(the runs' setting)",
    )
    quench, accept = boxes[2], boxes[3]
    figstyle.arrow(ax, (quench[0] + 0.15, row), (census[0] + census[2] - 0.1, census[1] + census[3]))
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

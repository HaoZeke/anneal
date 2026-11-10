#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Draw the arms of the recommended hop and the law of its burst length.

(a) The arms among which the Thompson draw of the depth posterior chooses, the single
quench every proposal meets, and the single acceptance test after it, whose depth
updates the posterior of the arm that played. (b) The burst law of eq:burst, exact in
rationals: one surface relocation, continued with probability 2/3 up to eight, so the
capped law puts the geometric tail beyond seven at eight; the uncapped geometric law is
drawn beside it. No run enters the figure.

Usage: fig_rec_arms.py OUT.pdf
"""
from __future__ import annotations

import argparse
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figstyle

ARMS = ("single", "surface", "sym", "burst", "soap", "resample")
CONTINUE = Fraction(2, 3)
CAP = 8


def burst() -> list[Fraction]:
    """P(K=k) for k = 1 to CAP under eq:burst."""
    law = [(1 - CONTINUE) * CONTINUE ** (k - 1) for k in range(1, CAP)]
    return [*law, CONTINUE ** (CAP - 1)]


def geometric() -> list[Fraction]:
    return [(1 - CONTINUE) * CONTINUE ** (k - 1) for k in range(1, CAP + 1)]


def draw(out: Path) -> None:
    import matplotlib.pyplot as plt

    figstyle.setup()
    fig = plt.figure(figsize=(figstyle.DOUBLE, 2.3))
    ax = figstyle.canvas(fig, (0.0, 0.0, 0.62, 1.0))
    left, right, gap = 0.1, 3.88, 0.07
    width = (right - left - gap * (len(ARMS) - 1)) / len(ARMS)
    top = figstyle.box(
        ax, left, 1.86, right - left, 0.3,
        "Thompson draw: one predictive draw from the Normal-Gamma posterior\nof each arm; the largest plays",
        fill=figstyle.SKY + "40",
    )
    quench = figstyle.box(
        ax, left, 0.92, right - left, 0.3, "one quench: a screen of $S$ steps, finished to $R$ unless the screen stops it",
        fill=figstyle.TEAL + "33",
    )
    accept = figstyle.box(
        ax, 0.7, 0.3, 2.4, 0.32, "one acceptance test on the quenched energy\nplus the bias, at temperature $T$", fill=figstyle.SUNSHINE + "66"
    )
    for i, name in enumerate(ARMS):
        x = left + i * (width + gap)
        figstyle.box(ax, x, 1.4, width, 0.3, name)
        centre = x + width / 2
        figstyle.arrow(ax, (centre, top[1]), (centre, 1.7), linestyle=(0, (2, 1.5)))
        figstyle.arrow(ax, (centre, 1.4), (centre, quench[1] + quench[3]))
    middle = accept[0] + accept[2] / 2
    figstyle.arrow(ax, (middle, quench[1]), (middle, accept[1] + accept[3]))
    figstyle.arrow(
        ax, (accept[0] + accept[2], accept[1] + accept[3] / 2), (right - 0.02, top[1] + 0.02),
        colour=figstyle.SKY, rad=0.42,
    )
    ax.text(
        3.3, 0.4, "depth $r$ updates the\nposterior of the arm\nthat played", fontsize=figstyle.SMALLEST,
        color=figstyle.SKY, ha="left", va="center",
    )
    ax.text(0.02, 2.2, "(a)", ha="left", va="bottom", fontsize=figstyle.BASE)

    law = fig.add_axes((0.75, 0.21, 0.235, 0.68))
    k = range(1, CAP + 1)
    law.bar(k, [float(p) for p in burst()], width=0.62, color=figstyle.SKY, label="capped at eight")
    law.plot(
        k, [float(p) for p in geometric()], linestyle="none", marker="o", markersize=3.5,
        markerfacecolor="white", markeredgecolor=figstyle.CORAL, label="uncapped geometric",
    )
    law.set_xticks(list(k))
    law.set_xlabel("Surface relocations in the burst, $k$")
    law.set_ylabel("$P(K=k)$")
    law.legend(loc="upper right")
    law.text(-0.3, 1.02, "(b)", transform=law.transAxes, ha="left", va="bottom", fontsize=figstyle.BASE)
    figstyle.save(fig, out)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    draw(parser.parse_args(argv).out)


if __name__ == "__main__":
    main(sys.argv[1:])

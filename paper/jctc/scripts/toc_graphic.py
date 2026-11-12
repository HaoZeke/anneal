#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Table-of-contents graphic, 3.25 in by 1.75 in, with the LJ38 count filled in."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    dest = Path(args[0]) if args else Path(__file__).resolve().parents[1] / "figs" / "toc-graphic.pdf"
    dest.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(3.25, 1.75))
    axis.set_xlim(0, 3.25)
    axis.set_ylim(0, 1.75)
    axis.axis("off")
    boxes = [
        (0.10, 0.55, 0.95, 0.95, "#d9efe8", "LJ38\n6/6 hits\n400000 calls"),
        (1.15, 0.55, 0.95, 0.95, "#d6e6f5", "One hop\nquench, then\nMetropolis"),
        (2.20, 0.55, 0.95, 0.95, "#f8e3c4", "Library fixed\nsurface list\nchanges at 98"),
    ]
    for x, y, width, height, colour, text in boxes:
        axis.add_patch(
            FancyBboxPatch(
                (x, y), width, height,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor=colour, edgecolor="#222222", linewidth=0.8,
            )
        )
        axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=7)
    for left, right in ((1.05, 1.15), (2.10, 2.20)):
        axis.add_patch(
            FancyArrowPatch((left, 1.02), (right, 1.02), arrowstyle="-|>", mutation_scale=8, linewidth=0.8, color="#222222")
        )
    axis.text(0.10, 0.22, "Shared starts. GMIN hits 3/6 at this size.", fontsize=7)
    figure.savefig(dest)
    plt.close(figure)


if __name__ == "__main__":
    main()

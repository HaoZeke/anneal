#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Draw the six-seed CREST counter chart.

This is not the 192-seed figure figs/water_crest.pdf.

The bar labels are the same integers as the manuscript table.
From the repository root:

    python3 paper/jctc/scripts/water_crest_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hexamer_numbers import load


def main() -> None:
    rows = load()
    seeds = list(range(6))
    old = [rows[seed]["old"] for seed in seeds]
    engrad = [rows[seed]["engrad"] for seed in seeds]
    figure, axis = plt.subplots(figsize=(6.6, 3.4))
    positions = list(range(6))
    width = 0.36
    axis.bar(
        [p - width / 2 for p in positions],
        old,
        width=width,
        color="#4c78a8",
        label="Old counter",
    )
    axis.bar(
        [p + width / 2 for p in positions],
        engrad,
        width=width,
        color="#f58518",
        label="Energy+gradient",
    )
    for position, value in zip(positions, old):
        axis.text(
            position - width / 2,
            value + 2500,
            str(value),
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )
    for position, value in zip(positions, engrad):
        axis.text(
            position + width / 2,
            value + 2500,
            str(value),
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels([str(seed) for seed in seeds])
    axis.set_xlabel("Seed")
    axis.set_ylabel("CREST calls")
    axis.set_ylim(0, 250000)
    axis.legend(frameon=False)
    axis.set_title("Water hexamer, both CREST counters")
    figure.tight_layout()
    dest = Path(__file__).resolve().parents[1] / "figs" / "water_crest_counters.pdf"
    figure.savefig(dest)
    plt.close(figure)


if __name__ == "__main__":
    main()

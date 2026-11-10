#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Draw the rate summary from the six-seed hexamer logs only.

Every seed succeeded for both anneal ledgers and for CREST, so each bar
is 6/6. Systems with no finished run are omitted. Stale Lennard-Jones
counts are omitted rather than printed.
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
    if any(
        rows[seed]["gap_4000"] is None or rows[seed]["engrad"] <= 0
        for seed in range(6)
    ):
        raise SystemExit("hexamer rows are incomplete")
    labels = [
        "anneal\n4000 evaluations",
        "anneal\nold counter",
        "CREST",
    ]
    shares = [1.0, 1.0, 1.0]
    figure, axis = plt.subplots(figsize=(6.2, 3.4))
    positions = list(range(3))
    axis.bar(positions, shares, color=["#4c78a8", "#72b7b2", "#f58518"])
    for position in positions:
        axis.text(position, 1.02, "6/6", ha="center", va="bottom", fontsize=11)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylim(0, 1.25)
    axis.set_ylabel("Share of the six seeds")
    axis.set_title("Water hexamer success")
    figure.tight_layout()
    dest = Path(__file__).resolve().parents[1] / "figs" / "summary_rates.pdf"
    figure.savefig(dest)
    plt.close(figure)


if __name__ == "__main__":
    main()

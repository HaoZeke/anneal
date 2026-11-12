#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Encounter curves of the stated-preset Lennard-Jones table.

LJ38, LJ55 and LJ75 are the per-seed rows in bench/gmin-h2h/stated-38-55-75-rows.
LJ98 and LJ104 are bench/gmin-wall-v8/results.csv. A miss does not step the curve.

Usage: fig_stated_ecdf.py OUT.pdf
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figstyle

ROOT = Path(__file__).resolve().parents[3]
ROWS = ROOT / "bench" / "gmin-h2h" / "stated-38-55-75-rows"
V8 = ROOT / "bench" / "gmin-wall-v8" / "results.csv"
SIZES = (38, 55, 75, 98, 104)


def _hit(text: str) -> bool:
    return text in ("1", "true", "True")


def load_rows() -> dict[int, dict[str, list[tuple[bool, float]]]]:
    found: dict[int, dict[str, list[tuple[bool, float]]]] = {n: {"anneal": [], "gmin": []} for n in (38, 55, 75)}
    for path in sorted(ROWS.glob("lj*/anneal-*.row")):
        n = int(path.parent.name.removeprefix("lj"))
        row = next(csv.DictReader(path.open()))
        found[n]["anneal"].append((_hit(row["hit"]), float(row["evals_to_hit"])))
    for path in sorted(ROWS.glob("lj*/gmin-*.row")):
        n = int(path.parent.name.removeprefix("lj"))
        text = path.read_text()
        fields = dict(part.split("=", 1) for part in text.split()[1:])
        found[n]["gmin"].append((_hit(fields["hit"]), float(fields["evals_to_hit"])))
    found[98] = {"anneal": [], "gmin": []}
    found[104] = {"anneal": [], "gmin": []}
    for row in csv.DictReader(V8.open()):
        n = int(row["system"].removeprefix("lj"))
        found[n][row["arm"]].append((_hit(row["hit"]), float(row["evals_to_hit"])))
    return found


def ecdf(runs: list[tuple[bool, float]]) -> tuple[list[float], list[float]]:
    calls = sorted(calls for hit, calls in runs if hit)
    n = len(runs)
    return [float(c) for c in calls], [(i + 1) / n for i in range(len(calls))]


def draw(out: Path) -> None:
    import matplotlib.pyplot as plt

    figstyle.setup()
    data = load_rows()
    fig, axes = plt.subplots(1, 5, figsize=(figstyle.DOUBLE, 1.85), sharey=True, layout="constrained")
    for ax, n, letter in zip(axes, SIZES, "abcde"):
        for arm, colour, dash, name in (
            ("anneal", figstyle.SKY, "-", "preset"),
            ("gmin", figstyle.CORAL, "-.", "GMIN"),
        ):
            x, y = ecdf(data[n][arm])
            if x:
                ax.step([x[0], *x], [0.0, *y], where="post", color=colour, linestyle=dash, label=name)
            else:
                ax.plot([], [], color=colour, linestyle=dash, label=name)
        ax.set_xscale("log")
        figstyle.plain_log(ax.xaxis)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"LJ{n}", fontsize=figstyle.SMALLEST)
        figstyle.label(ax, letter)
    axes[0].set_ylabel("Share of runs that hit")
    axes[2].set_xlabel("Force calls to the first hit")
    axes[0].legend(loc="upper left", handlelength=1.6, fontsize=figstyle.SMALLEST)
    figstyle.save(fig, out)


def main(argv: list[str]) -> None:
    draw(Path(argv[0]))


if __name__ == "__main__":
    main(sys.argv[1:])

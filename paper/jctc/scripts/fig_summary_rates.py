#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Draw the share of seeds that reach the reference, by class of tbl:systems and by method.

A cell is filled only from runs the repository archives and a script recounts:

  LJ75, hop and BH: the recommended hop and GMIN with core orbits, read through
      lj75_gmin_table.load, which holds them to the sealed verdict of their audit;
  LJ38, LJ55 and LJ98, hop: the recommended preset, read through lj_csm_sizes_table.load
      from the sealed board of each size, and only from a board the rerun reproduced;
  hexamer at GFN2-xTB, hop and CREST: the default hop and CREST of water_v45b
      on anneal 36086e98, seeds 0 to 191, read through water_results_table.load;
  LJ38 and LJ98, BH: GMIN of anneal_repro v4.5b, read from data/gmin_lj_sizes_v45b.

Every other cell is hatched as not yet run; the campaign at the caps of tbl:systems
fills them. Lennard-Jones takes one row a size; no other class pools more than the one
system named here.

Usage: fig_summary_rates.py [--lj75 DIR] [--sizes DIR] [--water DIR] OUT.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figstyle
import lj75_gmin_table
import lj_csm_sizes_table
import water_results_table

METHODS = ("hop", "BH", "PT", "PA", "CSA", "DH-GA", "BCGA", "MH", "GOFEE", "CREST")
LENNARD_JONES = ("LJ\N{SUBSCRIPT THREE}\N{SUBSCRIPT EIGHT}", "LJ\N{SUBSCRIPT FIVE}\N{SUBSCRIPT FIVE}",
                 "LJ\N{SUBSCRIPT SEVEN}\N{SUBSCRIPT FIVE}", "LJ\N{SUBSCRIPT NINE}\N{SUBSCRIPT EIGHT}")
ROW = dict(zip(("LJ38", "LJ55", "LJ75", "LJ98"), LENNARD_JONES, strict=True))
CLASSES = (
    *LENNARD_JONES, "Morse", "binary LJ", "Sutton-Chen", "effective medium", "embedded atom",
    "Si and C", "TIP4P water", "hexamer at GFN2-xTB", "hexamer at PBE", "hexamer at PET-MAD",
    "molecules at GFN-FF", "Pt(111) heptamer",
)


def cells(lj75: Path, sizes: Path, water: Path) -> dict[tuple[str, str], tuple[int, int]]:
    """The hits and the seeds of each filled cell."""
    import csv

    _, arms = lj75_gmin_table.load(lj75)
    by_label = {arm.label: arm for arm in arms}
    _, sealed, _ = water_results_table.load(water)
    by_folder = {arm.folder: arm for arm in sealed}
    found = {}
    for key, arm in (
        ((ROW["LJ75"], "hop"), by_label["anneal, recommended hop"]),
        ((ROW["LJ75"], "BH"), by_label["GMIN, core orbits"]),
    ):
        found[key] = (arm.hits, len(arm.runs))
    for key, folder in (
        (("hexamer at GFN2-xTB", "hop"), "anneal"),
        (("hexamer at GFN2-xTB", "CREST"), water_results_table.CREST_FOLDER),
    ):
        found[key] = (by_folder[folder].summary["hits"], by_folder[folder].summary["n"])
    for cell in lj_csm_sizes_table.load(sizes)[2]:
        found[(ROW[cell.system], cell.method)] = (cell.board.hits, cell.board.seeds)
    gmin_csv = Path(__file__).resolve().parents[1] / "data" / "gmin_lj_sizes_v45b" / "gmin-lj-sizes.csv"
    if gmin_csv.is_file():
        for row in csv.DictReader(gmin_csv.open()):
            system = {"lj38": "LJ38", "lj98": "LJ98"}[row["system"]]
            found[(ROW[system], "BH")] = (int(row["hits"]), int(row["n"]))
    return found


def draw(lj75: Path, sizes: Path, water: Path, out: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch, Rectangle

    figstyle.setup()
    filled = cells(lj75, sizes, water)
    colours = figstyle.ramp()
    scale = Normalize(0, 1)
    fig, ax = plt.subplots(figsize=(figstyle.DOUBLE, 4.15), layout="constrained")
    for j, name in enumerate(CLASSES):
        for i, method in enumerate(METHODS):
            cell = filled.get((name, method))
            if cell is None:
                ax.add_patch(
                    Rectangle((i, j), 1, 1, facecolor="white", edgecolor=figstyle.GREY, hatch="////", linewidth=0.4)
                )
                continue
            hits, seeds = cell
            share = hits / seeds
            ax.add_patch(Rectangle((i, j), 1, 1, facecolor=colours(scale(share)), edgecolor="white", linewidth=0.4))
            ink = "white" if share < 0.6 else "black"
            ax.text(i + 0.5, j + 0.5, f"{hits}/{seeds}", ha="center", va="center", fontsize=figstyle.SMALLEST, color=ink)
    ax.set_xlim(0, len(METHODS))
    ax.set_ylim(len(CLASSES), 0)
    ax.set_xticks([i + 0.5 for i in range(len(METHODS))], METHODS)
    ax.set_yticks([j + 0.5 for j in range(len(CLASSES))], CLASSES)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=scale, cmap=colours), ax=ax, shrink=0.62, pad=0.015, anchor=(0, 1))
    # A colorbar of 50 or more levels rasterises itself; the figure stays vector.
    bar.solids.set_rasterized(False)
    bar.solids.set_edgecolor("face")
    bar.set_label("Share of the seeds that hit")
    bar.outline.set_linewidth(0.4)
    fig.legend(
        handles=[
            Patch(facecolor="white", edgecolor=figstyle.GREY, hatch="////", linewidth=0.4, label="not yet run"),
        ],
        loc="outside lower center", ncols=1, fontsize=figstyle.SMALLEST, handlelength=1.6,
    )
    figstyle.save(fig, out)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lj75", type=Path, default=lj75_gmin_table.DATA)
    parser.add_argument("--sizes", type=Path, default=lj_csm_sizes_table.DATA)
    parser.add_argument("--water", type=Path, default=water_results_table.DATA)
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    draw(args.lj75, args.sizes, args.water, args.out)


if __name__ == "__main__":
    main(sys.argv[1:])

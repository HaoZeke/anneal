#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Draw fig:disconnectivity: the approximate disconnectivity graph of the LJ38 minima.

The steps and settings are those of campaign_disconnectivity.py of the paper at 028d995,
on the LJ38 dumps of data/landscapes_7695255/ and the fcc and icosahedral references of
data/system_visuals/sketchmap/, each read through system_visuals.py against its digest:
the dumps of basin hopping, the recommended hop and random structure search pooled, at
most 1200 drawn by the random stream of seed 11 with the deepest kept, and those at or
below -170 epsilon kept; the references first; the sorted smoothed coordination numbers
at cutoff 1.5 as the descriptor; minima within 0.15 of a deeper one merged; and, through
chemparseplot, the merge tree of the 6-nearest-neighbour graph over 48 energy levels.
The graph records no transition state, so a merge stands at the level at which two sets
of minima first join in the graph, not at a barrier. The stems are coloured by five
spectral basins of the same graph in the five colours of the Ruhi palette, as chemparseplot
paints its basins, named by their deepest minimum and its reference, and the four deepest
leaves are labelled.

Usage: fig_disconnectivity.py [--data DIR] out.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figstyle
import system_visuals

SIZE = 38
SEED = 11
CAP = 1200
EMAX = -170.0
CN_CUTOFF = 1.5
DEDUP = 0.15
KNN = 6
LEVELS = 48
BASINS = 5
LABELLED = 4
NAMES = {"fcc": "fcc global minimum", "icosahedral": "icosahedral minimum"}
# Given in order of the deepest minimum of each basin, so the pale sunshine falls on the shallowest.
PALETTE = (figstyle.SKY, figstyle.MAGENTA, figstyle.CORAL, figstyle.TEAL, figstyle.SUNSHINE)
HEIGHT = 3.9


def _rows(text: str):
    import numpy as np

    rows = [line.split() for line in text.splitlines()]
    return [(float(row[0]), np.array([float(v) for v in row[1:]])) for row in rows if len(row) >= 4]


def tree(data: Path):
    """The merge tree, its basins, the energies and descriptors of its minima, and the references."""
    import numpy as np
    import sketchmap_landscapes as run
    from chemparseplot.plot.disconnectivity import (
        dedup,
        knn_graph,
        layout,
        merge_tree,
        spectral_basins,
    )
    from chemparseplot.plot.landscape import cn_matrix

    _, digests = system_visuals.provenance(data)
    arms, refs = run.SIZES[SIZE]
    rng = np.random.default_rng(SEED)
    references = [_rows(system_visuals._read(data, digests, f"{run.REFS}/{name}"))[0] for _, name in refs]
    pooled = [row for _, name in arms for row in _rows(system_visuals._read(data, digests, f"{run.RUNS}/{name}"))]
    energies = np.array([energy for energy, _ in pooled])
    if len(pooled) > CAP:
        pick = rng.choice(len(pooled), size=CAP, replace=False)
        if energies.argmin() not in pick:
            pick[0] = energies.argmin()
        pooled = [pooled[i] for i in pick]
    pooled = [row for row in pooled if row[0] <= EMAX]
    structures = [coords for _, coords in references + pooled]
    energies = np.array([energy for energy, _ in references + pooled])
    matrix = cn_matrix(structures, ["X"] * SIZE, CN_CUTOFF, None)
    ref_matrix = matrix[: len(references)].copy()
    kept = dedup(matrix, energies, DEDUP)
    matrix, energies = matrix[kept], energies[kept]
    adjacency = knn_graph(matrix, KNN)
    root = merge_tree(adjacency, energies, np.linspace(energies.min(), energies.max(), LEVELS)[1:])
    layout(root)
    basins = spectral_basins(adjacency, BASINS, rng, matrix=matrix)
    names = [NAMES[label] for label, _ in refs]
    return root, basins, energies, matrix, ref_matrix, names, len(pooled), len(kept)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=system_visuals.DATA)
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    system_visuals.check_chemparseplot()
    root, basins, energies, matrix, ref_matrix, names, pooled, distinct = tree(args.data)
    figstyle.setup()
    import matplotlib.pyplot as plt
    import numpy as np
    from chemparseplot.plot.disconnectivity import draw, iter_nodes, paint
    from matplotlib.lines import Line2D

    def kind(i: int) -> str:
        return names[int(np.argmin([np.linalg.norm(matrix[i] - ref) for ref in ref_matrix]))]

    fig, ax = plt.subplots(figsize=(figstyle.DOUBLE, HEIGHT))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.98, bottom=0.03)
    deepest = {int(b): int(np.where(basins == b)[0][np.argmin(energies[basins == b])]) for b in np.unique(basins)}
    order = sorted(deepest, key=lambda b: energies[deepest[b]])
    colours = [figstyle.GREY] * (int(basins.max()) + 1)
    for rank, b in enumerate(order):
        colours[b] = PALETTE[rank % len(PALETTE)]
    draw(ax, root, paint(root, basins, colours), lw=0.8)
    e0 = float(energies.min())
    handles = [Line2D([], [], color=colours[b], lw=1.6,
                      label=f"basin of the {kind(deepest[b])}, \N{GREEK CAPITAL LETTER DELTA}E = {energies[deepest[b]] - e0:.3f}")
               for b in order]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 0.07), fontsize=figstyle.SMALLEST,
              handlelength=1.6, borderaxespad=0.2)
    leaves = {node.leaf: node for node in iter_nodes(root) if node.leaf is not None}
    span = float(np.ptp(energies))
    for rank, i in enumerate(np.argsort(energies)[:LABELLED]):
        ax.annotate(f"{kind(i)}, \N{GREEK CAPITAL LETTER DELTA}E = {energies[i] - e0:.3f}",
                    xy=(leaves[i].x, energies[i]), xytext=(0.04 * len(leaves), e0 - (0.05 + 0.05 * rank) * span),
                    ha="left", va="center", fontsize=figstyle.SMALLEST,
                    arrowprops={"arrowstyle": "-", "color": figstyle.GREY, "lw": 0.5,
                                "connectionstyle": "angle,angleA=0,angleB=90,rad=3"},
                    annotation_clip=False)
    ax.text(0.99, 0.02, f"{pooled} minima at or below \N{MINUS SIGN}170 \N{GREEK SMALL LETTER EPSILON} and 2 references, "
            f"{distinct} distinct", transform=ax.transAxes, ha="right", va="bottom", fontsize=figstyle.SMALLEST)
    ax.margins(y=0.08)
    ax.set_ylim(e0 - (0.1 + 0.05 * LABELLED) * span, None)
    ax.set_yticks(np.arange(np.floor(e0), np.floor(energies.max()) + 1))
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_ylabel("energy (\N{GREEK SMALL LETTER EPSILON})")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}".replace("-", "\N{MINUS SIGN}"))
    figstyle.save(fig, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Draw fig:landscapes: the sketch-map landscape, its minima and the landfold map of LJ38, LJ75 and LJ98.

Each column is one size: the projection of the paper at 028d995 as sketchmap_landscapes.py
wrote it into data/system_visuals/sketchmap/, read by system_visuals.py, which checks each
landmark against its line of its dump; and under it the landfold map of the same minima,
the fold that the occupancy book of anneal makes of them after the sigmoid switch. The drawing is that of campaign_landscape.py of that
commit: the energy of the landmarks interpolated cubically over the plane, smoothed by a
gaussian of four grid steps and clipped to the 97th percentile, as filled contours with
thin lines; the landmarks of each arm; the deepest landmark of each arm ringed; the
pinned references; under each map the landmarks of each arm and the energy of its deepest
above the global minimum; and the four deepest landmarks that are not references and stand
apart by a fifth of the span of the first coordinate, numbered on the map and drawn below
it in the style of system_visuals.py, each named by the reference whose coordination
descriptor is within half of sigma of its own. The surface is in the ramp of the Ruhi
palette, whose lightness rises with the energy.

Usage: fig_landscapes.py [--data DIR] out.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figstyle
import system_visuals

# The marker, fill and edge of the landmarks of each arm, and the marker, fill and area of each reference.
ARM_STYLE = {
    "basin hopping": ("o", figstyle.CORAL, "white"),
    "recommended": ("^", figstyle.MAGENTA, "white"),
    "random structure search": ("s", "white", figstyle.BLACK),
}
REF_STYLE = (("*", figstyle.SUNSHINE, 70), ("D", "white", 22))
# The reference minima, named by their structure.
REF_NAME = {"fcc": "fcc", "icosahedral": "icosahedral", "Marks": "decahedral", "Leary": "tetrahedral"}
GALLERY = 4
GRID = 240
LEVELS = 24
CN_CUTOFF = 1.5
TOP = 0.62
MAP = 1.62
GAP = 0.62
BAR = 0.06
SCALE = 0.36
LINE = 0.115
CELL = 0.86
FOLD = 0.95
# Under the fold: its axis labels and the key of its markers.
FOOT = 0.62
FOLD_KEY = (
    ({"s": 12, "color": figstyle.SKY, "alpha": 0.55, "linewidths": 0}, "largest community"),
    ({"s": 12, "color": figstyle.CORAL, "alpha": 0.9, "linewidths": 0}, "other communities"),
    ({"s": 40, "facecolors": "none", "edgecolors": figstyle.BLACK, "linewidths": 0.8}, "lowest family"),
    ({"s": 15, "color": figstyle.GREY, "linewidths": 0}, "a family of 10 minima"),
)


def surface(coords, energies, vmin, vmax):
    """The surface of campaign_landscape.py: cubic, linear where cubic leaves a hole, smoothed."""
    import numpy as np
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter

    gx, gy = np.meshgrid(np.linspace(coords[:, 0].min(), coords[:, 0].max(), GRID),
                         np.linspace(coords[:, 1].min(), coords[:, 1].max(), GRID))
    z = np.clip(energies, vmin, vmax)
    cubic = griddata(coords, z, (gx, gy), method="cubic")
    cubic = np.where(np.isnan(cubic), griddata(coords, z, (gx, gy), method="linear"), cubic)
    hole = np.isnan(cubic)
    smoothed = gaussian_filter(np.where(hole, np.nanmean(cubic), cubic), sigma=4.0)
    return gx, gy, np.where(hole, np.nan, np.clip(smoothed, vmin, vmax))


def gallery(landscape):
    """The four deepest landmarks that are not references, each a fifth of the span apart."""
    import numpy as np

    coords = np.array([(m.x, m.y) for m in landscape.landmarks])
    span = np.ptp(coords[:, 0])
    chosen = []
    for k in np.argsort([m.energy for m in landscape.landmarks]):
        if landscape.landmarks[k].label.startswith("ref:"):
            continue
        if all(np.linalg.norm(coords[k] - coords[j]) > 0.2 * span for j in chosen):
            chosen.append(int(k))
        if len(chosen) == GALLERY:
            break
    return chosen


def kind(landscape, coords, data) -> str:
    """The reference within half of sigma of a minimum in the coordination descriptor, if any."""
    import numpy as np
    from chemparseplot.plot.landscape import cn_matrix

    refs = [m for m in landscape.landmarks if m.label.startswith("ref:")]
    rows = [np.ravel(system_visuals.landmark_coords(m, data)) for m in refs] + [np.ravel(coords)]
    matrix = cn_matrix(rows, ["X"] * landscape.size, CN_CUTOFF, None)
    distances = [float(np.linalg.norm(matrix[-1] - matrix[k])) for k in range(len(refs))]
    k = int(np.argmin(distances))
    return REF_NAME[refs[k].label[4:]] if distances[k] <= 0.5 * landscape.sigma else ""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=system_visuals.DATA)
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    system_visuals.check_chemparseplot()
    landscapes = system_visuals.landscapes(args.data)
    folds = {fold.size: fold for fold in system_visuals.folds(args.data)}
    published = {int(row["size"]): float(row["energy"]) for row in system_visuals.references(args.data)
                 if row["class"] == "Lennard-Jones"}
    figstyle.setup()
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    cmap = figstyle.ramp()
    width = figstyle.DOUBLE / len(landscapes)
    stats = LINE * max(len(landscape.arms) for landscape in landscapes) + 0.08
    fold_block = FOOT + FOLD + 0.42
    height = TOP + MAP + GAP + BAR + SCALE + stats + 2 * CELL + fold_block
    fig = plt.figure(figsize=(figstyle.DOUBLE, height))

    def box(x, y, w, h):
        return fig.add_axes((x / figstyle.DOUBLE, y / height, w / figstyle.DOUBLE, h / height))

    panels = []
    for c, landscape in enumerate(landscapes):
        left = c * width
        marks = landscape.landmarks
        coords = np.array([(m.x, m.y) for m in marks])
        energies = np.array([m.energy for m in marks])
        labels = [m.label for m in marks]
        e0 = float(energies.min())
        if marks[0].energy != e0 or abs(e0 - published[landscape.size]) > 1e-6:
            raise SystemExit(f"fig_landscapes: the first reference of LJ{landscape.size} is not its global minimum")
        de = energies - e0
        vmax = float(np.nanpercentile(de, 97))
        ax = box(left + 0.36, height - TOP - MAP, width - 0.46, MAP)
        gx, gy, z = surface(coords, de, 0.0, vmax)
        levels = np.linspace(0.0, vmax, LEVELS)
        filled = ax.contourf(gx, gy, z, levels=levels, cmap=cmap, vmin=0.0, vmax=vmax, alpha=0.85)
        ax.contour(gx, gy, z, levels=levels, colors=figstyle.BLACK, linewidths=0.15, alpha=0.45)
        for arm in landscape.arms:
            marker, face, edge = ARM_STYLE[arm]
            own = np.array([label == arm for label in labels])
            ax.scatter(coords[own, 0], coords[own, 1], marker=marker, s=3.5, color=face, alpha=0.9,
                       edgecolors=edge, linewidths=0.15 if edge == "white" else 0.3, zorder=3)
        for n, arm in enumerate(landscape.arms):
            own = [k for k, label in enumerate(labels) if label == arm]
            best = min(own, key=lambda k: energies[k])
            fig.text((left + width / 2) / figstyle.DOUBLE,
                     (height - TOP - MAP - GAP - BAR - SCALE - n * LINE) / height,
                     f"{arm} {len(own)}, deepest +{de[best]:.3f}", ha="center", va="top",
                     fontsize=figstyle.SMALLEST)
            ax.scatter([coords[best, 0]], [coords[best, 1]], marker=ARM_STYLE[arm][0], s=42, facecolors="none",
                       edgecolors=figstyle.BLACK, linewidths=0.9, zorder=6)
        for k, ref in enumerate(landscape.references):
            marker, color, size = REF_STYLE[k]
            at = labels.index(f"ref:{ref}")
            ax.scatter([coords[at, 0]], [coords[at, 1]], marker=marker, s=size, color=color,
                       edgecolors=figstyle.BLACK, linewidths=0.6, zorder=7)
        lo, hi = coords.min(axis=0), coords.max(axis=0)
        pad = 0.06 * (hi - lo)
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.set_xlabel("sketch-map 1", labelpad=1)
        if c == 0:
            ax.set_ylabel("sketch-map 2", labelpad=1)
        ax.tick_params(pad=1.5)
        ax.text(0.0, 1.03, f"({'abc'[c]}) {landscape.label}", transform=ax.transAxes, ha="left", va="bottom",
                fontsize=figstyle.BASE)
        bar = box(left + 0.36, height - TOP - MAP - GAP, width - 0.46, BAR)
        colorbar = fig.colorbar(filled, cax=bar, orientation="horizontal")
        colorbar.set_ticks(np.linspace(0.0, vmax, 4))
        colorbar.ax.xaxis.set_major_formatter(lambda v, _: f"{v:.1f}")
        colorbar.ax.tick_params(pad=1.5, length=2)
        colorbar.outline.set_linewidth(0.4)
        colorbar.set_label("energy above the global minimum (\N{GREEK SMALL LETTER EPSILON})", labelpad=1)
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        middle = (sum(xlim) / 2, sum(ylim) / 2)
        for n, k in enumerate(gallery(landscape), 1):
            point = coords[k]
            offset = (9 if point[0] <= middle[0] else -9, 9 if point[1] <= middle[1] else -9)
            ax.annotate(str(n), xy=tuple(point), xytext=offset, textcoords="offset points", ha="center",
                        va="center", fontsize=figstyle.SMALLEST, zorder=9,
                        bbox={"boxstyle": "circle,pad=0.12", "facecolor": "white", "edgecolor": figstyle.BLACK,
                              "linewidth": 0.6},
                        arrowprops={"arrowstyle": "-", "color": figstyle.BLACK, "linewidth": 0.5,
                                    "shrinkA": 4, "shrinkB": 1.5})
            atoms = system_visuals.landmark_coords(marks[k], args.data)
            name = kind(landscape, atoms, args.data)
            structure = system_visuals.Structure(
                f"lj{landscape.size}_landmark{k}", "", "Lennard-Jones", ("X",) * len(atoms), tuple(atoms),
                marks[k].energy, "\N{GREEK SMALL LETTER EPSILON}", "tilt")
            row, col = divmod(n - 1, 2)
            bottom = fold_block + (1 - row) * CELL
            cell = box(left + col * width / 2 + 0.08, bottom + 0.24, width / 2 - 0.16, CELL - 0.3)
            cell.set_axis_off()
            panels.append((cell, system_visuals.render(structure)))
            head = f"{n}: {name}" if name else str(n)
            fig.text((left + (col + 0.5) * width / 2) / figstyle.DOUBLE, (bottom + 0.1) / height,
                     f"{head}, \N{GREEK CAPITAL LETTER DELTA}E = {de[k]:.3f}", ha="center", va="center",
                     fontsize=figstyle.SMALLEST)
        fold = folds[landscape.size]
        ax = box(left + 0.42, FOOT, width - 0.52, FOLD)
        system_visuals.draw_fold(ax, fold, ylabel=c == 0)
        ax.tick_params(pad=1.5)
        ax.set_title(f"{fold.minima} minima, {len(fold.families)} families, {fold.communities} communities",
                     fontsize=figstyle.SMALLEST, pad=3)
    second = next(landscape for landscape in landscapes if len(landscape.references) > 1)
    handles = [Line2D([], [], linestyle="none", marker=marker, color=face, markeredgecolor=edge,
                      markeredgewidth=0.2 if edge == "white" else 0.4, markersize=3.5, label=arm)
               for arm, (marker, face, edge) in ARM_STYLE.items()]
    handles.append(Line2D([], [], linestyle="none", marker="o", markerfacecolor="none",
                          markeredgecolor=figstyle.BLACK, markersize=6, markeredgewidth=0.9,
                          label="deepest landmark of each arm, in its marker"))
    handles += [Line2D([], [], linestyle="none", marker=marker, color=color, markeredgecolor=figstyle.BLACK,
                       markeredgewidth=0.6, markersize=7 if marker == "*" else 4, label=label)
                for (marker, color, _), label in zip(REF_STYLE, ("global minimum, pinned",
                                                                  f"{REF_NAME[second.references[1]]} minimum of {second.label}, pinned"))]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), fontsize=figstyle.SMALLEST,
               handletextpad=0.3, columnspacing=1.2, borderaxespad=0.1)
    key = fig.add_axes((0.0, 0.0, 1.0, 0.16 / height))
    key.set_axis_off()
    key.set_xlim(0, figstyle.DOUBLE)
    key.set_ylim(0, 1)
    for j, (style, text) in enumerate(FOLD_KEY):
        x = 0.9 + 1.4 * j
        key.scatter([x], [0.5], **style)
        key.text(x + 0.12, 0.5, text, va="center", fontsize=figstyle.SMALLEST)
    system_visuals.place(fig, panels, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

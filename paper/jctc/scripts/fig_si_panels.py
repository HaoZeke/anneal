#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Draw the supporting panels from the same sealed inputs as the main figures.

lj38_sketchmap.png is the LJ38 landmark scatter, coloured by energy on the Ruhi ramp.
lj38_landscape.png, lj75_landscape.png and lj98_landscape.png are the energy surfaces.
occupancy_landfold.pdf collects the three occupancy folds.
occupancy_landfold_lj38.pdf, occupancy_landfold_lj75.pdf and occupancy_landfold_lj98.pdf
are those folds one size at a time.

Usage: fig_si_panels.py [--data DIR] OUT_DIR
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fig_landscapes
import figstyle
import system_visuals


def _landscape_png(landscape, published: float, out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    figstyle.setup()
    marks = landscape.landmarks
    coords = np.array([(m.x, m.y) for m in marks])
    energies = np.array([m.energy for m in marks])
    labels = [m.label for m in marks]
    e0 = float(energies.min())
    if abs(e0 - published) > 1e-6:
        raise SystemExit(f"fig_si_panels: LJ{landscape.size} does not pin the published minimum")
    de = energies - e0
    vmax = float(np.nanpercentile(de, 97))
    fig, ax = plt.subplots(figsize=(figstyle.SINGLE, 2.6), layout="constrained")
    gx, gy, z = fig_landscapes.surface(coords, de, 0.0, vmax)
    levels = np.linspace(0.0, vmax, fig_landscapes.LEVELS)
    filled = ax.contourf(gx, gy, z, levels=levels, cmap=figstyle.ramp(), vmin=0.0, vmax=vmax, alpha=0.85)
    ax.contour(gx, gy, z, levels=levels, colors=figstyle.BLACK, linewidths=0.15, alpha=0.45)
    for arm in landscape.arms:
        marker, face, edge = fig_landscapes.ARM_STYLE[arm]
        own = np.array([label == arm for label in labels])
        ax.scatter(
            coords[own, 0],
            coords[own, 1],
            marker=marker,
            s=8,
            color=face,
            edgecolors=edge,
            linewidths=0.2,
            zorder=3,
            label=arm,
        )
    ax.set_xlabel("sketch-map 1")
    ax.set_ylabel("sketch-map 2")
    bar = fig.colorbar(filled, ax=ax, fraction=0.046, pad=0.04)
    bar.set_label("energy above the global minimum")
    ax.legend(loc="best", fontsize=figstyle.SMALLEST)
    fig.savefig(out, format="png", dpi=200)
    plt.close(fig)


def _sketch_png(landscape, out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    figstyle.setup()
    marks = landscape.landmarks
    coords = np.array([(m.x, m.y) for m in marks])
    energies = np.array([m.energy for m in marks])
    fig, ax = plt.subplots(figsize=(figstyle.SINGLE, 2.4), layout="constrained")
    points = ax.scatter(coords[:, 0], coords[:, 1], c=energies, cmap=figstyle.ramp(), s=12, linewidths=0)
    bar = fig.colorbar(points, ax=ax, fraction=0.046, pad=0.04)
    bar.set_label("quenched energy")
    ax.set_xlabel("sketch-map 1")
    ax.set_ylabel("sketch-map 2")
    fig.savefig(out, format="png", dpi=200)
    plt.close(fig)


def _fold_pdf(fold, out: Path) -> None:
    import matplotlib.pyplot as plt

    figstyle.setup()
    fig, ax = plt.subplots(figsize=(figstyle.SINGLE, 2.2), layout="constrained")
    system_visuals.draw_fold(ax, fold, ylabel=True)
    figstyle.save(fig, out)


def draw(data: Path, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    system_visuals.check_chemparseplot()
    landscapes = system_visuals.landscapes(data)
    folds = {fold.size: fold for fold in system_visuals.folds(data)}
    published = {
        int(row["size"]): float(row["energy"])
        for row in system_visuals.references(data)
        if row["class"] == "Lennard-Jones"
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    by_size = {landscape.size: landscape for landscape in landscapes}
    _sketch_png(by_size[38], out_dir / "lj38_sketchmap.png")
    for size in (38, 75, 98):
        _landscape_png(by_size[size], published[size], out_dir / f"lj{size}_landscape.png")
        _fold_pdf(folds[size], out_dir / f"occupancy_landfold_lj{size}.pdf")
    figstyle.setup()
    fig, axes = plt.subplots(1, 3, figsize=(figstyle.DOUBLE, 2.15), layout="constrained")
    for ax, size in zip(axes, (38, 75, 98)):
        system_visuals.draw_fold(ax, folds[size], ylabel=size == 38)
        ax.set_title(folds[size].label, fontsize=figstyle.SMALLEST)
    figstyle.save(fig, out_dir / "occupancy_landfold.pdf")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=system_visuals.DATA)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args(argv)
    draw(args.data, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

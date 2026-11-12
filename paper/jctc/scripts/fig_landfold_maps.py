#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Draw the landfold maps of the C26 campaigns from sealed landscape.ld / fes.csv.

Reads data/landfold_maps_v7/: PROVENANCE.json of every campaign that wrote a map, held to
the digests of provenance.txt. When landscape.ld is present it is drawn as a scatter on
the Ruhi free-energy ramp, recomputed on a uniform grid from the sealed coordinates when
fes.csv is missing or failed. When only the sealed landscape.png of the handoff is present,
that PNG is shown. A map the seal skips is no panel.

Usage: fig_landfold_maps.py [--data DIR] OUT.pdf
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figstyle
from paper_data import fetch_paper_data
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "landfold_maps_v7"
JOBS = (
    ("lj_csm/lj38/derived", "LJ\N{SUBSCRIPT THREE}\N{SUBSCRIPT EIGHT} derived"),
    ("lj_csm/lj38/recommended", "LJ\N{SUBSCRIPT THREE}\N{SUBSCRIPT EIGHT} recommended"),
    ("lj_csm/lj55/derived", "LJ\N{SUBSCRIPT FIVE}\N{SUBSCRIPT FIVE} derived"),
    ("lj_csm/lj55/recommended", "LJ\N{SUBSCRIPT FIVE}\N{SUBSCRIPT FIVE} recommended"),
    ("lj_csm/lj75/derived", "LJ\N{SUBSCRIPT SEVEN}\N{SUBSCRIPT FIVE} derived"),
    ("lj_csm/lj75/recommended", "LJ\N{SUBSCRIPT SEVEN}\N{SUBSCRIPT FIVE} recommended"),
    ("lj_csm/lj98/derived", "LJ\N{SUBSCRIPT NINE}\N{SUBSCRIPT EIGHT} derived"),
    ("lj_csm_c26/lj98/recommended", "LJ\N{SUBSCRIPT NINE}\N{SUBSCRIPT EIGHT} recommended"),
    ("tip4p/hexamer/for_tip4p", "TIP4P hexamer"),
)
KINDS = ("representative", "aggregate")
COLUMNS = 4
KT = 0.2


def die(message: str) -> NoReturn:
    raise SystemExit(f"fig_landfold_maps: {message}")


@dataclass(frozen=True)
class Map:
    job: str
    name: str
    kind: str
    seeds: str
    rows: int
    dimension: int
    grid: int
    folder: Path


def provenance(data: Path) -> tuple[dict[str, str], dict[str, str]]:
    keys: dict[str, str] = {}
    digests: dict[str, str] = {}
    for line in (data / "provenance.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        first, _, rest = line.partition(" ")
        if len(first) == 64 and all(c in "0123456789abcdef" for c in first):
            digests[rest.strip()] = first
        else:
            keys[first] = rest.strip()
    return keys, digests


def load(data: Path = DATA) -> tuple[dict[str, str], list[Map], list[str]]:
    fetch_paper_data()
    keys, digests = provenance(data)
    for name, want in digests.items():
        path = data / name
        if not path.is_file():
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != want:
            die(f"{name} does not match the digest provenance.txt records")
    maps, skipped = [], []
    for job, name in JOBS:
        seal = json.loads((data / job / "PROVENANCE.json").read_text())
        if not seal["run"] or seal["landfold_commit"] != keys["landfold_commit"]:
            die(f"{job} is not a run of landfold {keys['landfold_commit'][:8]}")
        for kind in KINDS:
            record = seal.get(kind) or {}
            if record.get("map") != "wrote landscape.png and fes.png":
                skipped.append(f"{name} {kind}")
                continue
            folder = data / job / kind
            if not (folder / "landscape.ld").exists() and not (folder / "landscape.png").exists():
                die(f"{job}/{kind}: neither landscape.ld nor landscape.png is present")
            seeds = f"seed {seal['first_hit_seed']}" if kind == "representative" else f"{seal['seeds_finished']} seeds"
            maps.append(
                Map(
                    job,
                    name,
                    kind,
                    seeds,
                    record["rows"],
                    record["dimension"],
                    record["fes_nx"],
                    folder,
                )
            )
    return keys, maps, skipped


def read_landscape(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = [float(x) for x in line.split()]
        if len(parts) < 2:
            continue
        rows.append(parts[:2])
    if not rows:
        die(f"{path} holds no coordinates")
    return np.asarray(rows, dtype=float)


def read_fes(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    xs, ys, fs = [], [], []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        x, y, f = float(parts[0]), float(parts[1]), float(parts[2])
        if not np.isfinite(f):
            continue
        xs.append(x)
        ys.append(y)
        fs.append(f)
    if len(fs) < 4:
        return None
    x = np.asarray(xs)
    y = np.asarray(ys)
    f = np.asarray(fs)
    ux, uy = np.unique(np.round(x, 12)), np.unique(np.round(y, 12))
    if len(ux) < 2 or len(uy) < 2:
        return None
    # Require a rectangular grid.
    if len(x) != len(ux) * len(uy):
        return None
    order = np.lexsort((x, y))
    grid = f[order].reshape(len(uy), len(ux))
    return np.sort(ux), np.sort(uy), grid


def fes_from_points(coords: np.ndarray, nx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniform-grid free energy F = -kT ln(rho/rho_max) from the sealed embedding."""
    nx = max(8, int(nx) or 48)
    x, y = coords[:, 0], coords[:, 1]
    pad_x = 0.05 * (x.max() - x.min() or 1.0)
    pad_y = 0.05 * (y.max() - y.min() or 1.0)
    xedges = np.linspace(x.min() - pad_x, x.max() + pad_x, nx + 1)
    yedges = np.linspace(y.min() - pad_y, y.max() + pad_y, nx + 1)
    hist, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
    rho = hist.T
    floor = max(rho.max() * 0.05, 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        fes = -KT * np.log(np.maximum(rho, floor) / rho.max())
    fes = np.where(rho > 0, fes, np.nan)
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    return xc, yc, fes


def draw_panel(ax, entry: Map):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    ld = entry.folder / "landscape.ld"
    fes_path = entry.folder / "fes.csv"
    png = entry.folder / "landscape.png"
    ax.set_title(f"{entry.name}\n{entry.kind}, {entry.seeds}", fontsize=figstyle.SMALLEST, linespacing=1.1)
    if ld.exists():
        coords = read_landscape(ld)
        if len(coords) != entry.rows:
            die(f"{ld}: {len(coords)} rows, seal records {entry.rows}")
        grid = None
        if fes_path.exists():
            grid = read_fes(fes_path)
        if grid is None:
            grid = fes_from_points(coords, entry.grid)
        xc, yc, fes = grid
        finite = fes[np.isfinite(fes)]
        vmax = float(finite.max()) if finite.size else 1.0
        norm = Normalize(0.0, vmax if vmax > 0 else 1.0)
        mesh = ax.pcolormesh(xc, yc, fes, cmap=figstyle.ramp(), norm=norm, shading="auto")
        ax.scatter(coords[:, 0], coords[:, 1], s=2.5, c=figstyle.BLACK, linewidths=0, alpha=0.35, zorder=3)
        # Mark the lowest free-energy cell.
        if finite.size:
            jy, jx = np.unravel_index(np.nanargmin(fes), fes.shape)
            ax.plot(xc[jx], yc[jy], marker="*", color=figstyle.CORAL, markersize=5, zorder=4)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("landfold coordinate 1", fontsize=figstyle.SMALLEST, labelpad=1)
        ax.set_ylabel("landfold coordinate 2", fontsize=figstyle.SMALLEST, labelpad=1)
        return mesh
    # Sealed campaign PNG for maps whose tables are still on the campaign disk.
    image = plt.imread(png)
    ax.imshow(image, origin="upper")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(True)
        side.set_color(figstyle.GREY)
    ax.set_xlabel("landfold coordinate 1", fontsize=figstyle.SMALLEST, labelpad=1)
    ax.set_ylabel("landfold coordinate 2", fontsize=figstyle.SMALLEST, labelpad=1)
    return plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=figstyle.ramp())


# Late import annotation for type checkers; matplotlib is imported inside draw.
import matplotlib  # noqa: E402


def draw(data: Path, out: Path) -> None:
    import matplotlib.pyplot as plt

    figstyle.setup()
    _, maps, _ = load(data)
    rows = -(-len(maps) // COLUMNS)
    fig, axes = plt.subplots(rows, COLUMNS, figsize=(figstyle.DOUBLE, 1.62 * rows + 0.55), layout="constrained")
    flat = axes.ravel()
    mappable = None
    for ax, entry in zip(flat, maps):
        mappable = draw_panel(ax, entry) or mappable
    for ax in flat[len(maps) :]:
        ax.set_axis_off()
    if mappable is not None:
        bar = fig.colorbar(mappable, ax=list(flat[: len(maps)]), orientation="horizontal", location="bottom", shrink=0.5, aspect=40, pad=0.01)
        bar.solids.set_rasterized(False)
        bar.solids.set_edgecolor("face")
        bar.set_label("free energy of the map, lowest to highest", fontsize=figstyle.SMALLEST)
        bar.outline.set_linewidth(0.4)
    figstyle.save(fig, out)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--list", action="store_true", help="print the maps drawn and the maps skipped")
    parser.add_argument("out", type=Path, nargs="?")
    args = parser.parse_args(argv)
    if args.list:
        _, maps, skipped = load(args.data)
        print("\n".join([f"map {m.name} {m.kind} {m.rows}" for m in maps] + [f"skipped {s}" for s in skipped]))
        return
    if args.out is None:
        parser.error("OUT.pdf is required")
    draw(args.data, args.out)


if __name__ == "__main__":
    main(sys.argv[1:])

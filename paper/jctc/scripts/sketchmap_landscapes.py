#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Project the Lennard-Jones landscape runs with sketch-map, as campaign_landscape.py did at 028d995.

The inputs are the dump selections of data/landscapes_7695255/ whose digests its
figure provenance records, and the reference minima that script pinned, copied byte for
byte from scripts/ of 028d995 into data/system_visuals/sketchmap/. The steps are those of
the script of record, in its order and with its random stream: at most CAP minima of each
arm, its deepest kept; the references and each arm's deepest minimum pinned as the first
landmarks; the sorted smoothed coordination numbers of chemparseplot as the descriptor;
the median distance of 400 rows drawn from that stream as sigma; and dimlandmark then
dimred of lab-cosmo/sketchmap through chemparseplot.plot.landscape.run_sketchmap.

It writes one CSV per size into the output folder, a row per landmark in projection
order: the label of its arm or reference, the file under data/ and the line it came from,
its energy, and the two coordinates dimred printed. The sigma of each size goes to
stdout, for provenance.txt.

Usage: sketchmap_landscapes.py --sketchmap-bin DIR --lapack-lib DIR [--workdir DIR] out-dir
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CAP = 800
LANDMARKS = 600
SEED = 7
CN_CUTOFF = 1.5
SIGMA_SAMPLE = 400
CHEMPARSEPLOT = "1.11.0"
RUNS = "landscapes_7695255"
REFS = "system_visuals/sketchmap"
# The arms and pinned references of the three figures of record, in the order the script
# of record took them.
SIZES = {
    38: (
        (("basin hopping", "lj38_bh_s0.min"), ("recommended", "lj38_recommended_s0.min"),
         ("random structure search", "lj38_rss_s0.min")),
        (("fcc", "ref_lj38_fcc.min"), ("icosahedral", "ref_lj38_ico.min")),
    ),
    75: (
        (("basin hopping", "lj75_bh_s0.min"), ("recommended", "lj75_recommended_s0.min")),
        (("Marks", "ref_lj75_marks.min"),),
    ),
    98: (
        (("basin hopping", "lj98_bh_s0.min"), ("recommended", "lj98_recommended_s0.min")),
        (("Leary", "ref_lj98_tetra.min"),),
    ),
}
COLUMNS = ("landmark", "label", "source", "line", "energy", "x", "y")


def rows_of(path: Path) -> list[tuple[int, float, np.ndarray]]:
    """Each minimum of a dump: its line, its energy and its flat coordinates."""
    import sealed_parts

    path = sealed_parts.materialize(path)
    out = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        fields = line.split()
        if len(fields) < 4:
            continue
        out.append((number, float(fields[0]), np.array([float(v) for v in fields[1:]])))
    return out


def capped(rows: list, rng: np.random.Generator) -> list:
    """At most CAP rows, drawn as the script of record drew them, the deepest kept."""
    if len(rows) <= CAP:
        return rows
    energies = np.array([energy for _, energy, _ in rows])
    pick = rng.choice(len(rows), size=CAP, replace=False)
    if energies.argmin() not in pick:
        pick[0] = energies.argmin()
    return [rows[i] for i in pick]


def project(size: int, sketchmap_bin: Path, lapack_lib: Path, workdir: Path):
    from chemparseplot.plot.landscape import cn_matrix, run_sketchmap

    arms, refs = SIZES[size]
    rng = np.random.default_rng(SEED)
    entries = []
    for label, name in refs:
        number, energy, coords = rows_of(DATA / REFS / name)[0]
        entries.append((f"ref:{label}", f"{REFS}/{name}", number, energy, coords))
    for label, name in arms:
        for number, energy, coords in capped(rows_of(DATA / RUNS / name), rng):
            entries.append((label, f"{RUNS}/{name}", number, energy, coords))
    pinned = list(range(len(refs)))
    for label, _ in arms:
        own = [i for i, entry in enumerate(entries) if entry[0] == label]
        pinned.append(min(own, key=lambda i: entries[i][3]))
    order = pinned + [i for i in range(len(entries)) if i not in set(pinned)]
    entries = [entries[i] for i in order]
    matrix = cn_matrix([entry[4] for entry in entries], ["X"] * size, CN_CUTOFF, None)
    sub = matrix[rng.choice(len(matrix), size=min(len(matrix), SIGMA_SAMPLE), replace=False)]
    distances = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=-1)
    sigma = float(np.median(distances[np.triu_indices(len(sub), 1)]))
    coords, idx = run_sketchmap(
        matrix, sketchmap_bin, lapack_lib, min(LANDMARKS, len(matrix)), sigma,
        workdir / f"lj{size}", n_pinned=len(pinned),
    )
    rows = [
        (k, entries[i][0], entries[i][1], entries[i][2], repr(entries[i][3]), repr(float(x)), repr(float(y)))
        for k, (i, (x, y)) in enumerate(zip(idx, coords))
    ]
    return sigma, rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sketchmap-bin", type=Path, required=True)
    parser.add_argument("--lapack-lib", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, default=Path("smwork"))
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    from importlib.metadata import version

    if version("chemparseplot") != CHEMPARSEPLOT:
        raise SystemExit(f"sketchmap_landscapes: chemparseplot {version('chemparseplot')}, not {CHEMPARSEPLOT}")
    args.out.mkdir(parents=True, exist_ok=True)
    for size in SIZES:
        sigma, rows = project(size, args.sketchmap_bin, args.lapack_lib, args.workdir)
        with (args.out / f"lj{size}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(COLUMNS)
            writer.writerows(rows)
        print(f"sketchmap_lj{size}_sigma {sigma!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

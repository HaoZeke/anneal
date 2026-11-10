#!/usr/bin/env python3
# Copyright (c) anneal developers
"""The structures and maps of the benchmark systems, for the figures of the main text and the SI.

Reads rewrite_2026/data/system_visuals/: structures/ holds the reference minima that the
tests of anneal check, copied byte for byte from tests/fixtures/ of the published anneal
commit, and landfold/ the output of occupancy_landfold_book of that commit on the
archived minima dumps of data/landscapes_7695255/. provenance.txt records the commit,
the build, the command of each map and the digest of every file this script reads,
including the water geometries of data/hq_sci_h2o6/, data/w6gofee4k/ and data/w6nwpbe/.

ccd/ holds the coordinates of the Cambridge Cluster Database for every other reference of
the tables of the Supporting Information that needs only coordinates, at the paths of
their URLs, and references.csv the published energy of each reference structure, as the
table of the database it names prints it. sketchmap/ holds the reference minima the
sketch-map runs of the paper at 028d995 pinned and the projection of each size that
sketchmap_landscapes.py writes.

Each reference structure is at its published energy, recomputed from its coordinates: the
12-6 energy of a Lennard-Jones file to 1e-5 epsilon; a Morse file, which the database
stores at the range of its first line, relaxed at the range of its row and then to 1e-5
epsilon; the
Sutton--Chen 10-8 energy of a Pt file to 1e-4 epsilon; and the TIP4P energy of rigid
molecules within 5e-3 kJ/mol and no more than 1e-3 above the published value, the test of
anneal, since the files round the coordinates to seven places. A hexamer of each class is
the lowest of the archived minima of one engine whose oxygens make that many contacts
under 3.2 Angstrom, the convention of the verifier: 9 prism, 8 cage, 7 book, 6 ring. A
map is one size's fold, whose minima count is the rows of the dumps it read. A landmark of
a sketch-map is a line of a dump whose digest provenance.txt records, at the energy of
that line.

The script stops when a file is not the one provenance.txt records, when an energy or a
geometry is not what its file says, when a class has no archived minimum, or when xyzrender
or cairosvg, or for a picture of solvis solvis-tools, pyvista or vtk, is not the version
that drew the committed figures.

MANIFEST gives, for each class of tbl:systems, what the figures draw and what each
missing structure or map waits on; table() renders tbl:si-visuals from it.

Usage: system_visuals.py [--table] [data-dir]
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "system_visuals"
XYZRENDER = "0.3.8"
CAIROSVG = "2.9.1"
# Nearest neighbours of a Lennard-Jones minimum, in sigma, drawn as sticks.
LJ_NEIGHBOUR = 1.35
# The pictures of a one-species model: for each style, the renderer, the longest pair in
# sigma it joins (0 joins none), and its settings. xyzrender takes command-line flags;
# solvis takes the sphere and stick radii in sigma, as its AtomicPlotter adds them, and
# whether it casts shadows: off as chemparseplot draws balls and sticks, on as solvis does.
ONE_SPECIES_STYLES = {
    "paton": ("xyzrender", LJ_NEIGHBOUR, ("-a", "1.2", "-b", "1.5")),
    "thin": ("xyzrender", 1.2, ("-a", "1.6", "-b", "0.6", "--bond-color", "#7a7a7a", "--fog", "-F", "1.0",
                                "--grad", "--mol-color", "#1E88E5")),
    "bare": ("xyzrender", 0.0, ("-a", "2.4", "--fog", "-F", "1.0", "--grad", "--mol-color", "#1E88E5")),
    "solvis-ball": ("solvis", 1.2, (0.25, 0.05, False)),
    "solvis-sphere": ("solvis", 0.0, (0.5, 0.0, True)),
}
# The style the figures draw, chosen on the page of lj_render_compare.py.
ONE_SPECIES_STYLE = "thin"
SOLVIS = "0.1.5"
PYVISTA = "0.49.1"
VTK = "9.7.1"
# solvis writes a raster; its square side in pixels, 600 dpi over the widest panel, 4 in.
SOLVIS_PIXELS = 2400
OH_BOND = 1.25
OO_CONTACT = 3.2
CONTACT_CLASS = {9: "prism", 8: "cage", 7: "book", 6: "ring"}
HEXAMER_CLASSES = ("prism", "cage", "book", "ring")
# TIP4P of Jorgensen and co-workers: O--H 0.9572 Angstrom, H--O--H 104.52 degrees.
TIP4P_OH = 0.9572
TIP4P_HOH = 104.52
# The water minima of each engine, as provenance.txt names their folders.
ENGINES = {
    "GFN2-xTB": ("hq_sci_h2o6/best_h2o6_eindir_s*.xyz", "w6gofee4k/rec4k_w6_*.min"),
    "PBE/6-31G*": ("w6nwpbe/best_h2o6_rgpot_s*.xyz", "w6nwpbe/w6_nwpbe_*.min"),
}
LJ_FILES = ("lj38_fcc", "lj75_marks", "lj98_gm", "lj104_gm")
TIP4P_FILES = ("tip4p-10", "tip4p-15", "tip4p-20")
MAP_SIZES = (38, 75, 98)
REFERENCES = "system_visuals/references.csv"
REFERENCE_CLASSES = ("Lennard-Jones", "Morse", "Sutton--Chen", "TIP4P water")
# The first line of a Sutton--Chen file of the database: epsilon, c, a, n, m of Pt.
SUTTON_CHEN = (1.0, 34.408, 1.0, 10.0, 8.0)
# TIP4P as GMIN (DEFTIP4) and anneal evaluate it: the O--O 12-6 term in kJ/mol and
# Angstrom, the charges in e on each H and on M, 0.15 Angstrom from O on the bisector,
# and the Coulomb constant in kJ/mol Angstrom per e^2.
TIP4P_C12 = 2510.4e3
TIP4P_C6 = 2552.24
TIP4P_COULOMB = 1389.354848
TIP4P_QH = 0.52
TIP4P_M = 0.15
# The lengths of each one-species model, scaled so that its nearest neighbours stand as
# far apart as those of Lennard-Jones, 2^(1/6) sigma: r_e of Morse and a/sqrt(2), the
# neighbour distance of the fcc lattice, of Sutton--Chen.
DRAWN_SCALE = {"Lennard-Jones": 1.0, "Morse": 2 ** (1 / 6), "Sutton--Chen": 2 ** (1 / 6) * math.sqrt(2)}
ENERGY_DIGITS = {"Lennard-Jones": 6, "Morse": 6, "Sutton--Chen": 4, "TIP4P water": 4}
VIEWS = {"lj38_fcc": "tilt", "lj98_gm": "pair", "lj13": "tilt", "lj26": "tilt", "lj55": "tilt"}
CHEMPARSEPLOT = "1.11.0"
SVG_ID = re.compile(r"x[0-9a-f]{32}")
SUBSCRIPT = str.maketrans("0123456789", "\N{SUBSCRIPT ZERO}\N{SUBSCRIPT ONE}\N{SUBSCRIPT TWO}\N{SUBSCRIPT THREE}"
                          "\N{SUBSCRIPT FOUR}\N{SUBSCRIPT FIVE}\N{SUBSCRIPT SIX}\N{SUBSCRIPT SEVEN}\N{SUBSCRIPT EIGHT}\N{SUBSCRIPT NINE}")
TABLE = "tbl:si-visuals"
FIGURES = {
    "fig:systems": "wiley/imgs/systems.pdf",
    "fig:landscapes": "wiley/imgs/landscapes.pdf",
    "fig:landfold-maps": "wiley/imgs/landfold_maps.pdf",
    "fig:disconnectivity": "wiley/imgs/disconnectivity.pdf",
    "fig:si-structures": "wiley/imgs/si_structures.pdf",
    "fig:si-morse": "wiley/imgs/si_morse.pdf",
    "fig:si-water": "wiley/imgs/si_water.pdf",
}
# The figures of the main text, and of them those that map the landscapes.
MAIN_FIGURES = ("fig:systems", "fig:landscapes", "fig:landfold-maps", "fig:disconnectivity")
LANDSCAPE_FIGURES = ("fig:landscapes", "fig:landfold-maps", "fig:disconnectivity")


def die(message: str) -> NoReturn:
    raise SystemExit(f"system_visuals: {message}")


def digest_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(data: Path = DATA) -> tuple[dict[str, str], dict[str, str]]:
    """The keyed lines of provenance.txt, and the digest it records for each file."""
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


def _resolve(data: Path, name: str) -> Path:
    """A file provenance.txt names, under data/ of the repository or, for its own folder, data."""
    head, _, rest = name.partition("/")
    return data / rest if head == data.name or head == DATA.name else ROOT / "data" / name


def _read(data: Path, digests: dict[str, str], name: str) -> str:
    if name not in digests:
        die(f"provenance.txt records no digest of {name}")
    path = _resolve(data, name)
    if not path.is_file() or digest_of(path) != digests[name]:
        die(f"{name} does not match the digest provenance.txt records")
    return path.read_text()


def _named(digests: dict[str, str], pattern: str) -> list[str]:
    regex = re.compile(re.escape(pattern).replace(r"\*", r"[^/]*") + "$")
    found = sorted(name for name in digests if regex.match(name))
    if not found:
        die(f"provenance.txt records no file of {pattern}")
    return found


@dataclass(frozen=True)
class Structure:
    key: str
    label: str
    cls: str
    symbols: tuple[str, ...]
    coords: tuple[tuple[float, float, float], ...]
    energy: float
    unit: str
    view: str
    range: float | None = None


@dataclass(frozen=True)
class Landmark:
    index: int
    label: str
    source: str
    line: int
    energy: float
    x: float
    y: float


@dataclass(frozen=True)
class Landscape:
    size: int
    label: str
    sigma: float
    arms: tuple[str, ...]
    references: tuple[str, ...]
    landmarks: tuple[Landmark, ...]


@dataclass(frozen=True)
class Family:
    family: int
    community: int
    x: float
    asinh_x: float
    wells: int
    min_e: float


@dataclass(frozen=True)
class Fold:
    size: int
    label: str
    minima: int
    families: tuple[Family, ...]
    communities: int

    @property
    def lowest(self) -> Family:
        return min(self.families, key=lambda family: (family.min_e, family.family))


def _xyz(text: str) -> tuple[str, list[str], list[tuple[float, float, float]]]:
    lines = text.splitlines()
    count = int(lines[0].split()[0])
    rows = [line.split() for line in lines[2 : 2 + count]]
    if len(rows) != count:
        die(f"an xyz file holds {len(rows)} atoms, not the {count} it declares")
    return lines[1], [row[0] for row in rows], [tuple(map(float, row[1:4])) for row in rows]


def lj_energy(coords) -> float:
    total = 0.0
    for i, a in enumerate(coords):
        for b in coords[i + 1 :]:
            inverse = 1.0 / math.dist(a, b) ** 6
            total += 4.0 * (inverse * inverse - inverse)
    return total


def _waters(symbols, coords) -> list[tuple[int, int, int]]:
    """Each oxygen with its two hydrogens, or a stop when the atoms are not whole molecules."""
    oxygens = [i for i, s in enumerate(symbols) if s.upper() == "O"]
    hydrogens = [i for i, s in enumerate(symbols) if s.upper() == "H"]
    taken: set[int] = set()
    molecules = []
    for o in oxygens:
        near = [h for h in hydrogens if math.dist(coords[o], coords[h]) < OH_BOND]
        if len(near) != 2 or taken & set(near):
            die("a water geometry is not whole molecules")
        taken |= set(near)
        molecules.append((o, *near))
    if len(oxygens) + len(hydrogens) != len(symbols) or len(taken) != len(hydrogens):
        die("a water geometry is not whole molecules")
    return molecules


def contact_class(symbols, coords) -> str:
    oxygens = [c for s, c in zip(symbols, coords) if s.upper() == "O"]
    contacts = sum(
        math.dist(a, b) < OO_CONTACT for i, a in enumerate(oxygens) for b in oxygens[i + 1 :]
    )
    return CONTACT_CLASS.get(contacts, f"other({contacts})")


def morse_energy(coords, rho: float, gradient: bool = False):
    """The Morse energy in epsilon at r_e = 1, and with gradient its derivative in the coordinates."""
    import numpy as np

    xyz = np.asarray(coords, dtype=float).reshape(-1, 3)
    i, j = np.triu_indices(len(xyz), 1)
    delta = xyz[i] - xyz[j]
    r = np.linalg.norm(delta, axis=1)
    e = np.exp(rho * (1.0 - r))
    energy = float(np.sum(e * (e - 2.0)))
    if not gradient:
        return energy
    pull = ((-2.0 * rho * e * (e - 1.0)) / r)[:, None] * delta
    grad = np.zeros_like(xyz)
    np.add.at(grad, i, pull)
    np.add.at(grad, j, -pull)
    return energy, grad.ravel()


def _relaxed(coords, rho: float):
    import numpy as np
    from scipy.optimize import minimize

    found = minimize(morse_energy, np.asarray(coords, dtype=float).ravel(), args=(rho, True), jac=True,
                     method="L-BFGS-B", options={"gtol": 1e-10, "ftol": 1e-15, "maxiter": 20000})
    return [tuple(map(float, row)) for row in found.x.reshape(-1, 3)]


def sutton_chen_energy(coords, c: float, n: float, m: float) -> float:
    """The Sutton--Chen energy in epsilon, lengths in the lattice constant a."""
    import numpy as np

    xyz = np.asarray(coords, dtype=float)
    r = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    np.fill_diagonal(r, np.inf)
    return float(0.5 * np.sum(r**-n) - c * np.sum(np.sqrt(np.sum(r**-m, axis=1))))


def tip4p_energy(symbols, coords) -> float:
    """The TIP4P energy in kJ/mol of whole rigid molecules, every pair of molecules counted once."""
    import numpy as np

    xyz = np.asarray(coords, dtype=float)
    molecules = []
    for o, h1, h2 in _waters(symbols, coords):
        bisector = (xyz[h1] - xyz[o]) + (xyz[h2] - xyz[o])
        m = xyz[o] + TIP4P_M * bisector / np.linalg.norm(bisector)
        molecules.append((xyz[o], ((xyz[h1], TIP4P_QH), (xyz[h2], TIP4P_QH), (m, -2.0 * TIP4P_QH))))
    total = 0.0
    for a, (oxygen_a, charges_a) in enumerate(molecules):
        for oxygen_b, charges_b in molecules[a + 1 :]:
            r = float(np.linalg.norm(oxygen_a - oxygen_b))
            total += TIP4P_C12 / r**12 - TIP4P_C6 / r**6
            for site_a, q_a in charges_a:
                for site_b, q_b in charges_b:
                    total += TIP4P_COULOMB * q_a * q_b / float(np.linalg.norm(site_a - site_b))
    return total


def _points(text: str, name: str, size: int, head: int):
    """The head values of a points file of the database and its size atoms."""
    lines = [line.split() for line in text.splitlines() if line.strip()]
    rows = [tuple(map(float, line)) for line in lines[1 if head else 0 :]]
    if len(rows) != size or any(len(row) != 3 for row in rows) or (head and len(lines[0]) != head):
        die(f"{name} is not {size} atoms of three coordinates")
    return (tuple(map(float, lines[0])) if head else ()), rows


def _rigid(name: str, symbols, coords) -> None:
    for o, h1, h2 in _waters(symbols, coords):
        lengths = (math.dist(coords[o], coords[h1]), math.dist(coords[o], coords[h2]))
        u = [b - a for a, b in zip(coords[o], coords[h1])]
        v = [b - a for a, b in zip(coords[o], coords[h2])]
        angle = math.degrees(math.acos(sum(p * q for p, q in zip(u, v)) / (lengths[0] * lengths[1])))
        if max(abs(length - TIP4P_OH) for length in lengths) > 1e-4 or abs(angle - TIP4P_HOH) > 1e-2:
            die(f"{name} holds a molecule that is not rigid TIP4P")


def _references(data, digests) -> list[dict[str, str]]:
    rows = list(csv.DictReader(io.StringIO(_read(data, digests, REFERENCES))))
    if any(row["class"] not in REFERENCE_CLASSES for row in rows):
        die(f"{REFERENCES} names a class that is not one of {REFERENCE_CLASSES}")
    return rows


def references(data: Path = DATA) -> list[dict[str, str]]:
    """Each row of references.csv: the class, size, range and published energy of a reference."""
    _, digests = provenance(data)
    return _references(data, digests)


def _reference(data, digests, row: dict[str, str]) -> Structure:
    """The structure of one row of references.csv, held to the energy the row publishes."""
    cls, name, size = row["class"], row["file"], int(row["size"])
    published = float(row["energy"])
    text = _read(data, digests, name)
    rho = None
    if cls == "TIP4P water":
        header, symbols, coords = _xyz(text)
        stated = float(re.search(r"Energy\s*=\s*(-?\d+\.\d+)", header)[1])
        _rigid(name, symbols, coords)
        energy = tip4p_energy(symbols, coords)
        if len(symbols) != 3 * size or abs(stated - published) > 5e-6:
            die(f"{name} is not the (H2O){size} at the {row['energy']} kJ/mol of {REFERENCES}")
        if abs(energy - published) >= 5e-3 or energy > published + 1e-3:
            die(f"{name} is at {energy:.5f} kJ/mol, not the {row['energy']} of {REFERENCES}")
        key, label, unit = f"tip4p-{size}", f"(H2O){size}".translate(SUBSCRIPT), "kJ/mol"
    else:
        if name.endswith(".xyz"):
            header, _, coords = _xyz(text)
            stated = float(re.search(r"e=(-?\d+\.\d+)", header)[1])
            if abs(lj_energy(coords) - stated) > 1e-5:
                die(f"{name} is at {lj_energy(coords):.6f} epsilon, not the {stated} its header gives")
            if len(coords) != size:
                die(f"{name} is not {size} atoms")
        else:
            head, coords = _points(text, name, size, {"Lennard-Jones": 0, "Morse": 1, "Sutton--Chen": 5}[cls])
        if cls == "Lennard-Jones":
            energy, tolerance = lj_energy(coords), 1e-5
            key = Path(name).stem if name.endswith(".xyz") else f"lj{size}"
            label = f"LJ{size}".translate(SUBSCRIPT)
        elif cls == "Morse":
            if len(head) != 1 or head[0] <= 0:
                die(f"{name} gives no range rho_0 on its first line")
            rho = float(row["range"])
            coords = _relaxed(coords, rho)
            energy, tolerance = morse_energy(coords, rho), 1e-5
            key, label = f"m{size}_rho{row['range']}", f"M{size}".translate(SUBSCRIPT)
        else:
            if head != SUTTON_CHEN:
                die(f"{name} is not Sutton--Chen 10-8 of Pt")
            energy, tolerance = sutton_chen_energy(coords, *SUTTON_CHEN[1:2], *SUTTON_CHEN[3:]), 1e-4
            key, label = f"pt{size}_sc", f"Pt{size}".translate(SUBSCRIPT)
        if abs(energy - published) > tolerance:
            die(f"{name} is at {energy:.6f} epsilon, not the {row['energy']} of {REFERENCES}")
        symbols = ["X"] * size
        unit = "\N{GREEK SMALL LETTER EPSILON}"
    return Structure(key, label, cls, tuple(symbols), tuple(coords), published, unit, VIEWS.get(key, "axis" if cls == "Lennard-Jones" else "tilt"), rho)


def _hexamer_candidates(data, digests, engine: str):
    """Every archived minimum of one engine: its energy, its order of reading, atoms and coordinates."""
    best, dumps = ENGINES[engine]
    found = []
    for name in _named(digests, best):
        header, symbols, coords = _xyz(_read(data, digests, name))
        found.append((float(header.split()[1]), (name, 0), symbols, coords))
    for name in _named(digests, dumps):
        for row, line in enumerate(_read(data, digests, name).splitlines()):
            values = [float(value) for value in line.split()]
            if len(values) != 55:
                die(f"{name} row {row} holds {len(values) - 1} coordinates, not 54")
            coords = [tuple(values[1 + 3 * i : 4 + 3 * i]) for i in range(18)]
            found.append((values[0], (name, row + 1), ["O", "H", "H"] * 6, coords))
    return found


def _hexamers(data, digests) -> list[Structure]:
    drawn = []
    for engine in ENGINES:
        lowest: dict[str, tuple] = {}
        for energy, where, symbols, coords in _hexamer_candidates(data, digests, engine):
            cls = contact_class(symbols, coords)
            if cls in HEXAMER_CLASSES and (cls not in lowest or (energy, where) < lowest[cls][:2]):
                lowest[cls] = (energy, where, symbols, coords)
        for cls in HEXAMER_CLASSES:
            if cls not in lowest:
                die(f"no archived {engine} minimum of the hexamer is a {cls}")
            energy, _, symbols, coords = lowest[cls]
            _waters(symbols, coords)
            key = f"w6_{engine.split('/')[0].split('-')[0].lower()}_{cls}"
            drawn.append(Structure(key, f"{cls}, {engine}", "Water hexamer", tuple(symbols), tuple(coords),
                                   energy, "eV", "tilt"))
    return drawn


def structures(data: Path = DATA) -> list[Structure]:
    """Every reference structure of references.csv, in its order, and then the hexamers."""
    _, digests = provenance(data)
    return [_reference(data, digests, row) for row in _references(data, digests)] + _hexamers(data, digests)


def _dump_lines(data, digests, name: str, cache: dict[str, list[str]]) -> list[str]:
    if name not in cache:
        cache[name] = _read(data, digests, name).splitlines()
    return cache[name]


def landmark_coords(landmark: Landmark, data: Path = DATA):
    """The coordinates of the minimum a landmark projects, read from its line of its dump."""
    _, digests = provenance(data)
    values = [float(value) for value in _dump_lines(data, digests, landmark.source, {})[landmark.line - 1].split()]
    return [tuple(values[1 + 3 * i : 4 + 3 * i]) for i in range((len(values) - 1) // 3)]


def landscapes(data: Path = DATA) -> list[Landscape]:
    """The sketch-map of each size that sketchmap_landscapes.py projected, each landmark checked
    against its line of its dump: the references and each arm's deepest minimum first, in the
    order the run pinned them, and the deepest of each arm the deepest of its dump."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import sketchmap_landscapes as run

    keys, digests = provenance(data)
    cache: dict[str, list[str]] = {}
    found = []
    for size, (arms, refs) in run.SIZES.items():
        name = f"system_visuals/sketchmap/lj{size}.csv"
        rows = list(csv.reader(io.StringIO(_read(data, digests, name))))
        if tuple(rows[0]) != run.COLUMNS or len(rows) - 1 != run.LANDMARKS:
            die(f"{name} is not {run.LANDMARKS} landmarks of the columns {run.COLUMNS}")
        sources = {f"ref:{label}": f"{run.REFS}/{file}" for label, file in refs}
        sources |= {label: f"{run.RUNS}/{file}" for label, file in arms}
        landmarks = []
        for k, (index, label, source, line, energy, x, y) in enumerate(rows[1:]):
            if int(index) != k or sources.get(label) != source:
                die(f"{name} row {k + 1} is not landmark {k} of a dump its run read")
            lines = _dump_lines(data, digests, source, cache)
            if not 1 <= int(line) <= len(lines) or float(lines[int(line) - 1].split()[0]) != float(energy):
                die(f"{name} row {k + 1} is not at the energy of line {line} of {source}")
            landmarks.append(Landmark(k, label, source, int(line), float(energy), float(x), float(y)))
        pinned = [f"ref:{label}" for label, _ in refs] + [label for label, _ in arms]
        for k, label in enumerate(pinned):
            deepest = min(float(text.split()[0]) for text in _dump_lines(data, digests, sources[label], cache))
            if landmarks[k].label != label or landmarks[k].energy != deepest:
                die(f"{name} does not pin the deepest minimum of {label} as landmark {k}")
        found.append(Landscape(size, f"LJ{size}".translate(SUBSCRIPT), float(keys[f"sketchmap_lj{size}_sigma"]),
                               tuple(label for label, _ in arms), tuple(label for label, _ in refs), tuple(landmarks)))
    return found


def check_chemparseplot() -> None:
    from importlib.metadata import version

    if version("chemparseplot") != CHEMPARSEPLOT:
        die(f"chemparseplot {version('chemparseplot')} is not the {CHEMPARSEPLOT} of the sketch-map runs")


def folds(data: Path = DATA) -> list[Fold]:
    keys, digests = provenance(data)
    found = []
    for size in MAP_SIZES:
        record = json.loads(_read(data, digests, f"system_visuals/landfold/lj{size}.json"))
        if record["kind"] != "occupancy_landfold" or record["label"] != f"LJ{size}":
            die(f"lj{size}.json is not the occupancy landfold of LJ{size}")
        if record["switch_l2"] != 0.0 or any(point["y"] != 0.0 for point in record["points"]):
            die(f"lj{size}.json folds to two dimensions, and its map draws one")
        inputs = keys[f"landfold_lj{size}_inputs"].split()
        rows = sum(len(_read(data, digests, name).splitlines()) for name in inputs)
        if record["n_in"] != rows:
            die(f"lj{size}.json folds {record['n_in']} minima, and its dumps hold {rows}")
        hist = {}
        for line in _read(data, digests, f"system_visuals/landfold/lj{size}_hist.txt").splitlines():
            if not line.startswith("#"):
                family, wells, min_e = line.split()[:3]
                hist[int(family)] = (int(wells), float(min_e))
        families = []
        for point in record["points"]:
            wells, min_e = hist[point["family"]]
            if wells != point["wells"]:
                die(f"lj{size}_hist.txt gives family {point['family']} {wells} wells, its fold {point['wells']}")
            families.append(Family(point["family"], point["community"], point["x"], point["asinh_x"], wells, min_e))
        if len(families) != record["occupied"] or len(hist) != record["occupied"]:
            die(f"lj{size}.json does not fold each occupied family once")
        found.append(Fold(size, f"LJ{size}".translate(SUBSCRIPT), rows, tuple(families), record["communities"]))
    return found


def _check_tools() -> tuple[str, str]:
    exe = shutil.which("xyzrender")
    if exe is None:
        die("xyzrender is not on PATH")
    version = subprocess.run([exe, "--version"], capture_output=True, text=True, check=True).stdout.split()[-1]
    import cairosvg

    if version != XYZRENDER or cairosvg.__version__ != CAIROSVG:
        die(f"xyzrender {version} and cairosvg {cairosvg.__version__} are not the {XYZRENDER} and {CAIROSVG} that drew the figures")
    return exe, version


def _rotation(axis: int, degrees: float):
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    if axis == 0:
        return ((1, 0, 0), (0, c, -s), (0, s, c))
    return ((c, 0, s), (0, 1, 0), (-s, 0, c))


def _apply(matrix, points):
    return [tuple(sum(m * p for m, p in zip(row, point)) for row in matrix) for point in points]


def viewed(structure: Structure):
    """The coordinates turned to the view xyzrender draws, down z.

    axis looks down the unique axis of the second moments, the five-fold axis of a
    decahedron; pair down the longest sum of the lines to two of the atoms farthest from
    the centre, the first such pair in file order, which sets the lattice of the Leary
    tetrahedron in rows; tilt turns the principal axes by 25 and 15 degrees.
    """
    import numpy as np

    xyz = np.asarray(structure.coords, dtype=float)
    xyz = xyz - xyz.mean(axis=0)
    values, vectors = np.linalg.eigh(xyz.T @ xyz)
    if structure.view == "axis":
        z = vectors[:, 2] if values[2] - values[1] > values[1] - values[0] else vectors[:, 0]
    elif structure.view == "pair":
        radii = np.linalg.norm(xyz, axis=1)
        outer = [i for i in range(len(xyz)) if radii[i] > radii.max() - 1e-6]
        z = max(
            (xyz[i] + xyz[j] for n, i in enumerate(outer) for j in outer[n + 1 :]),
            key=lambda v: round(float(np.linalg.norm(v)), 6),
        )
    else:
        turned = xyz @ vectors[:, ::-1]
        return _apply(_rotation(1, 15), _apply(_rotation(0, 25), turned.tolist()))
    z = z / np.linalg.norm(z)
    t = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(t, z)
    x /= np.linalg.norm(x)
    frame = np.array([x, np.cross(z, x), z])
    return (xyz @ frame.T).tolist()


def _bare_pdf(pdf: bytes) -> bytes:
    import pymupdf

    document = pymupdf.open("pdf", pdf)
    document.set_metadata({key: "" for key in ("producer", "creationDate", "creator", "modDate", "title", "author", "subject", "keywords")})
    return document.tobytes(garbage=4, deflate=True, no_new_id=True)


def render(structure: Structure, style: str = ONE_SPECIES_STYLE) -> bytes:
    """The picture of a structure as a PDF, a one-species model in the named style.

    Water is always the vector xyzrender picture with its hydrogen bonds dotted; oxygen and
    hydrogen take Ruhi teal and sky rather than the default NCI element palette. A style of
    solvis gives a raster of SOLVIS_PIXELS on a side, placed in a page of 4 in.
    """
    import cairosvg
    import figstyle

    lj = structure.cls in DRAWN_SCALE
    coords = [tuple(DRAWN_SCALE[structure.cls] * v for v in point) for point in viewed(structure)] if lj else viewed(structure)
    if lj:
        renderer, joined, settings = ONE_SPECIES_STYLES[style]
        pairs = [(i, j) for i in range(len(coords)) for j in range(i + 1, len(coords))
                 if math.dist(coords[i], coords[j]) < joined]
        config = "paton"
    else:
        renderer, joined, settings = (
            "xyzrender",
            0.0,
            ("--nci", "--nci-color", figstyle.CORAL, "--hl", "O", figstyle.TEAL, "--hl", "H", figstyle.SKY),
        )
        pairs = [(o, h) for molecule in _waters(list(structure.symbols), coords)
                 for o, h in ((molecule[0], molecule[1]), (molecule[0], molecule[2]))]
        config = "paton"
    if renderer == "solvis":
        return _render_solvis(coords, pairs, *settings)
    exe, _ = _check_tools()
    # xyzrender knows no element X, so an atom of a one-species model is drawn as argon.
    symbols = ["Ar"] * len(coords) if lj else list(structure.symbols)
    bonds = ["--bond", *(f"{i + 1}-{j + 1}" for i, j in pairs)] if pairs else []
    with tempfile.TemporaryDirectory() as scratch:
        xyz = Path(scratch) / f"{structure.key}.xyz"
        svg = Path(scratch) / f"{structure.key}.svg"
        # Ruhi element colours for water: JSON color_overrides, not the NCI CPK palette.
        cfg = Path(scratch) / "ruhi_water.json"
        if not lj:
            cfg.write_text(
                '{"colors": {"O": "%s", "H": "%s"}}\n' % (figstyle.TEAL, figstyle.SKY)
            )
            config = str(cfg)
        xyz.write_text(
            f"{len(coords)}\n{structure.key}\n"
            + "".join(f"{s} {x:.8f} {y:.8f} {z:.8f}\n" for s, (x, y, z) in zip(symbols, coords))
        )
        subprocess.run(
            [exe, "--config", config, "-t", "--no-orient", *settings, "--unbond", "all",
             *bonds, "-o", str(svg), "--", str(xyz)],
            check=True, capture_output=True,
        )
        text = SVG_ID.sub("xyzr", svg.read_text())
    # cairo dates the PDF it writes, so the picture is saved again without its metadata.
    return _bare_pdf(cairosvg.svg2pdf(bytestring=text.encode()))


def _render_solvis(coords, pairs, radius: float, stick: float, shadows: bool) -> bytes:
    """The solvis picture of one-species coordinates in sigma, in its own lighting.

    The view is parallel, down z with x to the right as xyzrender draws it, framed on the
    atoms, on a transparent ground, in the sky of the Ruhi palette.
    """
    import numpy as np
    import pymupdf
    from PIL import Image

    _check_solvis()
    import figstyle
    from solvis.visualization import AtomicPlotter

    colour = figstyle.SKY
    xyz = np.asarray(coords, dtype=float)
    plotter = AtomicPlotter(interactive_mode=False, window_size=[SOLVIS_PIXELS, SOLVIS_PIXELS],
                            depth_peeling=True, shadows=shadows)
    plotter.add_atoms_as_spheres(xyz, [colour] * len(xyz), radius=radius)
    for n, (i, j) in enumerate(pairs):
        plotter.add_bond(xyz[i], xyz[j], colour, colour, f"bond_{n}", radius=stick)
    plotter.plotter.enable_parallel_projection()
    plotter.plotter.view_xy()
    plotter.plotter.reset_camera(bounds=_square(xyz, radius))
    plotter.plotter.camera.parallel_scale = float(np.abs(xyz[:, :2]).max() + radius) * 1.02
    image = plotter.plotter.screenshot(None, transparent_background=True, return_img=True)
    plotter.close()
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG", optimize=False)
    document = pymupdf.open()
    page = document.new_page(width=288, height=288)
    page.insert_image(page.rect, stream=buffer.getvalue())
    return _bare_pdf(document.tobytes())


def _square(xyz, radius: float):
    half = float(abs(xyz).max()) + radius
    return (-half, half, -half, half, -half, half)


def _check_solvis() -> None:
    from importlib.metadata import version

    found = {name: version(name) for name in ("solvis-tools", "pyvista", "vtk")}
    if found != {"solvis-tools": SOLVIS, "pyvista": PYVISTA, "vtk": VTK}:
        die(f"solvis-tools, pyvista and vtk are {found}, not the {SOLVIS}, {PYVISTA} and {VTK} that drew the figures")


def place(figure, panels, out: Path) -> None:
    """Save a matplotlib figure with each (axes, PDF) of panels drawn into its axes box."""
    import matplotlib.pyplot as plt
    import pymupdf

    width, height = figure.get_size_inches()
    boxes = [ax.get_position() for ax, _ in panels]
    buffer = io.BytesIO()
    figure.savefig(buffer, format="pdf", metadata={"CreationDate": None, "ModDate": None, "Creator": None})
    plt.close(figure)
    document = pymupdf.open("pdf", buffer.getvalue())
    page = document[0]
    for box, (_, pdf) in zip(boxes, panels):
        rect = pymupdf.Rect(box.x0 * width * 72, (1 - box.y1) * height * 72, box.x1 * width * 72, (1 - box.y0) * height * 72)
        page.show_pdf_page(rect, pymupdf.open("pdf", pdf), 0)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(document.tobytes(garbage=4, deflate=True, no_new_id=True))


def caption(structure: Structure) -> str:
    """The line under a structure: its name and, but for water at an electronic structure, its energy."""
    if structure.cls == "Water hexamer":
        return structure.label
    energy = f"{structure.energy:.{ENERGY_DIGITS[structure.cls]}f}".replace("-", "\N{MINUS SIGN}")
    return f"{structure.label}, {energy} {structure.unit}"


def draw_fold(ax, fold: Fold, ylabel: bool = True) -> None:
    """One size's map: each family at its fold coordinate and its lowest energy above the lowest,
    its area the minima it holds; the largest community sky, the others coral, the lowest
    family ringed."""
    import figstyle

    sizes = {}
    for family in fold.families:
        sizes[family.community] = sizes.get(family.community, 0) + 1
    largest = max(sorted(sizes), key=lambda community: sizes[community])
    base = fold.lowest.min_e
    for inside in (True, False):
        chosen = [f for f in fold.families if (f.community == largest) == inside]
        ax.scatter(
            [f.x for f in chosen], [f.min_e - base for f in chosen],
            s=[1.5 * f.wells for f in chosen], color=figstyle.SKY if inside else figstyle.CORAL,
            alpha=0.55 if inside else 0.9, linewidths=0, zorder=2 if inside else 3,
        )
    low = fold.lowest
    ax.scatter([low.x], [0.0], s=40, facecolors="none", edgecolors=figstyle.BLACK,
               linewidths=0.8, zorder=4)
    ax.set_xlabel("fold coordinate")
    if ylabel:
        ax.set_ylabel("energy above lowest (\N{GREEK SMALL LETTER EPSILON})")


# What each class of tbl:systems draws, and what each missing structure or map waits on.
MANIFEST = (
    ("Lennard-Jones", "LJ\\(_N\\) at every \\(N\\) of Table~\\ref{tbl:lj}",
     "landfold and sketch-map of LJ\\(_{38}\\), LJ\\(_{75}\\), LJ\\(_{98}\\); disconnectivity of LJ\\(_{38}\\)",
     ("the accepted-minima dump, ANNEAL\\_MIN\\_DUMP, of one recommended seed at each size without a "
      "map; the sealed fes.csv and landscape.ld of each landfold map of the main text, which "
      "the landfold handoff does not carry")),
    ("Morse", "M\\(_N\\)~\\cite{morseDiatomicMoleculesAccording1929} at every \\(N\\) and \\(\\rho_0\\) of Table~\\ref{tbl:morse}", "",
     "the accepted-minima dump of one recommended seed of each cluster and range"),
    ("Binary LJ", "", "",
     ("the sizes and ratios of the class, the reference minimum of each as an xyz file, and the "
      "accepted-minima dump of one recommended seed of each, folded by a packing book that tells "
      "the two species apart")),
    ("Sutton--Chen", "Pt\\(_{38}\\), Pt\\(_{75}\\)", "",
     "the accepted-minima dump of one recommended seed of each"),
    ("Effective medium", "", "",
     ("the lowest minimum over all codes of each Cu\\(_n\\) and Ag\\(_n\\), its reference, as an xyz "
      "file from the campaign, since the Cambridge Cluster Database~\\cite{walesCambridgeClusterDatabase} "
      "holds no effective-medium minima; and the accepted-minima dump of one recommended seed of each")),
    ("Embedded atom", "", "",
     ("the lowest minimum over all codes of Na\\(_{13}\\), its reference, as an xyz file from the "
      "campaign, since the Cambridge Cluster Database holds no embedded-atom minimum of it; and the "
      "accepted-minima dump of one recommended seed")),
    ("Covalent", "", "",
     ("the sizes of the class, the lowest minimum over all codes of each as an xyz file, from the "
      "campaign, and the accepted-minima dump of one recommended seed of each")),
    ("TIP4P water", "(H\\(_2\\)O)\\(_n\\) at every \\(n\\) of Table~\\ref{tbl:tip4p}", "",
     ("the accepted-minima dump of one recommended seed of each \\(n\\), folded by a packing book "
      "that tells oxygen from hydrogen; the sealed fes.csv and landscape.ld of the hexamer maps of "
      "the main text")),
    ("Water hexamer", "prism, cage, book and ring at GFN2-xTB and at PBE/6-31G*", "",
     ("the four references of the metrics at each engine as xyz files; the minima of each class at "
      "PET-MAD; the accepted-minima dumps of the sealed water arms, folded by a packing book that tells "
      "oxygen from hydrogen")),
    ("Molecules", "", "",
     ("the molecules and peptides of the class, the lowest minimum over all codes of each as an xyz "
      "file, from the campaign, and the accepted-minima dump of one recommended seed of each")),
    ("Surface", "", "",
     ("the OptBench minimum of Pt\\(_7\\) on Pt(111) as a geometry file, and the accepted-minima "
      "dump of one recommended seed, folded by a packing book that tells the adatoms from the slab")),
)
CLASSES = tuple(row[0] for row in MANIFEST)


def marker(waits: str) -> str:
    return f"\\textcolor{{red}}{{[TODO(results): {waits}]}}"


def table() -> str:
    """tbl:si-visuals as the Supporting Information prints it, an org export block."""
    rows = "\n".join(
        f"{cls} & {drawn or 'none'} & {mapped or 'none'} & {marker(waits)}\\\\"
        for cls, drawn, mapped, waits in MANIFEST
    )
    return (
        "#+BEGIN_EXPORT latex\n"
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\scriptsize\n"
        "\\setlength{\\tabcolsep}{2.5pt}\n"
        "\\caption{The structures and maps of each class of the benchmark. Drawn are the "
        "structures of Figures~\\ref{fig:si-structures} to~\\ref{fig:si-water} and, in the main "
        "text, the maps of Lennard-Jones; each other structure or map waits on the run or file its "
        "last cell names.}\n"
        f"\\label{{{TABLE}}}\n"
        "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{1.9cm}>{\\raggedright\\arraybackslash}p{3.4cm}"
        ">{\\raggedright\\arraybackslash}p{2.2cm}>{\\raggedright\\arraybackslash}p{7.8cm}@{}}\n"
        "\\toprule\n"
        "Class & Structures drawn & Maps drawn & Waits on\\\\\n"
        "\\midrule\n"
        f"{rows}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
        "#+END_EXPORT"
    )


def main(argv: list[str]) -> int:
    if argv[:1] == ["--table"]:
        print(table())
        return 0
    data = Path(argv[0]) if argv else DATA
    for structure in structures(data):
        print(f"{structure.key} {structure.cls} {len(structure.coords)} atoms {structure.energy} {structure.unit}")
    for fold in folds(data):
        low = fold.lowest
        print(f"LJ{fold.size} {fold.minima} minima {len(fold.families)} families {fold.communities} communities lowest {low.min_e}")
    for landscape in landscapes(data):
        print(f"sketch-map LJ{landscape.size} {len(landscape.landmarks)} landmarks sigma {landscape.sigma}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

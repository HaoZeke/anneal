#!/usr/bin/env python3
"""Regenerate the hexamer inputs shared by both code versions.

water-v10.xyz comes from the isolated-water template and the molecule
spacing that ``reference_coordinates`` in ``src/catalog/molecular.rs`` uses.
The three isomer references are deposited minima rewritten so that every
molecule is a contiguous O, H, H triple with the two hydrogens bonded to its
oxygen, as the references of the rebuild are. The example drivers check the
monomers by counting the hydrogens within 1.2 A of every oxygen and match a
reference with a species-aware IRA, so the order of the atoms of a reference
does not change a verdict; the controls of this A/B check that.
"""

import argparse
import hashlib
import math
import re
import sys
from pathlib import Path

MOLECULES = 6
OH_LIMIT = 1.2


def template_from_catalog(source: Path):
    text = source.read_text()
    block = re.search(
        r"const WATER: \[\[f64; 3\]; 3\] = \[(.*?)\];", text, flags=re.DOTALL
    )
    if block is None:
        sys.exit("water template not found in the catalog source")
    atoms = re.findall(
        r"\[\s*(-?[0-9.]+)\s*,\s*(-?[0-9.]+)\s*,\s*(-?[0-9.]+)\s*\]", block.group(1)
    )
    if len(atoms) != 3:
        sys.exit("water template must hold three atoms")
    spacing = re.search(r"let shift = ([0-9.]+) \* molecule as f64;", text)
    if spacing is None:
        sys.exit("molecule spacing not found in the catalog source")
    return [tuple(float(v) for v in atom) for atom in atoms], float(spacing.group(1))


def write_template(path: Path, atoms, spacing):
    lines = [str(3 * MOLECULES), "(H2O)6 reference_coordinates template"]
    for molecule in range(MOLECULES):
        for symbol, atom in zip("OHH", atoms):
            x = atom[0] + spacing * molecule
            lines.append(f"{symbol} {x:.6f} {atom[1]:.6f} {atom[2]:.6f}")
    path.write_text("\n".join(lines) + "\n")


def read_xyz(path: Path):
    rows = path.read_text().splitlines()
    count = int(rows[0].split()[0])
    atoms = []
    for row in rows[2 : 2 + count]:
        fields = row.split()
        atoms.append((fields[0], tuple(float(v) for v in fields[1:4])))
    return atoms


def reorder_oh_h(atoms):
    oxygens = [i for i, (symbol, _) in enumerate(atoms) if symbol == "O"]
    hydrogens = [i for i, (symbol, _) in enumerate(atoms) if symbol == "H"]
    if len(oxygens) != MOLECULES or len(hydrogens) != 2 * MOLECULES:
        sys.exit("a hexamer needs six O and twelve H")
    owner = {}
    for h in hydrogens:
        distances = [(math.dist(atoms[h][1], atoms[o][1]), o) for o in oxygens]
        distance, oxygen = min(distances)
        if distance > OH_LIMIT:
            sys.exit("a hydrogen has no oxygen within the O-H limit")
        owner.setdefault(oxygen, []).append(h)
    if any(len(owner.get(o, [])) != 2 for o in oxygens):
        sys.exit("every oxygen must own exactly two hydrogens")
    order = []
    for o in oxygens:
        order.extend([o, *sorted(owner[o])])
    return [atoms[i] for i in order]


def write_reference(path: Path, atoms, label):
    lines = [str(len(atoms)), label]
    for symbol, (x, y, z) in atoms:
        lines.append(f"{symbol} {x:.6f} {y:.6f} {z:.6f}")
    path.write_text("\n".join(lines) + "\n")


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-source", type=Path, required=True)
    parser.add_argument("--prism", type=Path, required=True)
    parser.add_argument("--cage", type=Path, required=True)
    parser.add_argument("--book", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    atoms, spacing = template_from_catalog(args.catalog_source)
    template_path = args.out_dir / "water-v10.xyz"
    write_template(template_path, atoms, spacing)
    written = [template_path]
    for name in ("prism", "cage", "book"):
        source = getattr(args, name)
        reference = reorder_oh_h(read_xyz(source))
        target = args.out_dir / f"{name}.xyz"
        write_reference(target, reference, f"{name} GFN2 minimum")
        written.append(target)
    for path in written:
        print(f"{sha256(path)}  {path.name}")


if __name__ == "__main__":
    main()

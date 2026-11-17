#!/usr/bin/env python3
# Copyright (c) anneal developers
"""The panels of landfold_maps.pdf and the caption text that names them.

The figure has no panel titles, so the caption lists each panel by letter. The list is read
from the seals of data/landfold_maps_v7, which fig_landfold_maps.py draws from, and the
manuscript caption is checked against it by scripts/paper_audit.py.

Usage: landfold_panels.py [--data DIR]
"""
from __future__ import annotations

import argparse
import json
import string
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "landfold_maps_v7"
JOBS = (
    ("lj_csm/lj38/derived", "LJ", "38", "derived"),
    ("lj_csm/lj38/recommended", "LJ", "38", "recommended"),
    ("lj_csm/lj55/derived", "LJ", "55", "derived"),
    ("lj_csm/lj55/recommended", "LJ", "55", "recommended"),
    ("lj_csm/lj75/derived", "LJ", "75", "derived"),
    ("lj_csm/lj75/recommended", "LJ", "75", "recommended"),
    ("lj_csm/lj98/derived", "LJ", "98", "derived"),
    ("lj_csm_c26/lj98/recommended", "LJ", "98", "recommended"),
    ("tip4p/hexamer/for_tip4p", "TIP4P", "", "hexamer"),
)
KINDS = ("representative", "aggregate")
DRAWN = "wrote landscape.png and fes.png"


def panels(data: Path = DATA) -> list[dict[str, str]]:
    """One row per drawn map, in the order of the figure."""
    rows = []
    for job, family, size, role in JOBS:
        seal = json.loads((data / job / "PROVENANCE.json").read_text())
        for kind in KINDS:
            if (seal.get(kind) or {}).get("map") != DRAWN:
                continue
            seeds = f"seed {seal['first_hit_seed']}" if kind == "representative" else f"{seal['seeds_finished']} seeds"
            rows.append({"family": family, "size": size, "role": role, "kind": kind, "seeds": seeds})
    return rows


def raster_letters(data: Path = DATA) -> str:
    """The letters of the panels whose landscape.ld is not committed, so only the sealed raster map exists."""
    letters = []
    rows = []
    for job, _family, _size, _role in JOBS:
        seal = json.loads((data / job / "PROVENANCE.json").read_text())
        for kind in KINDS:
            if (seal.get(kind) or {}).get("map") == DRAWN:
                rows.append(data / job / kind / "landscape.ld")
    for letter, path in zip(string.ascii_lowercase, rows):
        if not path.exists():
            letters.append(letter)
    return "".join(letters)


def system_tex(row: dict[str, str]) -> str:
    return f"{row['family']}\\(_{{{row['size']}}}\\)" if row["size"] else row["family"]


def caption_tex(data: Path = DATA) -> str:
    """The panel list of the manuscript caption, one clause per panel."""
    clauses = []
    for letter, row in zip(string.ascii_lowercase, panels(data)):
        clauses.append(f"({letter}) {system_tex(row)} {row['role']}, {row['kind']}, {row['seeds']}")
    return "; ".join(clauses) + "."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args()
    print(caption_tex(args.data))
    print("raster panels:", raster_letters(args.data))


if __name__ == "__main__":
    main()

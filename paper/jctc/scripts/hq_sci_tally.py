#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Rebuild the tally from the committed rec logs.

A hexamer hit is a finite best within 0.01 eV of the lowest finite best.
An inf best is recorded as best=None and is not a hit.
A copper hit is a finite best below the reference plus its tolerance.
The reference is -415.971529 eV.
scripts/elja_jcc_molslab_ensemble.sh line 79 sets that target.
scripts/elja_hq_sci_molslab_bank.sh line 118 sets it too.
Commit a8f1092fc8d6937b34e34a18bc50394a559ca9f3 names it the start energy.
The tolerance is 0.001 eV, from TARGET_TOLERANCE and TARGET_TOL in those scripts.
The set_relative_0.01 line is the separate 0.01 eV figure.
The copper block prefers cuh2_fcc_rec_N.out and falls back to .txt.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HIT_ABOVE_EV = 0.01
REFERENCE_EV = -415.971529
REFERENCE_TOL_EV = 1e-3
BEST = re.compile(r"seed \d+: best (\S+) eV")


def best_token(path: Path) -> str:
    match = BEST.search(path.read_text())
    if match is None:
        raise SystemExit(f"no best line in {path.name}")
    return match.group(1)


def seed_file(directory: Path, stem: str, index: int) -> Path:
    for suffix in (".out", ".txt"):
        path = directory / f"{stem}_{index}{suffix}"
        if path.is_file():
            return path
    raise SystemExit(f"missing {stem}_{index}.out")


def block(directory: Path, stem: str, label: str) -> str:
    tokens = []
    for index in range(8):
        tokens.append(best_token(seed_file(directory, stem, index)))
    finite = [float(token) for token in tokens if token != "inf"]
    if not finite:
        raise SystemExit("no finite best")
    floor = min(finite)
    lines = [label]
    hits = 0
    for index, token in enumerate(tokens):
        if token == "inf":
            shown = "None"
            hit = False
        else:
            shown = token
            hit = float(token) - floor <= HIT_ABOVE_EV
        hits += int(hit)
        lines.append(f"  {stem}_{index}.out best={shown} hit={hit}")
    lines.append(f"  {hits}/8={hits}/8")
    return "\n".join(lines) + "\n"


def copper_block(directory: Path) -> str:
    tokens = []
    for index in range(8):
        tokens.append(best_token(seed_file(directory, "cuh2_fcc_rec", index)))
    lines = ["cu hq_sci_cuh2"]
    hits = 0
    finite = []
    for index, token in enumerate(tokens):
        if token == "inf":
            shown = "None"
            hit = False
        else:
            shown = token
            value = float(token)
            finite.append(value)
            hit = value < REFERENCE_EV + REFERENCE_TOL_EV
        hits += int(hit)
        gap = None if token == "inf" else value - REFERENCE_EV
        lines.append(f"  cuh2_fcc_rec_{index}.out best={shown} hit={hit}")
        if gap is not None:
            lines.append(f"  gap_to_reference {index} {gap:.6f}")
    lines.append(f"  {hits}/8={hits}/8")
    floor = min(finite) if finite else None
    relative = 0
    if floor is not None:
        relative = sum(item - floor <= HIT_ABOVE_EV for item in finite)
    lines.append(f"  set_relative_0.01 {relative}/8")
    return "\n".join(lines) + "\n"


def tally(hexamer: Path, copper: Path) -> str:
    return block(hexamer, "h2o6_rec", "hexamer hq_sci_h2o6") + copper_block(
        copper
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    hexamer = Path(args[0]) if args else Path("paper/jctc/data/hq_sci_h2o6")
    copper = Path(args[1]) if len(args) > 1 else Path("paper/jctc/data/hq_sci_cuh2")
    sys.stdout.write(tally(hexamer, copper))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

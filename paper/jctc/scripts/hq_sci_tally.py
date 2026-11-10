#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Rebuild the tally from the committed rec logs.

A finite best within 0.01 eV of the lowest finite best is a hit.
An inf best is recorded as best=None and is not a hit.
The copper block is read from the eight committed cuh2 logs.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HIT_ABOVE_EV = 0.01
BEST = re.compile(r"seed \d+: best (\S+) eV")


def best_token(path: Path) -> str:
    match = BEST.search(path.read_text())
    if match is None:
        raise SystemExit(f"no best line in {path.name}")
    return match.group(1)


def block(directory: Path, stem: str, label: str) -> str:
    tokens = []
    for index in range(8):
        path = directory / f"{stem}_{index}.txt"
        if not path.is_file():
            raise SystemExit(f"missing {path.name}")
        tokens.append(best_token(path))
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


def tally(hexamer: Path, copper: Path) -> str:
    return block(hexamer, "h2o6_rec", "hexamer hq_sci_h2o6") + block(
        copper, "cuh2_fcc_rec", "cu hq_sci_cuh2"
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    hexamer = Path(args[0]) if args else Path("paper/jctc/data/hq_sci_h2o6")
    copper = Path(args[1]) if len(args) > 1 else Path("paper/jctc/data/hq_sci_cuh2")
    sys.stdout.write(tally(hexamer, copper))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

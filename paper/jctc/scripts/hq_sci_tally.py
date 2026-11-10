#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Rebuild the hexamer tally from the shipped rec logs.

A finite best within 0.01 eV of the lowest finite best is a hit.
An inf best is recorded as best=None and is not a hit.
The copper block is omitted when no cuh2 rec log is present.
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


def hexamer_block(directory: Path) -> str:
    tokens = []
    for index in range(8):
        path = directory / f"h2o6_rec_{index}.txt"
        if not path.is_file():
            raise SystemExit(f"missing {path.name}")
        tokens.append(best_token(path))
    finite = [float(token) for token in tokens if token != "inf"]
    if not finite:
        raise SystemExit("no finite best")
    floor = min(finite)
    lines = ["hexamer hq_sci_h2o6"]
    hits = 0
    for index, token in enumerate(tokens):
        if token == "inf":
            shown = "None"
            hit = False
        else:
            shown = token
            hit = float(token) - floor <= HIT_ABOVE_EV
        hits += int(hit)
        lines.append(
            f"  h2o6_rec_{index}.out best={shown} hit={hit}"
        )
    lines.append(f"  {hits}/8={hits}/8")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    directory = Path(args[0]) if args else Path("paper/jctc/data/hq_sci_h2o6")
    sys.stdout.write(hexamer_block(directory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

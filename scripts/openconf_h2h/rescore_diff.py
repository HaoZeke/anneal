#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Compare the scored cells of two ``rows.csv`` files of the same runs.

Rows pair on (molecule, arm, seed). Per molecule and arm the table gives the
hits and the mirror hits of each file and how many scored cells differ, over
``hit``, ``hit_mirror``, ``calls_to_hit``, ``delta_min_kcal``, ``candidates``
and ``invalid``. ``--cells`` lists every differing cell. A run in one file
only is an error.

    rescore_diff.py --before old/rows.csv --after new/rows.csv --out table.tsv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

CELLS = ("hit", "hit_mirror", "calls_to_hit", "delta_min_kcal", "candidates", "invalid")
FIELDS = ["molecule", "arm", "n", "hits_before", "hits_after", "hit_mirror_before", "hit_mirror_after", "cells_differing"]


def read(path: Path) -> dict[tuple[str, str, str], dict]:
    with path.open(newline="") as handle:
        return {(r["molecule"], r["arm"], r["seed"]): r for r in csv.DictReader(handle)}


def differing(before: dict, after: dict) -> list[tuple[str, str, str, str, str, str]]:
    if before.keys() != after.keys():
        raise SystemExit(f"runs differ: {len(before.keys() ^ after.keys())} in one file only")
    return [
        (*key, cell, before[key][cell], after[key][cell])
        for key in sorted(before, key=lambda k: (k[0], k[1], int(k[2])))
        for cell in CELLS
        if before[key][cell] != after[key][cell]
    ]


def table(before: dict, after: dict) -> list[dict]:
    differing(before, after)
    out = []
    for molecule, arm in dict.fromkeys((m, a) for m, a, _ in before):
        keys = [k for k in before if k[:2] == (molecule, arm)]
        out.append(
            {
                "molecule": molecule,
                "arm": arm,
                "n": len(keys),
                "hits_before": sum(int(before[k]["hit"]) for k in keys),
                "hits_after": sum(int(after[k]["hit"]) for k in keys),
                "hit_mirror_before": sum(int(before[k]["hit_mirror"]) for k in keys),
                "hit_mirror_after": sum(int(after[k]["hit_mirror"]) for k in keys),
                "cells_differing": len(differing({k: before[k] for k in keys}, {k: after[k] for k in keys})),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cells", type=Path, help="write every differing cell here")
    args = parser.parse_args()
    before, after = read(args.before), read(args.after)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(table(before, after))
    if args.cells:
        with args.cells.open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["molecule", "arm", "seed", "cell", "before", "after"])
            writer.writerows(differing(before, after))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

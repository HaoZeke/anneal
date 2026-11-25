#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Run the serial time-to-hit worker-scaling campaign."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

SEEDS = tuple(range(2000, 2012))
WORKERS = (1, 2, 4, 8)
ASCENDING = (1, 2, 4, 8)
DESCENDING = (8, 4, 2, 1)


def run_one(binary: Path, rows: Path, system: str, seed: int, workers: int) -> None:
    subprocess.run(
        [str(binary), system, str(seed), str(workers), str(rows)],
        check=True,
    )


def hits(rows: Path, system: str, workers: int) -> int:
    with rows.open(newline="") as handle:
        found = [
            row
            for row in csv.DictReader(handle)
            if row["system"] == system and int(row["workers"]) == workers
        ]
    if len(found) != len(SEEDS):
        raise SystemExit(f"{system} p={workers} has {len(found)} rows")
    return sum(int(row["hit"]) for row in found)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    args = parser.parse_args()

    args.rows.parent.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        run_one(args.binary, args.rows, "au38", seed, 1)
    au_hits = hits(args.rows, "au38", 1)
    second = "au38" if au_hits >= 10 else "m55r6"
    print(f"Au38 p=1 hits {au_hits}/12; second system {second}", flush=True)

    for system in ("lj38", second):
        for index, seed in enumerate(SEEDS):
            order = ASCENDING if index % 2 == 0 else DESCENDING
            for workers in order:
                run_one(args.binary, args.rows, system, seed, workers)
    if second == "m55r6":
        for seed in SEEDS:
            run_one(args.binary, args.rows, second, seed, 1)
        print(f"M55 rho=6 p=1 hits {hits(args.rows, second, 1)}/12", flush=True)


if __name__ == "__main__":
    main()

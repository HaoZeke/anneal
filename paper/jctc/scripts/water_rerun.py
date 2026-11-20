#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Read the 192-seed water hexamer rerun against CREST from its receipts.

Reads data/water_v45b/rerun-9a30f319/: one status receipt per arm and seed, every
receipt held to SHA256SUMS. Both arms of a seed start from the same geometry, so the
arms pair by seed, and every statistic comes from scripts/paired_stats.py.

Usage: water_rerun.py [--data DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import paired_stats  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "water_v45b" / "rerun-9a30f319"
ARMS = ("anneal", "crest")
SEEDS = tuple(range(192))
BUDGET = 4000


def die(message: str) -> NoReturn:
    raise SystemExit(f"water_rerun: {message}")


@dataclass(frozen=True)
class Rerun:
    hits: dict[str, int]
    wilson: dict[str, tuple[float, float]]
    walls: dict[str, float]
    table: tuple[int, int, int, int]
    p: float
    newcombe: tuple[float, float]


def receipts(data: Path) -> dict[str, list[dict]]:
    sums = {}
    for line in (data / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        sums[name.lstrip("*").removeprefix("./")] = digest
    out: dict[str, list[dict]] = {}
    for arm in ARMS:
        rows = []
        for seed in SEEDS:
            name = f"{arm}/seed-{seed:04d}.json"
            path = data / name
            if sums.get(name) != hashlib.sha256(path.read_bytes()).hexdigest():
                die(f"{name} does not match SHA256SUMS")
            row = json.loads(path.read_text())
            if row["arm"] != arm or row["seed"] != seed or row["budget"] != BUDGET:
                die(f"{name} is not {arm} seed {seed} at {BUDGET}")
            rows.append(row)
        out[arm] = rows
    if len(sums) != len(ARMS) * len(SEEDS):
        die(f"SHA256SUMS lists {len(sums)} receipts")
    return out


def load(data: Path = DATA) -> Rerun:
    rows = receipts(data)
    for first, second in zip(rows["anneal"], rows["crest"]):
        if first["start_sha256"] != second["start_sha256"]:
            die(f"seed {first['seed']} starts differ")
    if any(row["force_count"] != BUDGET for row in rows["anneal"]):
        die("an anneal receipt spent other than the budget")
    if any(row["calls"] != BUDGET or not row["engrad_cap_reached"] for row in rows["crest"]):
        die("a CREST receipt did not stop at the energy and gradient cap")
    flags = {arm: [bool(row["hit"]) for row in rows[arm]] for arm in ARMS}
    table = paired_stats.outcome_table(flags["anneal"], flags["crest"])
    hits = {arm: sum(flags[arm]) for arm in ARMS}
    return Rerun(
        hits=hits,
        wilson={arm: paired_stats.wilson(hits[arm], len(SEEDS)) for arm in ARMS},
        walls={arm: statistics.quantiles([float(row["wall_seconds"]) for row in rows[arm]], n=2,
                                         method="inclusive")[0] for arm in ARMS},
        table=table,
        p=paired_stats.mcnemar_exact(table[1], table[2]),
        newcombe=paired_stats.newcombe_paired(*table),
    )


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DATA)
    rerun = load(parser.parse_args(argv).data)
    print(f"hits anneal {rerun.hits['anneal']}/192 crest {rerun.hits['crest']}/192")
    print("table both {} anneal-only {} crest-only {} neither {}".format(*rerun.table))
    print(f"mcnemar p {rerun.p:.6g}")
    print(f"newcombe {rerun.newcombe[0]:.6f} {rerun.newcombe[1]:.6f}")
    print(f"median wall anneal {rerun.walls['anneal']:.6f} crest {rerun.walls['crest']:.6f}")


if __name__ == "__main__":
    main(sys.argv[1:])

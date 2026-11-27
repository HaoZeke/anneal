#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Merge the four NH3 octamer Pareto shard tables into one rows file.

The machine column is the literal unrecorded. This harness does not read
a machine identifier.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARDS = HERE / "shards"
OUT = HERE / "rows.csv"
FIELDS = (
    "seed",
    "arm",
    "hit",
    "calls",
    "wall_s",
    "best_energy",
    "revision",
    "shard",
    "machine_id",
)
MACHINE_ID = "unrecorded"
LETTERS = ("A", "B", "C", "D")
HIT_YES = {"yes", "1"}
HIT_NO = {"no", "0"}


def hit_bit(value: str) -> str:
    text = value.strip().lower()
    if text in HIT_YES:
        return "1"
    if text in HIT_NO:
        return "0"
    raise SystemExit(f"unexpected hit value {value!r}")


def censored_calls(budget: str, charged: str) -> str:
    if budget.strip().isdigit():
        return str(int(budget.strip()) + 1)
    if charged.strip().isdigit():
        return str(int(charged.strip()) + 1)
    raise SystemExit(f"miss has no numeric budget ({budget!r})")


def calls_of(row: dict[str, str], hit: str) -> str:
    first = row["calls_to_first_hit"].strip()
    if hit == "1":
        if not first:
            raise SystemExit(f"hit seed {row['seed']} has no calls_to_first_hit")
        return first
    censored = censored_calls(row["budget"], row["calls_charged"])
    if first and first != censored:
        raise SystemExit(
            f"miss seed {row['seed']} records {first}, budget plus 1 is {censored}"
        )
    return censored


def load_shard(letter: str) -> list[dict[str, str]]:
    path = SHARDS / f"rows-{letter}.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{path.name} is empty")
    return rows


def merged_row(letter: str, row: dict[str, str]) -> dict[str, str]:
    hit = hit_bit(row["hit"])
    return {
        "seed": row["seed"].strip(),
        "arm": row["method"].strip(),
        "hit": hit,
        "calls": calls_of(row, hit),
        "wall_s": row["wall_s"].strip(),
        "best_energy": row["best_energy_Eh"].strip(),
        "revision": row["revision"].strip(),
        "shard": letter,
        "machine_id": MACHINE_ID,
    }


def merge_rows() -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    for letter in LETTERS:
        for row in load_shard(letter):
            merged.append(merged_row(letter, row))
    return merged


def write_rows(rows: list[dict[str, str]], path: Path = OUT) -> None:
    if any(row["machine_id"] != MACHINE_ID for row in rows):
        raise SystemExit("machine_id must be the literal unrecorded")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str]) -> None:
    path = OUT
    if argv:
        if len(argv) == 2 and argv[0] == "--out":
            path = Path(argv[1])
        else:
            raise SystemExit("usage: run.py [--out PATH]")
    rows = merge_rows()
    write_rows(rows, path)
    print(f"ROWS {len(rows)}")


if __name__ == "__main__":
    main(sys.argv[1:])

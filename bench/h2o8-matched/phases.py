#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Phase split of a matched (H2O)8 run.

Reads each seed's run.txt. The per-seed logs are not committed. A new run
appends hop_end, polish_begin, pre_Eh, first_hit, where, and margin, and
those lines are used when they are present. Older logs are derived from
search_calls, polished_from_eV, hartree_eV, first_hit_call, lowest_Eh, and
threshold_Eh.
"""

from __future__ import annotations

import csv
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = tuple(range(2000, 2012))
COLUMNS = (
    "seed",
    "hop_end",
    "polish_begin",
    "pre_Eh",
    "first_hit",
    "where",
    "lowest_Eh",
    "margin",
)


def fields_of(path: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        found[key] = value.strip()
    return found


def _margin(threshold: str, lowest: str) -> str:
    return f"{Decimal(threshold) - Decimal(lowest):.3e}"


def _pre_eh(polished_from_ev: str, hartree_ev: str) -> str:
    return f"{Decimal(polished_from_ev) / Decimal(hartree_ev):.12f}"


def phase_of(seed: int, record: dict[str, str]) -> dict[str, str]:
    if "hop_end" in record:
        return {column: record[column] if column != "seed" else str(seed) for column in COLUMNS}
    hop_end = int(record["search_calls"])
    raw_hit = record["first_hit_call"]
    hit = int(record["hit"])
    if hit == 0 or raw_hit == "none":
        first_hit = int(record["calls_to_hit"])
        where = "none"
    else:
        first_hit = int(raw_hit)
        where = "polish" if first_hit > hop_end else "hop"
    return {
        "seed": str(seed),
        "hop_end": str(hop_end),
        "polish_begin": str(hop_end + 1),
        "pre_Eh": _pre_eh(record["polished_from_eV"], record["hartree_eV"]),
        "first_hit": str(first_hit),
        "where": where,
        "lowest_Eh": record["lowest_Eh"],
        "margin": _margin(record["threshold_Eh"], record["lowest_Eh"]),
    }


def phase_lines(seed: int, record: dict[str, str]) -> str:
    phase = phase_of(seed, record)
    keys = ("hop_end", "polish_begin", "pre_Eh", "first_hit", "where", "margin")
    return "".join(f"{key}: {phase[key]}\n" for key in keys)


def append_phase(path: Path, seed: int) -> None:
    text = path.read_text()
    if "\nhop_end: " in f"\n{text}":
        return
    record = fields_of(path)
    path.write_text(text if text.endswith("\n") else text + "\n")
    with path.open("a") as handle:
        handle.write(phase_lines(seed, record))


def write_phases(path: Path) -> None:
    rows = []
    for seed in SEEDS:
        record = fields_of(HERE / "runs" / str(seed) / "run.txt")
        rows.append(phase_of(seed, record))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    write_phases(HERE / "phases.csv")


if __name__ == "__main__":
    main()

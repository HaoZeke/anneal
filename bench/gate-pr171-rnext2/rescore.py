# Copyright (c) anneal developers
"""Rescore committed gate rows against the current stop levels.

The original rows.csv stays unchanged.  Every row's hit is recomputed
with the harness rule: a clean exit with the observed and returned
best energies at or below the cell's stop level.  The rescored rows go
to rows-rescored.csv with the original hit kept as hit_recorded, and
the full summary goes to results-rescored.json.  Nothing is rerun.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cells as gate
import summarize

HERE = Path(__file__).resolve().parent


def rescore(row: dict[str, str]) -> str:
    cell = next(cell for cell in gate.CELLS if cell.name == row["cell"])
    stop = gate.stop_level(cell)
    hit = (
        int(row["exit_code"]) == 0
        and float(row["seen_lowest"]) <= stop
        and float(row["out_best"]) <= stop
    )
    return "1" if hit else "0"


def main() -> int:
    with (HERE / "rows.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    changed = []
    for row in rows:
        row["hit_recorded"] = row["hit"]
        row["hit"] = rescore(row)
        if row["hit"] != row["hit_recorded"]:
            changed.append((row["cell"], row["arm"], row["seed"]))
    out_fields = fields + ["hit_recorded"]
    with (HERE / "rows-rescored.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    found = {(r["cell"], r["arm"], int(r["seed"])): r for r in rows}
    summary, verdict = summarize.evaluate(found)
    cells_changed = sorted({cell for cell, _, _ in changed})
    result = {
        "source": "rows.csv",
        "stop_levels": {cell.name: gate.stop_level(cell) for cell in gate.CELLS},
        "rows_changed": len(changed),
        "cells_changed": cells_changed,
        "verdict": verdict,
        "cells": summary,
    }
    with (HERE / "results-rescored.json").open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(verdict, len(changed), ",".join(cells_changed))
    return 0


if __name__ == "__main__":
    sys.exit(main())

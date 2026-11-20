#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Append the replay columns to the G3 rows.

Usage: add_replay.py ROWS.csv REPLAY.csv OUT.csv

Adds ``best_bits``, ``hop_first_best``, ``replay_trace_match`` and
``replay_best_match`` from ``replay.py`` to every row; the stored columns
are copied unchanged.
"""

from __future__ import annotations

import csv
import sys

ADDED = {
    "best_bits": "best_bits",
    "hop_first_best": "hop_first_best",
    "replay_trace_match": "trace_match",
    "replay_best_match": "best_match",
}


def main() -> int:
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    rows_path, replay_path, out_path = sys.argv[1:]
    with open(rows_path) as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames) + [name for name in ADDED if name not in reader.fieldnames]
        rows = list(reader)
    with open(replay_path) as handle:
        replay = {(r["system"], r["method"], r["seed"]): r for r in csv.DictReader(handle)}
    for row in rows:
        found = replay[(row["system"], row["method"], row["seed"])]
        for name, source in ADDED.items():
            row[name] = found[source]
    with open(out_path, "w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())

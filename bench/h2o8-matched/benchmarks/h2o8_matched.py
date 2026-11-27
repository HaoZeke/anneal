# Copyright (c) anneal developers
"""Track calls_to_hit from the committed rows.

Reading the table is the whole benchmark. It does not start a search.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROWS = Path(__file__).resolve().parents[1] / "rows.csv"
SEEDS = list(range(2000, 2012))


def calls_to_hit(seed: int) -> float:
    with ROWS.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["seed"]) == seed:
                return float(row["calls_to_hit"])
    raise RuntimeError(f"seed {seed} is missing from {ROWS.name}")


class H2o8Matched:
    params = SEEDS
    param_names = ["seed"]
    timeout = 30

    def track_calls_to_hit(self, seed: int) -> float:
        return calls_to_hit(int(seed))

# Copyright (c) anneal developers
"""Paired statistics of the move-a rows with scripts/paired_stats.py.

Per system and pooled over the shard: hits per arm, the paired outcome
table, the two-sided exact McNemar test, and the Newcombe method-10
interval for the stock hit rate less the knockout hit rate.

    python3 paper/jctc/data/knockouts/move-a/stats.py \
        paper/jctc/data/knockouts/move-a/rows.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "scripts"))
import paired_stats  # noqa: E402

SYSTEMS = ("lj38", "lj55", "lj75", "au55", "m55r6", "m55r14", "m80r6", "m80r14")
STOCK = "stock"
KNOCKOUT = "H2H_KNOCKOUT=move_choice"


def main(path: str) -> int:
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["system"], row["arm"], row["seed"])
        if key in by_key:
            raise SystemExit(f"duplicate row {key}")
        if row["hit"] not in ("0", "1"):
            raise SystemExit(f"row without an outcome {key}")
        by_key[key] = row
    paired_stats.self_test()
    print("system,n,stock_hits,knockout_hits,both,stock_only,knockout_only,neither,"
          "mcnemar_p,newcombe_low,newcombe_high")
    pooled_stock: list[bool] = []
    pooled_knock: list[bool] = []
    for system in SYSTEMS + ("all",):
        if system == "all":
            stock, knock = pooled_stock, pooled_knock
        else:
            seeds = sorted({s for (sy, _a, s) in by_key if sy == system}, key=int)
            stock = [by_key[(system, STOCK, s)]["hit"] == "1" for s in seeds]
            knock = [by_key[(system, KNOCKOUT, s)]["hit"] == "1" for s in seeds]
            pooled_stock += stock
            pooled_knock += knock
        both, only_stock, only_knock, neither = paired_stats.outcome_table(stock, knock)
        p = paired_stats.mcnemar_exact(only_stock, only_knock)
        low, high = paired_stats.newcombe_paired(both, only_stock, only_knock, neither)
        print(f"{system},{len(stock)},{sum(stock)},{sum(knock)},{both},{only_stock},{only_knock},{neither},"
              f"{p:.6g},{low:.4f},{high:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))

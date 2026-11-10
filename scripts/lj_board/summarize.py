#!/usr/bin/env python3
"""Build NUMBERS.tsv from ASE and AGOX logs.

Quartiles are Hyndman-Fan type 7.
The call column is the logged count. A miss is not rewritten as the paper budget.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from boardlib import median, type7_iqr


def parse_log(path: Path) -> dict:
    fields = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        for token in line.split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            fields[key] = value
    return fields


def main() -> int:
    paths = [Path(arg) for arg in sys.argv[1:]]
    if not paths:
        raise SystemExit("usage: summarize.py LOG [LOG...]")
    groups = defaultdict(list)
    for path in paths:
        fields = parse_log(path)
        if fields.get("exit_code") not in {"0", "75"}:
            raise SystemExit(f"{path} did not finish cleanly")
        if fields.get("stop") == "error":
            raise SystemExit(f"{path} recorded an error")
        key = (fields["code"], int(fields["n"]))
        groups[key].append(fields)
    print("kind\tsize\tcode\tmetric\tvalue\tsource")
    for code, size in sorted(groups, key=lambda item: (item[1], item[0])):
        rows = groups[(code, size)]
        if len(rows) != 6:
            raise SystemExit(f"{code} {size} has {len(rows)} logs, expected 6")
        calls = [float(row["calls"]) for row in rows]
        charged = [float(row["charged"]) for row in rows]
        walls = [float(row["wall_s"]) for row in rows]
        hits = sum(int(row["hit"]) for row in rows)
        source = f"results/lj-board/{code}/lj{size}-seed*.log"
        cells = {
            "hits": float(hits),
            "median_calls": median(calls),
            "iqr_calls": type7_iqr(calls),
            "median_charged": median(charged),
            "iqr_charged": type7_iqr(charged),
            "median_wall_s": median(walls),
            "iqr_wall_s": type7_iqr(walls),
        }
        for metric, value in cells.items():
            print(f"h2h\t{size}\t{code}\t{metric}\t{value:.6f}\t{source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

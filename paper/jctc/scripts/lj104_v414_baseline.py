#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Print the old-method LJ104 baseline from the measure v4.14 receipts.

Reads logs/lj104-v414/<seed>.log for seeds 2000 through 2005. Each file is a
copy of that seed's budget receipt. The summary row follows the header
system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best. The hit count is
how many of those rows record hit=true. The best energy and its seed are taken
from the same rows, as written. The reference token is the reference line in
logs/lj104-seed0.log. Every recorded best lies above that reference.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(2000, 2006))


def _summary(path: Path) -> dict[str, str]:
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("system,code,seed,"):
            continue
        header = line.split(",")
        fields = lines[index + 1].split(",")
        wanted = ("seed", "hit", "best")
        missing = [name for name in wanted if name not in header]
        if missing:
            raise SystemExit(f"{path} summary lacks {missing[0]}")
        return {name: fields[header.index(name)] for name in wanted} | {"path": path}
    raise SystemExit(f"no summary header in {path}")


def _reference(path: Path) -> str:
    marker = "reference "
    for line in path.read_text().splitlines():
        if marker not in line:
            continue
        return line.split(marker, 1)[1].strip().split()[0]
    raise SystemExit(f"no reference line in {path}")


def baseline(root: Path | None = None) -> dict:
    logs = root or (ROOT / "logs")
    rows = []
    for seed in SEEDS:
        path = logs / "lj104-v414" / f"{seed}.log"
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        row = _summary(path)
        if row["seed"] != str(seed):
            raise SystemExit(f"{path} records seed {row['seed']}")
        rows.append(row)
    hits = sum(row["hit"] == "true" for row in rows)
    best = min(rows, key=lambda row: float(row["best"]))
    reference = _reference(logs / "lj104-seed0.log")
    if any(float(row["best"]) <= float(reference) for row in rows):
        raise SystemExit("a receipt best is at or below the reference")
    return {
        "hits": str(hits),
        "n": str(len(rows)),
        "best": best["best"],
        "seed": best["seed"],
        "reference": reference,
        "rows": rows,
    }


def main() -> int:
    found = baseline()
    print(f"hits {found['hits']} of {found['n']}")
    print(f"best {found['best']} seed {found['seed']}")
    print(f"reference {found['reference']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

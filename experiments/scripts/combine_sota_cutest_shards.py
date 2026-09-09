"""Combine sota_cutest.py shards. Keys are problem, method, seed."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from experiments.scripts.sota_cutest import FIELDNAMES


def _row_key(row: dict) -> tuple[str, str, str]:
    return (str(row["problem"]), str(row["method"]), str(row["seed"]))


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDNAMES:
            raise ValueError(f"{path} has unexpected field names: {reader.fieldnames}")
        return list(reader)


def combine_rows(paths: list[Path]) -> list[dict]:
    by_key: dict[tuple[str, str, str], dict] = {}
    for path in paths:
        for row in load_rows(path):
            key = _row_key(row)
            existing = by_key.get(key)
            if existing is not None and existing != row:
                raise ValueError(f"conflicting duplicate SOTA cell: {key} in {path}")
            by_key[key] = row
    rows = list(by_key.values())
    rows.sort(key=lambda row: (row["problem"], int(row["seed"]), row["method"]))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = combine_rows(args.shards)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

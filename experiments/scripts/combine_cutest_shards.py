"""Combine sharded CUTEst benchmark CSVs into one long-form CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from experiments.scripts.run_cutest_full_suite import FIELD_NAMES, write_rows


def _row_key(row: dict) -> tuple[str, str, str]:
    return (str(row["problem"]), str(row["driver"]), str(row["seed"]))


def _sort_key(row: dict) -> tuple[str, int, str]:
    try:
        seed = int(row["seed"])
    except (TypeError, ValueError):
        seed = 0
    return (str(row["problem"]), seed, str(row["driver"]))


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELD_NAMES:
            raise ValueError(f"{path} has unexpected field names: {reader.fieldnames}")
        return list(reader)


def combine_rows(paths: list[Path]) -> list[dict]:
    by_key: dict[tuple[str, str, str], dict] = {}
    for path in paths:
        for row in load_rows(path):
            key = _row_key(row)
            existing = by_key.get(key)
            if existing is not None and existing != row:
                raise ValueError(
                    "conflicting duplicate CUTEst cell: "
                    f"{key[0]} / {key[1]} / seed {key[2]}"
                )
            by_key[key] = row
    return sorted(by_key.values(), key=_sort_key)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = combine_rows(args.csv)
    write_rows(str(args.out), rows)
    print(f"Wrote {len(rows)} row(s) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

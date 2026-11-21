#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Top symbols by the change in outside self share, B against A."""

from __future__ import annotations

import argparse
import csv


def load(path: str) -> dict[str, dict[str, float]]:
    rows = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            rows[row["symbol"]] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    left = load(args.a)
    right = load(args.b)
    names = set(left) | set(right)

    def share(table: dict[str, dict[str, str]], name: str) -> float:
        row = table.get(name)
        if row is None:
            return 0.0
        return float(row["self_share"])

    def samples(table: dict[str, dict[str, str]], name: str) -> int:
        row = table.get(name)
        if row is None:
            return 0
        return int(row["self_samples"])

    ranked = sorted(names, key=lambda name: share(right, name) - share(left, name), reverse=True)
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "symbol",
                "self_share_a",
                "self_share_b",
                "delta_share",
                "self_samples_a",
                "self_samples_b",
                "self_per_call_a",
                "self_per_call_b",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for name in ranked[: args.limit]:
            a_row = left.get(name, {})
            b_row = right.get(name, {})
            writer.writerow(
                {
                    "symbol": name,
                    "self_share_a": f"{share(left, name):.6e}",
                    "self_share_b": f"{share(right, name):.6e}",
                    "delta_share": f"{share(right, name) - share(left, name):.6e}",
                    "self_samples_a": samples(left, name),
                    "self_samples_b": samples(right, name),
                    "self_per_call_a": a_row.get("self_per_call_s", "0"),
                    "self_per_call_b": b_row.get("self_per_call_s", "0"),
                }
            )


if __name__ == "__main__":
    main()

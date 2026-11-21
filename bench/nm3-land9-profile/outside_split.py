#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Outside-potential time per call from folded stacks and run rows.

Sample fractions are scaled by the timer's outside seconds, then
divided by the charged potential calls.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict


def load_folded(path: str) -> list[tuple[list[str], int]]:
    rows = []
    with open(path, newline="\n") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            stack, _, count = line.rpartition(" ")
            rows.append((stack.split(";"), int(count)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folded", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    seeds = {int(part) for part in args.seeds.split(",")}
    outside = 0.0
    charged = 0
    with open(args.runs, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["arm"] != args.arm or row["system"] != args.system:
                continue
            if int(row["seed"]) not in seeds:
                continue
            outside += float(row["outside_s"])
            charged += int(row["charged"])
    stacks = load_folded(args.folded)
    total = sum(count for _, count in stacks)
    self_counts: dict[str, int] = defaultdict(int)
    incl_counts: dict[str, int] = defaultdict(int)
    for frames, count in stacks:
        if not frames:
            continue
        self_counts[frames[-1]] += count
        seen = set()
        for frame in frames:
            if frame in seen:
                continue
            seen.add(frame)
            incl_counts[frame] += count
    names = sorted(set(self_counts) | set(incl_counts), key=lambda name: -self_counts[name])
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "arm",
                "symbol",
                "self_samples",
                "incl_samples",
                "self_share",
                "incl_share",
                "self_s",
                "incl_s",
                "self_per_call_s",
                "incl_per_call_s",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for name in names:
            self_n = self_counts[name]
            incl_n = incl_counts[name]
            self_share = self_n / total if total else 0.0
            incl_share = incl_n / total if total else 0.0
            self_s = self_share * outside
            incl_s = incl_share * outside
            writer.writerow(
                {
                    "arm": args.arm,
                    "symbol": name,
                    "self_samples": self_n,
                    "incl_samples": incl_n,
                    "self_share": f"{self_share:.6e}",
                    "incl_share": f"{incl_share:.6e}",
                    "self_s": f"{self_s:.6f}",
                    "incl_s": f"{incl_s:.6f}",
                    "self_per_call_s": f"{(self_s / charged) if charged else 0.0:.9e}",
                    "incl_per_call_s": f"{(incl_s / charged) if charged else 0.0:.9e}",
                }
            )


if __name__ == "__main__":
    main()

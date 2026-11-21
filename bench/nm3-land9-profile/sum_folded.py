#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Add per-seed folded stacks into one file per arm."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def load(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    text = path.read_text()
    for line in text.splitlines():
        if not line:
            continue
        stack, _, count = line.rpartition(" ")
        counts[stack] += int(count)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()
    total: Counter[str] = Counter()
    for name in args.inputs:
        total.update(load(Path(name)))
    lines = [f"{stack} {count}\n" for stack, count in sorted(total.items())]
    Path(args.out).write_text("".join(lines))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Collapse perf script output to folded stacks.

A sample is outside the potential when no frame name contains one of
the inside markers. Paths and addresses are dropped.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter


def symbol_of(line: str) -> str:
    text = line.strip()
    if not text or text.startswith("#"):
        return ""
    # Drop a leading address.
    parts = text.split()
    if parts and all(c in "0123456789abcdefABCDEFxX" for c in parts[0]):
        parts = parts[1:]
    if not parts:
        return ""
    name = parts[0]
    if name.startswith("("):
        return ""
    plus = name.find("+")
    if plus > 0:
        name = name[:plus]
    if name.startswith("[") and name.endswith("]"):
        return ""
    return name


def frames_of(block: list[str]) -> list[str]:
    frames: list[str] = []
    for line in block:
        if not line.startswith("\t") and not line.startswith("        "):
            continue
        name = symbol_of(line)
        if name:
            frames.append(name)
    frames.reverse()
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inside", required=True, help="comma separated markers")
    parser.add_argument("--all-out", required=True)
    parser.add_argument("--outside-out", required=True)
    args = parser.parse_args()
    markers = [item for item in args.inside.split(",") if item]
    all_counts: Counter[str] = Counter()
    out_counts: Counter[str] = Counter()
    block: list[str] = []

    def flush() -> None:
        if not block:
            return
        frames = frames_of(block)
        block.clear()
        if not frames:
            return
        stack = ";".join(frames)
        all_counts[stack] += 1
        inside = any(any(marker in frame for marker in markers) for frame in frames)
        if not inside:
            out_counts[stack] += 1

    for line in sys.stdin:
        if line.startswith("\n") or line == "\n":
            flush()
            continue
        if line.startswith("\t") or (block and line.startswith(" ")):
            block.append(line)
            continue
        if block and not line.startswith("\t"):
            flush()
        block.append(line)
    flush()

    def write(path: str, counts: Counter[str]) -> None:
        lines = [f"{stack} {count}\n" for stack, count in sorted(counts.items())]
        with open(path, "w", newline="\n") as handle:
            handle.writelines(lines)

    write(args.all_out, all_counts)
    write(args.outside_out, out_counts)


if __name__ == "__main__":
    main()

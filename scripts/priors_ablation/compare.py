#!/usr/bin/env python3
# Copyright (c) anneal developers
"""One arm at two revisions, system by system, from rows.csv.

Each cell is hits out of n, the median calls to hit with a miss counted as
the full budget, and the median wall, inside (potential) and outside
seconds, as summarize.py scores them. The two revisions are compared with
the two-sided Fisher exact test of the hits and the two-sided exact
Mann-Whitney test of the calls to hit. Only rows at the two named
revisions are read, and a seed recorded twice at one revision is scored
from its last row.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from summarize import ORDER, cell, read, seed_cells, tests  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--arm", default="default")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    latest = {}
    for r in read(args.rows):
        if r["arm"] != args.arm or r["code_revision"] not in (args.base, args.head):
            continue
        r["hit"] = r["hit_binary"]
        latest[(r["system"], r["code_revision"], r["seed"])] = r
    lines = [
        f"| system | `{args.base}` | `{args.head}` | Fisher / MW |",
        "|---|---|---|---|",
    ]
    for system in ORDER:
        cells = {}
        for revision in (args.base, args.head):
            found = [r for (s, rev, _), r in latest.items() if s == system and rev == revision]
            cells[revision] = seed_cells(found) if found else None
        if not all(cells.values()):
            continue
        fisher_p, mw_p = tests(cells[args.base], cells[args.head])
        lines.append(
            f"| {system} | {cell(cells[args.base])} | {cell(cells[args.head])} "
            f"| {fisher_p:.3f} / {mw_p:.3f} |"
        )
    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

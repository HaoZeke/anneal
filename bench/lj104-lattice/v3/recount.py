#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Rebuild the development cut tables from raw logs.

Usage: recount.py LOGDIR OUTDIR

LOGDIR holds lowest-index/ and uniform/. Each seed log has
charged: and wall_seconds:. The matching requench file is
LOGDIR/../requench/<cut>-<seed>.txt with an energy: line.
"""

import re
import statistics
import sys
from pathlib import Path

STOP = -582.086641


def grab(text, pattern):
    match = re.search(pattern, text, re.M)
    if match is None:
        raise SystemExit(f"missing {pattern}")
    return match.group(1)


def load(logdir, cut):
    rows = []
    for seed in range(3000, 3030):
        log = (logdir / cut / f"{seed}.log").read_text()
        quench = (logdir.parent / "requench" / f"{cut}-{seed}.txt").read_text()
        energy = float(grab(quench, r"^energy:\s*(\S+)"))
        rows.append(
            {
                "seed": seed,
                "hit": int(energy <= STOP),
                "charged": int(grab(log, r"^charged:\s*(\d+)")),
                "wall": float(grab(log, r"^wall_seconds:\s*(\S+)")),
                "energy": energy,
                "start": grab(log, r"^start=(\S+)"),
                "end": grab(log, r"^end=(\S+)"),
            }
        )
    return rows


def median(values):
    return statistics.median(values)


def write_cut(path, rows):
    lines = ["seed\thit\tcharged\twall\tenergy\tstart\tend\n"]
    for row in rows:
        lines.append(
            f"{row['seed']}\t{row['hit']}\t{row['charged']}\t"
            f"{row['wall']:.3f}\t{row['energy']:.10f}\t"
            f"{row['start']}\t{row['end']}\n"
        )
    path.write_text("".join(lines))
    hits = [row["charged"] for row in rows if row["hit"]]
    print(
        path.name,
        "hits",
        sum(row["hit"] for row in rows),
        "median",
        median(hits) if hits else "none",
        "span",
        rows[0]["start"],
        rows[-1]["end"],
    )


def main():
    logdir = Path(sys.argv[1])
    outdir = Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    lowest = load(logdir, "lowest-index")
    uniform = load(logdir, "uniform")
    write_cut(outdir / "lowest-index.tsv", lowest)
    write_cut(outdir / "uniform.tsv", uniform)
    lines = ["seed\tlowest_hit\tlowest_charged\tuniform_hit\tuniform_charged\n"]
    for left, right in zip(lowest, uniform):
        lines.append(
            f"{left['seed']}\t{left['hit']}\t{left['charged']}\t"
            f"{right['hit']}\t{right['charged']}\n"
        )
    (outdir / "charged-diff.tsv").write_text("".join(lines))


if __name__ == "__main__":
    main()

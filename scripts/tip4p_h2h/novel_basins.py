#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Rows and summary of the novel-basin reward and basin-archive taboo trials.

``rows``
    One row per (variant, system, seed) from ``<runs>/<variant>/<system>-s<seed>.log``.
    TIP4P logs are ``h2h_seed`` output (the ``h2h_row`` line), LJ logs are
    ``h2h_gmin_proof`` output (the ``,anneal,`` line, with ``outside_s`` as
    wall minus ``pot_s``), and Sutton-Chen logs are ``h2h_sc`` output.

``summary``
    Hits, best, median best, median charged calls and median time inside and
    outside the potential per variant and system. Each variant other than
    ``base`` carries the two-sided Fisher exact p on its hits against
    ``base``, and the seeds it loses and gains.

The variant builds were scratch builds; only their logs are read here.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

from scipy.stats import fisher_exact

SEEDS = tuple(range(2000, 2012))
VARIANTS = ("base", "a", "b", "ab")
SYSTEMS = ("tip4p21", "tip4p20", "tip4p15", "tip4p8", "lj38", "lj75", "sc96-80")
ROW_FIELDS = ["variant", "system", "seed", "budget", "hit", "charged", "best", "wall_s",
              "pot_s", "outside_s"]


def row_from_log(variant: str, system: str, seed: int, path: Path) -> list[str]:
    text = path.read_text().splitlines()
    if system.startswith("tip4p"):
        line = next(l for l in text if l.startswith("h2h_row "))
        d = dict(token.split("=", 1) for token in line.split()[1:])
        return [variant, system, str(seed), d["budget"], d["hit"], d["charged"], d["best"],
                d["wall_s"], d["pot_s"], d["outside_s"]]
    if system.startswith("lj"):
        fields = next(l for l in text if ",anneal," in l).strip().split(",")
        name, _code, seed_s, budget, hit, _evals, charged, wall, best = fields[:9]
        pot = float(fields[13])
        return [variant, name, seed_s, budget, "1" if hit == "true" else "0", charged, best,
                wall, f"{pot:.6f}", f"{float(wall) - pot:.6f}"]
    header = next(l for l in text if l.startswith("system,"))
    data = next(l for l in text if l.startswith(f"{system},"))
    d = next(csv.DictReader([header, data]))
    return [variant, system, str(seed), d["budget"], "1" if d["hit"] == "true" else "0",
            d["charged"], d["best"], d["wall_s"], d["pot_s"], d["outside_s"]]


def cmd_rows(args: argparse.Namespace) -> None:
    writer = csv.writer(args.output, lineterminator="\n")
    writer.writerow(ROW_FIELDS)
    for variant in VARIANTS:
        for system in SYSTEMS:
            for seed in SEEDS:
                path = args.runs / variant / f"{system}-s{seed}.log"
                if path.is_file():
                    writer.writerow(row_from_log(variant, system, seed, path))


def cmd_summary(args: argparse.Namespace) -> None:
    group: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in csv.DictReader(args.rows):
        group.setdefault((row["variant"], row["system"]), []).append(row)
    writer = csv.writer(args.output, lineterminator="\n")
    writer.writerow(["variant", "system", "n", "hits", "best", "median_best",
                     "median_charged", "median_pot_s", "median_outside_s",
                     "fisher_p_vs_base", "lost_seeds", "gained_seeds"])
    for variant in VARIANTS:
        for system in SYSTEMS:
            rows = group.get((variant, system))
            if not rows:
                continue
            n = len(rows)
            hits = sum(int(r["hit"]) for r in rows)
            best = [float(r["best"]) for r in rows]
            base = group.get(("base", system), [])
            p = lost = gained = ""
            if variant != "base" and len(base) == n:
                base_hits = sum(int(r["hit"]) for r in base)
                p = f"{fisher_exact([[hits, n - hits], [base_hits, n - base_hits]])[1]:.4f}"
                base_hit = {r["seed"]: int(r["hit"]) for r in base}
                lost = " ".join(r["seed"] for r in rows if base_hit[r["seed"]] and not int(r["hit"]))
                gained = " ".join(r["seed"] for r in rows if int(r["hit"]) and not base_hit[r["seed"]])
            writer.writerow([
                variant, system, n, f"{hits}/{n}", f"{min(best):.10f}",
                f"{statistics.median(best):.6f}",
                f"{statistics.median(int(r['charged']) for r in rows):.1f}",
                f"{statistics.median(float(r['pot_s']) for r in rows):.6f}",
                f"{statistics.median(float(r['outside_s']) for r in rows):.6f}",
                p, lost, gained,
            ])


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    rows = sub.add_parser("rows")
    rows.add_argument("--runs", type=Path, required=True)
    rows.add_argument("--output", type=argparse.FileType("w"), default=sys.stdout)
    summary = sub.add_parser("summary")
    summary.add_argument("rows", type=argparse.FileType("r"))
    summary.add_argument("--output", type=argparse.FileType("w"), default=sys.stdout)
    args = parser.parse_args(argv[1:])
    {"rows": cmd_rows, "summary": cmd_summary}[args.command](args)


if __name__ == "__main__":
    main(sys.argv)

#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Gupta Au55, twelve seeds, the paper budget.

The search sphere and the neighbour shell follow the opening quench
when that quench is outside the reduced unit. A hit is a quenched
energy within 0.001 of the Cambridge minimum. GMIN rows already stored
in ``bench/au55-gupta`` are copied and not launched again.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
from pathlib import Path

SYSTEM = "au55"
BUDGET = 1_000_000
REFERENCE = -191.03085280
SEEDS = list(range(2000, 2012))
SEED_FIELDS = [
    "system",
    "code",
    "seed",
    "budget",
    "hit",
    "evals_to_hit",
    "charged",
    "wall_s",
    "best",
    "pot_s",
    "outside_s",
]


def median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 1:
        return ordered[0]
    h = 1.0 + (n - 1) * 0.5
    lower = int(h)
    frac = h - lower
    index = lower - 1
    if frac == 0.0:
        return ordered[index]
    return ordered[index] * (1.0 - frac) + ordered[index + 1] * frac


def fisher_two_sided(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    a, b = hits_a, n_a - hits_a
    c, d = hits_b, n_b - hits_b

    def prob(aa: int, bb: int, cc: int, dd: int) -> float:
        return (
            math.comb(aa + bb, aa)
            * math.comb(cc + dd, cc)
            / math.comb(aa + bb + cc + dd, aa + cc)
        )

    total = a + b + c + d
    row = a + b
    col = a + c
    observed = prob(a, b, c, d)
    score = 0.0
    for aa in range(max(0, row + col - total), min(row, col) + 1):
        bb = row - aa
        cc = col - aa
        dd = total - row - cc
        value = prob(aa, bb, cc, dd)
        if value <= observed + 1e-15:
            score += value
    return score


def binary_path(root: Path) -> Path:
    target = root / ".." / "tmp" / "anneal-target" / "release" / "examples" / "h2h_metal"
    if target.is_file():
        return target
    return Path("h2h_metal")


def parse_row(text: str) -> dict[str, str]:
    for line in text.splitlines():
        if line.startswith("au55,anneal,"):
            parts = line.split(",")
            hit = "1" if parts[4] in ("1", "true") else "0"
            return {
                "system": "au55",
                "code": "anneal",
                "seed": parts[2],
                "budget": parts[3],
                "hit": hit,
                "evals_to_hit": parts[5],
                "charged": parts[6],
                "wall_s": f"{float(parts[7]):.6f}",
                "best": f"{float(parts[8]):.8f}",
                "pot_s": f"{float(parts[9]):.6f}",
                "outside_s": f"{float(parts[10]):.6f}",
            }
    raise RuntimeError(text[-500:])


def run_anneal(binary: Path, seed: int) -> dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["RAYON_NUM_THREADS"] = "1"
    proc = subprocess.run(
        ["taskset", "-c", "0", str(binary), "run", SYSTEM, str(seed)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd="/work",
    )
    text = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise SystemExit(text[-800:])
    return parse_row(text)


def stored_gmin(root: Path) -> list[dict[str, str]]:
    path = root / "bench" / "au55-gupta" / "seeds.csv"
    if not path.is_file():
        return []
    rows = []
    for row in csv.DictReader(path.open()):
        if row["code"] != "gmin" or int(row["budget"]) != BUDGET:
            continue
        row = {field: row[field] for field in SEED_FIELDS}
        row["hit"] = "1" if row["hit"] in ("1", "true") else "0"
        rows.append(row)
    return rows


def summarise(anneal: list[dict[str, str]], gmin: list[dict[str, str]], path: Path) -> None:
    def hits(rows: list[dict[str, str]]) -> int:
        return sum(row["hit"] == "1" for row in rows)

    def column(rows: list[dict[str, str]], name: str) -> list[float]:
        return [float(row[name]) for row in rows]

    anneal_hits = hits(anneal)
    gmin_hits = hits(gmin)
    fields = [
        "arm",
        "hits",
        "n",
        "fisher_p_vs_anneal",
        "evals_median",
        "wall_median_s",
        "pot_median_s",
        "outside_median_s",
        "reference",
    ]
    lines = [
        {
            "arm": "anneal",
            "hits": str(anneal_hits),
            "n": str(len(anneal)),
            "fisher_p_vs_anneal": "",
            "evals_median": f"{median(column(anneal, 'evals_to_hit')):.6f}",
            "wall_median_s": f"{median(column(anneal, 'wall_s')):.6f}",
            "pot_median_s": f"{median(column(anneal, 'pot_s')):.6f}",
            "outside_median_s": f"{median(column(anneal, 'outside_s')):.6f}",
            "reference": f"{REFERENCE:.8f}",
        }
    ]
    if gmin:
        lines.append(
            {
                "arm": "gmin",
                "hits": str(gmin_hits),
                "n": str(len(gmin)),
                "fisher_p_vs_anneal": f"{fisher_two_sided(anneal_hits, len(anneal), gmin_hits, len(gmin)):.6g}",
                "evals_median": f"{median(column(gmin, 'evals_to_hit')):.6f}",
                "wall_median_s": f"{median(column(gmin, 'wall_s')):.6f}",
                "pot_median_s": f"{median(column(gmin, 'pot_s')):.6f}",
                "outside_median_s": f"{median(column(gmin, 'outside_s')):.6f}",
                "reference": f"{REFERENCE:.8f}",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    binary = binary_path(root)
    if args.seed is not None:
        row = run_anneal(binary, args.seed)
        print(",".join(row[field] for field in SEED_FIELDS), flush=True)
        return 0
    anneal = [run_anneal(binary, seed) for seed in SEEDS]
    gmin = stored_gmin(root)
    rows = []
    for seed in SEEDS:
        rows.extend(row for row in anneal if int(row["seed"]) == seed)
        rows.extend(row for row in gmin if int(row["seed"]) == seed)
    out = root / "bench" / "au55-scale"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "seeds.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEED_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summarise(anneal, gmin, out / "summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())

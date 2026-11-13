#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Morse rho = 14, M55 and M80, twelve seeds, paper budgets.

Anneal runs serially. A GMIN row already stored at the same budget is
copied and not launched again. Hits are a quenched energy within 0.001
of the Cambridge table. Medians are Hyndman-Fan type 7. Fisher is the
two-sided exact test on hits. Mann-Whitney is two-sided on the call
counts and on the wall, misses stored as the budget.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
from pathlib import Path

SYSTEMS = {
    "m55r14": {"n": 55, "rho": 14.0, "budget": 1_000_000, "reference": -220.646208},
    "m80r14": {"n": 80, "rho": 14.0, "budget": 4_000_000, "reference": -340.811371},
}
SEEDS = list(range(2000, 2012))
HIT_TOL = 0.001
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


def type7(ordered: list[float], p: float) -> float:
    n = len(ordered)
    if n == 1:
        return ordered[0]
    h = 1.0 + (n - 1) * p
    lower = int(h)
    frac = h - lower
    index = lower - 1
    if frac == 0.0:
        return ordered[index]
    return ordered[index] * (1.0 - frac) + ordered[index + 1] * frac


def median(values: list[float]) -> float:
    return type7(sorted(values), 0.5)


def comb(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def fisher_two_sided(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    a, b = hits_a, n_a - hits_a
    c, d = hits_b, n_b - hits_b

    def prob(aa: int, bb: int, cc: int, dd: int) -> float:
        total = aa + bb + cc + dd
        return comb(aa + bb, aa) * comb(cc + dd, cc) / comb(total, aa + cc)

    base = prob(a, b, c, d)
    total = 0.0
    for aa in range(a + b + 1):
        cc = a + c - aa
        bb = a + b - aa
        dd = c + d - cc
        if min(aa, bb, cc, dd) < 0:
            continue
        p = prob(aa, bb, cc, dd)
        if p <= base + 1e-12:
            total += p
    return min(1.0, total)


def mann_whitney(left: list[float], right: list[float]) -> float:
    from scipy.stats import mannwhitneyu

    if not left or not right:
        return float("nan")
    return float(mannwhitneyu(left, right, alternative="two-sided").pvalue)


def binary_path(root: Path) -> Path:
    candidates = [
        root.parent / "tmp" / "anneal-target" / "release" / "examples" / "h2h_morse",
        root / "target" / "release" / "examples" / "h2h_morse",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise SystemExit("h2h_morse release binary is not built")


def load_gmin(archive: Path) -> dict[tuple[str, int], dict[str, str]]:
    rows = {}
    with archive.open() as handle:
        for row in csv.DictReader(handle):
            if row["code"] != "gmin":
                continue
            system = row["system"]
            if system not in SYSTEMS:
                continue
            if int(row["budget"]) != SYSTEMS[system]["budget"]:
                continue
            rows[(system, int(row["seed"]))] = row
    return rows


def correct_evals(hit: bool, evals: int, charged: int, budget: int) -> int:
    if not hit:
        return budget
    if evals < budget:
        return evals
    return min(charged, budget)


def parse_anneal(text: str, system: str) -> dict[str, str]:
    rows = [line for line in text.splitlines() if line.startswith(system + ",")]
    if not rows:
        raise SystemExit(f"no anneal row\n{text[-600:]}")
    cols = rows[-1].split(",")
    budget = int(cols[3])
    hit = cols[4] == "true"
    evals = correct_evals(hit, int(cols[5]), int(cols[6]), budget)
    return {
        "system": system,
        "code": "anneal",
        "seed": cols[2],
        "budget": cols[3],
        "hit": "1" if hit else "0",
        "evals_to_hit": str(evals),
        "charged": cols[6],
        "wall_s": cols[7],
        "best": cols[8],
        "pot_s": cols[9],
        "outside_s": cols[10],
    }


def run_anneal(binary: Path, system: str, seed: int) -> dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["RAYON_NUM_THREADS"] = "1"
    proc = subprocess.run(
        ["taskset", "-c", "0", str(binary), "run", system, str(seed)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    text = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise SystemExit(text[-800:])
    return parse_anneal(text, system)


def gmin_row(stored: dict[str, str]) -> dict[str, str]:
    return {field: stored[field] for field in SEED_FIELDS}


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEED_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summarise(rows: list[dict[str, str]], path: Path) -> None:
    fields = [
        "system",
        "anneal_hits",
        "anneal_n",
        "gmin_hits",
        "gmin_n",
        "gmin_reused",
        "fisher_p",
        "mannwhitney_p_evals",
        "mannwhitney_p_wall",
        "anneal_evals_median",
        "gmin_evals_median",
        "anneal_wall_median_s",
        "gmin_wall_median_s",
        "anneal_pot_median_s",
        "anneal_outside_median_s",
        "gmin_pot_median_s",
        "gmin_outside_median_s",
    ]
    lines = []
    for system in SYSTEMS:
        anneal = [row for row in rows if row["system"] == system and row["code"] == "anneal"]
        gmin = [row for row in rows if row["system"] == system and row["code"] == "gmin"]
        shared = sorted(set(int(row["seed"]) for row in anneal) & set(int(row["seed"]) for row in gmin))
        anneal_shared = [row for row in anneal if int(row["seed"]) in shared]
        gmin_shared = [row for row in gmin if int(row["seed"]) in shared]

        def hits(group: list[dict[str, str]]) -> int:
            return sum(row["hit"] == "1" for row in group)

        def column(group: list[dict[str, str]], name: str) -> list[float]:
            return [float(row[name]) for row in group]

        lines.append(
            {
                "system": system,
                "anneal_hits": str(hits(anneal)),
                "anneal_n": str(len(anneal)),
                "gmin_hits": str(hits(gmin_shared)),
                "gmin_n": str(len(gmin_shared)),
                "gmin_reused": " ".join(str(seed) for seed in shared),
                "fisher_p": f"{fisher_two_sided(hits(anneal_shared), len(anneal_shared), hits(gmin_shared), len(gmin_shared)):.6g}",
                "mannwhitney_p_evals": f"{mann_whitney(column(anneal_shared, 'evals_to_hit'), column(gmin_shared, 'evals_to_hit')):.6g}",
                "mannwhitney_p_wall": f"{mann_whitney(column(anneal_shared, 'wall_s'), column(gmin_shared, 'wall_s')):.6g}",
                "anneal_evals_median": f"{median(column(anneal, 'evals_to_hit')):.6f}",
                "gmin_evals_median": f"{median(column(gmin, 'evals_to_hit')):.6f}" if gmin else "",
                "anneal_wall_median_s": f"{median(column(anneal, 'wall_s')):.6f}",
                "gmin_wall_median_s": f"{median(column(gmin, 'wall_s')):.6f}" if gmin else "",
                "anneal_pot_median_s": f"{median(column(anneal, 'pot_s')):.6f}",
                "anneal_outside_median_s": f"{median(column(anneal, 'outside_s')):.6f}",
                "gmin_pot_median_s": f"{median(column(gmin, 'pot_s')):.6f}" if gmin else "",
                "gmin_outside_median_s": f"{median(column(gmin, 'outside_s')):.6f}" if gmin else "",
            }
        )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(lines)


def panel(root: Path) -> list[dict[str, str]]:
    archive = load_gmin(root / "bench" / "morse-h2h" / "seeds.csv")
    binary = binary_path(root)
    rows: list[dict[str, str]] = []
    for system in SYSTEMS:
        for seed in SEEDS:
            anneal = run_anneal(binary, system, seed)
            print(",".join(anneal[field] for field in SEED_FIELDS), flush=True)
            rows.append(anneal)
            stored = archive.get((system, seed))
            if stored is not None:
                copied = gmin_row(stored)
                print("reused " + ",".join(copied[field] for field in SEED_FIELDS), flush=True)
                rows.append(copied)
    return rows


def one(root: Path, system: str, seed: int, arm: str) -> None:
    if arm == "gmin":
        stored = load_gmin(root / "bench" / "morse-h2h" / "seeds.csv").get((system, seed))
        if stored is None:
            raise SystemExit(
                f"no stored GMIN row for {system} seed {seed} at budget {SYSTEMS[system]['budget']}"
            )
        print(",".join(gmin_row(stored)[field] for field in SEED_FIELDS))
        return
    if arm != "anneal":
        raise SystemExit(f"unknown arm {arm}")
    print(",".join(run_anneal(binary_path(root), system, seed)[field] for field in SEED_FIELDS))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--system")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--arm")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.system or args.seed or args.arm:
        if not (args.system and args.seed is not None and args.arm):
            raise SystemExit("one job needs --system, --seed and --arm")
        one(root, args.system, args.seed, args.arm)
        return 0
    out = root / "bench" / "morse14-h2h"
    rows = panel(root)
    write_rows(out / "seeds.csv", rows)
    summarise(rows, out / "summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())

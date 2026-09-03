#!/usr/bin/env python3
"""Summarize the Cu(111)+H6 plain / recommended / random-relax comparison."""

from __future__ import annotations

import pathlib
import re
import statistics
import sys

ENERGY_TOL = 1e-4

seed_re = re.compile(
    r"seed (?P<seed>\d+): best (?P<best>[-+0-9.eE]+|[-+]?inf|nan) eV\s+"
    r"hops (?P<hops>\d+)\s+charged (?P<charged>\d+)\s+basins (?P<basins>\d+)",
    re.IGNORECASE,
)
improve_re = re.compile(
    r"improve hops=\d+ charged=(?P<charged>\d+) basins=\d+ e=(?P<energy>[-+0-9.eE]+)"
)
relax_re = re.compile(
    r"relax (?P<idx>\d+): e=(?P<energy>[-+0-9.eE]+) charged=(?P<charged>\d+)"
    r"(?: \|g\|=(?P<ginf>\S+) (?P<status>accepted|rejected))?"
)
summary_re = re.compile(
    r"SUMMARY distinct=(?P<distinct>\d+) lowest=(?P<lowest>\S+) "
    r"hits=(?P<hits>\d+)/(?P<starts>\d+) charged=(?P<charged>\d+)"
)
basin_re = re.compile(r"basin (?P<energy>[-+0-9.eE]+)\s+n=(?P<count>\d+)")


def finite(value: float) -> bool:
    return value == value and abs(value) != float("inf")


def parse_hop(directory: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(directory.glob("seed_*.out")):
        text = path.read_text(errors="replace")
        seed_m = seed_re.search(text)
        if not seed_m:
            continue
        best = float(seed_m.group("best"))
        charged_to_best = None
        visited: set[float] = set()
        if finite(best):
            for match in improve_re.finditer(text):
                energy = float(match.group("energy"))
                visited.add(round(energy / ENERGY_TOL) * ENERGY_TOL)
                if abs(energy - best) <= 1e-6:
                    charged_to_best = int(match.group("charged"))
        rows.append(
            {
                "seed": int(seed_m.group("seed")),
                "best": best,
                "charged": int(seed_m.group("charged")),
                "basins": int(seed_m.group("basins")),
                "charged_to_best": charged_to_best,
                "visited": len(visited) if visited else int(seed_m.group("basins")),
            }
        )
    return rows


def parse_relax(directory: pathlib.Path) -> dict:
    text = ""
    for path in sorted(directory.glob("seed_*.out")):
        text += path.read_text(errors="replace")
    energies: list[float] = []
    charges: list[int] = []
    for match in relax_re.finditer(text):
        energy = float(match.group("energy"))
        status = match.group("status")
        if status == "rejected":
            continue
        if finite(energy):
            energies.append(energy)
            charges.append(int(match.group("charged")))
    summary = summary_re.search(text)
    basins = [(float(m.group("energy")), int(m.group("count"))) for m in basin_re.finditer(text)]
    lowest = min(energies) if energies else float("nan")
    hits = sum(1 for energy in energies if finite(lowest) and abs(energy - lowest) <= ENERGY_TOL)
    distinct = len(basins) if basins else len({round(e / ENERGY_TOL) for e in energies})
    if summary:
        distinct = int(summary.group("distinct"))
        lowest = float(summary.group("lowest"))
        hits = int(summary.group("hits"))
    return {
        "energies": energies,
        "charges": charges,
        "lowest": lowest,
        "hits": hits,
        "starts": len(energies),
        "distinct": distinct,
        "charged": int(summary.group("charged")) if summary else sum(charges),
        "hit_charges": [
            charge
            for energy, charge in zip(energies, charges)
            if finite(lowest) and abs(energy - lowest) <= ENERGY_TOL
        ],
    }


def median_or_nan(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: terra_summarize_slab_h6.py OUT_DIR", file=sys.stderr)
        return 2
    out = pathlib.Path(sys.argv[1])
    arms = {
        "plain": parse_hop(out / "cuh2_h6_plain"),
        "recommended": parse_hop(out / "cuh2_h6_recommended"),
    }
    relax_dir = out / "cuh2_h6_random_relax"
    relax = parse_relax(relax_dir) if relax_dir.is_dir() else None

    all_bests = [row["best"] for rows in arms.values() for row in rows if finite(row["best"])]
    if relax and finite(relax["lowest"]):
        all_bests.append(relax["lowest"])
    if not all_bests:
        print("no finite energies")
        return 1
    global_best = min(all_bests)

    print("=== Cu(111)+H6 cost to solution ===")
    print(f"lowest_seen {global_best:.6f} eV  (tol {ENERGY_TOL})")
    print("arm             seeds_at_lowest  median_evals_to_it  distinct_minima")
    for name, rows in arms.items():
        hits = [
            row
            for row in rows
            if finite(row["best"]) and abs(row["best"] - global_best) <= ENERGY_TOL
        ]
        charges = [float(row["charged_to_best"] or row["charged"]) for row in hits]
        distinct = sum(row["visited"] for row in rows)
        print(
            f"{name:14}  {len(hits)}/{len(rows):<14}  {median_or_nan(charges):<18}  {distinct}"
        )
        for row in rows:
            print(
                f"  seed {row['seed']}: best {row['best']:.6f}  "
                f"charged {row['charged']}  basins {row['basins']}"
            )
    if relax:
        hits = relax["hits"] if abs(relax["lowest"] - global_best) <= ENERGY_TOL else 0
        print(
            f"{'random_relax':14}  {hits}/{relax['starts']:<14}  "
            f"{median_or_nan([float(c) for c in relax['hit_charges']]):<18}  {relax['distinct']}"
        )
        print(
            f"  starts {relax['starts']}  charged {relax['charged']}  "
            f"lowest {relax['lowest']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

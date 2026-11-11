#!/usr/bin/env python3
"""Serial head-to-head of the lean preset and the published bonobo initialiser.

Lennard-Jones sizes 38, 75, 76 and 77. Six seeds on each arm. One local
relaxation budget for every run. Hits are counted against the Cambridge
Cluster Database energies in the fixture, with the same 1e-4 window the
cluster-search driver already uses. Nothing in this file is a paper count.
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent

# Copied from the Cambridge table before the runs. The fixture must match.
LOCKED = {
    38: -173.928427,
    75: -397.492331,
    76: -402.894866,
    77: -409.083517,
}
SIZES = (38, 75, 76, 77)
SEEDS = (0, 1, 2, 3, 4, 5)
# 100 of these are the published initial population. The rest are bonobo
# updates, twenty per generation. Anneal counts a full relax_steps quench
# the same way, and does not count its short screen. On LJ38 the lean
# preset reached the Cambridge minimum between 506 and 962 full
# relaxations, so 2000 sits past that and still finishes serially.
RELAXATION_BUDGET = 2000
HIT_TOL = 1.0e-4


def load_fixture(path: Path) -> dict[int, float]:
    found: dict[int, float] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("n\t"):
            continue
        n_text, _group, energy_text = line.split("\t")
        found[int(n_text)] = float(energy_text)
    if found != LOCKED:
        raise SystemExit(f"Cambridge fixture {path} does not match the locked table")
    return found


def parse_row(line: str) -> dict:
    row = {}
    for token in line.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        row[key] = value
    if "arm" not in row or "hit" not in row:
        raise ValueError(f"no result line in {line!r}")
    row["n"] = int(row["n"])
    row["seed"] = int(row["seed"])
    row["hit"] = int(row["hit"])
    row["relaxations"] = int(row["relaxations"])
    row["wall_s"] = float(row["wall_s"])
    hit_at = row.get("relaxations_to_hit", "")
    row["relaxations_to_hit"] = int(hit_at) if hit_at else None
    return row


def run_lean(binary: Path, n: int, seed: int, budget: int, reference: float, log: Path) -> dict:
    proc = subprocess.run(
        [str(binary), str(n), str(budget), str(seed), f"{reference:.10f}"],
        check=False,
        capture_output=True,
        text=True,
    )
    log.write_text(proc.stdout + "\n" + proc.stderr)
    lines = [line for line in proc.stdout.splitlines() if line.startswith("arm=")]
    if not lines:
        raise RuntimeError(f"lean seed {seed} n={n} produced no row:\n{proc.stderr[-2000:]}")
    return parse_row(lines[-1])


def run_bonobo(
    python: Path,
    arm: Path,
    n: int,
    seed: int,
    budget: int,
    reference: float,
    upstream: Path,
    log: Path,
    population: int | None,
    max_iter: int | None,
) -> dict:
    work = log.parent
    env = os.environ.copy()
    if population:
        env["LJ_BONOBO_POPULATION"] = str(population)
    if max_iter:
        env["LJ_BONOBO_MAX_ITER"] = str(max_iter)
    proc = subprocess.run(
        [
            str(python),
            str(arm),
            str(n),
            str(seed),
            str(budget),
            f"{reference:.10f}",
            str(upstream),
        ],
        cwd=work,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    log.write_text(proc.stdout + "\n" + proc.stderr)
    lines = [line for line in proc.stdout.splitlines() if line.startswith("arm=")]
    if not lines:
        raise RuntimeError(f"bonobo seed {seed} n={n} produced no row:\n{proc.stderr[-2000:]}")
    return parse_row(lines[-1])


def median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def fmt(value: float | None, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def summarise(rows: list[dict], budget: int) -> str:
    lines = [
        "n\tarm\thits\truns\tmedian_relaxations_to_hit\tmedian_wall_s\twall_ratio_bonobo_over_lean",
        f"# budget_relaxations\t{budget}",
        f"# hit_window\t{HIT_TOL}",
        "# wall_ratio is the median bonobo wall divided by the median lean wall",
    ]
    by_n: dict[int, dict[str, list]] = {}
    for row in rows:
        by_n.setdefault(row["n"], {}).setdefault(row["arm"], []).append(row)
    for n in sorted(by_n):
        arms = by_n[n]
        walls = {}
        for arm, group in arms.items():
            walls[arm] = median([item["wall_s"] for item in group])
        ratio = None
        if walls.get("lean") and walls.get("bonobo"):
            ratio = walls["bonobo"] / walls["lean"]
        for arm in ("lean", "bonobo"):
            group = arms.get(arm, [])
            hits = [item["relaxations_to_hit"] for item in group if item["hit"]]
            lines.append(
                "\t".join(
                    [
                        str(n),
                        arm,
                        str(sum(item["hit"] for item in group)),
                        str(len(group)),
                        fmt(median(hits), 1),
                        fmt(walls.get(arm), 6),
                        fmt(ratio, 3),
                    ]
                )
            )
    return "\n".join(lines) + "\n"


def seed_table(rows: list[dict]) -> str:
    lines = ["n\tarm\tseed\thit\trelaxations\trelaxations_to_hit\twall_s\tbest\tscreens"]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    str(row["n"]),
                    row["arm"],
                    str(row["seed"]),
                    str(row["hit"]),
                    str(row["relaxations"]),
                    "" if row["relaxations_to_hit"] is None else str(row["relaxations_to_hit"]),
                    f"{row['wall_s']:.6f}",
                    f"{float(row['best']):.6f}",
                    str(row.get("screens", "")),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    if os.uname().nodename != "bench":
        print("uts hostname is not bench", file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--arm", choices=("lean", "bonobo", "both"), default="both")
    parser.add_argument("--budget", type=int, default=RELAXATION_BUDGET)
    parser.add_argument("--population", type=int)
    parser.add_argument("--max-iter", type=int)
    parser.add_argument("--seeds-out", type=Path)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    targets = load_fixture(args.fixture)
    sizes = tuple(args.size) if args.size else SIZES
    seeds = tuple(args.seed) if args.seed else SEEDS
    arms = ("lean", "bonobo") if args.arm == "both" else (args.arm,)
    for n in sizes:
        if n not in targets:
            raise SystemExit(f"no Cambridge energy for LJ{n}")
    rows = []
    full = sizes == SIZES and seeds == SEEDS and arms == ("lean", "bonobo")
    for n in sizes:
        reference = targets[n]
        for arm in arms:
            for seed in seeds:
                log = args.runs / arm / f"n{n}" / f"seed{seed}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                print(f"start n={n} arm={arm} seed={seed}", flush=True)
                if arm == "lean":
                    row = run_lean(args.binary, n, seed, args.budget, reference, log)
                else:
                    row = run_bonobo(
                        Path(sys.executable),
                        HERE / "bonobo_arm.py",
                        n,
                        seed,
                        args.budget,
                        reference,
                        args.upstream,
                        log,
                        args.population,
                        args.max_iter,
                    )
                rows.append(row)
                print(
                    f"done n={n} arm={arm} seed={seed} hit={row['hit']} "
                    f"relaxations={row['relaxations']} to_hit={row['relaxations_to_hit']} "
                    f"wall_s={row['wall_s']:.3f} best={float(row['best']):.6f}",
                    flush=True,
                )
    text = summarise(rows, args.budget)
    if args.population or args.max_iter:
        text = (
            f"# population\t{args.population or ''}\n"
            f"# max_iter\t{args.max_iter or ''}\n"
            + text
        )
    sys.stdout.write(text)
    if args.summary and (full or args.seeds_out):
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(text)
        seeds_path = args.seeds_out or args.summary.with_name("seeds.tsv")
        seeds_path.write_text(seed_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

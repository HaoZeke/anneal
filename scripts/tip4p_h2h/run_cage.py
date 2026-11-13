#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Append the cage-archive anneal arm and the Lennard-Jones check.

Existing rows are left in place. New rows use code anneal-cage.
One core. The caller has already set the hostname to bench and the
working directory to /work.
"""

from __future__ import annotations

import csv
import os
import socket
import subprocess
import sys
from pathlib import Path

BUDGET_WATER = 2_000_000
REFERENCES = {15: -628.3722525077, 20: -872.9887544952}
HIT_TOL = 0.001
SEEDS = (2000, 2001, 2002, 2003, 2004, 2005)
CODE = "anneal-cage"
# Evaluation counts on 4fdc37e8 and on 91c8a9c3, one per seed 2000–2005.
# A miss is the full budget.
LJ_EXPECTED = {
    "lj38": (400_000, [11792, 16544, 8587, 7255, 18485, 8231]),
    "lj75": (4_000_000, [663625, 2594196, 11228, 1486000, 834735, 1123846]),
    "lj98": (4_000_000, [750015, 2949848, 612380, 1729651, 4000000, 1044686]),
    "lj104": (4_000_000, [1525401, 2420382, 635431, 2822730, 620506, 2099115]),
}
FIELDS = [
    "system",
    "code",
    "seed",
    "budget",
    "hit",
    "evals_to_hit",
    "charged",
    "best",
    "wall_s",
    "pot_s",
    "outside_s",
    "basins",
]


def type7(values: list[float], p: float = 0.5) -> float:
    ordered = sorted(values)
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


def env_of(budget: int) -> dict[str, str]:
    env = os.environ.copy()
    env["H2H_BUDGET"] = str(budget)
    env["OMP_NUM_THREADS"] = "1"
    env["RAYON_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    return env


def append_row(path: Path, row: dict[str, str]) -> None:
    new = not path.is_file() or path.stat().st_size == 0
    with path.open("a") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in FIELDS})


def run_water(binary: Path, n: int, seed: int, work: Path) -> dict[str, str]:
    prefix = work / f"cage-tip4p{n}-s{seed}"
    log = work / f"tip4p{n}-s{seed}-anneal-cage.log"
    stop = REFERENCES[n] + HIT_TOL
    with log.open("w") as handle:
        proc = subprocess.run(
            [
                "taskset",
                "-c",
                "0",
                str(binary),
                "tip4p",
                str(n),
                "0",
                str(BUDGET_WATER),
                str(seed),
                "recommended",
                repr(stop),
                str(prefix),
            ],
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env_of(BUDGET_WATER),
        )
    text = log.read_text()
    if proc.returncode != 0:
        raise RuntimeError(f"anneal-cage tip4p{n} seed={seed} exited {proc.returncode}\n{text[-800:]}")
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("h2h_row "):
            fields = dict(part.split("=", 1) for part in line.split()[1:])
            break
    if not fields:
        raise RuntimeError(f"no h2h_row\n{text[-800:]}")
    return {
        "system": f"tip4p{n}",
        "code": CODE,
        "seed": str(seed),
        "budget": str(BUDGET_WATER),
        "hit": fields["hit"],
        "evals_to_hit": fields["evals_to_hit"],
        "charged": fields["charged"],
        "best": fields["best"],
        "wall_s": fields["wall_s"],
        "pot_s": fields["pot_s"],
        "outside_s": fields["outside_s"],
        "basins": fields["basins"],
    }


def run_lj(binary: Path, system: str, seed: int, budget: int) -> dict[str, str]:
    proc = subprocess.run(
        ["taskset", "-c", "0", str(binary), "run", system, str(seed)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env_of(budget),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{CODE} {system} seed={seed} exited {proc.returncode}\n{proc.stdout[-800:]}")
    line = [row for row in proc.stdout.splitlines() if row.startswith(system + ",")]
    if len(line) != 1:
        raise RuntimeError(f"no row in {proc.stdout!r}")
    header = "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,head,tree,cpu,pins,pot_s"
    raw = next(csv.DictReader([header, line[0]]))
    wall = float(raw["wall_s"])
    pot = float(raw["pot_s"])
    hit = raw["hit"] in ("1", "true", "True")
    return {
        "system": system,
        "code": CODE,
        "seed": str(seed),
        "budget": raw["budget"],
        "hit": "1" if hit else "0",
        "evals_to_hit": raw["evals_to_hit"],
        "charged": raw["charged"],
        "best": raw["best"],
        "wall_s": f"{wall:.6f}",
        "pot_s": f"{pot:.6f}",
        "outside_s": f"{wall - pot:.6f}",
        "basins": "",
    }


def summarise(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["system"], row["code"]), []).append(row)
    for (system, code), arm in sorted(groups.items()):
        def med(key: str) -> str:
            values = [float(row[key]) for row in arm if row.get(key)]
            if len(values) != len(arm):
                return ""
            return f"{type7(values):.6f}"

        out.append(
            {
                "system": system,
                "code": code,
                "hits": str(sum(row["hit"] == "1" for row in arm)),
                "n": str(len(arm)),
                "budget": arm[0]["budget"],
                "evals_median": med("evals_to_hit"),
                "wall_median_s": med("wall_s"),
                "pot_median_s": med("pot_s"),
                "outside_median_s": med("outside_s"),
            }
        )
    return out


def main() -> int:
    if socket.gethostname() != "bench":
        print(f"hostname is {socket.gethostname()}, not bench", file=sys.stderr)
        return 2
    if Path.cwd() != Path("/work"):
        print(f"cwd is {Path.cwd()}, not /work", file=sys.stderr)
        return 2
    root = Path(os.environ["ANNEAL_ROOT"])
    rows_path = root / "bench/tip4p-h2h/rows.csv"
    summary_path = root / "bench/tip4p-h2h/summary-cage.csv"
    work = Path("/work/tip4p-cage")
    work.mkdir(parents=True, exist_ok=True)
    water = Path("/work/tip4p-fix/target/release/examples/h2h_seed")
    lj_bin = Path("/work/tip4p-fix/target/release/examples/h2h_gmin_proof")
    fresh: list[dict[str, str]] = []

    def keep(row: dict[str, str]) -> None:
        append_row(rows_path, row)
        fresh.append(row)
        print(
            f"{row['system']} {row['code']} seed={row['seed']} hit={row['hit']} "
            f"evals={row['evals_to_hit']} best={row['best']} outside={row['outside_s']}",
            flush=True,
        )

    for n in (15, 20):
        for seed in SEEDS:
            print(f"anneal-cage n={n} seed={seed}", flush=True)
            keep(run_water(water, n, seed, work))
    lj_ok = True
    for system, (budget, expected) in LJ_EXPECTED.items():
        for seed, want in zip(SEEDS, expected, strict=True):
            print(f"anneal-cage {system} seed={seed}", flush=True)
            row = run_lj(lj_bin, system, seed, budget)
            keep(row)
            got = int(row["evals_to_hit"])
            if got != want:
                lj_ok = False
                print(f"LJ MISMATCH {system} seed={seed} got={got} want={want}", flush=True)
            else:
                print(f"LJ MATCH {system} seed={seed} evals={got}", flush=True)
    summary = summarise(fresh)
    with summary_path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(row, flush=True)
    if not lj_ok:
        print("Lennard-Jones evaluation counts changed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

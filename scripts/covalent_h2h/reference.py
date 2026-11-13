#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Long unstopped runs that fix the best-known energy of each system.

Seeds 9000 and up, never the panel seeds. Every anneal arm, GMIN on the
Stillinger-Weber rows and ASE basin hopping on every row run with no
target, so each reports the lowest minimum it recorded. The lowest over
all of them is the reference the panel stops at. Jobs run in parallel:
these runs are not timed.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SYSTEMS = ("sw10", "sw13", "sw20", "tc10", "tc20")
ARMS = ("recommended", "nopriors", "general")
NO_TARGET = "-1e300"


def anneal_job(binary: Path, system: str, seed: int, arm: str, budget: int, out: Path):
    env = dict(os.environ, H2H_ARM=arm, H2H_BUDGET=str(budget), H2H_TARGET=NO_TARGET)
    best = out / f"{system}-anneal-{arm}-{seed}.xyz"
    text = subprocess.run(
        [str(binary), "run", system, str(seed), str(best)],
        check=True, capture_output=True, text=True, env=env,
    ).stdout
    row = next(csv.DictReader(text.splitlines()[:2]))
    return system, f"anneal-{arm}", seed, budget, float(row["best"])


def gmin_job(binary: Path, gmin: Path, data: Path, system: str, seed: int, budget: int, out: Path):
    work = out / f"{system}-gmin-{seed}"
    work.mkdir(parents=True, exist_ok=True)
    scales = subprocess.run(
        [str(binary), "scales", system], check=True, capture_output=True, text=True
    ).stdout.split()
    radius = float(scales[2])
    (work / "data").write_text(data.read_text() + f"RADIUS {radius:.10f}\nRANSEED {seed}\n")
    subprocess.run([str(binary), "coords", system, str(seed), str(work / "coords")], check=True)
    env = dict(
        os.environ,
        H2H_BUDGET=str(budget),
        H2H_SYSTEM=system,
        H2H_SEED=str(seed),
        H2H_TARGET=NO_TARGET,
        H2H_TOL="0.001",
        H2H_ROW=str(work / "row.txt"),
    )
    with (work / "log").open("w") as handle:
        subprocess.run([str(gmin)], check=False, stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=work)
    fields = dict(part.split("=", 1) for part in (work / "row.txt").read_text().split()[1:])
    return system, "gmin", seed, budget, float(fields["best"])


def ase_job(binary: Path, library: Path, script: Path, system: str, seed: int, budget: int):
    env = dict(os.environ, H2H_TARGET=NO_TARGET)
    text = subprocess.run(
        [sys.executable, str(script), "--binary", str(binary), "--library", str(library),
         "--system", system, "--seed", str(seed), "--budget", str(budget)],
        check=True, capture_output=True, text=True, env=env,
    ).stdout
    row = next(csv.DictReader(text.splitlines()[:2]))
    return system, "ase", seed, budget, float(row["best"])


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--gmin", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--anneal-budget", type=int, default=1_000_000)
    parser.add_argument("--gmin-budget", type=int, default=2_000_000)
    parser.add_argument("--ase-budget", type=int, default=400_000)
    args = parser.parse_args(argv[1:])
    args.out.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("ase_bh.py")
    seeds = range(9000, 9000 + args.seeds)
    jobs = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for system in SYSTEMS:
            for seed in seeds:
                for arm in ARMS:
                    jobs.append(pool.submit(anneal_job, args.binary, system, seed, arm, args.anneal_budget, args.out))
                if system.startswith("sw"):
                    jobs.append(pool.submit(gmin_job, args.binary, args.gmin, args.data, system, seed, args.gmin_budget, args.out))
                jobs.append(pool.submit(ase_job, args.binary, args.library, script, system, seed, args.ase_budget))
        rows = []
        for job in jobs:
            row = job.result()
            print(",".join(str(v) for v in row), flush=True)
            rows.append(row)
    rows.sort()
    with args.csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["system", "code", "seed", "budget", "best"])
        for system, code, seed, budget, best in rows:
            writer.writerow([system, code, seed, budget, f"{best:.8f}"])
    for system in SYSTEMS:
        mine = [row for row in rows if row[0] == system]
        low = min(mine, key=lambda row: row[4])
        print(f"lowest {system} {low[4]:.8f} {low[1]} seed {low[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

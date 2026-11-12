#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial TIP4P15 and TIP4P20, anneal against GMIN, one core.

Budgets are the campaign ceilings at which the manuscript scored the
loss: 1.04e7 calls at 15 molecules and 6e7 at 20. A hit is a quenched
energy at or below the GminTip4p header plus 0.001 kJ/mol. Six seeds.
The wall ratio is the GMIN median wall divided by the anneal median wall.
"""

from __future__ import annotations

import argparse
import csv
import os
import socket
import subprocess
import sys
from pathlib import Path

# The (H2O)20 paper ceiling. (H2O)15 is scored at the same ceiling:
# three of its four hits in the six-seed set fall between the 5e5
# rate-table budget and this one.
PAPER_BUDGETS = {15: 2_000_000, 20: 2_000_000}
REFERENCES = {15: -628.3722525077, 20: -872.9887544952}
HIT_TOL = 0.001
SEEDS = (2000, 2001, 2002, 2003, 2004, 2005)
SIZES = (15, 20)


def type7(sorted_values: list[float], p: float) -> float:
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    h = 1.0 + (n - 1) * p
    lower = int(h)
    frac = h - lower
    index = lower - 1
    if frac == 0.0:
        return sorted_values[index]
    return sorted_values[index] * (1.0 - frac) + sorted_values[index + 1] * frac


def parse_kv(text: str, prefix: str) -> dict[str, str]:
    for line in text.splitlines():
        if line.startswith(prefix):
            fields = {}
            for part in line.split()[1:]:
                key, value = part.split("=", 1)
                fields[key] = value
            return fields
    raise ValueError(f"no {prefix} row")


def env_of(budget: int) -> dict[str, str]:
    env = os.environ.copy()
    env["H2H_BUDGET"] = str(budget)
    env["OMP_NUM_THREADS"] = "1"
    env["RAYON_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    return env


def run_anneal(binary: Path, n: int, seed: int, budget: int, work: Path) -> dict[str, str]:
    prefix = work / f"tip4p{n}-s{seed}"
    log = work / f"tip4p{n}-s{seed}-anneal.log"
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
                str(budget),
                str(seed),
                "recommended",
                repr(stop),
                str(prefix),
            ],
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env_of(budget),
        )
    text = log.read_text()
    if proc.returncode != 0:
        raise RuntimeError(f"anneal n={n} seed={seed} exited {proc.returncode}\n{text[-500:]}")
    row = parse_kv(text, "h2h_row ")
    if int(row["budget"]) != budget:
        raise RuntimeError(f"anneal budget {row}")
    return row


def run_gmin(
    gmin: Path,
    data: Path,
    n: int,
    seed: int,
    budget: int,
    start: Path,
    work: Path,
) -> dict[str, str]:
    run = work / f"gmin-tip4p{n}-s{seed}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "coords").write_bytes(start.read_bytes())
    (run / "data").write_text(data.read_text() + f"RANSEED {seed}\n")
    log = work / f"tip4p{n}-s{seed}-gmin.log"
    env = env_of(budget)
    env.update(
        {
            "H2H_SYSTEM": f"tip4p{n}",
            "H2H_SEED": str(seed),
            "H2H_TARGET": f"{REFERENCES[n]:.10f}",
            "H2H_TOL": f"{HIT_TOL}",
            "H2H_ROW": str(run / "row.txt"),
        }
    )
    with log.open("w") as handle:
        proc = subprocess.run(
            ["taskset", "-c", "0", str(gmin)],
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=run,
        )
    row_path = run / "row.txt"
    if not row_path.is_file():
        tail = log.read_text()[-800:]
        raise RuntimeError(f"gmin n={n} seed={seed} exited {proc.returncode} with no row\n{tail}")
    row = parse_kv(row_path.read_text(), "h2h_row ")
    if int(row["budget"]) != budget:
        raise RuntimeError(f"gmin budget {row}")
    return row


def summarise(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for n in SIZES:
        anneal = [r for r in rows if r["system"] == f"tip4p{n}" and r["code"] == "anneal"]
        gmin = [r for r in rows if r["system"] == f"tip4p{n}" and r["code"] == "gmin"]
        if len(anneal) != len(SEEDS) or len(gmin) != len(SEEDS):
            continue
        def med(arm: list[dict[str, str]], key: str) -> float:
            return type7(sorted(float(r[key]) for r in arm), 0.5)

        aw, gw = med(anneal, "wall_s"), med(gmin, "wall_s")
        out.append(
            {
                "n": str(n),
                "budget": str(PAPER_BUDGETS[n]),
                "anneal_hits": str(sum(r["hit"] == "1" for r in anneal)),
                "gmin_hits": str(sum(r["hit"] == "1" for r in gmin)),
                "anneal_evals_median": f"{med(anneal, 'evals_to_hit'):.1f}",
                "gmin_evals_median": f"{med(gmin, 'evals_to_hit'):.1f}",
                "anneal_wall_median_s": f"{aw:.6f}",
                "gmin_wall_median_s": f"{gw:.6f}",
                "wall_ratio": f"{(gw / aw) if aw else float('inf'):.6f}",
                "anneal_pot_median_s": f"{med(anneal, 'pot_s'):.6f}",
                "anneal_outside_median_s": f"{med(anneal, 'outside_s'):.6f}",
                "gmin_pot_median_s": f"{med(gmin, 'pot_s'):.6f}",
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--gmin", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--size", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if socket.gethostname() != "bench":
        print(f"hostname is {socket.gethostname()}, not bench", file=sys.stderr)
        return 2
    if Path.cwd() != Path("/work"):
        print(f"cwd is {Path.cwd()}, not /work", file=sys.stderr)
        return 2
    sizes = (args.size,) if args.size else SIZES
    seeds = (args.seed,) if args.seed is not None else SEEDS
    args.work.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    if args.rows.is_file() and args.seed is None and args.size is None:
        args.rows.unlink()
    for n in sizes:
        for seed in seeds:
            print(f"anneal n={n} seed={seed}", flush=True)
            anneal = run_anneal(args.binary, n, seed, PAPER_BUDGETS[n], args.work)
            rows.append(anneal)
            start = args.work / f"tip4p{n}-s{seed}.start.gmin"
            print(f"gmin n={n} seed={seed}", flush=True)
            gmin = run_gmin(args.gmin, args.data, n, seed, PAPER_BUDGETS[n], start, args.work)
            rows.append(gmin)
            with args.rows.open("a") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(anneal.keys()))
                if handle.tell() == 0:
                    writer.writeheader()
                writer.writerow(anneal)
                writer.writerow(gmin)
    if args.seed is None and args.size is None:
        summary = summarise(rows)
        with args.summary.open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)
        for row in summary:
            print(row, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

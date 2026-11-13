#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Append TIP4P rows where GMIN opens anneal's rigid start.

``h2h_seed`` writes each start as the chart's centres and Rodrigues vectors,
the block NEWTIP reads. Before the searches GMIN evaluates that block once
(``CHECKD 0``) and the run stops if its energy is not anneal's start energy.
Every seed runs ``gmin-control-same-start`` and ``gmin-radius-same-start``
back to back; at 4 and 8 molecules ``anneal-bf8c3fe1`` runs first. Existing
rows are left in place. One core. The caller has already set the hostname
to bench and the working directory to /work.
"""

from __future__ import annotations

import csv
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

from run_radius import (
    HIT_TOL,
    PAPER_BUDGETS,
    RADII,
    REFERENCES,
    SEEDS,
    append_row,
    env_of,
    lj_container,
    run_gmin,
    type7,
)

ANNEAL_CODE = "anneal-bf8c3fe1"
ANNEAL_SIZES = (4, 8)
# GMIN's Coulomb literal is 1389.354848, GminTip4p's 1389.352660667.
START_TOL = 1.0e-3
START_FIELDS = ["system", "seed", "anneal_start_kj", "gmin_start_kj", "difference_kj"]


def write_start(binary: Path, n: int, seed: int, work: Path) -> tuple[Path, float]:
    prefix = work / f"tip4p{n}-s{seed}"
    proc = subprocess.run(
        [str(binary), "tip4p", str(n), "0", "1", str(seed), "recommended", "-", str(prefix)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env_of(1),
    )
    start = Path(f"{prefix}.start.gmin")
    found = re.search(r"^lowest: (\S+)$", proc.stdout, re.MULTILINE)
    if proc.returncode != 0 or not start.is_file() or found is None:
        raise RuntimeError(
            f"start n={n} seed={seed} exited {proc.returncode}\n{proc.stdout[-800:]}"
        )
    return start, float(found.group(1))


def gmin_start_energy(gmin: Path, data: Path, n: int, seed: int, start: Path, work: Path) -> float:
    run = work / f"checkd-tip4p{n}-s{seed}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "coords").write_bytes(start.read_bytes())
    (run / "data").write_text(data.read_text() + f"CHECKD 0\nRANSEED {seed}\n")
    env = os.environ.copy()
    env.pop("H2H_BUDGET", None)
    proc = subprocess.run(
        [str(gmin)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=run,
    )
    (work / f"tip4p{n}-s{seed}-checkd.log").write_text(proc.stdout)
    found = re.search(r"Energy  =\s*(\S+)", proc.stdout)
    if found is None:
        raise RuntimeError(f"CHECKD n={n} seed={seed} printed no energy\n{proc.stdout[-800:]}")
    return float(found.group(1))


def run_anneal(
    binary: Path, n: int, seed: int, work: Path, code: str = ANNEAL_CODE
) -> dict[str, str]:
    budget = PAPER_BUDGETS[n]
    prefix = work / f"anneal-tip4p{n}-s{seed}"
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
        raise RuntimeError(f"anneal n={n} seed={seed} exited {proc.returncode}\n{text[-800:]}")
    line = next((row for row in text.splitlines() if row.startswith("h2h_row ")), None)
    if line is None:
        raise RuntimeError(f"no h2h_row\n{text[-800:]}")
    fields = dict(part.split("=", 1) for part in line.split()[1:])
    if int(fields["budget"]) != budget:
        raise RuntimeError(f"anneal budget {fields}")
    return {
        "system": f"tip4p{n}",
        "code": code,
        "seed": str(seed),
        "budget": str(budget),
        "hit": fields["hit"],
        "evals_to_hit": fields["evals_to_hit"],
        "charged": fields["charged"],
        "best": fields["best"],
        "wall_s": fields["wall_s"],
        "pot_s": fields["pot_s"],
        "outside_s": fields["outside_s"],
        "basins": fields["basins"],
    }


def summarise(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[int, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((int(row["system"][5:]), row["code"]), []).append(row)
    out = []
    for (n, code), arm in sorted(groups.items()):
        if code.startswith("gmin-radius"):
            radius = f"{RADII[n]:.4f}"
        elif code.startswith("gmin-control"):
            radius = f"{lj_container(n):.4f}"
        else:
            radius = ""
        out.append(
            {
                "system": f"tip4p{n}",
                "code": code,
                "radius": radius,
                "hits": str(sum(row["hit"] == "1" for row in arm)),
                "n": str(len(arm)),
                "budget": arm[0]["budget"],
                "evals_median": f"{type7([float(r['evals_to_hit']) for r in arm]):.6f}",
                "wall_median_s": f"{type7([float(r['wall_s']) for r in arm]):.6f}",
                "best_min": f"{min(float(r['best']) for r in arm):.10f}",
                "best_median": f"{type7([float(r['best']) for r in arm]):.10f}",
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
    seed_bin = Path(os.environ["H2H_SEED_BIN"])
    rows_path = root / "bench/tip4p-h2h/rows.csv"
    starts_path = root / "bench/tip4p-h2h/starts-same-start.csv"
    summary_path = root / "bench/tip4p-h2h/summary-same-start.csv"
    work = Path("/work/tip4p-same-start")
    work.mkdir(parents=True, exist_ok=True)
    gmin = Path("/work/gmin-h2h/bin/GMIN")
    control = root / "examples/h2h_gmin_tip4p.data"
    fresh: list[dict[str, str]] = []

    def keep(row: dict[str, str]) -> None:
        append_row(rows_path, row)
        fresh.append(row)
        print(
            f"{row['system']} {row['code']} seed={row['seed']} hit={row['hit']} "
            f"evals={row['evals_to_hit']} best={row['best']} wall={row['wall_s']}",
            flush=True,
        )

    for n in PAPER_BUDGETS:
        arms = (
            ("gmin-control-same-start", control, lj_container(n)),
            (
                "gmin-radius-same-start",
                root / f"examples/h2h_gmin_tip4p_radius{n}.data",
                RADII[n],
            ),
        )
        for seed in SEEDS:
            start, anneal_start = write_start(seed_bin, n, seed, work)
            gmin_start = gmin_start_energy(gmin, control, n, seed, start, work)
            difference = gmin_start - anneal_start
            new = not starts_path.is_file() or starts_path.stat().st_size == 0
            with starts_path.open("a") as handle:
                writer = csv.DictWriter(handle, fieldnames=START_FIELDS, lineterminator="\n")
                if new:
                    writer.writeheader()
                writer.writerow(
                    {
                        "system": f"tip4p{n}",
                        "seed": str(seed),
                        "anneal_start_kj": f"{anneal_start:.10f}",
                        "gmin_start_kj": f"{gmin_start:.10f}",
                        "difference_kj": f"{difference:.3e}",
                    }
                )
            print(
                f"start n={n} seed={seed} anneal={anneal_start:.10f} gmin={gmin_start:.10f}",
                flush=True,
            )
            if abs(difference) > START_TOL:
                raise RuntimeError(f"start energies differ by {difference} at n={n} seed={seed}")
            if n in ANNEAL_SIZES:
                print(f"{ANNEAL_CODE} n={n} seed={seed}", flush=True)
                keep(run_anneal(seed_bin, n, seed, work))
                rerun = work / f"anneal-tip4p{n}-s{seed}.start.gmin"
                if rerun.read_bytes() != start.read_bytes():
                    raise RuntimeError(f"anneal opened another start at n={n} seed={seed}")
            for code, data, radius in arms:
                print(f"{code} n={n} seed={seed}", flush=True)
                row = run_gmin(gmin, data, code, radius, n, seed, start, work)
                keep(row)
    summary = summarise(fresh)
    with summary_path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(row, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

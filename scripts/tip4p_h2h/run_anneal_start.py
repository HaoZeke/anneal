#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Append anneal TIP4P rows at 15 and 20 molecules on the same-start harness.

Each seed takes the start ``h2h_seed`` writes, checks GMIN's ``CHECKD 0``
energy of it against anneal's as ``run_start.py`` does, and runs
``h2h_seed`` from it at the paper budget under the ``stop`` of the
reference plus the hit tolerance. Rows go to ``rows.csv`` as ``ANNEAL_CODE``.
In the same session GMIN reruns both same-start arms on those starts; those
rows go to ``gmin-recheck-same-start.csv`` beside the recorded
``gmin-*-same-start`` rows they repeat, not to ``rows.csv``. One core. The
caller has already set the hostname to bench and the working directory to
/work.
"""

from __future__ import annotations

import csv
import os
import socket
import sys
from pathlib import Path

from run_radius import FIELDS, RADII, SEEDS, append_row, lj_container, run_gmin
from run_start import START_TOL, gmin_start_energy, run_anneal, summarise, write_start

ANNEAL_CODE = "anneal-79c13d53"
SIZES = (15, 20)
RECHECK_FIELDS = [*FIELDS, "recorded_hit", "recorded_evals_to_hit", "recorded_best", "same"]


def recorded_rows(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    with path.open() as handle:
        return {(r["system"], r["code"], r["seed"]): r for r in csv.DictReader(handle)}


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
    recheck_path = root / "bench/tip4p-h2h/gmin-recheck-same-start.csv"
    summary_path = root / "bench/tip4p-h2h/summary-anneal-same-start.csv"
    work = Path("/work/tip4p-anneal-start")
    work.mkdir(parents=True, exist_ok=True)
    gmin = Path("/work/gmin-h2h/bin/GMIN")
    control = root / "examples/h2h_gmin_tip4p.data"
    recorded = recorded_rows(rows_path)
    fresh: list[dict[str, str]] = []
    repeated: list[dict[str, str]] = []

    for n in SIZES:
        arms = (
            ("gmin-control-same-start", control, lj_container(n)),
            ("gmin-radius-same-start", root / f"examples/h2h_gmin_tip4p_radius{n}.data", RADII[n]),
        )
        for seed in SEEDS:
            start, anneal_start = write_start(seed_bin, n, seed, work)
            gmin_start = gmin_start_energy(gmin, control, n, seed, start, work)
            print(
                f"start n={n} seed={seed} anneal={anneal_start:.10f} gmin={gmin_start:.10f}",
                flush=True,
            )
            if abs(gmin_start - anneal_start) > START_TOL:
                raise RuntimeError(f"start energies differ at n={n} seed={seed}")
            print(f"{ANNEAL_CODE} n={n} seed={seed}", flush=True)
            row = run_anneal(seed_bin, n, seed, work, ANNEAL_CODE)
            append_row(rows_path, row)
            fresh.append(row)
            print(
                f"{row['system']} {row['code']} seed={seed} hit={row['hit']} "
                f"evals={row['evals_to_hit']} best={row['best']} wall={row['wall_s']}",
                flush=True,
            )
            if (work / f"anneal-tip4p{n}-s{seed}.start.gmin").read_bytes() != start.read_bytes():
                raise RuntimeError(f"anneal opened another start at n={n} seed={seed}")
            for code, data, radius in arms:
                print(f"{code} recheck n={n} seed={seed}", flush=True)
                again = run_gmin(gmin, data, code, radius, n, seed, start, work)
                old = recorded[(again["system"], code, str(seed))]
                same = (
                    again["hit"] == old["hit"]
                    and again["evals_to_hit"] == old["evals_to_hit"]
                    and abs(float(again["best"]) - float(old["best"])) <= 1.0e-6
                )
                again |= {
                    "recorded_hit": old["hit"],
                    "recorded_evals_to_hit": old["evals_to_hit"],
                    "recorded_best": old["best"],
                    "same": str(int(same)),
                }
                repeated.append(again)
                new = not recheck_path.is_file() or recheck_path.stat().st_size == 0
                with recheck_path.open("a") as handle:
                    writer = csv.DictWriter(handle, fieldnames=RECHECK_FIELDS, lineterminator="\n")
                    if new:
                        writer.writeheader()
                    writer.writerow(again)
                print(
                    f"{code} n={n} seed={seed} hit={again['hit']} evals={again['evals_to_hit']} "
                    f"best={again['best']} same={same}",
                    flush=True,
                )
    summary = summarise(fresh)
    with summary_path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(row, flush=True)
    changed = [row for row in repeated if row["same"] != "1"]
    print(f"GMIN recheck: {len(repeated) - len(changed)}/{len(repeated)} rows repeat", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

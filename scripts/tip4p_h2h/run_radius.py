#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Append fixed-step GMIN TIP4P rows with and without a container radius.

Existing rows are left in place. ``gmin-radius`` reads
``examples/h2h_gmin_tip4p_radius<n>.data``. ``gmin-control`` reads
``examples/h2h_gmin_tip4p.data``, which leaves the Lennard-Jones container,
on the same binary in the same session. The two arms of a seed run back to
back from the start ``h2h_seed`` writes for that seed. One core. The caller
has already set the hostname to bench and the working directory to /work.
"""

from __future__ import annotations

import csv
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

# run_campaign.py scores 15 and 20 at the (H2O)20 paper ceiling. 4 and 8 are
# not in that map and use the same ceiling; a hit stops the run.
PAPER_BUDGETS = {4: 2_000_000, 8: 2_000_000, 15: 2_000_000, 20: 2_000_000}
REFERENCES = {
    4: -116.5904186160,
    8: -305.5183221676,
    15: -628.3722525077,
    20: -872.9887544952,
}
RADII = {4: 4.9229, 8: 5.4544, 15: 6.7744, 20: 7.5769}
HIT_TOL = 0.001
SEEDS = (2000, 2001, 2002, 2003, 2004, 2005)
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


def lj_container(n: int) -> float:
    # io1.f90 with RADIUS unset; NATOMS counts centre and angle-axis rows.
    return (2.0 + (3.0 * 2 * n / 17.77153175) ** (1.0 / 3.0)) * 2.0 ** (1.0 / 6.0)


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
        writer = csv.DictWriter(
            handle, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n"
        )
        if new:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in FIELDS})


def write_start(binary: Path, n: int, seed: int, work: Path) -> Path:
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
    if proc.returncode != 0 or not start.is_file():
        raise RuntimeError(
            f"start n={n} seed={seed} exited {proc.returncode}\n{proc.stdout[-800:]}"
        )
    return start


def run_gmin(
    gmin: Path, data: Path, code: str, radius: float, n: int, seed: int, start: Path, work: Path
) -> dict[str, str]:
    budget = PAPER_BUDGETS[n]
    run = work / f"{code}-tip4p{n}-s{seed}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "coords").write_bytes(start.read_bytes())
    (run / "data").write_text(data.read_text() + f"RANSEED {seed}\n")
    log = work / f"tip4p{n}-s{seed}-{code}.log"
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
    text = log.read_text()
    found = re.search(r"io1> Container radius=\s*(\S+)", text)
    if found is None or abs(float(found.group(1)) - radius) > 1e-4:
        raise RuntimeError(
            f"{code} n={n} seed={seed} container {found and found.group(1)}, want {radius}"
        )
    row_path = run / "row.txt"
    if not row_path.is_file():
        raise RuntimeError(f"{code} n={n} seed={seed} exited {proc.returncode}\n{text[-800:]}")
    raw = dict(part.split("=", 1) for part in row_path.read_text().split()[1:])
    if int(raw["budget"]) != budget:
        raise RuntimeError(f"{code} budget {raw}")
    # This build does not time NEWTIP, so the inside and outside cells stay blank.
    return {
        "system": f"tip4p{n}",
        "code": code,
        "seed": str(seed),
        "budget": str(budget),
        "hit": raw["hit"],
        "evals_to_hit": raw["evals_to_hit"],
        "charged": raw["charged"],
        "best": raw["best"],
        "wall_s": f"{float(raw['wall_s']):.6f}",
        "pot_s": "",
        "outside_s": "",
        "basins": "",
    }


def summarise(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[int, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((int(row["system"][5:]), row["code"]), []).append(row)
    out = []
    for (n, code), arm in sorted(groups.items()):
        out.append(
            {
                "system": f"tip4p{n}",
                "code": code,
                "radius": f"{RADII[n] if code == 'gmin-radius' else lj_container(n):.4f}",
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
    summary_path = root / "bench/tip4p-h2h/summary-radius.csv"
    work = Path("/work/tip4p-radius")
    work.mkdir(parents=True, exist_ok=True)
    gmin = Path("/work/gmin-h2h/bin/GMIN")
    control = root / "examples/h2h_gmin_tip4p.data"
    fresh: list[dict[str, str]] = []

    for n in PAPER_BUDGETS:
        arms = (
            ("gmin-control", control, lj_container(n)),
            ("gmin-radius", root / f"examples/h2h_gmin_tip4p_radius{n}.data", RADII[n]),
        )
        for seed in SEEDS:
            start = write_start(seed_bin, n, seed, work)
            for code, data, radius in arms:
                print(f"{code} n={n} seed={seed}", flush=True)
                row = run_gmin(gmin, data, code, radius, n, seed, start, work)
                append_row(rows_path, row)
                fresh.append(row)
                print(
                    f"{row['system']} {code} seed={seed} hit={row['hit']} "
                    f"evals={row['evals_to_hit']} best={row['best']} wall={row['wall_s']}",
                    flush=True,
                )
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

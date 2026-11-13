#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Append the adaptive-GMIN arm, the 4fdc37e8 anneal arm, and the LJ check.

Existing rows are left in place. New rows use a distinct code.
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
BUDGET_LJ = 4_000_000
REFERENCES = {15: -628.3722525077, 20: -872.9887544952}
LJ = {"lj75": (-397.492331, 75), "lj104": (-582.086642, 104)}
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


def parse_kv_line(text: str) -> dict[str, str]:
    for line in text.splitlines():
        if line.startswith("h2h_row "):
            return dict(part.split("=", 1) for part in line.split()[1:])
    raise ValueError("no h2h_row")


def parse_develop_log(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()
    return fields


def append_row(path: Path, row: dict[str, str]) -> None:
    new = not path.is_file() or path.stat().st_size == 0
    with path.open("a") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in FIELDS})


def run_gmin(gmin: Path, data: Path, n: int, seed: int, start: Path, work: Path) -> dict[str, str]:
    run = work / f"accrat-tip4p{n}-s{seed}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "coords").write_bytes(start.read_bytes())
    # `data` must be a file. A directory of that name makes GMIN stop.
    target = run / "data"
    if target.is_dir():
        raise RuntimeError(f"{target} is a directory")
    target.write_text(data.read_text() + f"RANSEED {seed}\n")
    log = work / f"tip4p{n}-s{seed}-gmin-accrat.log"
    env = env_of(BUDGET_WATER)
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
        raise RuntimeError(f"gmin-accrat n={n} seed={seed} exited {proc.returncode}\n{log.read_text()[-600:]}")
    raw = parse_kv_line(row_path.read_text())
    wall = float(raw["wall_s"])
    pot = float(raw.get("pot_s") or 0.0)
    return {
        "system": f"tip4p{n}",
        "code": "gmin-accrat",
        "seed": str(seed),
        "budget": str(BUDGET_WATER),
        "hit": raw["hit"],
        "evals_to_hit": raw["evals_to_hit"],
        "charged": raw["charged"],
        "best": raw["best"],
        "wall_s": f"{wall:.6f}",
        "pot_s": f"{pot:.6f}",
        "outside_s": f"{wall - pot:.6f}",
        "basins": "",
    }


def run_develop_water(binary: Path, n: int, seed: int, work: Path) -> dict[str, str]:
    prefix = work / f"dev-tip4p{n}-s{seed}"
    log = work / f"tip4p{n}-s{seed}-anneal-4fdc37e8.log"
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
        raise RuntimeError(f"develop tip4p{n} seed={seed} exited {proc.returncode}\n{text[-600:]}")
    fields = parse_develop_log(text)
    lowest = float(fields["lowest"])
    calls = int(fields["calls"])
    hit = lowest <= stop
    wall = float(fields["wall_seconds"])
    return {
        "system": f"tip4p{n}",
        "code": "anneal-4fdc37e8",
        "seed": str(seed),
        "budget": str(BUDGET_WATER),
        "hit": "1" if hit else "0",
        "evals_to_hit": str(calls if hit else BUDGET_WATER),
        "charged": fields["charged"],
        "best": f"{lowest:.10f}",
        "wall_s": f"{wall:.6f}",
        "pot_s": "",
        "outside_s": "",
        "basins": fields.get("basins", ""),
    }


def run_lj(binary: Path, system: str, seed: int, code: str) -> dict[str, str]:
    proc = subprocess.run(
        ["taskset", "-c", "0", str(binary), "run", system, str(seed)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env_of(BUDGET_LJ),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{code} {system} seed={seed} exited {proc.returncode}\n{proc.stdout[-600:]}")
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
        "code": code,
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
        if len(arm) != len(SEEDS):
            continue
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
    summary_path = root / "bench/tip4p-h2h/summary-followup.csv"
    work = Path("/work/tip4p-followup")
    work.mkdir(parents=True, exist_ok=True)
    gmin = Path("/work/gmin-h2h/bin/GMIN")
    data = root / "examples/h2h_gmin_tip4p_accrat.data"
    develop_water = Path("/work/anneal-4fdc37e8/target/release/examples/h2h_seed")
    develop_lj = Path("/work/anneal-4fdc37e8/target/release/examples/h2h_gmin_proof")
    tip_lj = Path("/work/tip4p-fix/target/release/examples/h2h_gmin_proof")
    starts = Path("/work/tip4p-h2h/runs")
    fresh: list[dict[str, str]] = []

    def keep(row: dict[str, str]) -> None:
        append_row(rows_path, row)
        fresh.append(row)
        print(
            f"{row['system']} {row['code']} seed={row['seed']} hit={row['hit']} "
            f"evals={row['evals_to_hit']} best={row['best']}",
            flush=True,
        )

    for n in (15, 20):
        for seed in SEEDS:
            print(f"gmin-accrat n={n} seed={seed}", flush=True)
            start = starts / f"tip4p{n}-s{seed}.start.gmin"
            keep(run_gmin(gmin, data, n, seed, start, work))
    for n in (15, 20):
        for seed in SEEDS:
            print(f"anneal-4fdc37e8 n={n} seed={seed}", flush=True)
            keep(run_develop_water(develop_water, n, seed, work))
    for system in ("lj75", "lj104"):
        for seed in SEEDS:
            print(f"tip {system} seed={seed}", flush=True)
            keep(run_lj(tip_lj, system, seed, "anneal"))
        for seed in SEEDS:
            print(f"develop {system} seed={seed}", flush=True)
            keep(run_lj(develop_lj, system, seed, "anneal-4fdc37e8"))

    summary = summarise(fresh)
    with summary_path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(row, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

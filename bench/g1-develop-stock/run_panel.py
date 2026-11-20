#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Run the ten added Morse cells with the stock preset."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ROWS = HERE / "anneal-rows.csv"
LOGS = HERE / "logs"
BASE_REVISION = "61fa67cf380e46876b569da45e7cc091647a775b"
SEEDS = tuple(range(2000, 2012))
SYSTEMS = (
    ("m30r6", 30, 400_000, -118.432844),
    ("m30r14", 30, 400_000, -106.835790),
    ("m38r6", 38, 400_000, -157.477108),
    ("m38r14", 38, 400_000, -144.321054),
    ("m70r3", 70, 1_000_000, -577.739782),
    ("m70r6", 70, 1_000_000, -325.887749),
    ("m70r10", 70, 1_000_000, -296.412149),
    ("m70r14", 70, 1_000_000, -292.462856),
    ("m80r3", 80, 4_000_000, -690.577890),
    ("m80r10", 80, 4_000_000, -344.911381),
)
FIELDS = [
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
    "code_revision",
    "host",
    "config",
]
THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "MKL_DOMAIN_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "POLARS_MAX_THREADS",
)


def child_env(budget: int) -> dict[str, str]:
    env = {"LC_ALL": "C", "H2H_BUDGET": str(budget)}
    for key in ("PATH", "HOME", "LD_LIBRARY_PATH"):
        if key in os.environ:
            env[key] = os.environ[key]
    for key in THREAD_KEYS:
        env[key] = "1"
    env["OMP_DYNAMIC"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"
    return env


def machine_hash() -> str:
    value = Path("/etc/machine-id").read_text().strip().encode()
    return hashlib.sha256(value).hexdigest()


def release_binary() -> Path:
    proc = subprocess.run(
        [
            "rustup",
            "run",
            "1.88.0",
            "cargo",
            "metadata",
            "--format-version",
            "1",
            "--no-deps",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    target = json.loads(proc.stdout)["target_directory"]
    return Path(target) / "release" / "examples" / "h2h_morse"


def read_rows() -> list[dict[str, str]]:
    if not ROWS.is_file():
        return []
    with ROWS.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_row(row: dict[str, str]) -> None:
    fresh = not ROWS.is_file()
    with ROWS.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def check_specs(binary: Path) -> None:
    for system, _n, budget, reference in SYSTEMS:
        proc = subprocess.run(
            [str(binary), "budget", system],
            cwd=ROOT,
            env=child_env(budget),
            check=True,
            capture_output=True,
            text=True,
        )
        lines = proc.stdout.splitlines()
        if int(lines[0]) != budget or abs(float(lines[1]) - reference) > 1e-12:
            raise SystemExit(f"{system}: binary specification differs")


def run_one(binary: Path, system: str, seed: int, budget: int) -> None:
    command = ["taskset", "-c", "0", str(binary), "run", system, str(seed)]
    proc = subprocess.run(
        command,
        cwd=ROOT,
        env=child_env(budget),
        check=False,
        capture_output=True,
        text=True,
    )
    stem = LOGS / f"{system}-{seed}"
    stem.with_suffix(".stdout").write_text(proc.stdout)
    stem.with_suffix(".stderr").write_text(proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(f"{system} {seed} exited {proc.returncode}")
    lines = [line for line in proc.stdout.splitlines() if line.startswith(f"{system},anneal,")]
    if len(lines) != 1:
        raise SystemExit(f"{system} {seed}: expected one result row")
    header = "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,pot_s,outside_s"
    row = next(csv.DictReader([header, lines[0]]))
    if int(row["budget"]) != budget:
        raise SystemExit(f"{system} {seed}: wrong budget")
    row["hit"] = "1" if row["hit"] == "true" else "0"
    row["code_revision"] = BASE_REVISION
    row["host"] = machine_hash()
    row["config"] = "recommended"
    append_row(row)


def main() -> None:
    if len(sys.argv) != 1:
        raise SystemExit("usage: run_panel.py")
    binary = release_binary()
    if not binary.is_file():
        raise SystemExit("release binary is missing")
    LOGS.mkdir(parents=True, exist_ok=True)
    check_specs(binary)
    done = {(row["system"], int(row["seed"])) for row in read_rows()}
    for system, _n, budget, _reference in SYSTEMS:
        for seed in SEEDS:
            if (system, seed) in done:
                continue
            print(f"run {system} {seed}", flush=True)
            run_one(binary, system, seed, budget)
            done.add((system, seed))


if __name__ == "__main__":
    main()

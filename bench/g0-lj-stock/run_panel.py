# Copyright (c) anneal developers
"""Run the stock panel one seed at a time.

The binary is examples/g0_lj_stock, already built. Each seed is
pinned to cpu 0. Thread variables are set to 1. A row already stored
for the same cell and seed is left in place.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ROWS = HERE / "rows.csv"
MISSES = HERE / "misses.csv"
STARTS = HERE / "starts"
MISS_DIR = HERE / "misses"
ROW_FIELDS = [
    "cell",
    "seed",
    "start",
    "hit",
    "calls_to_hit",
    "charged",
    "best",
    "best_bits",
    "wall_s",
    "pot_s",
    "outside_s",
    "hop_first_best",
    "preset_hash",
    "revision",
    "core",
    "affinity",
    "machine_id_sha256",
    "session",
    "sequence",
]
MISS_FIELDS = [
    "cell",
    "seed",
    "point_group",
    "tolerance",
    "cna_signature",
    "structure_type",
    "cutoff",
]
THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_NUM_THREADS",
    "MKL_DOMAIN_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "OMP_DYNAMIC",
    "MKL_DYNAMIC",
)


def child_env() -> dict[str, str]:
    env = {"LC_ALL": "C"}
    for key in ("PATH", "HOME", "LD_LIBRARY_PATH"):
        if key in os.environ:
            env[key] = os.environ[key]
    for key in THREAD_KEYS:
        env[key] = "1"
    env["OMP_DYNAMIC"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"
    return env


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append(path: Path, fields: list[str], row: dict[str, str]) -> None:
    fresh = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def release_binary() -> Path:
    proc = subprocess.run(
        ["rustup", "run", "1.88.0", "cargo", "metadata", "--format-version", "1", "--no-deps"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    target = json.loads(proc.stdout)["target_directory"]
    return Path(target) / "release" / "examples" / "g0_lj_stock"


def jobs(binary: Path) -> list[tuple[str, str]]:
    proc = subprocess.run(
        [str(binary), "list"],
        cwd=ROOT,
        env=child_env(),
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in proc.stdout.splitlines() if line and not line.startswith("cell,")]
    out = []
    for line in lines:
        cell = line.split(",", 1)[0]
        for seed in range(2000, 2012):
            out.append((cell, str(seed)))
    return out


def run_one(binary: Path, cell: str, seed: str, session: str, sequence: str) -> None:
    start = STARTS / f"{cell}-{seed}.txt"
    command = [
        "taskset",
        "-c",
        "0",
        str(binary),
        "run",
        cell,
        seed,
        str(start),
        session,
        sequence,
        str(MISS_DIR),
    ]
    proc = subprocess.run(command, cwd=ROOT, env=child_env(), capture_output=True, text=True, check=False)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(f"{cell} {seed} exited {proc.returncode}")
    row = None
    for line in proc.stdout.splitlines():
        if line.startswith("miss,"):
            fields = line.split(",")
            if len(fields) != 8:
                raise SystemExit(f"{cell} {seed}: bad miss line")
            append(
                MISSES,
                MISS_FIELDS,
                {
                    "cell": fields[1],
                    "seed": fields[2],
                    "point_group": fields[3],
                    "tolerance": fields[4],
                    "cna_signature": fields[5],
                    "structure_type": fields[6],
                    "cutoff": fields[7],
                },
            )
        elif line.startswith("row,"):
            fields = line.split(",")
            if len(fields) != 1 + len(ROW_FIELDS):
                raise SystemExit(f"{cell} {seed}: row has {len(fields)} fields")
            row = dict(zip(ROW_FIELDS, fields[1:], strict=True))
    if row is None:
        raise SystemExit(f"{cell} {seed}: no row")
    if row["cell"] != cell or row["seed"] != seed:
        raise SystemExit(f"{cell} {seed}: row identity mismatch")
    append(ROWS, ROW_FIELDS, row)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_panel.py SESSION")
    session = sys.argv[1]
    binary = release_binary()
    if not binary.is_file():
        raise SystemExit("release binary is missing")
    MISS_DIR.mkdir(parents=True, exist_ok=True)
    done = {(row["cell"], row["seed"]) for row in read_rows(ROWS)}
    sequence = len(done)
    for cell, seed in jobs(binary):
        if (cell, seed) in done:
            continue
        sequence += 1
        run_one(binary, cell, seed, session, str(sequence))
        done.add((cell, seed))


if __name__ == "__main__":
    main()

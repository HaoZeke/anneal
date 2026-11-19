#!/usr/bin/env python3
"""Run the three-arm shelf gate serially and append one row per process."""

from __future__ import annotations

import argparse
import csv
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class System:
    name: str
    kind: str
    n: int
    rho: float
    budget: int
    reference: float


SYSTEMS = (
    System("lj38", "lj", 38, 0.0, 400_000, -173.928427),
    System("lj75", "lj", 75, 0.0, 4_000_000, -397.492331),
    System("lj76", "lj", 76, 0.0, 4_000_000, -402.894866),
    System("lj77", "lj", 77, 0.0, 4_000_000, -409.083517),
    System("lj98", "lj", 98, 0.0, 4_000_000, -543.665361),
    System("lj102", "lj", 102, 0.0, 4_000_000, -569.363652),
    System("lj103", "lj", 103, 0.0, 4_000_000, -575.766131),
    System("lj104", "lj", 104, 0.0, 4_000_000, -582.086642),
    System("binary-lj", "binary", 100, 0.0, 2_000_000, -604.7963065717),
    System("m80r6", "morse", 80, 6.0, 4_000_000, -378.333471),
    System("m80r14", "morse", 80, 14.0, 4_000_000, -340.811371),
)
SEEDS = tuple(range(2000, 2012))
HIT_TOL = 0.001
FIELDS = (
    "system",
    "arm",
    "seed",
    "budget",
    "hit",
    "calls",
    "charged",
    "wall_s",
    "pot_s",
    "outside_s",
    "best",
    "code_revision",
    "host",
    "status",
)
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
}


@dataclass(frozen=True)
class Arm:
    name: str
    revision: str
    tree: Path
    atomic_binary: Path
    binary_lj_binary: Path


def parse_arm(values: list[str]) -> Arm:
    name, revision, tree, atomic, binary_lj = values
    return Arm(name, revision, Path(tree), Path(atomic), Path(binary_lj))


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if rows and tuple(rows[0]) != FIELDS:
        raise SystemExit(f"{path} has an unexpected header")
    return rows


def append_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def values_from_output(text: str) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.startswith("h2h_row ")]
    if len(lines) != 1:
        raise ValueError(f"expected one h2h_row, found {len(lines)}")
    return dict(token.split("=", 1) for token in lines[0].split()[1:])


def command_for(
    system: System,
    arm: Arm,
    seed: int,
    binary_inputs: Path,
    run_dir: Path,
) -> list[str]:
    level = system.reference + HIT_TOL
    if system.kind == "binary":
        start = binary_inputs / f"coords.{seed - SEEDS[0]}"
        if not start.is_file():
            raise FileNotFoundError(start)
        return [
            "taskset",
            "-c",
            "0",
            str(arm.binary_lj_binary),
            str(start),
            str(system.budget),
            f"{level:.12f}",
            str(seed),
        ]
    prefix = run_dir / "run"
    return [
        "taskset",
        "-c",
        "0",
        str(arm.atomic_binary),
        system.kind,
        str(system.n),
        str(system.rho),
        str(system.budget),
        str(seed),
        "recommended",
        f"{level:.12f}",
        str(prefix),
    ]


def validate_arms(arms: list[Arm]) -> dict[str, Arm]:
    by_name = {arm.name: arm for arm in arms}
    if tuple(sorted(by_name)) != ("A", "B", "C") or len(arms) != 3:
        raise SystemExit("the arms must be A, B and C exactly once")
    for arm in arms:
        head = subprocess.run(
            ["git", "-C", str(arm.tree), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if head != arm.revision:
            raise SystemExit(f"arm {arm.name} is at {head}, expected {arm.revision}")
        for binary in (arm.atomic_binary, arm.binary_lj_binary):
            if not binary.is_file() or not os.access(binary, os.X_OK):
                raise SystemExit(f"arm {arm.name} has no executable {binary}")
    return by_name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--binary-inputs", type=Path, required=True)
    parser.add_argument("--arm", nargs=5, action="append", default=[])
    args = parser.parse_args()

    if socket.gethostname() != "bench":
        raise SystemExit("the benchmark must run with UTS hostname bench")
    arms = validate_arms([parse_arm(values) for values in args.arm])
    stored = read_rows(args.rows)
    done = {(row["system"], row["arm"], int(row["seed"])) for row in stored}
    if len(done) != len(stored):
        raise SystemExit("rows contain a duplicate system, arm and seed")

    env = os.environ.copy()
    env.update(THREAD_ENV)
    failures = 0
    for system in SYSTEMS:
        for seed_index, seed in enumerate(SEEDS):
            order = ("A", "B", "C")
            order = order[seed_index % 3 :] + order[: seed_index % 3]
            for name in order:
                key = (system.name, name, seed)
                if key in done:
                    print(f"skip system={system.name} arm={name} seed={seed}", flush=True)
                    continue
                arm = arms[name]
                run_dir = args.runs / system.name / str(seed) / name
                log = args.logs / system.name / f"{seed}-{name}.log"
                run_dir.mkdir(parents=True, exist_ok=True)
                log.parent.mkdir(parents=True, exist_ok=True)
                row = {field: "" for field in FIELDS}
                row.update(
                    {
                        "system": system.name,
                        "arm": name,
                        "seed": str(seed),
                        "budget": str(system.budget),
                        "code_revision": arm.revision,
                        "host": socket.gethostname(),
                    }
                )
                try:
                    command = command_for(system, arm, seed, args.binary_inputs, run_dir)
                    process = subprocess.run(
                        command,
                        cwd=arm.tree,
                        env=env,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    log.write_text(process.stdout + process.stderr)
                    if process.returncode != 0:
                        raise RuntimeError(f"exit_{process.returncode}")
                    result = values_from_output(process.stdout)
                    row.update(
                        {
                            "hit": str(int(result["hit"])),
                            "calls": str(int(result["evals_to_hit"])),
                            "charged": str(int(result["charged"])),
                            "wall_s": f"{float(result['wall_s']):.6f}",
                            "pot_s": f"{float(result['pot_s']):.6f}",
                            "outside_s": f"{float(result['outside_s']):.6f}",
                            "best": f"{float(result['best']):.10f}",
                            "status": "ok",
                        }
                    )
                except Exception as error:
                    row["status"] = f"failed:{type(error).__name__}:{error}"
                    failures += 1
                append_row(args.rows, row)
                done.add(key)
                print(
                    f"done system={system.name} arm={name} seed={seed} "
                    f"status={row['status']} wall_s={row['wall_s'] or '-'}",
                    flush=True,
                )
    if failures:
        print(f"{failures} run(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

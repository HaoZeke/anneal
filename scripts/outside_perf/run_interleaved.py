#!/usr/bin/env python3
# Copyright (c) anneal developers
"""One serial three-arm outside-potential session.

code_revision identifies the source. machine_sha256 identifies the
machine without recording a host name. core and affinity come from the
pinned process.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import subprocess
import time
from pathlib import Path

SYSTEMS = {
    "lj150": {"binary": "h2h_gmin_proof", "paper_budget": 4_000_000},
    "lj104": {"binary": "h2h_gmin_proof", "paper_budget": 4_000_000},
    "m80r6": {"binary": "h2h_morse", "paper_budget": 4_000_000},
    "m80r14": {"binary": "h2h_morse", "paper_budget": 4_000_000},
    "au55": {"binary": "h2h_metal", "paper_budget": 1_000_000},
    "blj55s130": {"binary": "h2h_blj", "paper_budget": 1_000_000, "call_cap": 500_000},
    "blj38s105": {"binary": "h2h_blj", "paper_budget": 400_000, "call_cap": 500_000},
}
PAPER_BUDGETS = {name: spec["paper_budget"] for name, spec in SYSTEMS.items()}
SEEDS = tuple(range(2000, 2012))
ARMS = ("develop", "develop_prime", "new")
FIELDS = [
    "system",
    "arm",
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
    "binary_sha256",
    "core",
    "affinity",
    "machine_sha256",
    "session",
    "sequence",
]
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def arm_order(seed: int) -> tuple[str, str, str]:
    """ABBA rotation: reverse the second and third seeds of every block of four.

    The block's starting arm advances after each block, so twelve seeds put
    every arm in every position four times.
    """
    index = seed - SEEDS[0]
    shift = (index // 4) % len(ARMS)
    rotated = list(ARMS[shift:] + ARMS[:shift])
    if index % 4 in (1, 2):
        rotated.reverse()
    return tuple(rotated)


def schedule() -> list[tuple[str, int, str]]:
    positions = {arm: [0, 0, 0] for arm in ARMS}
    orders = {}
    for seed in SEEDS:
        order = arm_order(seed)
        if len(order) != len(ARMS) or set(order) != set(ARMS):
            raise RuntimeError(f"seed {seed} order {order}")
        orders[seed] = order
        for place, arm in enumerate(order):
            positions[arm][place] += 1
    if any(counts != [4, 4, 4] for counts in positions.values()):
        raise RuntimeError(f"unbalanced rotation: {positions}")
    return [
        (system, seed, arm)
        for system in SYSTEMS
        for seed in SEEDS
        for arm in orders[seed]
    ]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def machine_sha256() -> str:
    raw = Path("/etc/machine-id").read_bytes().strip()
    if not raw:
        raise SystemExit("machine id is empty")
    return hashlib.sha256(raw).hexdigest()


def pin_record() -> tuple[str, str]:
    allowed = sorted(os.sched_getaffinity(0))
    if allowed != [0]:
        raise SystemExit("process is not pinned to core 0")
    return "0", "0"


def parse(system: str, text: str, budget: int) -> dict[str, str]:
    rows = list(csv.DictReader(text.splitlines()))
    rows = [row for row in rows if row.get("system") == system and row.get("code") == "anneal"]
    if len(rows) != 1:
        raise RuntimeError(f"{system}: expected one result row")
    row = rows[0]
    wall = float(row["wall_s"])
    pot = float(row["pot_s"])
    if int(row["budget"]) != budget:
        raise RuntimeError(f"{system}: binary budget {row['budget']}, expected {budget}")
    return {
        "system": row["system"],
        "seed": row["seed"],
        "budget": row["budget"],
        "hit": row["hit"],
        "evals_to_hit": row["evals_to_hit"],
        "charged": row["charged"],
        "wall_s": f"{wall:.6f}",
        "best": row["best"],
        "pot_s": f"{pot:.6f}",
        "outside_s": f"{wall - pot:.6f}",
    }


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise SystemExit(f"unexpected row header: {reader.fieldnames}")
        return list(reader)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--develop-bin", type=Path, required=True)
    parser.add_argument("--develop-prime-bin", type=Path, required=True)
    parser.add_argument("--new-bin", type=Path, required=True)
    parser.add_argument("--develop-revision", required=True)
    parser.add_argument("--new-revision", required=True)
    parser.add_argument("--rows", type=Path, required=True)
    args = parser.parse_args()
    planned = schedule()
    core, affinity = pin_record()
    machine = machine_sha256()
    directories = {
        "develop": args.develop_bin,
        "develop_prime": args.develop_prime_bin,
        "new": args.new_bin,
    }
    revisions = {
        "develop": args.develop_revision,
        "develop_prime": args.develop_revision,
        "new": args.new_revision,
    }
    binaries: dict[tuple[str, str], tuple[Path, str]] = {}
    for arm, directory in directories.items():
        for executable in {spec["binary"] for spec in SYSTEMS.values()}:
            path = directory / executable
            if not path.is_file():
                raise SystemExit(f"missing {path}")
            binaries[(arm, executable)] = (path, digest(path))
    env = os.environ.copy()
    for name in THREAD_VARIABLES:
        env[name] = "1"
    for name in list(env):
        if name.startswith("H2H_"):
            del env[name]
    existing = load_rows(args.rows)
    if existing:
        session = existing[0]["session"]
        if any(row["session"] != session for row in existing):
            raise SystemExit("existing rows span more than one session")
        for row, expected in zip(existing, planned):
            got = (row["system"], int(row["seed"]), row["arm"])
            if got != expected:
                raise SystemExit(f"existing row {got} does not match schedule {expected}")
            if row["core"] != core or row["affinity"] != affinity or row["machine_sha256"] != machine:
                raise SystemExit("existing row was recorded on a different pin or machine")
    else:
        session = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    args.rows.parent.mkdir(parents=True, exist_ok=True)
    with args.rows.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        if not existing:
            writer.writeheader()
        for sequence, (system, seed, arm) in enumerate(planned, start=1):
            if sequence <= len(existing):
                continue
            spec = SYSTEMS[system]
            executable = spec["binary"]
            run_budget = int(spec.get("call_cap", spec["paper_budget"]))
            binary, binary_digest = binaries[(arm, executable)]
            command = ["taskset", "-c", "0", str(binary), "run", system, str(seed)]
            run_env = env.copy()
            call_cap = spec.get("call_cap")
            if call_cap is not None:
                run_env["H2H_BUDGET"] = str(call_cap)
                run_env["H2H_FORCE_FULL"] = "1"
            proc = subprocess.run(command, env=run_env, capture_output=True, text=True, check=True)
            row = parse(system, proc.stdout, run_budget)
            row["budget"] = str(spec["paper_budget"])
            if int(row["seed"]) != seed:
                raise RuntimeError(f"{system}: seed {row['seed']}, expected {seed}")
            row.update(
                {
                    "arm": arm,
                    "code_revision": revisions[arm],
                    "binary_sha256": binary_digest,
                    "core": core,
                    "affinity": affinity,
                    "machine_sha256": machine,
                    "session": session,
                    "sequence": str(sequence),
                }
            )
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
            print(
                f"{sequence:03d} {system} {seed} {arm} "
                f"inside={row['pot_s']} outside={row['outside_s']}",
                flush=True,
            )


if __name__ == "__main__":
    main()

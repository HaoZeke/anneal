#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Interleaved base/HEAD outside-potential timings on the GMIN panel."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import socket
import subprocess
import time
from pathlib import Path

SYSTEMS = {
    "lj150": ("h2h_gmin_proof", 4_000_000),
    "lj104": ("h2h_gmin_proof", 4_000_000),
    "m80r6": ("h2h_morse", 4_000_000),
    "au55": ("h2h_metal", 1_000_000),
}
SEEDS = tuple(range(2000, 2012))
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
    "host",
    "session",
    "sequence",
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def parse(system: str, text: str, budget: int) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.startswith(f"{system},anneal,")]
    if len(lines) != 1:
        raise RuntimeError(f"{system}: expected one result row")
    fields = lines[0].split(",")
    wall = float(fields[7])
    if system.startswith("lj"):
        pot = float(fields[13])
    else:
        pot = float(fields[9])
    if int(fields[3]) != budget:
        raise RuntimeError(f"{system}: binary budget {fields[3]}, expected {budget}")
    return {
        "system": fields[0],
        "seed": fields[2],
        "budget": fields[3],
        "hit": fields[4],
        "evals_to_hit": fields[5],
        "charged": fields[6],
        "wall_s": f"{wall:.6f}",
        "best": fields[8],
        "pot_s": f"{pot:.6f}",
        "outside_s": f"{wall - pot:.6f}",
    }


def main() -> None:
    if socket.gethostname() != "bench":
        raise SystemExit(f"hostname is {socket.gethostname()}, expected bench")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-bin", type=Path, required=True)
    parser.add_argument("--new-bin", type=Path, required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--new-revision", required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--systems", default=",".join(SYSTEMS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    args = parser.parse_args()
    systems = [name for name in args.systems.split(",") if name]
    seeds = [int(seed) for seed in args.seeds.split(",") if seed]
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
        env[name] = "1"
    for name in list(env):
        if name.startswith("H2H_"):
            del env[name]
    binaries: dict[tuple[str, str], tuple[Path, str, str]] = {}
    for arm, directory, revision in (
        ("base", args.base_bin, args.base_revision),
        ("new", args.new_bin, args.new_revision),
    ):
        for executable, _budget in set(SYSTEMS.values()):
            path = directory / executable
            if not path.is_file():
                raise SystemExit(f"missing {path}")
            binaries[(arm, executable)] = (path, revision, digest(path))
    session = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    args.rows.parent.mkdir(parents=True, exist_ok=True)
    sequence = 0
    with args.rows.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for system in systems:
            executable, budget = SYSTEMS[system]
            for seed in seeds:
                for arm in ("base", "new"):
                    binary, revision, binary_digest = binaries[(arm, executable)]
                    sequence += 1
                    command = ["taskset", "-c", "0", str(binary), "run", system, str(seed)]
                    proc = subprocess.run(
                        command,
                        env=env,
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    row = parse(system, proc.stdout, budget)
                    if int(row["seed"]) != seed:
                        raise RuntimeError(f"{system}: seed {row['seed']}, expected {seed}")
                    row.update(
                        {
                            "arm": arm,
                            "code_revision": revision,
                            "binary_sha256": binary_digest,
                            "host": socket.gethostname(),
                            "session": session,
                            "sequence": str(sequence),
                        }
                    )
                    writer.writerow(row)
                    handle.flush()
                    os.fsync(handle.fileno())
                    print(
                        f"{sequence:02d} {system} {seed} {arm} "
                        f"inside={row['pot_s']} outside={row['outside_s']}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()

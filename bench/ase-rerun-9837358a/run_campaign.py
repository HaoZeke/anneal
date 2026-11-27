#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Run and record the fixed ASE provenance campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from target_tol import DEFAULT_HIT_TOL, energy_hits  # noqa: E402

REVISION = "9837358ae1027067ff654fac8b48969020853452"
TREE = "15032ef401fb681e230d27c7b315d89f7ac53e4e"
ASE_VERSION = "3.29.0"
ASE_WHEEL_SHA256 = (
    "7b9dd103f007810339c24acfee2f6b677c0c48443b21d3c98e52959246cf4ebf"
)
HIT_TOL = DEFAULT_HIT_TOL
SYSTEMS = {
    "lj38": (400_000, -173.928427),
    "lj55": (1_000_000, -279.248470),
    "lj98": (4_000_000, -543.665361),
    "lj104": (4_000_000, -582.086642),
    "lj150": (4_000_000, -893.310258),
    "au55": (1_000_000, -191.03085280),
}
ARMS = ("anneal", "ase_bh", "ase_mh", "recommended")
SEEDS = tuple(range(2000, 2012))
FIELDS = (
    "system",
    "arm",
    "seed",
    "budget",
    "energy_calls",
    "gradient_calls",
    "calls_to_hit",
    "best_energy",
    "reference_energy",
    "hit",
    "wall_s",
    "wall_cap_s",
    "stop_reason",
    "anneal_commit",
    "anneal_tree",
    "ase_version",
    "ase_wheel_sha256",
    "machine_id",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-root", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--asg", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    return parser.parse_args()


def machine_id() -> str:
    return "unrecorded"


def existing_rows(path: Path) -> dict[tuple[str, str, int], dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise RuntimeError("unexpected output columns")
        rows = list(reader)
    keyed = {(row["system"], row["arm"], int(row["seed"])): row for row in rows}
    if len(keyed) != len(rows):
        raise RuntimeError("duplicate output rows")
    return keyed


def append_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    create = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        if create:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def write_log(path: Path, row: dict[str, str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(f"{field}={row[field]}\n" for field in FIELDS)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.sync()
    temporary.replace(path)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar_tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
    with sidecar_tmp.open("w", newline="\n") as handle:
        handle.write(f"{digest}  {path.name}\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.sync()
    sidecar_tmp.replace(sidecar)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise RuntimeError("log read-back hash mismatch")
    return digest


def quote(value: object) -> str:
    return shlex.quote(str(value))


def run_one(
    args: argparse.Namespace,
    root: Path,
    system: str,
    arm: str,
    seed: int,
    cap: str,
) -> dict[str, str]:
    raw_dir = args.work / "raw"
    run_dir = args.work / "runs"
    home = args.work / "home"
    numba = args.work / "numba"
    for directory in (raw_dir, run_dir, home, numba):
        directory.mkdir(parents=True, exist_ok=True)
    stem = f"{system}-{arm}-{seed}"
    stdout_path = raw_dir / f"{stem}.stdout.tmp"
    stderr_path = raw_dir / f"{stem}.stderr.tmp"
    exports = {
        "HOME": home,
        "NUMBA_CACHE_DIR": numba,
        "PYTHONPATH": args.python_root,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "OMP_PROC_BIND": "TRUE",
        "OMP_PLACES": "cores",
        "PYTHONUNBUFFERED": "1",
        "ASE_PANEL_WORK": run_dir,
        "ASE_PANEL": args.panel,
        "ASG_EVAL": args.asg,
        "ASE_PANEL_WALL_CAP": cap,
        "ASE_PANEL_REVISION": REVISION,
    }
    environment = " ".join(f"{key}={quote(value)}" for key, value in exports.items())
    command = (
        f"hostname bench; exec env {environment} taskset -c 0 "
        f"python3 {quote(root / 'scripts/ase_panel/run_one.py')} "
        f"{quote(system)} {quote(arm)} {seed}"
    )
    with stdout_path.open("w", newline="\n") as stdout, stderr_path.open(
        "w", newline="\n"
    ) as stderr:
        proc = subprocess.run(
            ["sudo", "-n", "unshare", "--uts", "sh", "-c", command],
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
        stdout.flush()
        stderr.flush()
        os.fsync(stdout.fileno())
        os.fsync(stderr.fileno())
    if proc.returncode != 0:
        raise RuntimeError(f"{stem} exited {proc.returncode}")
    lines = stdout_path.read_text().splitlines()
    raw = next((line for line in lines if line.startswith(f"{system},{arm},")), "")
    fields = raw.split(",")
    if len(fields) != 15 or fields[11] != REVISION:
        raise RuntimeError(f"{stem} produced an invalid row")
    charged = int(fields[6])
    if charged <= 0 or not fields[13]:
        raise RuntimeError(f"{stem} did not prove its code path")
    budget, reference = SYSTEMS[system]
    hit = fields[4]
    best = float(fields[7])
    if hit not in {"0", "1"}:
        raise RuntimeError(f"{stem} produced an invalid hit value")
    if energy_hits(best, reference, HIT_TOL) != (hit == "1"):
        raise RuntimeError(f"{stem} hit classification disagrees with the reference")
    return {
        "system": system,
        "arm": arm,
        "seed": str(seed),
        "budget": str(budget),
        "energy_calls": str(charged),
        "gradient_calls": str(charged),
        "calls_to_hit": fields[5],
        "best_energy": fields[7],
        "reference_energy": f"{reference:.8f}",
        "hit": hit,
        "wall_s": fields[8],
        "wall_cap_s": cap,
        "stop_reason": fields[13],
        "anneal_commit": REVISION,
        "anneal_tree": TREE,
        "ase_version": ASE_VERSION,
        "ase_wheel_sha256": ASE_WHEEL_SHA256,
        "machine_id": machine_id(),
    }


def pending_rows(path: Path, system: str) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise RuntimeError("unexpected pending columns")
        return [
            row
            for row in reader
            if row["system"] == system and row["arm"] == "anneal"
        ]


def main() -> int:
    args = arguments()
    root = Path(__file__).resolve().parents[2]
    actual_revision = subprocess.run(
        ["git", "-C", root, "rev-parse", REVISION],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    actual_tree = subprocess.run(
        ["git", "-C", root, "rev-parse", f"{REVISION}^{{tree}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if (actual_revision, actual_tree) != (REVISION, TREE):
        raise RuntimeError("source revision or tree mismatch")
    version = subprocess.run(
        [
            "env",
            f"PYTHONPATH={args.python_root}",
            "python3",
            "-c",
            "import ase; print(ase.__version__)",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if version != ASE_VERSION:
        raise RuntimeError("ASE version mismatch")

    stored = existing_rows(args.rows)
    pending_path = args.work / "pending-anneal.csv"
    for system in SYSTEMS:
        anneal = pending_rows(pending_path, system)
        pending_keys = {int(row["seed"]) for row in anneal}
        final_anneal = [
            row
            for (row_system, row_arm, _seed), row in stored.items()
            if row_system == system and row_arm == "anneal"
        ]
        if final_anneal:
            if len(final_anneal) != len(SEEDS):
                raise RuntimeError(f"{system} has a partial final anneal arm")
            cap = final_anneal[0]["wall_cap_s"]
        else:
            for seed in SEEDS:
                if seed not in pending_keys:
                    row = run_one(args, root, system, "anneal", seed, "")
                    append_row(pending_path, row)
                    anneal.append(row)
            anneal.sort(key=lambda row: int(row["seed"]))
            if tuple(int(row["seed"]) for row in anneal) != SEEDS:
                raise RuntimeError(f"{system} has incomplete anneal seeds")
            cap = f"{3.0 * max(float(row['wall_s']) for row in anneal):.6f}"
            for row in anneal:
                row["wall_cap_s"] = cap
                append_row(args.rows, row)
                write_log(
                    args.logs / f"{system}-anneal-{row['seed']}.log",
                    row,
                )
                stored[(system, "anneal", int(row["seed"]))] = row

        for arm in ARMS[1:]:
            for seed in SEEDS:
                key = (system, arm, seed)
                if key in stored:
                    continue
                row = run_one(args, root, system, arm, seed, cap)
                append_row(args.rows, row)
                digest = write_log(args.logs / f"{system}-{arm}-{seed}.log", row)
                stored[key] = row
                print(f"recorded {system} {arm} {seed} {digest}", flush=True)

    if len(stored) != len(SYSTEMS) * len(ARMS) * len(SEEDS):
        raise RuntimeError(f"incomplete campaign: {len(stored)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

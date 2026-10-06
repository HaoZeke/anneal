#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "cyclopts>=3",
#   "qcg-pilotjob>=0.13",
# ]
# ///
"""One Slurm job per Lennard-Jones seed, run by QCG-PilotJob.

The science runs on the compute node's /scratch/users/$USER disk.
A finished log is the only file copied back, at the filer bandwidth cap.
`plan` prints the jobs. `submit` queues them only with --yes.
`cell` is what each job runs, and it refuses to start without SLURM_JOB_ID.

    uv run --script scripts/elja_qcg_cell.py plan --n 75 --budget 4000000 \
        --seeds 1 --arm rec --binary "$LJ_BIN" --record "$LJ_RECORD"
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import cyclopts

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts" / "elja_scratch.sh"
app = cyclopts.App(help=__doc__)


def _sbatch(n: int, budget: int, arm: str, seed: int, binary: Path, record: Path) -> str:
    script = Path(__file__).resolve()
    quoted = shlex.quote(str(script))
    binary_q = shlex.quote(str(binary))
    record_q = shlex.quote(str(record))
    return f"""#!/bin/bash
#SBATCH --job-name=lj{n}-{arm}-{seed}
#SBATCH --partition=s-normal
#SBATCH --account=chem-ui
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --output={record}/slurm-lj{n}-{arm}-{seed}-%j.out
set -euo pipefail
exec uv run --script {quoted} cell --n {n} --budget {budget} --arm {arm} --seed {seed} --binary {binary_q} --record {record_q}
"""


def _check_common(seeds: int, arm: str, binary: Path) -> None:
    if seeds < 1 or seeds > 48:
        raise SystemExit("seeds must be from 1 to 48")
    if not arm or any(ch.isspace() for ch in arm):
        raise SystemExit("arm must be one token")
    if not binary.is_file():
        raise SystemExit(f"missing binary {binary}")


@app.command
def plan(
    n: int,
    budget: int,
    seeds: int,
    arm: str,
    binary: Path,
    record: Path,
) -> None:
    """Print one Slurm job per seed. Submit nothing."""
    _check_common(seeds, arm, binary)
    for seed in range(seeds):
        sys.stdout.write(_sbatch(n, budget, arm, seed, binary, record))
        sys.stdout.write("\n")


@app.command
def submit(
    n: int,
    budget: int,
    seeds: int,
    arm: str,
    binary: Path,
    record: Path,
    *,
    yes: bool = False,
) -> None:
    """Queue one Slurm job per seed. Refuses without --yes."""
    if not yes:
        raise SystemExit("submit refuses without --yes; plan prints the jobs")
    _check_common(seeds, arm, binary)
    record.mkdir(parents=True, exist_ok=True)
    for seed in range(seeds):
        text = _sbatch(n, budget, arm, seed, binary, record)
        done = subprocess.run(
            ["sbatch"], input=text, text=True, capture_output=True, check=False
        )
        if done.returncode != 0:
            sys.stderr.write(done.stderr)
            raise SystemExit(done.returncode)
        print(done.stdout.strip())


def _enter_scratch() -> Path:
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("cell: SLURM_JOB_ID is required")
    done = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; elja_enter_scratch; printf %s "$ELJA_SCRATCH"',
            "elja",
            str(HELPER),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    path = Path(done.stdout.strip())
    if not str(path).startswith("/scratch/users/"):
        raise SystemExit(f"scratch is not the node disk: {path}")
    return path


def _leave_scratch(path: Path) -> None:
    subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; export ELJA_SCRATCH="$2"; elja_leave_scratch',
            "elja",
            str(HELPER),
            str(path),
        ],
        check=False,
    )


@app.command
def cell(
    n: int,
    budget: int,
    arm: str,
    seed: int,
    binary: Path,
    record: Path,
) -> None:
    """Run one seed inside this allocation, on the node scratch disk."""
    _check_common(1, arm, binary)
    scratch = _enter_scratch()
    name = f"lj{n}-{arm}-{seed}"
    body = (
        f"export SEED_OFFSET={seed}\n"
        f"exec {shlex.quote(str(binary))} {n} {budget} 1 {shlex.quote(arm)}\n"
    )
    manager = None
    try:
        from qcg.pilotjob.api.job import Jobs
        from qcg.pilotjob.api.manager import LocalManager

        manager = LocalManager(
            server_args=[
                "--wd",
                str(scratch),
                "--resources",
                "slurm",
                "--report-format",
                "json",
                "--report-file",
                "jobs.report",
                "--log",
                "warning",
            ]
        )
        manager.submit(
            Jobs().add(
                name=name,
                script=body,
                stdout="seed.log",
                stderr="seed.err",
                wd=str(scratch),
                numCores={"exact": 1},
                model="default",
            )
        )
        manager.wait4all()
        state = manager.status(name)
    finally:
        if manager is not None:
            manager.finish()
    log = scratch / "seed.log"
    if log.is_file():
        record.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "rsync",
                "-a",
                "--bwlimit=40000",
                str(log),
                str(record / f"n{n}-{arm}-seed{seed}.log"),
            ],
            check=True,
        )
    _leave_scratch(scratch)
    text = str(state)
    if "SUCCEED" not in text:
        print(text, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    app()

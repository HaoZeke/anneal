#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Append one machine-readable record for a reproducibility session.

The caller should run this process under the same ``taskset`` mask as the
campaign.  The resulting JSON Lines file is append-only: one line identifies
one invocation, the kernel boot, the effective CPU affinity, the exact
top-level reproduce command, the compilers and the Python that build and
drive it, the OpenMP runtime and its thread settings, and, with
``--anneal-repro``, the revision of that checkout.  A tool that is not
installed is recorded as null, not omitted.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path

COMPILERS = {
    "rustc": ["rustc", "-V"],
    "cargo": ["cargo", "-V"],
    "gfortran": ["gfortran", "--version"],
    "cc": ["cc", "--version"],
}
OPENMP_LIBRARIES = ("libgomp", "libomp", "libiomp5")


def read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def git_revision(root: Path, ref: str = "HEAD") -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", ref],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def first_line(command: list[str]) -> str | None:
    """The first output line of a version command, or None when the tool is absent or fails."""
    if shutil.which(command[0]) is None:
        return None
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    lines = (proc.stdout or proc.stderr).strip().splitlines()
    return lines[0].strip() if proc.returncode == 0 and lines else None


def compilers() -> dict[str, str | None]:
    return {name: first_line(command) for name, command in COMPILERS.items()}


def openmp_runtime() -> dict[str, object]:
    """The OpenMP libraries the dynamic loader can resolve, and the OMP_* settings of this session."""
    found: dict[str, str] = {}
    proc = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, check=False) \
        if shutil.which("ldconfig") else None
    for line in (proc.stdout.splitlines() if proc and proc.returncode == 0 else []):
        name, _, path = line.strip().partition(" => ")
        library = name.split(" ")[0]
        stem = library.split(".so")[0]
        if stem in OPENMP_LIBRARIES and stem not in found:
            found[stem] = str(Path(path).resolve()) if path else library
    settings = {key: value for key, value in sorted(os.environ.items()) if key.startswith(("OMP_", "GOMP_", "KMP_"))}
    return {
        "libraries": found,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "settings": settings,
    }


def repro_checkout(path: Path | None) -> dict[str, str | None] | None:
    if path is None:
        return None
    path = path.resolve()
    return {
        "path": str(path),
        "revision": git_revision(path),
        "tree": git_revision(path, "HEAD^{tree}"),
    }


def cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    except OSError:
        pass
    return platform.processor() or None


def record(campaign: str, command: list[str], root: Path, anneal_repro: Path | None = None) -> dict[str, object]:
    affinity = sorted(os.sched_getaffinity(0))
    return {
        "schema": "anneal-session-v2",
        "campaign": campaign,
        "session_utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "boot_id": read_text(Path("/proc/sys/kernel/random/boot_id")),
        "hostname": socket.gethostname(),
        "kernel": platform.release(),
        "cpu_model": cpu_model(),
        "cpu_affinity": affinity,
        "cpu_affinity_text": ",".join(str(cpu) for cpu in affinity),
        "command": command,
        "cwd": str(root),
        "code_revision": git_revision(root),
        "compilers": compilers(),
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation(),
                   "executable": sys.executable},
        "openmp": openmp_runtime(),
        "anneal_repro": repro_checkout(anneal_repro),
    }


def append(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    fd = os.open(path, flags, 0o644)
    try:
        os.write(fd, line.encode())
        os.fsync(fd)
    finally:
        os.close(fd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--anneal-repro", type=Path, help="record the revision and tree of this anneal_repro checkout")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("the exact reproduce command is required after --")
    append(args.output, record(args.campaign, command, args.root.resolve(), args.anneal_repro))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

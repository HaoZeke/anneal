#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Append one machine-readable record for a reproducibility session.

The caller should run this process under the same ``taskset`` mask as the
campaign.  The resulting JSON Lines file is append-only: one line identifies
one invocation, the kernel boot, the effective CPU affinity, the exact
top-level reproduce command, the compilers and the Python that build and
drive it, the OpenMP runtime and its thread settings, whether the checkout
had uncommitted changes, and, with ``--anneal-repro``, the revision of that
checkout.  A tool that is not installed is recorded as null, not omitted.
A runtime that cannot be identified is recorded as "unknown".
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from machine_provenance import host_label, machine_columns  # noqa: E402

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


def git_dirty(root: Path) -> bool | None:
    """True when ``git status --porcelain`` lists any change or untracked file, None when git cannot say."""
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return bool(proc.stdout.strip())


def first_line(command: list[str]) -> str | None:
    """The first output line of a version command, or None when the tool is absent or fails."""
    if shutil.which(command[0]) is None:
        return None
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    lines = (proc.stdout or proc.stderr).strip().splitlines()
    return lines[0].strip() if proc.returncode == 0 and lines else None


def compilers() -> dict[str, str | None]:
    return {name: first_line(command) for name, command in COMPILERS.items()}


def openmp_stem(name: str) -> str | None:
    """The OpenMP runtime a shared library file name belongs to, or None."""
    stem = Path(name).name.split(".so")[0].split(".dylib")[0]
    return stem if stem in OPENMP_LIBRARIES else None


def loaded_openmp(maps_text: str) -> dict[str, str]:
    """The OpenMP runtimes mapped into a process, from the text of /proc/PID/maps."""
    found: dict[str, str] = {}
    for line in maps_text.splitlines():
        parts = line.split(None, 5)
        if len(parts) == 6 and (stem := openmp_stem(parts[5])) and stem not in found:
            found[stem] = parts[5].strip()
    return found


def needed_libraries(executable: str) -> list[str] | None:
    """Shared libraries an executable asks the loader for, read without running it.

    readelf reads the dynamic section on Linux and otool the load commands on
    macOS. None means neither tool could read the file, as for a script.
    """
    path = shutil.which(executable)
    if path is None:
        return None
    for command, pattern in (
        (["readelf", "-d", path], r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]"),
        (["otool", "-L", path], r"^\s+(\S+) \(compatibility"),
    ):
        if shutil.which(command[0]) is None:
            continue
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            return re.findall(pattern, proc.stdout, re.MULTILINE)
    return None


def identify_openmp(command: list[str] | None, maps: Path = Path("/proc/self/maps")) -> dict[str, object]:
    """Name the OpenMP runtime from evidence, or record "unknown".

    The recorder's own address space is read first: a runtime mapped there is
    the one this process loaded.  Otherwise the dynamic section of the command's
    executable names the runtime it will load.  A script, an executable the
    tools cannot read, and a binary that links no OpenMP runtime give "unknown".
    Nothing is inferred from which libraries happen to be installed.
    """
    text = read_text(maps)
    candidates = loaded_openmp(text) if text else {}
    if candidates:
        source = "loaded in the recorder process"
    else:
        names = needed_libraries(command[0]) if command else None
        candidates = {stem: name for name in (names or []) if (stem := openmp_stem(name))}
        source = f"dynamic section of {command[0]}" if candidates else "no runtime found in the recorder or the command"
    if len(candidates) == 1:
        (library, path), = candidates.items()
        return {"library": library, "path": path if path.startswith("/") else None, "source": source}
    return {"library": "unknown", "path": None, "source": source, "candidates": sorted(candidates)}


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
        "dirty": git_dirty(path),
    }


def cpu_model() -> str | None:
    try:
        return machine_columns()["cpu_model"]
    except (OSError, RuntimeError):
        return platform.processor() or None


def cpu_affinity() -> list[int]:
    """The CPUs this process may run on; macOS has no affinity call, so every CPU counts."""
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


def record(campaign: str, command: list[str], root: Path, anneal_repro: Path | None = None) -> dict[str, object]:
    affinity = cpu_affinity()
    return {
        "schema": "anneal-session-v2",
        "campaign": campaign,
        "session_utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "boot_id": read_text(Path("/proc/sys/kernel/random/boot_id")),
        "hostname": host_label(socket.gethostname()),
        "kernel": platform.release(),
        "cpu_model": cpu_model(),
        "cpu_affinity": affinity,
        "cpu_affinity_text": ",".join(str(cpu) for cpu in affinity),
        "command": command,
        "cwd": str(root),
        "code_revision": git_revision(root),
        "dirty": git_dirty(root),
        "compilers": compilers(),
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation(),
                   "executable": sys.executable},
        "openmp": openmp_runtime(),
        "openmp_in_use": identify_openmp(command),
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

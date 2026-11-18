#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Append one machine-readable record for a reproducibility session.

The caller should run this process under the same ``taskset`` mask as the
campaign.  The resulting JSON Lines file is append-only: one line identifies
one invocation, the kernel boot, the effective CPU affinity, and the exact
top-level reproduce command.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import socket
import subprocess
from pathlib import Path


def read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def git_revision(root: Path) -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    except OSError:
        pass
    return platform.processor() or None


def record(campaign: str, command: list[str], root: Path) -> dict[str, object]:
    affinity = sorted(os.sched_getaffinity(0))
    return {
        "schema": "anneal-session-v1",
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
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("the exact reproduce command is required after --")
    append(args.output, record(args.campaign, command, args.root.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

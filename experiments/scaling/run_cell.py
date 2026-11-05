#!/usr/bin/env python3
"""Run one strong scaling cell and write a raw log.

The log records the commit, the tree, the command, and the exit code.
"""

from __future__ import annotations

import subprocess
import sys


def git_line(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def main() -> int:
    if len(sys.argv) != 5:
        print("usage: run_cell.py METHOD WORKERS REPEAT OUTPUT", file=sys.stderr)
        return 2
    method, workers, repeat, output = sys.argv[1:]
    command = (
        "experiments/scaling/bin/pop_strong_scale "
        f"--method {method} --workers {workers} --repeat {repeat}"
    )
    head = git_line(["rev-parse", "HEAD"])
    tree = git_line(["rev-parse", "HEAD^{tree}"])
    completed = subprocess.run(command.split(), capture_output=True, text=True)
    lines = [
        f"HEAD: {head}",
        f"tree: {tree}",
        f"command: {command}",
        f"exit: {completed.returncode}",
        completed.stdout.rstrip("\n"),
    ]
    if completed.stderr.strip():
        lines.append("stderr:")
        lines.append(completed.stderr.rstrip("\n"))
    text = "\n".join(lines) + "\n"
    with open(output, "w", encoding="utf-8") as handle:
        handle.write(text)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

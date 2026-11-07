#!/usr/bin/env python3
"""List commits from BASE to HEAD and the current ladder medians."""

import subprocess
import sys
from pathlib import Path


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "e22393bc03960fde656dd0e7bf10f583c5751464"
    report = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    print(f"BASE {base}")
    tip = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], text=True).strip()
    print(f"HEAD {tip}")
    print(f"TREE {tree}")
    log = subprocess.check_output(
        ["git", "log", "--format=%H %s", f"{base}..HEAD"], text=True
    )
    print(log, end="" if log.endswith("\n") else "\n")
    if report is not None and report.is_file():
        print("report")
        print(report.read_text(encoding="utf-8"), end="")
    print("exit_code=0")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""List commits from BASE to HEAD and the current ladder medians."""

import subprocess
import sys
from pathlib import Path


def _medians(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if "median_us=" not in line or " n=" not in line:
            continue
        parts = line.split()
        kind = parts[0]
        method = parts[1] if len(parts) > 1 else ""
        fields = {}
        for token in parts:
            if "=" in token:
                key, val = token.split("=", 1)
                fields[key] = val
        if "n" in fields and "threads" in fields and "median_us" in fields:
            rows.append((kind, method, fields["n"], fields["threads"], fields["median_us"]))
    return rows


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "e22393bc03960fde656dd0e7bf10f583c5751464"
    tip_log = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    base_log = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    print(f"BASE {base}")
    tip = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], text=True).strip()
    print(f"HEAD {tip}")
    print(f"TREE {tree}")
    log = subprocess.check_output(
        ["git", "log", "--format=%H %s", f"{base}..HEAD"], text=True
    )
    print(log, end="" if log.endswith("\n") else "\n")
    if tip_log is not None and base_log is not None and tip_log.is_file() and base_log.is_file():
        base_rows = {(a, b, c, d): e for a, b, c, d, e in _medians(base_log)}
        print("kind method n threads tip_us base_us")
        for kind, method, n, threads, tip_us in _medians(tip_log):
            key = (kind, method, n, threads)
            base_us = base_rows.get(key, "missing")
            print(f"{kind} {method} {n} {threads} {tip_us} {base_us}")
    print("exit_code=0")


if __name__ == "__main__":
    main()

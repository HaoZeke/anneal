#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Install the registered driver in one clean benchmark arm."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--arm", choices=("A", "A2", "B"), required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    source = Path(__file__).resolve().parent / "driver"
    target = repo / "bench" / "scan-safe6-vs-safe4" / "driver"
    if target.exists():
        raise SystemExit("refusing to replace an existing arm driver")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)

    symmetrise = repo / "src" / "symmetrise.rs"
    text = symmetrise.read_text(encoding="utf-8")
    native = "pub fn safe6_nearest_index_calls() -> u64"
    if args.arm == "B" and text.count(native) != 1:
        raise SystemExit("B does not expose exactly one safe6 path counter")
    if args.arm != "B" and native in text:
        raise SystemExit(f"{args.arm} unexpectedly exposes the safe6 path counter")

    print(f"commit={git(repo, 'rev-parse', 'HEAD')}")
    print(f"tree={git(repo, 'rev-parse', 'HEAD^{tree}')}")
    print(f"safe6_counter={int(args.arm == 'B')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

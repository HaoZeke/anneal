#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Hash each figure PDF and compare it with the same path on main.

The tip hash is the git blob of the file in figs/.
The main hash is that path on the given commit, or absent.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MAIN = "3675c9c375bbe560773cdf860bc672b8cbfe74d2"


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def blob(path: Path) -> str:
    proc = git("hash-object", str(path))
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip())
    return proc.stdout.strip()


def main_blob(name: str, commit: str) -> str | None:
    spec = f"{commit}:paper/jctc/figs/{name}"
    proc = git("rev-parse", "--verify", spec)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def records(figs: Path, commit: str) -> str:
    lines = []
    for path in sorted(figs.glob("*.pdf")):
        tip = blob(path)
        other = main_blob(path.name, commit)
        if other is None:
            lines.append(f"{path.name} tip {tip} main absent equal False")
        else:
            same = tip == other
            lines.append(
                f"{path.name} tip {tip} main {other} equal {same}"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    figs = Path(args[0]) if args else Path("paper/jctc/figs")
    commit = args[1] if len(args) > 1 else MAIN
    sys.stdout.write(records(figs, commit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

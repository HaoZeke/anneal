#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Compare the timed commit with HEAD.

The receipt tree must equal the timed commit's tree.
Every path is compared.
No path under scripts is excluded.

The only excluded paths are the paper files and the receipt
outputs the timed run itself writes. Each one is named here.

Paper text:
  NOTES.md
  REPRODUCE.md
  PROGRESS.md
  REPORT.md

Receipt outputs:
  logs/budget/2000/curve.txt
  logs/budget/2000/minima.txt
  logs/budget/2000/receipt.log
  logs/budget/2001/curve.txt
  logs/budget/2001/minima.txt
  logs/budget/2001/receipt.log
  logs/budget/2002/curve.txt
  logs/budget/2002/minima.txt
  logs/budget/2002/receipt.log
  logs/budget/2003/curve.txt
  logs/budget/2003/minima.txt
  logs/budget/2003/receipt.log
  logs/budget/2004/curve.txt
  logs/budget/2004/minima.txt
  logs/budget/2004/receipt.log
  logs/budget/2005/curve.txt
  logs/budget/2005/minima.txt
  logs/budget/2005/receipt.log
"""

import subprocess
import sys
from pathlib import Path

PAPER = (
    "NOTES.md",
    "REPRODUCE.md",
    "PROGRESS.md",
    "REPORT.md",
)

RECEIPTS = tuple(
    f"logs/budget/{seed}/{name}"
    for seed in range(2000, 2006)
    for name in ("curve.txt", "minima.txt", "receipt.log")
)

EXCLUDED = set(PAPER) | set(RECEIPTS)


def git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    cmd = ["git", "-c", f"safe.directory={repo.resolve()}", *args]
    return subprocess.run(
        cmd,
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def fail(what: str) -> None:
    print(what, file=sys.stderr)
    raise SystemExit(1)


def git_text(repo: Path, *args: str) -> str:
    result = git(repo, *args)
    if result.returncode != 0:
        sys.stderr.buffer.write(result.stderr)
        fail("git failed")
    return result.stdout.decode()


def covered(path: str) -> bool:
    if path.startswith("scripts/"):
        return True
    return path not in EXCLUDED


def ls_rows(repo: Path, rev: str) -> dict[str, str]:
    rows = {}
    for line in git_text(repo, "ls-tree", "-r", rev).splitlines():
        meta, path = line.split("\t", 1)
        rows[path] = meta
    return rows


def field(text: str, name: str) -> str:
    prefix = name + "="
    found = [
        line[len(prefix):]
        for line in text.splitlines()
        if line.startswith(prefix)
    ]
    if len(found) != 1 or not found[0]:
        fail(f"receipt field {name} is missing")
    return found[0]


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "usage: lj104_tree_check.py <receipt-root> <timed-commit>",
            file=sys.stderr,
        )
        return 2
    root_arg, timed = sys.argv[1], sys.argv[2]
    if root_arg.startswith("/"):
        fail("receipt root must be a repo-relative path")
    repo = Path(".")
    root = Path(root_arg)
    timed_tree = git_text(
        repo, "rev-parse", "--verify", f"{timed}^{{tree}}"
    ).strip()
    timed_full = git_text(
        repo, "rev-parse", "--verify", f"{timed}^{{commit}}"
    ).strip()
    print("producer=python3 -B scripts/lj104_tree_check.py")
    print(f"receipt_root={root_arg}")
    print(f"timed={timed_full}")
    print(f"timed_tree={timed_tree}")
    receipts = sorted(root.glob("**/receipt.log"))
    if len(receipts) != 6:
        fail("expected six receipts")
    problems = 0
    for path in receipts:
        text = path.read_text()
        receipt_head = field(text, "head")
        receipt_tree = field(text, "tree")
        print(f"receipt {path.as_posix()}")
        print(f"receipt_head={receipt_head}")
        print(f"receipt_tree={receipt_tree}")
        head_match = receipt_head == timed_full
        tree_match = receipt_tree == timed_tree
        print(f"head_match={'yes' if head_match else 'no'}")
        print(f"tree_match={'yes' if tree_match else 'no'}")
        if not head_match or not tree_match:
            problems += 1
    tip_rows = ls_rows(repo, "HEAD")
    timed_rows = ls_rows(repo, timed_full)
    changed = sorted(
        path
        for path in set(timed_rows) | set(tip_rows)
        if covered(path) and timed_rows.get(path) != tip_rows.get(path)
    )
    print(f"covered_diff_count={len(changed)}")
    for path in changed:
        print(f"diff {path}")
    if changed:
        problems += len(changed)
    if problems:
        print(f"tree check failed: {problems}")
        return 1
    print(f"tree check ok: {len(receipts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

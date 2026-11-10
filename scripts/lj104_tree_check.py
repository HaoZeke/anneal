#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Compare receipt head and tree values with git ls-tree.

Each budget receipt has a head, a tree, and a timed head.
The head's tree in git must equal the receipt tree.
Outside scripts, that head and the timed head must list the same blobs.
The tip may differ from that head only by the two copyright lines.
"""

import subprocess
import sys
from pathlib import Path

TIMED = "6f74d43958d6da93a1aba25f7b2e08ef0fcdf067"
COPYRIGHT = (
    "examples/h2h_gmin_proof.rs",
    "tests/stall_growth_hold.rs",
)
COPYRIGHT_LINE = "// Copyright (c) anneal developers"


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
        fail(f"git {' '.join(args)} failed")
    return result.stdout.decode()


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


def ls_rows(repo: Path, rev: str) -> dict[str, str]:
    rows = {}
    for line in git_text(repo, "ls-tree", "-r", rev).splitlines():
        meta, path = line.split("\t", 1)
        if path.startswith("scripts/"):
            continue
        rows[path] = meta
    return rows


def copyright_only(repo: Path, old: str, path: str) -> bool:
    diff = git_text(repo, "diff", "-U0", old, "HEAD", "--", path)
    added = []
    removed = []
    for line in diff.splitlines():
        if line.startswith(("+++", "---", "@@", "diff ", "index ")):
            continue
        if line.startswith("+"):
            added.append(line[1:])
        elif line.startswith("-"):
            removed.append(line[1:])
    return removed == [] and added == [COPYRIGHT_LINE, ""]


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: lj104_tree_check.py <kit> <repo>", file=sys.stderr)
        return 2
    kit_arg, repo_arg = sys.argv[1], sys.argv[2]
    if kit_arg.startswith("/") or repo_arg.startswith("/"):
        fail("arguments must be relative")
    kit = Path(kit_arg)
    repo = Path(repo_arg)
    head = git_text(repo, "rev-parse", "HEAD").strip()
    print(f"producer=python3 -B scripts/lj104_tree_check.py {kit_arg} {repo_arg}")
    print(f"head={head}")
    print(f"timed={TIMED}")
    receipts = sorted((kit / "logs" / "budget").glob("*/receipt.log"))
    if len(receipts) != 6:
        fail("expected six receipts")
    problems = 0
    engine_rows = None
    engine_head = None
    for path in receipts:
        text = path.read_text()
        receipt_head = field(text, "head")
        receipt_tree = field(text, "tree")
        timed_head = field(text, "timed_head")
        git_tree = git_text(repo, "rev-parse", "--verify", f"{receipt_head}^{{tree}}").strip()
        timed_type = git_text(repo, "cat-file", "-t", timed_head).strip()
        rel = str(path.relative_to(kit))
        print(f"receipt {rel}")
        print(f"receipt_head={receipt_head}")
        print(f"receipt_tree={receipt_tree}")
        print(f"git_tree={git_tree}")
        print(f"timed_head={timed_head}")
        tree_match = git_tree == receipt_tree
        timed_match = timed_head == TIMED and timed_type == "commit"
        print(f"tree_match={'yes' if tree_match else 'no'}")
        print(f"timed_match={'yes' if timed_match else 'no'}")
        if not tree_match or not timed_match:
            problems += 1
        rows = ls_rows(repo, receipt_head)
        timed_rows = ls_rows(repo, timed_head)
        same = rows == timed_rows
        print(f"ls_tree_outside_scripts={'match' if same else 'differ'}")
        if not same:
            problems += 1
        if engine_rows is None:
            engine_rows = rows
            engine_head = receipt_head
        elif rows != engine_rows or receipt_head != engine_head:
            print("receipts disagree")
            problems += 1
    tip_rows = ls_rows(repo, "HEAD")
    assert engine_rows is not None and engine_head is not None
    changed = sorted(
        path
        for path in set(engine_rows) | set(tip_rows)
        if engine_rows.get(path) != tip_rows.get(path)
    )
    print("tip_outside_scripts=" + ",".join(changed))
    if changed != sorted(COPYRIGHT):
        problems += 1
    for path in COPYRIGHT:
        only = copyright_only(repo, engine_head, path)
        print(f"copyright_only {path}={'yes' if only else 'no'}")
        if not only:
            problems += 1
    if problems:
        print(f"tree check failed: {problems}")
        return 1
    print("tree check ok: 6")
    return 0


if __name__ == "__main__":
    sys.exit(main())

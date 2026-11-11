#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Resolve the tip a generated note describes.

A trailing commit that only refreshes generated notes is not a new tip.
Re-running a writer on that commit reproduces the committed note.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

GENERATED = {
    "docs/prereg/lj104-sota/SCOPE.md",
    "docs/prereg/lj104-sota/bundle-verify.txt",
    "docs/prereg/lj104-sota/TREE_DIFF_STAT.txt",
}
STAT = "docs/prereg/lj104-sota/TREE_DIFF_STAT.txt"


def git(root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip() or "git failed")
    return proc.stdout.strip()


def names_of(root: Path, commit: str) -> set[str]:
    text = git(root, "diff-tree", "--no-commit-id", "--name-only", "-r", commit)
    return {line for line in text.splitlines() if line}


def content_tip(root: Path) -> str:
    """Latest commit that changes more than a generated note."""
    commit = git(root, "rev-parse", "HEAD")
    while True:
        files = names_of(root, commit)
        if not files or not files <= GENERATED:
            return commit
        parent = git(root, "rev-parse", f"{commit}^")
        if parent == commit:
            return commit
        commit = parent


def stat_tree(root: Path) -> str:
    """Commit whose numstat the diff note records."""
    commit = git(root, "rev-parse", "HEAD")
    if names_of(root, commit) == {STAT}:
        return git(root, "rev-parse", f"{commit}^")
    return commit

#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Write the scope of each commit from 91b15a02 to 5fb7368d.

Added and modified paths are read from git diff-tree. The twelve
subjects are the commit subjects in that range.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BASE = "91b15a024749320b9463e4abbdf10d37e933174b"
TIP = "5fb7368d5ea15cdf517c8bc48e2f56286ccb8b33"


def repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / ".git").exists():
            return candidate
    raise SystemExit("not inside a git work tree")


def git(root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout


def names(root: Path, commit: str, diff_filter: str) -> list[str]:
    text = git(
        root,
        "diff-tree",
        "--no-commit-id",
        f"--diff-filter={diff_filter}",
        "--name-only",
        "-r",
        commit,
    )
    return [line for line in text.splitlines() if line]


def sentences(text: str) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        trial = " ".join([*current, word])
        if current and len(trial) + 1 > 72:
            lines.append(" ".join(current) + ".")
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current) + ".")
    for line in lines:
        if len(line) > 72:
            raise SystemExit(f"scope line is {len(line)} columns: {line}")
    return lines


def main() -> None:
    root = repo_root()
    listed = git(root, "rev-list", "--reverse", f"{BASE}..{TIP}")
    commits = [line for line in listed.splitlines() if line]
    if len(commits) != 12:
        raise SystemExit(f"expected 12 commits, found {len(commits)}")
    out = [
        "Copyright (c) anneal developers",
        "",
        "Scope of the twelve commits after 91b15a02 through 5fb7368d.",
        "Subjects and paths are taken from git.",
        "An added path is a diff-tree addition.",
        "A modified path is a diff-tree modification.",
        "",
    ]
    for commit in commits:
        subject = git(root, "log", "-1", "--format=%s", commit).strip()
        added = names(root, commit, "A")
        modified = names(root, commit, "M")
        out.append(f"Commit {commit}.")
        out.append("Subject follows.")
        out.extend(sentences(subject))
        if added:
            out.append(f"Adds {len(added)} paths.")
            out.extend(f"Path {path}." for path in added)
        else:
            out.append("Adds no path.")
        if modified:
            word = "path" if len(modified) == 1 else "paths"
            out.append(f"Modifies {len(modified)} {word}.")
            out.extend(f"Path {path}." for path in modified)
        out.append("")
    text = "\n".join(out)
    if not text.endswith("\n"):
        text += "\n"
    dest = Path(__file__).resolve().parent / "SCOPE.md"
    dest.write_text(text, encoding="utf-8")
    sys.stdout.write(text)


if __name__ == "__main__":
    main()

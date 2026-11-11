#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Write the scope of every commit from 91b15a02 through HEAD.

Added and modified paths are read from git diff-tree. Subjects are
copied from git with no extra periods.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BASE = "91b15a024749320b9463e4abbdf10d37e933174b"
TIP = "HEAD"


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


def main() -> None:
    root = repo_root()
    from note_tip import content_tip

    tip = content_tip(root)
    listed = git(root, "rev-list", "--reverse", f"{BASE}..{tip}")
    commits = [line for line in listed.splitlines() if line]
    if not commits:
        raise SystemExit("the range after 91b15a02 is empty")
    intro = [
        "Copyright (c) anneal developers",
        "",
        "We read every commit after 91b15a02 through the tip.",
        "The tip identifier follows.",
        f"{tip}.",
        "Git supplies each subject and each path.",
        "diff-tree uses A for an addition.",
        "The same command uses M for a modification.",
        "The listing is one code block.",
        "Each subject is copied whole.",
        "",
        "```",
    ]
    block = []
    for commit in commits:
        subject = git(root, "log", "-1", "--format=%s", commit).strip()
        if ".." in subject:
            raise SystemExit(f"subject has a doubled period: {commit}")
        added = names(root, commit, "A")
        modified = names(root, commit, "M")
        block.append(f"Commit {commit}")
        block.append(f"Subject: {subject}")
        if added:
            block.append(f"Adds {len(added)} paths.")
            block.extend(added)
        else:
            block.append("Adds no path.")
        if modified:
            word = "path" if len(modified) == 1 else "paths"
            block.append(f"Modifies {len(modified)} {word}.")
            block.extend(modified)
        block.append("")
    out = intro + block + ["```", ""]
    text = "\n".join(out)
    if not text.endswith("\n"):
        text += "\n"
    dest = Path(__file__).resolve().parent / "SCOPE.md"
    dest.write_text(text, encoding="utf-8")
    sys.stdout.write(text)


if __name__ == "__main__":
    main()

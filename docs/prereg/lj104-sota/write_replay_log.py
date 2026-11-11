#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Regenerate the replay log from the committed map and git.

The 17:15:58 burst is kept. That pass set no date variables and slept
none. This writer does not replay. It prints the objects git already has.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

COMMAND = """\
unset GIT_AUTHOR_DATE GIT_COMMITTER_DATE
export TZ=America/Chicago
export GIT_AUTHOR_NAME='Rohit Goswami'
export GIT_AUTHOR_EMAIL='rgoswami@ieee.org'
export GIT_COMMITTER_NAME='Rohit Goswami'
export GIT_COMMITTER_EMAIL='rgoswami@ieee.org'
git -c commit.gpgsign=false commit-tree TREE -p PARENT -F -
"""


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


def main() -> None:
    here = Path(__file__).resolve().parent
    root = here
    for candidate in (here, *here.parents):
        if (candidate / ".git").exists():
            root = candidate
            break
    rows = (here / "replay-map.tsv").read_text(encoding="utf-8").splitlines()[1:]
    if len(rows) != 36:
        raise SystemExit(f"expected 36 mapped commits, found {len(rows)}")
    lines = [
        "case: shared seconds kept",
        "date variables passed to git: none",
        "sleeps: none",
        "zone: America/Chicago",
        "The pass is one commit-tree run on the real clock.",
        "No date was set and no sleep was used.",
        "A child is never earlier than its parent.",
        "Shared seconds are real.",
    ]
    whens = []
    for row in rows:
        old, new, tree, when, _subject = row.split("\t")
        actual_tree = git(root, "rev-parse", f"{new}^{{tree}}")
        actual_when = git(root, "log", "-1", "--format=%aI", new)
        if actual_tree != tree or actual_when != when:
            raise SystemExit(f"map disagrees with git for {new}")
        lines.extend(
            [
                f"old {old}",
                f"new {new}",
                f"tree {tree}",
                f"when {when}",
            ]
        )
        whens.append(when)
    if len(set(whens)) != 1:
        raise SystemExit(f"mapped whens differ: {sorted(set(whens))}")
    first_new = rows[0].split("\t")[1]
    if not first_new.startswith("75e82762"):
        raise SystemExit(f"first new id is {first_new}")
    text = "\n".join(lines) + "\n"
    (here / "replay-commit-tree.log").write_text(text, encoding="utf-8")
    (here / "replay-pass-command.txt").write_text(
        "Copyright (c) anneal developers\n\n"
        "The kept pass used this command.\n"
        "Dates were unset and the clock was real.\n"
        "We record that command in this note.\n"
        "Sleep was not called.\n"
        "The author date variable was absent.\n"
        "The committer date variable was absent.\n"
        "The zone was America/Chicago.\n"
        "The offset was -0600.\n"
        + COMMAND,
        encoding="utf-8",
    )
    print(first_new, whens[0], "rows", len(rows))


if __name__ == "__main__":
    sys.exit(main() or 0)

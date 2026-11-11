#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Replay the v4.13 history with git commit-tree.

No sleep is used. No author date or committer date is set.
The seconds are the ones git writes.
A child earlier than its parent is rejected.
Two messages lose one flagged word each.
Every tree is the archived tree.
"""

import os
import subprocess
import sys

BASE = "91b15a024749320b9463e4abbdf10d37e933174b"
OLD_TIP = "cebb875ddd00364924fca91a1c95c1bc4f0865d1"
NEW_REF = "refs/heads/fix/lj104-measure-v4.14"
ARCHIVE_V413 = "refs/archive/fix/lj104-measure-v4.13"
PROOF = "d9eec16a32dfe85e30c06837dc4e2a2eebc8fdd1"
HOLDS = "cebb875ddd00364924fca91a1c95c1bc4f0865d1"

PROOF_MESSAGE = """test(lj104): record the compiler and cover the example

The receipt prints rustc -V and cargo -V.
The example tests the published budgets.
It also tests a hit count and a type 7 sample.
The profile is the llvm-cov data for that example.
Line cover on the example is 98.39 percent.
rust-toolchain.toml is absent, and the pin was not added.
"""

HOLDS_MESSAGE = """docs(lj104): keep the tree check and the prose extract

The report records the tree check transcript.
The extract records the prose files and the commit messages.
"""


def git(*args: str, input_text: str | None = None) -> str:
    env = os.environ.copy()
    env.pop("GIT_AUTHOR_DATE", None)
    env.pop("GIT_COMMITTER_DATE", None)
    env["TZ"] = "America/Chicago"
    env["GIT_AUTHOR_NAME"] = "Rohit Goswami"
    env["GIT_AUTHOR_EMAIL"] = "rgoswami@ieee.org"
    env["GIT_COMMITTER_NAME"] = "Rohit Goswami"
    env["GIT_COMMITTER_EMAIL"] = "rgoswami@ieee.org"
    result = subprocess.run(
        ["git", "-c", "commit.gpgsign=false", *args],
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    return result.stdout


def main() -> int:
    olds = git("rev-list", "--reverse", f"{BASE}..{OLD_TIP}").split()
    parent = BASE
    print("The replay sets no dates.")
    print("The replay does not sleep.")
    print("The seconds are the ones git writes.")
    print("Archive refs are never pushed.")
    print("The push plan names only refs/heads/fix/lj104-measure-v4.14.")
    print(f"old_tip={OLD_TIP}")
    for old in olds:
        tree = git("rev-parse", f"{old}^{{tree}}").strip()
        if old == PROOF:
            message = PROOF_MESSAGE
            changed = "yes"
        elif old == HOLDS:
            message = HOLDS_MESSAGE
            changed = "yes"
        else:
            message = git("log", "-1", "--format=%B", old)
            changed = "no"
        new = git("commit-tree", tree, "-p", parent, input_text=message).strip()
        parent_ct = int(git("log", "-1", "--format=%ct", parent).strip())
        new_ct = int(git("log", "-1", "--format=%ct", new).strip())
        if new_ct < parent_ct:
            print(f"earlier than parent {old} {new}", file=sys.stderr)
            return 1
        same = git("rev-parse", f"{new}^{{tree}}").strip() == tree
        if not same:
            print(f"tree mismatch {old}", file=sys.stderr)
            return 1
        print(
            f"{old} {new} {tree} committer={new_ct} "
            f"parent_committer={parent_ct} message_changed={changed}"
        )
        parent = new
    git("update-ref", NEW_REF, parent)
    git("update-ref", ARCHIVE_V413, OLD_TIP)
    print(f"ref={NEW_REF}")
    print(f"tip={parent}")
    print(f"archive_v413={OLD_TIP}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

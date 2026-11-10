#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Replay the v4.11 history with git commit-tree.

No sleep is used. No author date or committer date is set.
The seconds are the ones git writes.
A child earlier than its parent is rejected.
One message is rewrapped so its lines stay within 72 columns.
Every tree is the archived tree.
"""

import os
import subprocess
import sys

BASE = "91b15a024749320b9463e4abbdf10d37e933174b"
OLD_TIP = "3f51ecf4e19796366ceac4005e17499e0ec2a16e"
NEW_REF = "refs/heads/fix/lj104-measure-v4.13"
REWRAP = "243afc94749722c9b6567a59884a4570cf5d6181"

MESSAGE = """bench(h2h): count both searches on one budget

The example runs Config::recommended for one Lennard-Jones seed.
Each quench returns energy and gradient as one ledger charge.
Each climb gradient is one charge.
A miss is stored as the full paper budget.
The shell runs that binary and the basin-hopping program from one start.
The order is anneal, then GMIN, one seed at a time, on one thread.
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
        ["git", *args],
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
    print("dates=unset")
    print("The replay does not sleep.")
    print("The seconds are the ones git writes.")
    print("Archive refs are never pushed.")
    print("Push plan: refs/heads/fix/lj104-measure-v4.13")
    print("archive=refs/archive/fix/lj104-measure-v4.9")
    print("archive=refs/archive/fix/lj104-measure-v4.10 87147482ce9c25b51d26b19e9d3cc2be1ad0c1d8")
    print("archive=refs/archive/fix/lj104-measure-v4.11 3f51ecf4e19796366ceac4005e17499e0ec2a16e")
    print("archive=refs/archive/fix/lj104-measure-v4.12 3f51ecf4e19796366ceac4005e17499e0ec2a16e")
    print(f"old_tip={OLD_TIP}")
    for old in olds:
        tree = git("rev-parse", f"{old}^{{tree}}").strip()
        if old == REWRAP:
            message = MESSAGE
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
    print(f"ref={NEW_REF}")
    print(f"tip={parent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

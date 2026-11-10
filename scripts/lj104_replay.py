#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Replay archive commits with git commit-tree.

No sleep is used. No author date or committer date is set.
The seconds in the new commits are the seconds git writes.
A child is rejected when its committer clock is earlier than its parent.
"""

import os
import subprocess
import sys

BASE = "91b15a024749320b9463e4abbdf10d37e933174b"
ARCHIVE = "refs/archive/fix/lj104-measure-v4.9"
NEW_REF = "refs/heads/fix/lj104-measure-v4.11"

SHA_MAP = """fix(lj104-kit): add the sha-map script

The script scripts/lj104_sha_map.py is new in this commit.
It rewrites the sha-map from the kit files.
It checks each row against the file, the sums line, and the tip blob.
A mismatch makes the script fail.
"""

WRAP_42 = """docs: close three prose findings in the measure notes

Three sentences in the measure notes are split.
The subject sits next to the verb.
Adjacent paragraphs share a landing.
"""

PLAIN_560 = """fix(lj104-kit): check the sha-map before any write

The check mode reads logs/sha-map.tsv and does not rewrite it.
It compares every row with the file, the sums line, and the tip blob.
A mismatch is named and the script exits non-zero.
The refresh mode is the only mode that rewrites the map.
A git error is fatal, and a missing tip path is not.
The script names the safe.directory setting for a store mount.
The script gains the copyright and licence header.
The notes name the corrected row and the check mode.
The reproduce file names them too.
"""

REDACT = """chore(lj104): remove the log redactor

scripts/redact.py is deleted.
The deletion is 71 lines.
The file rewrote path tokens and a commit token.
The measure no longer rewrites logs.
"""

REPLACEMENTS = {
    "d7597244d124ab7b7945cac2d755883d2a11109a": SHA_MAP,
    "42e60888c3f74b7e54b6cc1f32af020099e5e03c": WRAP_42,
    "560e1989f66629354ab68ce3a54f4d2b8aaf2d0a": PLAIN_560,
    "ac6bb80b42c46c7fba3889e444b4b5349bb76714": REDACT,
}


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
    olds = git(
        "rev-list", "--reverse", f"{BASE}..{ARCHIVE}"
    ).split()
    parent = BASE
    print("clock=git")
    print("dates=unset")
    print("sleeps=none")
    print("Archive refs are never pushed.")
    print("Push plan: refs/heads/fix/lj104-measure-v4.11")
    print(f"archive={ARCHIVE}")
    print(f"old_tip={git('rev-parse', ARCHIVE).strip()}")
    for old in olds:
        tree = git("rev-parse", f"{old}^{{tree}}").strip()
        if old in REPLACEMENTS:
            message = REPLACEMENTS[old]
            changed = "yes"
        else:
            message = git("log", "-1", "--format=%B", old)
            changed = "no"
        new = git(
            "commit-tree", tree, "-p", parent, input_text=message
        ).strip()
        parent_ct = int(git("log", "-1", "--format=%ct", parent).strip())
        new_ct = int(git("log", "-1", "--format=%ct", new).strip())
        if new_ct < parent_ct:
            print(f"earlier than parent {old} {new}", file=sys.stderr)
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

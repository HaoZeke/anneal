#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Replay the branch in one commit-tree pass.

The clock supplies every stamp. This script does not sleep and does
not set author or committer dates. Shared seconds stay as the clock
wrote them. Archive refs are never pushed.
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(ROOT, "..", "..", ".."))
BASE = "91b15a024749320b9463e4abbdf10d37e933174b"
BRANCH = "refs/heads/fix/lj104-lattice-v3"


def git(*args, env=None, input_text=None):
    cmd = ["git", "-c", "commit.gpgsign=false"]
    run = subprocess.run(
        cmd + list(args),
        cwd=REPO,
        env=env,
        input=input_text,
        text=True,
        capture_output=True,
        check=True,
    )
    return run.stdout


def reflow(message):
    message = message.replace(
        "A short state is skipped.",
        "The short state stays on the hold path.",
    )
    message = message.replace(
        "The propose\nmatch does not build a geometry.",
        "The propose\nmatch stays closed.",
    )
    message = message.replace(
        "match does not build a geometry.",
        "match stays closed.",
    )
    lines = message.splitlines()
    subject = lines[0].rstrip()
    body = lines[1:]
    if body and body[0].strip() == "":
        body = body[1:]
    blocks = []
    buf = []

    def pathish(line):
        text = line.strip()
        return (
            text.startswith("Files")
            or text.startswith("src/")
            or text.startswith("bench/")
            or text.startswith("examples/")
            or text.startswith("tests/")
        )

    def flush():
        nonlocal buf
        if not buf:
            return
        if all(pathish(line) for line in buf):
            blocks.append("\n".join(line.rstrip() for line in buf))
        else:
            words = " ".join(line.strip() for line in buf)
            blocks.append(wrap(words))
        buf = []

    for line in body:
        if line.strip() == "":
            flush()
        else:
            buf.append(line)
    flush()
    text = subject + "\n\n" + "\n\n".join(blocks)
    if not text.endswith("\n"):
        text += "\n"
    return text


def wrap(words):
    out = []
    line = ""
    for word in words.split():
        trial = word if not line else line + " " + word
        if len(trial) <= 72:
            line = trial
        else:
            if line:
                out.append(line)
            line = word
    if line:
        out.append(line)
    return "\n".join(out)


def main():
    env = os.environ.copy()
    for key in ("GIT_AUTHOR_DATE", "GIT_COMMITTER_DATE"):
        env.pop(key, None)
    env["TZ"] = "America/Chicago"
    env["GIT_AUTHOR_NAME"] = "Rohit Goswami"
    env["GIT_AUTHOR_EMAIL"] = "rgoswami@ieee.org"
    env["GIT_COMMITTER_NAME"] = "Rohit Goswami"
    env["GIT_COMMITTER_EMAIL"] = "rgoswami@ieee.org"
    olds = git(
        "rev-list", "--reverse", f"{BASE}..{BRANCH}"
    ).split()
    parent = BASE
    rows = []
    for old in olds:
        message = git("log", "-1", "--format=%B", old)
        tree = git("rev-parse", f"{old}^{{tree}}").strip()
        new = git(
            "commit-tree",
            tree,
            "-p",
            parent,
            env=env,
            input_text=reflow(message),
        ).strip()
        stamp = git(
            "log",
            "-1",
            "--format=%ad %cd",
            "--date=format:%Y-%m-%dT%H:%M:%S%z",
            new,
        ).strip()
        parent_stamp = git(
            "log",
            "-1",
            "--format=%ad",
            "--date=format:%Y-%m-%dT%H:%M:%S%z",
            parent,
        ).strip()
        if stamp.split()[0] < parent_stamp:
            sys.exit(
                "child earlier than parent: "
                + new
                + " "
                + stamp
                + " parent "
                + parent_stamp
            )
        rows.append((old, new, tree, stamp))
        parent = new
    git("update-ref", BRANCH, parent)
    log_path = os.path.join(ROOT, "replay.log")
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write("replay commit-tree pass\n")
        handle.write("dates come from the clock\n")
        handle.write("shared seconds are real\n")
        handle.write("archive refs are never pushed\n")
        for old, new, tree, stamp in rows:
            handle.write(f"old {old}\n")
            handle.write(f"new {new}\n")
            handle.write(f"tree {tree}\n")
            handle.write(f"stamp {stamp}\n")
    map_path = os.path.join(ROOT, "REPLAY.txt")
    with open(map_path, "w", encoding="utf-8") as handle:
        handle.write("Old chain to new chain.\n\n")
        handle.write("Shared seconds are real.\n")
        handle.write("Archive refs are never pushed.\n\n")
        for old, new, tree, stamp in rows:
            handle.write(f"{old}\n{new}\n{tree}\n{stamp}\n\n")
    print(f"tip {parent}")
    print(f"commits {len(rows)}")


if __name__ == "__main__":
    main()

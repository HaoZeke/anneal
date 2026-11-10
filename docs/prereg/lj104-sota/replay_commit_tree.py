#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Replay the pre-registration commits with git commit-tree.

One pass. No sleep. GIT_AUTHOR_DATE and GIT_COMMITTER_DATE are never
set. The seconds are the ones git writes. A child may share its
parent's second and must not be earlier.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

BASE = "91b15a024749320b9463e4abbdf10d37e933174b"
ARCHIVE_V13 = "835b90741211f2fd00b5b9ea68cde4cc877b31af"
ARCHIVE_V131 = "6cde8551b2791cc33b1060311dd334a04d3d1042"
BRANCH = "refs/heads/prereg/lj104-sota-v1"

STEPS: list[tuple[str, str, int]] = [
    (
        "lj104-prereg: record predictions and McNemar table",
        "The registration records the held-out predictions, the\n"
        "one-sided exact McNemar rule, and the table printed by the\n"
        "threshold script. The added files are\n"
        "docs/prereg/lj104-sota/PREDICTIONS.md,\n"
        "docs/prereg/lj104-sota/mcnemar_threshold.py, and\n"
        "docs/prereg/lj104-sota/mcnemar_threshold.tsv.",
        72,
    ),
    (
        "lj104-prereg: add the serial runner and its tests",
        "The serial runner refuses a second scored start, and each\n"
        "scored row records the runner blob, HEAD, and TREE. The added\n"
        "files are docs/prereg/lj104-sota/lj104_sota_runner.py and\n"
        "docs/prereg/lj104-sota/test_lj104_sota_runner.py.",
        72,
    ),
    (
        "lj104-prereg: name panel seeds 6000-6005",
        "Panel seeds 6000 through 6005 are named before any panel run,\n"
        "and the note states that those seeds were never used for\n"
        "tuning. The edit is docs/prereg/lj104-sota/PREDICTIONS.md.\n"
        "This commit adds no path.",
        72,
    ),
    (
        "lj104-prereg: bind lj_requench for the LJ104 minimum",
        "lj_requench is bound to WarmLbfgs minimize, and the LJ104\n"
        "minimum test requires energy at or below -582.086641. The\n"
        "function and the test are in src/methods/lattice_search.rs.\n"
        "The citation is in docs/prereg/lj104-sota/PREDICTIONS.md.",
        72,
    ),
    (
        "lj104-prereg: set the plain DLS prediction to 20/30",
        "Plain DLS is registered at 20 of 30 from the lattice v2\n"
        "recompute on seeds 3000 through 3029. The sentence is in\n"
        "docs/prereg/lj104-sota/PREDICTIONS.md.",
        72,
    ),
    (
        "lj104-prereg: run the default panel from the runner",
        "The runner gains the 24-job default panel, still inside the\n"
        "serial lock, and the tests cover that command. The floor\n"
        "paragraph in docs/prereg/lj104-sota/PREDICTIONS.md names the\n"
        "budgets and the 0.001 window. The code edits are\n"
        "docs/prereg/lj104-sota/lj104_sota_runner.py and\n"
        "docs/prereg/lj104-sota/test_lj104_sota_runner.py. The raw\n"
        "deposit is a later commit.",
        72,
    ),
    (
        "lj104-prereg: restore the LJ104 occupation test",
        "The test attribute is restored on the LJ104 occupation test\n"
        "in src/methods/lattice_search.rs. This commit adds no path.",
        72,
    ),
    (
        "lj104-prereg: commit raw panel traces and receipts",
        "The raw default-panel deposit is stored. Each of the\n"
        "twenty-four jobs contributes a span, a stderr note, a trace,\n"
        "and coordinates. The deposit also holds the scored rows, the\n"
        "session note, and the start and end receipts.",
        72,
    ),
    (
        "lj104-prereg: add the untouched panel stdout logs",
        "The untouched panel stdout logs are stored beside the traces.\n"
        "They are the twenty-four run logs, and nothing in them is\n"
        "edited.",
        72,
    ),
    (
        "lj104-prereg: write floors from the raw panel logs",
        "Floors are derived from the raw panel logs. The writer is\n"
        "docs/prereg/lj104-sota/write_floors.py and the window-count\n"
        "table is docs/prereg/lj104-sota/floors.md. The registration in\n"
        "docs/prereg/lj104-sota/PREDICTIONS.md cites those counts.",
        72,
    ),
    (
        "lj104-prereg: score hits by re-quench at 1e-6",
        "A hit is a re-quenched energy at or below the known minimum\n"
        "plus 1e-6. The known energies are\n"
        "docs/prereg/lj104-sota/e_known.tsv. The driver is\n"
        "docs/prereg/lj104-sota/requench_panel.py. The example that\n"
        "prints energy and calls is examples/lj_requench_xyz.rs. The\n"
        "function and the LJ104 test live in\n"
        "src/methods/lattice_search.rs. The rule is stated in\n"
        "docs/prereg/lj104-sota/PREDICTIONS.md.",
        72,
    ),
    (
        "lj104-prereg: record re-quenched energies and calls",
        "The re-quenched energies and the separate call counts are\n"
        "recorded. The table is docs/prereg/lj104-sota/requench.tsv and\n"
        "the readable note is docs/prereg/lj104-sota/requench.md. The\n"
        "driver docs/prereg/lj104-sota/requench_panel.py writes both,\n"
        "and docs/prereg/lj104-sota/PREDICTIONS.md records the floors.",
        72,
    ),
    (
        "lj104-prereg: list commit subjects and added paths",
        "The scope note lists each subject and each added path for the\n"
        "twelve commits from 91b15a02 through 5fb7368d. The note is\n"
        "docs/prereg/lj104-sota/SCOPE.md and the generator is\n"
        "docs/prereg/lj104-sota/scope_paths.py.",
        72,
    ),
    (
        "lj104-prereg: pin the GMIN digest and charge line",
        "The GMIN digest and the charge citation at lattice_search.rs\n"
        "line 1388 are recorded in docs/prereg/lj104-sota/PREDICTIONS.md.\n"
        "The shared keyword block is\n"
        "docs/prereg/lj104-sota/gmin_best.keywords. The binary search\n"
        "list in docs/prereg/lj104-sota/requench_panel.py no longer\n"
        "names a relative target fallback.",
        72,
    ),
    (
        "lj104-prereg: record replay notes and map",
        "The replay map, the tree-equality note, the diff stat, the\n"
        "replay note, and four checker notes are stored. The map is\n"
        "docs/prereg/lj104-sota/replay-map.tsv. The equality note is\n"
        "docs/prereg/lj104-sota/TREE_EQUAL.txt. The stat is\n"
        "docs/prereg/lj104-sota/TREE_DIFF_STAT.txt. The replay note is\n"
        "docs/prereg/lj104-sota/REPLAY.txt. The four notes say the prose\n"
        "commands had not been run. They are\n"
        "docs/prereg/lj104-sota/prose/proseguard-rohit-paper.txt,\n"
        "docs/prereg/lj104-sota/prose/proseguard-prose-human.txt,\n"
        "docs/prereg/lj104-sota/prose/snapper.txt, and\n"
        "docs/prereg/lj104-sota/prose/snapper-fmt.txt.",
        52,
    ),
]


def git_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("GIT_AUTHOR_DATE", None)
    env.pop("GIT_COMMITTER_DATE", None)
    env["TZ"] = "America/Chicago"
    env["GIT_AUTHOR_NAME"] = "Rohit Goswami"
    env["GIT_AUTHOR_EMAIL"] = "rgoswami@ieee.org"
    env["GIT_COMMITTER_NAME"] = "Rohit Goswami"
    env["GIT_COMMITTER_EMAIL"] = "rgoswami@ieee.org"
    if "GIT_AUTHOR_DATE" in env or "GIT_COMMITTER_DATE" in env:
        raise SystemExit("date variables are set")
    return env


def git(root: Path, env: dict[str, str], *args: str, input_text: str | None = None) -> str:
    proc = subprocess.run(
        ["git", "-C", str(root), "-c", "commit.gpgsign=false", *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        input=input_text,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip() or proc.stdout.strip())
    return proc.stdout


def added_paths(root: Path, env: dict[str, str], commit: str) -> list[str]:
    text = git(root, env, "diff-tree", "--diff-filter=A", "--name-only", "-r", commit)
    lines = [line for line in text.splitlines() if line]
    if lines and lines[0] == commit:
        lines = lines[1:]
    return lines


def build_message(subject: str, prose: str, cap: int, paths: list[str]) -> str:
    if len(subject) > cap:
        raise SystemExit(f"subject is {len(subject)} columns, cap {cap}: {subject}")
    if not subject.startswith("lj104-prereg: "):
        raise SystemExit(f"subject is not scoped: {subject}")
    body = prose.rstrip() + "\n"
    missing = [path for path in paths if path not in body]
    if missing:
        body += "The diff adds these paths.\n" + "\n".join(missing) + "\n"
    for path in paths:
        if path not in body:
            raise SystemExit(f"message omits {path}")
    text = subject + "\n\n" + body
    if not text.endswith("\n"):
        text += "\n"
    for index, line in enumerate(text.splitlines(), start=1):
        if len(line) > 72:
            raise SystemExit(f"message line {index} is {len(line)} columns: {line}")
    return text


def main() -> None:
    root = Path(git(Path("."), git_env(), "rev-parse", "--show-toplevel").strip())
    env = git_env()
    tip = git(root, env, "rev-parse", "HEAD").strip()
    if tip != ARCHIVE_V131:
        raise SystemExit(f"expected tip {ARCHIVE_V131}, found {tip}")
    listed = git(root, env, "rev-list", "--reverse", f"{BASE}..{tip}")
    olds = [line for line in listed.splitlines() if line]
    if len(olds) != len(STEPS):
        raise SystemExit(f"expected {len(STEPS)} commits, found {len(olds)}")
    print("date variables passed to git: none")
    print("zone: America/Chicago")
    git(root, env, "update-ref", "refs/archive/prereg/lj104-sota-v1.3", ARCHIVE_V13)
    git(root, env, "update-ref", "refs/archive/prereg/lj104-sota-v1.3.1", ARCHIVE_V131)
    print(f"archive v1.3 {ARCHIVE_V13}")
    print(f"archive v1.3.1 {ARCHIVE_V131}")
    parent = BASE
    previous = int(git(root, env, "log", "-1", "--format=%at", BASE).strip())
    for old, (subject, prose, cap) in zip(olds, STEPS):
        tree = git(root, env, "rev-parse", f"{old}^{{tree}}").strip()
        paths = added_paths(root, env, old)
        message = build_message(subject, prose, cap, paths)
        new = git(
            root,
            env,
            "commit-tree",
            tree,
            "-p",
            parent,
            "-F",
            "-",
            input_text=message,
        ).strip()
        got_tree = git(root, env, "rev-parse", f"{new}^{{tree}}").strip()
        if got_tree != tree:
            raise SystemExit(f"tree mismatch for {old}")
        when = git(root, env, "log", "-1", "--format=%at %ct %aI %cI", new).strip()
        author_at, committer_at, author_iso, committer_iso = when.split()
        if author_at != committer_at or author_iso != committer_iso:
            raise SystemExit(f"author and committer clocks differ for {new}")
        if int(author_at) < previous:
            raise SystemExit(f"{new} is earlier than its parent")
        sig = git(root, env, "cat-file", "commit", new)
        if "gpgsig" in sig:
            raise SystemExit(f"{new} is signed")
        print(f"old {old}")
        print(f"new {new}")
        print(f"tree {tree}")
        print(f"when {author_iso}")
        print(f"paths {len(paths)}")
        print(message, end="" if message.endswith("\n") else "\n")
        print("---")
        parent = new
        previous = int(author_at)
    git(root, env, "update-ref", BRANCH, parent)
    git(root, env, "reset", "--hard", parent)
    print(f"branch {BRANCH}")
    print(f"tip {parent}")


if __name__ == "__main__":
    main()

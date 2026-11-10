#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Count Cargo.lock entries two ways.

Usage: lockfile_delta.py OLD_REV NEW_REV
The package-block count must match the version lines in the unified diff.
"""

from __future__ import annotations

import subprocess
import sys
from collections import defaultdict


def blocks(text: str):
    current: list[str] = []
    groups: list[list[str]] = []
    for line in text.splitlines():
        if line.startswith("[[package]]"):
            if current:
                groups.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        groups.append(current)
    pairs = []
    by_name = defaultdict(set)
    for group in groups:
        name = version = None
        for line in group:
            if line.startswith("name = "):
                name = line.split("=", 1)[1].strip().strip('"')
            elif line.startswith("version = "):
                version = line.split("=", 1)[1].strip().strip('"')
        if name and version:
            pairs.append((name, version))
            by_name[name].add(version)
    return pairs, by_name


def show(rev: str) -> str:
    return subprocess.check_output(["git", "show", f"{rev}:Cargo.lock"], text=True)


def minor(version: str) -> tuple[str, str]:
    base = version.split("+", 1)[0]
    parts = base.split(".")
    return (parts[0], parts[1] if len(parts) > 1 else "0")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: lockfile_delta.py OLD_REV NEW_REV")
    old_rev, new_rev = sys.argv[1], sys.argv[2]
    old_pairs, _old_names = blocks(show(old_rev))
    new_pairs, new_names = blocks(show(new_rev))
    removed = sorted(set(old_pairs) - set(new_pairs))
    added = sorted(set(new_pairs) - set(old_pairs))
    diff = subprocess.check_output(
        ["git", "diff", "-U0", old_rev, new_rev, "--", "Cargo.lock"], text=True
    )
    missing = [item for item in removed if f'-version = "{item[1]}"' not in diff]
    extra = [item for item in added if f'+version = "{item[1]}"' not in diff]
    gone = []
    partial = []
    moved = []
    for name, version in removed:
        stay = sorted(new_names.get(name, ()))
        news = sorted(ver for crate, ver in added if crate == name)
        if not stay:
            gone.append((name, version))
        elif len(news) == 1 and minor(version) != minor(news[0]):
            moved.append((name, version, news[0]))
        else:
            partial.append((name, version, stay))
    print(f"removed_entries {len(removed)}")
    print(f"added_entries {len(added)}")
    print(f"packages_leave {len(gone)}")
    for name, version in gone:
        print(f"leave {name} {version}")
    print(f"version_drops {len(partial)}")
    for name, version, stay in partial:
        print(f"drop {name} {version} stays {' '.join(stay)}")
    print(f"minor_boundary {len(moved)}")
    for name, old, new in moved:
        print(f"minor {name} {old} {new}")
    if missing or extra:
        print("DIFF_MISMATCH")
        raise SystemExit(1)
    print("diff_check ok")


if __name__ == "__main__":
    main()

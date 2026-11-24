#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Complete a staged directory of Python scripts with their local imports.

The enter_*.sh wrappers copy a campaign script into a scratch directory and
run it there, away from the checkout. A script that imports a sibling helper
by name then needs that helper beside it. Listing the helpers by hand in each
wrapper goes stale whenever a script gains an import, so the wrapper copies
its scripts and then calls this tool, which reads the imports of every
``*.py`` already in the directory and copies the missing local modules until
none are left.

A module is local when ``<name>.py`` exists in one of the ``--from``
directories. Everything else (the standard library, installed packages) is
left alone.

    stage_python.py DEST [--from DIR]...
"""

from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path


def imported_modules(path: Path) -> set[str]:
    """Top-level module names imported anywhere in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def find_local(name: str, search: list[Path]) -> Path | None:
    for directory in search:
        candidate = directory / f"{name}.py"
        if candidate.is_file():
            return candidate
    return None


def missing_local_imports(dest: Path, search: list[Path]) -> dict[str, Path]:
    """Local modules some script in ``dest`` imports that ``dest`` lacks."""
    missing: dict[str, Path] = {}
    for script in sorted(dest.glob("*.py")):
        for name in sorted(imported_modules(script)):
            if (dest / f"{name}.py").exists():
                continue
            found = find_local(name, search)
            if found is not None:
                missing[name] = found
    return missing


def complete(dest: Path, search: list[Path]) -> list[Path]:
    """Copy missing local imports into ``dest`` until a fixed point."""
    copied: list[Path] = []
    while True:
        missing = missing_local_imports(dest, search)
        if not missing:
            return copied
        for name, source in sorted(missing.items()):
            target = dest / f"{name}.py"
            shutil.copy2(source, target)
            copied.append(target)


def main(argv: list[str]) -> int:
    args = list(argv)
    search: list[Path] = []
    positional: list[str] = []
    while args:
        arg = args.pop(0)
        if arg == "--from":
            if not args:
                print("stage_python.py: --from needs a directory", file=sys.stderr)
                return 2
            search.append(Path(args.pop(0)))
        else:
            positional.append(arg)
    if len(positional) != 1:
        print("usage: stage_python.py DEST [--from DIR]...", file=sys.stderr)
        return 2
    dest = Path(positional[0])
    if not dest.is_dir():
        print(f"stage_python.py: {dest} is not a directory", file=sys.stderr)
        return 2
    for target in complete(dest, search):
        print(target.name)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

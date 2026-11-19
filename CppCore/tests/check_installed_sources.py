#!/usr/bin/env python3
"""List implementation sources, or fail if a staged install contains one.

install_subdir compares exclude_files to literal paths relative to the
installed directory. A glob is not one of those paths, so it drops
nothing. --list prints each implementation file as that literal path.

The install check runs meson install --destdir and prints every staged
file. It fails when a staged file is an implementation source. .cc is
the suffix the header tree was leaking. The other suffixes are the ones
the old globs were meant to drop. Installed libraries stay allowed.

Usage:
  check_installed_sources.py --list DIR
  check_installed_sources.py BUILDDIR MESON
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Source files under CppCore/rgpot. .o, .a and .so are named here so a
# build product left in the source tree is not installed as a header.
IMPL_SUFFIXES = {
    ".cc",
    ".cpp",
    ".cxx",
    ".c",
    ".f90",
    ".F",
    ".py",
    ".o",
    ".a",
    ".so",
}

# The prefix is supposed to contain libraries. Only sources fail the
# staged tree.
STAGED_SOURCE_SUFFIXES = IMPL_SUFFIXES - {".o", ".a", ".so"}


def list_impl(root: Path) -> int:
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 1
    paths: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if Path(name).suffix in IMPL_SUFFIXES:
                rel = Path(dirpath, name).relative_to(root)
                paths.append(rel.as_posix())
    for path in sorted(paths):
        print(path)
    return 0


def check_install(build: Path, meson: list[str]) -> int:
    with tempfile.TemporaryDirectory(prefix="rgpot-destdir-") as dest:
        try:
            proc = subprocess.run(
                [
                    *meson,
                    "install",
                    "-C",
                    str(build),
                    "--destdir",
                    dest,
                    "--no-rebuild",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except OSError as exc:
            print(f"could not run meson install: {exc}", file=sys.stderr)
            return 1
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout)
            print("meson install --destdir failed", file=sys.stderr)
            return 1
        root = Path(dest)
        files = sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        )
        for path in files:
            print(path)
        bad = [
            path
            for path in files
            if Path(path).suffix in STAGED_SOURCE_SUFFIXES
        ]
        if bad:
            print("installed implementation sources:", file=sys.stderr)
            for path in bad:
                print(f"  {path}", file=sys.stderr)
            return 1
    print(f"installed {len(files)} files and no implementation sources")
    return 0


def _meson_argv(argv: list[str]) -> list[str]:
    if len(argv) == 1 and Path(argv[0]).name.lower().startswith("python"):
        return [argv[0], "-m", "mesonbuild.mesonmain"]
    return argv


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--list":
        return list_impl(Path(sys.argv[2]))
    if len(sys.argv) >= 3 and sys.argv[1] != "--list":
        return check_install(Path(sys.argv[1]), _meson_argv(sys.argv[2:]))
    print(
        "usage: check_installed_sources.py --list DIR\n"
        "       check_installed_sources.py BUILDDIR MESON [MESON_ARGS...]",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())

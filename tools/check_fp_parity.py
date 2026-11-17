#!/usr/bin/env python3
"""Compare the floating-point flags meson and CMake compile each source with.

  check_fp_parity.py MESON_COMPILE_COMMANDS CMAKE_COMPILE_COMMANDS

Both arguments are the compile_commands.json a configured build writes.
For every source in this repository that both build systems compile, the
contraction (-ffp-contract=, /fp:) and fast-math (-ffast-math,
-fno-fast-math) flags must agree: a source compiled several times must
have the same set of settings in each build. Tests compare results bit
for bit, and an FMA-baseline target such as arm64 contracts a*b+c unless
told not to, so a source that only one build compiles unfused passes on
x86-64 and fails there.
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def setting(argv: list[str]) -> tuple[tuple[str, str], ...]:
    """The last flag of each family, as (family, flag) pairs."""
    last: dict[str, str] = {}
    for arg in argv:
        if arg.startswith("-ffp-contract="):
            last["contract"] = arg
        elif arg in ("-ffast-math", "-fno-fast-math"):
            last["fast-math"] = arg
        elif arg.startswith("/fp:"):
            last["msvc-fp"] = arg
    return tuple(sorted(last.items()))


def settings(path: Path) -> dict[str, set[tuple[tuple[str, str], ...]]]:
    """Each repository source in `path` and the settings it is compiled with."""
    found: dict[str, set[tuple[tuple[str, str], ...]]] = {}
    for entry in json.loads(path.read_text(encoding="utf-8")):
        argv = entry.get("arguments") or shlex.split(entry["command"])
        source = (Path(entry["directory"]) / entry["file"]).resolve()
        try:
            rel = source.relative_to(ROOT).as_posix()
        except ValueError:
            continue
        found.setdefault(rel, set()).add(setting(argv))
    return found


def show(values: set[tuple[tuple[str, str], ...]]) -> str:
    return " | ".join(sorted(" ".join(flag for _, flag in v) or "(none)" for v in values))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("meson", type=Path)
    parser.add_argument("cmake", type=Path)
    args = parser.parse_args()
    meson = settings(args.meson)
    cmake = settings(args.cmake)
    shared = sorted(meson.keys() & cmake.keys())
    differ = [src for src in shared if meson[src] != cmake[src]]
    for src in differ:
        print(f"{src}: meson {show(meson[src])}; cmake {show(cmake[src])}", file=sys.stderr)
    if differ:
        print(f"{len(differ)} of {len(shared)} shared sources differ in their "
              "floating-point flags", file=sys.stderr)
        return 1
    print(f"{len(shared)} shared sources have the same floating-point flags")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""A parent project that vendors rgpot must not install rgpot's engines.

gpridge and similar parents pull rgpot in with subproject() and build only
the targets they link. The engine drivers and plugins under engines/ are
not among them, so if rgpot still marks them for install, meson install in
the parent stops at the first one it cannot find. This configures a parent
with rgpot as its subproject, builds nothing, and requires that the install
plan holds no engines/ output and that meson install --dry-run succeeds.
Usage: test_optional_engine_install.py <root> <meson>
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PARENT = """project('rgpot_parent', 'c', 'cpp')
subproject('rgpot', default_options: ['with_tests=false'])
"""


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _meson_argv(argv: list[str]) -> list[str]:
    if len(argv) == 1 and Path(argv[0]).name.lower().startswith("python"):
        return [argv[0], "-m", "mesonbuild.mesonmain"]
    return argv


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    meson = _meson_argv(sys.argv[2:])
    with tempfile.TemporaryDirectory(prefix="rgpot-parent-") as work:
        parent = Path(work)
        (parent / "subprojects").mkdir()
        (parent / "subprojects" / "rgpot").symlink_to(root, target_is_directory=True)
        (parent / "meson.build").write_text(PARENT)
        build = parent / "build"
        setup = run([*meson, "setup", str(build)], parent)
        if setup.returncode != 0:
            print(setup.stdout, file=sys.stderr)
            if "Could not detect Ninja" in setup.stdout:
                print(
                    "optional_engine_install requires ninja on PATH or in NINJA",
                    file=sys.stderr,
                )
            elif "Unknown compiler(s)" in setup.stdout:
                print(
                    "optional_engine_install requires C and C++ compilers on PATH or in CC/CXX",
                    file=sys.stderr,
                )
            else:
                print("parent project did not configure", file=sys.stderr)
            return 1
        listed = run([*meson, "introspect", str(build), "--installed"], parent)
        if listed.returncode != 0:
            print(listed.stdout, file=sys.stderr)
            return 1
        engines = build / "subprojects" / "rgpot" / "engines"
        planned = sorted(
            Path(src).relative_to(build).as_posix()
            for src in json.loads(listed.stdout)
            if Path(src).is_relative_to(engines)
        )
        install = run(
            [
                *meson,
                "install",
                "-C",
                str(build),
                "--no-rebuild",
                "--dry-run",
                "--destdir",
                str(parent / "dest"),
            ],
            parent,
        )
    failed = False
    if planned:
        print("a subproject build plans to install rgpot engines:", file=sys.stderr)
        for path in planned:
            print(f"  {path}", file=sys.stderr)
        failed = True
    if install.returncode != 0:
        print(install.stdout, file=sys.stderr)
        print("meson install --dry-run failed in the parent project", file=sys.stderr)
        failed = True
    if failed:
        return 1
    print("subproject build installs no rgpot engines")
    return 0


if __name__ == "__main__":
    sys.exit(main())

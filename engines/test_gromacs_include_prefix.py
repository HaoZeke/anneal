#!/usr/bin/env python3
"""A Python prefix inside the parent source tree must not fail configure.

engines/ probes sys.prefix/include for the optional Gromacs header. Meson
rejects include_directories() of an absolute path inside the source tree,
which is where a pixi or venv prefix lives. The probe has to skip that
path. The header test stays optional.
Usage: test_gromacs_include_prefix.py <root> <meson>
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import venv
except ImportError:
    venv = None  # type: ignore[assignment]

PARENT = """project('rgpot_parent', 'c', 'cpp')
subproject(
    'rgpot',
    default_options: [
        'with_tests=false',
        'with_eigen=false',
        'with_rpc=false',
        'with_fortran_pots=disabled',
        'with_fast_nlist=disabled',
        'with_metatomic=false',
    ],
)
"""


def _meson_argv(argv: list[str]) -> list[str]:
    if len(argv) == 1 and Path(argv[0]).name.lower().startswith("python"):
        return [argv[0], "-m", "mesonbuild.mesonmain"]
    return argv


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    if venv is None:
        print(
            "gromacs_include_prefix requires the Python standard library venv module "
            "(on Debian/Ubuntu, install python3-venv)",
            file=sys.stderr,
        )
        return 2
    root = Path(sys.argv[1]).resolve()
    meson = _meson_argv(sys.argv[2:])
    with tempfile.TemporaryDirectory(prefix="rgpot-gromacs-") as work:
        parent = Path(work)
        try:
            venv.create(parent / ".venv", with_pip=False)
        except Exception as exc:
            print(
                f"gromacs_include_prefix could not create a Python venv ({exc}); "
                "ensure python3-venv is installed",
                file=sys.stderr,
            )
            return 1
        py = parent / ".venv" / "bin" / "python"
        prefix = subprocess.check_output(
            [str(py), "-c", "import sys; print(sys.prefix)"],
            text=True,
        ).strip()
        if not prefix.startswith(str(parent)):
            print(f"venv prefix {prefix} is outside {parent}", file=sys.stderr)
            return 1
        (parent / "subprojects").mkdir()
        (parent / "subprojects" / "rgpot").symlink_to(root, target_is_directory=True)
        (parent / "meson.build").write_text(PARENT)
        env = os.environ.copy()
        env["PATH"] = str(parent / ".venv" / "bin") + os.pathsep + env.get("PATH", "")
        setup = subprocess.run(
            [*meson, "setup", str(parent / "build")],
            cwd=parent,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if "absolute path to a dir in the source tree" in setup.stdout:
            print(setup.stdout, file=sys.stderr)
            print(
                "configure rejected the in-tree Python prefix",
                file=sys.stderr,
            )
            return 1
        if setup.returncode != 0:
            print(setup.stdout, file=sys.stderr)
            if "Could not detect Ninja" in setup.stdout:
                print(
                    "gromacs_include_prefix requires ninja on PATH or in NINJA",
                    file=sys.stderr,
                )
            elif "Unknown compiler(s)" in setup.stdout:
                print(
                    "gromacs_include_prefix requires C and C++ compilers on PATH or in CC/CXX",
                    file=sys.stderr,
                )
            else:
                print(
                    "parent project did not configure with the in-tree Python prefix",
                    file=sys.stderr,
                )
            return 1
    print("in-tree Python prefix does not fail configure")
    return 0


if __name__ == "__main__":
    sys.exit(main())

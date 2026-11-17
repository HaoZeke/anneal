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
import venv
from pathlib import Path

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


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    meson = sys.argv[2]
    with tempfile.TemporaryDirectory(prefix="rgpot-gromacs-") as work:
        parent = Path(work)
        venv.create(parent / ".venv", with_pip=False)
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
            [meson, "setup", str(parent / "build")],
            cwd=parent,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if setup.returncode != 0 or "absolute path to a dir in the source tree" in setup.stdout:
            print(setup.stdout, file=sys.stderr)
            print(
                "configure rejected the in-tree Python prefix",
                file=sys.stderr,
            )
            return 1
    print("in-tree Python prefix does not fail configure")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""with_eigen must configure when pkg-config has no eigen3.

dependency(..., subproject: subproject('eigen')) is not a dependency()
keyword. The wrap exposes eigen_dep and does not override the eigen3
name, so the lookup has to be fallback: ['eigen', 'eigen_dep'].
Usage: test_eigen_subproject_fallback.py <root> <meson>
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

PARENT = """project('rgpot_parent', 'c', 'cpp')
subproject(
    'rgpot',
    default_options: [
        'with_tests=false',
        'with_eigen=true',
        'with_rpc=false',
        'with_fortran_pots=disabled',
        'with_fast_nlist=disabled',
        'with_metatomic=false',
    ],
)
"""


def _wrapper(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    meson = sys.argv[2]
    real_pc = shutil.which("pkg-config")
    real_cm = shutil.which("cmake")
    if real_pc is None or real_cm is None:
        print("pkg-config and cmake are required", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory(prefix="rgpot-eigen-") as work:
        parent = Path(work)
        bindir = parent / "bin"
        bindir.mkdir()
        _wrapper(
            bindir / "pkg-config",
            "#!/bin/sh\n"
            "for a in \"$@\"; do\n"
            "  if [ \"$a\" = eigen3 ]; then exit 1; fi\n"
            "done\n"
            f"exec {real_pc} \"$@\"\n",
        )
        _wrapper(
            bindir / "cmake",
            "#!/bin/sh\n"
            "if grep -q 'find_package(Eigen3' CMakeLists.txt 2>/dev/null; then\n"
            "  exit 1\n"
            "fi\n"
            f"exec {real_cm} \"$@\"\n",
        )
        (parent / "subprojects").mkdir()
        (parent / "subprojects" / "rgpot").symlink_to(root, target_is_directory=True)
        (parent / "meson.build").write_text(PARENT)
        env = os.environ.copy()
        env["PATH"] = str(bindir) + os.pathsep + env.get("PATH", "")
        setup = subprocess.run(
            [meson, "setup", str(parent / "build")],
            cwd=parent,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        out = setup.stdout
        if setup.returncode != 0 or "eigen.wrap" not in out:
            print(out, file=sys.stderr)
            print(
                "with_eigen did not fall back to the eigen wrap",
                file=sys.stderr,
            )
            return 1
        if "unknown keyword arguments" in out:
            print(out, file=sys.stderr)
            return 1
    print("eigen wrap supplies eigen3 when pkg-config does not")
    return 0


if __name__ == "__main__":
    sys.exit(main())

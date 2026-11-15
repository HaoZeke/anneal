#!/usr/bin/env python3
"""The generated Stillinger-Weber source must not share a library with its modules.

rgpot_sw.f90 is a custom_target copy, so meson cannot scan its use lines
until the copy exists. Compiled into rgpot_fortran_kernels it races
rgpot_kinds and rgpot_neighbors. It has to be its own library, ordered
after those modules.
Usage: test_fortran_sw_module_order.py <root> <meson>
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    meson = sys.argv[2]
    with tempfile.TemporaryDirectory(prefix="rgpot-sw-") as work:
        build = Path(work) / "build"
        setup = run(
            [
                meson,
                "setup",
                str(build),
                str(root),
                "-Dwith_tests=false",
                "-Dwith_eigen=false",
                "-Dwith_rpc=false",
                "-Dwith_fast_nlist=disabled",
                "-Dwith_metatomic=false",
                "-Dwith_fortran_pots=enabled",
            ],
            root,
        )
        if setup.returncode != 0:
            print(setup.stdout, file=sys.stderr)
            print("rgpot did not configure with Fortran pots", file=sys.stderr)
            return 1
        ninja_file = (build / "build.ninja").read_text(errors="replace")
        sw = []
        for line in ninja_file.splitlines():
            if not line.startswith("build ") or ": " not in line:
                continue
            name = line[len("build ") :].split(":", 1)[0].strip()
            if name.endswith("librgpot_sw_lib.a"):
                sw.append(name)
        if not sw:
            print(
                "generated rgpot_sw.f90 is not in its own library after the kernels",
                file=sys.stderr,
            )
            return 1
        built = run(["ninja", "-C", str(build), sw[0]], root)
        if built.returncode != 0:
            print(built.stdout, file=sys.stderr)
            print(f"failed to build {sw[0]}", file=sys.stderr)
            return 1
    print(f"built {sw[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

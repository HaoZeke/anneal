#!/usr/bin/env python3
"""A consumer of rgpot_dep_no_openmp compiles and links without OpenMP.

A parent project takes rgpot as a subproject, once with a shared and once
with a static librgpot, and builds the same program against rgpot_dep and
against rgpot_dep_no_openmp. The rgpot_dep program's compile and link
lines carry the OpenMP flag. The rgpot_dep_no_openmp program's carry none,
and the program fails to compile if _OPENMP is defined. Both programs run
and read OMP_NUM_THREADS back through librgpot, which only an OpenMP
librgpot does, so the library's own build is unchanged.
With --pc, the Libs of that rgpot.pc carry no OpenMP flag either, nor do
its Cflags unless --pc-static says librgpot is static: meson writes the
compile flag of a static library's archives there.
With --compilers CC CXX the nested builds use those compilers instead of the
environment's. A static librgpot links the OpenMP runtime its compiler
selects, which for Clang is libomp and not libgomp, so a build that ran only
under GCC would not notice a runtime paired with the wrong compiler.
Exit 77 (skip) when rgpot finds no OpenMP, or when the compilers given cannot
link -fopenmp.
Usage: test_consumer_no_openmp.py [--compilers CC CXX]
       [--pc rgpot.pc [--pc-static]] <root> <meson>
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

PARENT = """project('rgpot_consumer', 'cpp', default_options: ['cpp_std=c++20'])
rgpot_proj = subproject(
    'rgpot',
    default_options: [
        'default_library={library}',
        'with_tests=false',
        'with_rpc=false',
        'with_fortran_pots=disabled',
        'with_fast_nlist=disabled',
    ],
)
executable(
    'consumer_openmp',
    'consumer.cc',
    cpp_args: ['-DEXPECT_OPENMP'],
    dependencies: rgpot_proj.get_variable('rgpot_dep'),
)
executable(
    'consumer_no_openmp',
    'consumer.cc',
    dependencies: rgpot_proj.get_variable('rgpot_dep_no_openmp'),
)
"""

CONSUMER = r"""#include <cmath>
#include <cstdio>

#include "rgpot/Slices.hpp"
#include "rgpot/SuttonChen/SuttonChenPot.hpp"

#if defined(EXPECT_OPENMP) != defined(_OPENMP)
#error "consumer compiled with the wrong OpenMP setting"
#endif

int main() {
  int ran[4] = {};
  rgpot::forEachSlice(4, [&](int slice) { ran[slice] = 1; });
  const int slices = ran[0] + ran[1] + ran[2] + ran[3];
  const rgpot_sc_config cfg{1.0, 34.408, 1.0, 10, 8, 0.0, 0};
  const double x[6] = {0.0, 0.0, 0.0, 1.1, 0.0, 0.0};
  double forces[6] = {};
  double energy = 0.0;
  if (rgpot_sutton_chen_energy_forces(&cfg, 2, x, nullptr, &energy,
                                      forces) != 0 ||
      !std::isfinite(energy) || slices != 4) {
    return 1;
  }
  std::printf("pool %d\n", rgpot::poolThreads());
  return 0;
}
"""

# -fopenmp, -fopenmp=libomp, -qopenmp, /openmp, -openmp and the like.
OPENMP_FLAG = re.compile(r"^[-/][A-Za-z]*openmp", re.IGNORECASE)


def _pool_lines(output: str) -> list[str]:
    # The runtime may print diagnostics of its own around the program's line.
    return [ln.strip() for ln in output.splitlines() if ln.strip().startswith("pool ")]


def _meson_argv(argv: list[str]) -> list[str]:
    if len(argv) == 1 and Path(argv[0]).name.lower().startswith("python"):
        return [argv[0], "-m", "mesonbuild.mesonmain"]
    return argv


def _run(argv: list[str], **kw) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kw
    )


def _commands(build: Path, target: str) -> tuple[str, str]:
    """The compile line of consumer.cc and the link line of `target`."""
    ninja = os.environ.get("NINJA", "ninja")
    out = _run([ninja, "-C", str(build), "-t", "commands", target], check=True)
    obj = f"{target}.p"
    compile_line = link_line = ""
    for line in out.stdout.splitlines():
        if obj not in line:
            continue
        if " -c " in line and "consumer.cc" in line:
            compile_line = line
        elif re.search(rf"-o\s+{re.escape(target)}(\.exe)?(\s|$)", line):
            link_line = line
    if not compile_line or not link_line:
        raise RuntimeError(f"no compile or link line for {target}:\n{out.stdout}")
    return compile_line, link_line


def _openmp_flags(line: str) -> list[str]:
    return [tok for tok in shlex.split(line) if OPENMP_FLAG.match(tok)]


def _case(root: Path, meson: list[str], work: Path, library: str) -> int:
    parent = work / library
    (parent / "subprojects").mkdir(parents=True)
    (parent / "subprojects" / "rgpot").symlink_to(root, target_is_directory=True)
    pkgcache = root / "subprojects" / "packagecache"
    if pkgcache.is_dir():
        (parent / "subprojects" / "packagecache").symlink_to(
            pkgcache, target_is_directory=True
        )
    (parent / "meson.build").write_text(PARENT.format(library=library))
    (parent / "consumer.cc").write_text(CONSUMER)
    build = parent / "build"
    setup = _run([*meson, "setup", str(build)], cwd=parent)
    if setup.returncode != 0:
        print(setup.stdout, file=sys.stderr)
        return 1

    lines = {
        target: _commands(build, target)
        for target in ("consumer_openmp", "consumer_no_openmp")
    }
    with_omp = lines["consumer_openmp"]
    if not _openmp_flags(with_omp[0]):
        print(f"[{library}] rgpot found no OpenMP: nothing to compare")
        return 77
    failed = False
    if not _openmp_flags(with_omp[1]):
        print(f"[{library}] rgpot_dep link line lost the OpenMP flag:", file=sys.stderr)
        print(with_omp[1], file=sys.stderr)
        failed = True
    for kind, line in zip(("compile", "link"), lines["consumer_no_openmp"]):
        flags = _openmp_flags(line)
        print(f"[{library}] rgpot_dep_no_openmp {kind} OpenMP flags: {flags}")
        if flags:
            print(line, file=sys.stderr)
            failed = True
    if failed:
        return 1

    built = _run(
        [*meson, "compile", "-C", str(build), "consumer_openmp", "consumer_no_openmp"]
    )
    if built.returncode != 0:
        print(built.stdout, file=sys.stderr)
        return 1
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "3"
    for target in ("consumer_openmp", "consumer_no_openmp"):
        ran = _run([str(build / target)], env=env)
        print(f"[{library}] {target}: {ran.stdout.strip()}")
        if ran.returncode != 0 or _pool_lines(ran.stdout) != ["pool 3"]:
            print(
                f"[{library}] {target} exited {ran.returncode} without "
                "reaching librgpot's OpenMP pool",
                file=sys.stderr,
            )
            return 1
    return 0


def _pc_flags(pc: Path) -> dict[str, list[str]]:
    flags = {}
    for line in pc.read_text().splitlines():
        key, sep, value = line.partition(":")
        if sep and key in ("Cflags", "Libs"):
            flags[key] = [tok for tok in value.split() if OPENMP_FLAG.match(tok)]
    return flags


def _use_compilers(cc: str, cxx: str) -> bool:
    """Point the nested builds at another compiler, if it can build OpenMP."""
    with tempfile.TemporaryDirectory(prefix="rgpot-omp-probe-") as probe:
        src = Path(probe) / "probe.cc"
        src.write_text("int main() { return 0; }\n")
        built = _run([cxx, "-fopenmp", str(src), "-o", str(Path(probe) / "probe")])
    if built.returncode != 0:
        print(f"{cxx} cannot link -fopenmp: nothing to compare")
        return False
    os.environ["CC"] = cc
    os.environ["CXX"] = cxx
    return True


def main() -> int:
    args = sys.argv[1:]
    pc = None
    pc_static = False
    if len(args) >= 3 and args[0] == "--compilers":
        if not _use_compilers(args[1], args[2]):
            return 77
        args = args[3:]
    if len(args) >= 2 and args[0] == "--pc":
        pc = Path(args[1])
        args = args[2:]
        if args[:1] == ["--pc-static"]:
            pc_static = True
            args = args[1:]
    if len(args) < 2:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    root = Path(args[0]).resolve()
    meson = _meson_argv(args[1:])
    if pc is not None and pc.is_file():
        flags = _pc_flags(pc)
        print(f"rgpot.pc OpenMP flags: {flags}")
        checked = ["Libs"] if pc_static else ["Cflags", "Libs"]
        if set(flags) != {"Cflags", "Libs"} or any(flags[k] for k in checked):
            print(pc.read_text(), file=sys.stderr)
            return 1
    with tempfile.TemporaryDirectory(prefix="rgpot-no-openmp-") as work:
        results = [
            _case(root, meson, Path(work), library) for library in ("shared", "static")
        ]
    if any(r not in (0, 77) for r in results):
        return 1
    if all(r == 77 for r in results):
        return 77
    print("rgpot_dep_no_openmp consumers compile and link without OpenMP")
    return 0


if __name__ == "__main__":
    sys.exit(main())

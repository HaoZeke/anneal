#!/usr/bin/env python3
"""A CMake consumer of an installed static rgpot links without -fopenmp.

Builds and installs rgpot with CMake as a static library, then configures a
consumer with find_package(rgpot) and links rgpot::rgpot. The consumer's
compile and link lines carry no OpenMP flag, and the program fails to
compile if _OPENMP is defined. It runs and reads OMP_NUM_THREADS back
through librgpot, which only an OpenMP librgpot does.
Exit 77 (skip) when CMake finds no OpenMP for rgpot. CMake records a miss as
the value NOTFOUND, or as <variable>-NOTFOUND, in the cache, so those values
do not count as a flag.
Usage: test_cmake_consumer_openmp.py <root> <cmake>
       test_cmake_consumer_openmp.py --self-test
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CONSUMER_CMAKE = """cmake_minimum_required(VERSION 3.20)
project(rgpot_cmake_consumer CXX)
set(CMAKE_CXX_STANDARD 20)
find_package(rgpot REQUIRED)
add_executable(consumer consumer.cc)
target_link_libraries(consumer PRIVATE rgpot::rgpot)
"""

CONSUMER = r"""#include <cstdio>

#include "rgpot/Slices.hpp"

#if defined(_OPENMP)
#error "consumer compiled with OpenMP"
#endif

int main() {
  std::printf("pool %d\n", rgpot::poolThreads());
  return 0;
}
"""

OPENMP_FLAG = re.compile(r"^[-/][A-Za-z]*openmp", re.IGNORECASE)


def _run(argv: list[str], **kw) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kw
    )


def _cmake_found_openmp(cache: str) -> bool:
    """True when the cache holds an OpenMP_CXX_FLAGS value CMake found."""
    match = re.search(r"^OpenMP_CXX_FLAGS:\w+=(\S*)", cache, re.MULTILINE)
    if match is None:
        return False
    value = match.group(1)
    return value != "" and value != "NOTFOUND" and not value.endswith("-NOTFOUND")


def _self_test() -> int:
    cases = [
        ("OpenMP_CXX_FLAGS:STRING=-fopenmp\n", True),
        ("OpenMP_CXX_FLAGS:STRING=-fopenmp=libomp\n", True),
        ("OpenMP_CXX_FLAGS:STRING=-Xpreprocessor -fopenmp\n", True),
        ("OpenMP_CXX_FLAGS:STRING=\n", False),
        ("OpenMP_CXX_FLAGS:STRING=NOTFOUND\n", False),
        ("OpenMP_CXX_FLAGS:STRING=OpenMP_CXX_FLAGS-NOTFOUND\n", False),
        ("OpenMP_CXX_FLAGS:UNINITIALIZED=NOTFOUND\n", False),
        ("OpenMP_C_FLAGS:STRING=-fopenmp\n", False),
        ("OpenMP_gomp_LIBRARY:FILEPATH=/usr/lib/libgomp.so\n", False),
        ("A:STRING=1\nOpenMP_CXX_FLAGS:STRING=NOTFOUND\nB:STRING=2\n", False),
        ("A:STRING=1\nOpenMP_CXX_FLAGS:STRING=-fopenmp\nB:STRING=2\n", True),
    ]
    bad = [(c, w) for c, w in cases if _cmake_found_openmp(c) != w]
    for cache, want in bad:
        print(f"expected {want} for {cache!r}", file=sys.stderr)
    if not bad:
        print(f"{len(cases)} cache cases classified")
    return 1 if bad else 0


def _pool_lines(output: str) -> list[str]:
    # The runtime may print diagnostics of its own around the program's line.
    return [ln.strip() for ln in output.splitlines() if ln.strip().startswith("pool ")]


def _checked(argv: list[str], **kw) -> str | None:
    done = _run(argv, **kw)
    if done.returncode != 0:
        print(shlex.join(argv), file=sys.stderr)
        print(done.stdout, file=sys.stderr)
        return None
    return done.stdout


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        return _self_test()
    if len(sys.argv) != 3:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    cmake = sys.argv[2]
    ninja = os.environ.get("NINJA") or shutil.which("ninja")
    if ninja is None:
        print("test_cmake_consumer_openmp needs ninja", file=sys.stderr)
        return 2
    gen = ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}"]
    jobs = str(os.cpu_count() or 2)
    with tempfile.TemporaryDirectory(prefix="rgpot-cmake-openmp-") as work:
        work = Path(work)
        build, prefix = work / "rgpot-build", work / "prefix"
        configured = _checked(
            [
                cmake,
                "-S",
                str(root),
                "-B",
                str(build),
                *gen,
                "-DCMAKE_BUILD_TYPE=Release",
                "-DBUILD_SHARED_LIBS=OFF",
                "-DRGPOT_BUILD_TESTS=OFF",
                "-DRGPOT_BUILD_EXAMPLES=OFF",
                "-DRGPOT_WITH_CACHE=OFF",
                "-DRGPOT_WITH_RPC=OFF",
                "-DRGPOT_WITH_HIGHWAY=OFF",
                f"-DCMAKE_INSTALL_PREFIX={prefix}",
            ]
        )
        if configured is None:
            return 1
        cache = (build / "CMakeCache.txt").read_text()
        if not _cmake_found_openmp(cache):
            print("CMake found no OpenMP for rgpot: nothing to compare")
            return 77
        if _checked([cmake, "--build", str(build), "-j", jobs]) is None:
            return 1
        if _checked([cmake, "--install", str(build)]) is None:
            return 1

        src, cbuild = work / "consumer", work / "consumer-build"
        src.mkdir()
        (src / "CMakeLists.txt").write_text(CONSUMER_CMAKE)
        (src / "consumer.cc").write_text(CONSUMER)
        if (
            _checked(
                [
                    cmake,
                    "-S",
                    str(src),
                    "-B",
                    str(cbuild),
                    *gen,
                    f"-DCMAKE_PREFIX_PATH={prefix}",
                ]
            )
            is None
        ):
            return 1
        commands = _checked([ninja, "-C", str(cbuild), "-t", "commands", "consumer"])
        if commands is None:
            return 1
        lines = [line for line in commands.splitlines() if "consumer" in line]
        flags = [
            tok
            for line in lines
            for tok in shlex.split(line)
            if OPENMP_FLAG.match(tok)
        ]
        print(f"consumer OpenMP flags: {flags}")
        if flags or len(lines) < 2:
            print(commands, file=sys.stderr)
            return 1
        if _checked([cmake, "--build", str(cbuild)]) is None:
            return 1
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "3"
        ran = _run([str(cbuild / "consumer")], env=env)
        print(f"consumer: {ran.stdout.strip()}")
        if ran.returncode != 0 or _pool_lines(ran.stdout) != ["pool 3"]:
            print("consumer did not reach librgpot's OpenMP pool", file=sys.stderr)
            return 1
    print("an installed static rgpot links its CMake consumer without -fopenmp")
    return 0


if __name__ == "__main__":
    sys.exit(main())

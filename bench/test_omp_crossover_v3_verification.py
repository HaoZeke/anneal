#!/usr/bin/env python3
"""The verification note must take its counts from the test logs."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "bench" / "record_omp_crossover_v3_verification.py"


def load_recorder():
    spec = importlib.util.spec_from_file_location("record_omp_crossover_v3_verification", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def toolchain() -> dict[str, str]:
    return {
        "cc": "gcc (GCC) 13.3.0",
        "cxx": "g++ (GCC) 13.3.0",
        "gfortran": "no",
        "rustc": "rustc 1.83.0 (90b35a623 2024-11-26)",
        "ase": "no",
        "pytest": "no",
        "mpi": "no",
        "meson": "1.5.2",
        "cmake": "3.28.3",
        "ninja": "1.11.1",
    }


def meson_options() -> list[dict[str, object]]:
    values: list[tuple[str, object]] = [
        ("with_tests", True),
        ("with_examples", False),
        ("with_xtensor", False),
        ("with_eigen", False),
        ("with_rpc", True),
        ("with_cache", True),
        ("pure_lib", False),
        ("with_fortran_pots", "disabled"),
        ("with_fast_nlist", "disabled"),
        ("with_mpi", "disabled"),
        ("with_ipi", "enabled"),
    ]
    return [{"name": name, "value": value} for name, value in values]


def meson_log() -> str:
    decoy = (
        "Ok:                 1\n"
        "Expected Fail:      0\n"
        "Fail:               1\n"
        "Unexpected Pass:    0\n"
        "Skipped:            0\n"
        "Timeout:            0\n"
    )
    final = (
        "Ok:                 10\n"
        "Expected Fail:      2\n"
        "Fail:               0\n"
        "Unexpected Pass:    0\n"
        "Skipped:            1\n"
        "Timeout:            0\n"
    )
    return "noise\n" + decoy + "1/13 sample OK\n" + final


def cmake_cache() -> str:
    return "\n".join(
        [
            "// comment",
            "CMAKE_CXX_COMPILER_LAUNCHER:STRING=ccache",
            "CMAKE_C_COMPILER_LAUNCHER:STRING=ccache",
            "CMAKE_BUILD_TYPE:STRING=Release",
            "RGPOT_BUILD_TESTS:BOOL=ON",
            "RGPOT_BUILD_EXAMPLES:BOOL=OFF",
            "RGPOT_WITH_XTENSOR:BOOL=OFF",
            "RGPOT_WITH_EIGEN:BOOL=OFF",
            "RGPOT_WITH_RPC:BOOL=ON",
            "RGPOT_WITH_CACHE:BOOL=ON",
            "RGPOT_PURE_LIB:BOOL=OFF",
            "CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/c++",
            "",
        ]
    )


def ctest_log() -> str:
    return (
        "50% tests passed, 1 tests failed out of 2\n"
        "100% tests passed, 0 tests failed out of 8\n"
    )


def build_fixture(directory: Path) -> tuple[Path, Path, Path]:
    write(directory / "toolchain.json", json.dumps(toolchain()) + "\n")
    write(directory / "hidden-meson.log", meson_log())
    write(directory / "hidden-options.json", json.dumps(meson_options()))
    write(directory / "hidden-ctest.log", ctest_log())
    write(directory / "hidden-cache.txt", cmake_cache())
    meson_leg = {
        "title": "Meson, RPC and cache enabled",
        "runner": "meson",
        "build": "build/meson-on",
        "setup": [
            "meson",
            "setup",
            "build/meson-on",
            "-Dwith_tests=True",
            "-Dwith_examples=False",
            "-Dwith_xtensor=False",
            "-Dwith_eigen=False",
            "-Dwith_rpc=true",
            "-Dwith_cache=true",
            "-Dpure_lib=False",
        ],
        "compile": ["meson", "compile", "-C", "build/meson-on"],
        "test": ["meson", "test", "-C", "build/meson-on", "--print-errorlogs"],
        "log": "hidden-meson.log",
        "configured": "hidden-options.json",
    }
    cmake_leg = {
        "title": "CMake CTest, RPC and cache enabled",
        "runner": "cmake",
        "build": "build/cmake-on",
        "setup": [
            "cmake",
            "-S",
            ".",
            "-B",
            "build/cmake-on",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
            "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DRGPOT_BUILD_TESTS=True",
            "-DRGPOT_BUILD_EXAMPLES=False",
            "-DRGPOT_WITH_XTENSOR=OFF",
            "-DRGPOT_WITH_EIGEN=OFF",
            "-DRGPOT_WITH_RPC=ON",
            "-DRGPOT_WITH_CACHE=ON",
            "-DRGPOT_PURE_LIB=OFF",
        ],
        "compile": ["cmake", "--build", "build/cmake-on", "--parallel", "8"],
        "test": ["ctest", "--test-dir", "build/cmake-on", "--output-on-failure"],
        "log": "hidden-ctest.log",
        "configured": "hidden-cache.txt",
    }
    meson_path = directory / "meson-leg.json"
    cmake_path = directory / "cmake-leg.json"
    write(meson_path, json.dumps(meson_leg) + "\n")
    write(cmake_path, json.dumps(cmake_leg) + "\n")
    return directory / "toolchain.json", meson_path, cmake_path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def require_exit(message: str) -> None:
    raise SystemExit(message)


def main() -> None:
    recorder = load_recorder()
    meson_counts = recorder.parse_meson_summary(meson_log())
    require(
        meson_counts == {"Ok": 10, "Expected Fail": 2, "Fail": 0, "Unexpected Pass": 0, "Skipped": 1, "Timeout": 0},
        f"meson parser kept the wrong summary: {meson_counts}",
    )
    ctest_counts = recorder.parse_ctest_summary(ctest_log())
    require(
        ctest_counts == {"passed": 8, "failed": 0, "total": 8},
        f"ctest parser kept the wrong summary: {ctest_counts}",
    )
    try:
        recorder.parse_meson_summary("Ok: 1\n")
    except SystemExit as exc:
        require(str(exc) == "Meson log has no summary", f"incomplete meson log said {exc}")
    else:
        require_exit("incomplete meson log was accepted")
    try:
        recorder.parse_ctest_summary("99% tests passed, 0 tests failed out of 2\n")
    except SystemExit as exc:
        require("percent" in str(exc), f"inconsistent ctest percent said {exc}")
    else:
        require_exit("inconsistent ctest percent was accepted")

    with tempfile.TemporaryDirectory() as directory_name:
        directory = Path(directory_name)
        toolchain_path, meson_path, cmake_path = build_fixture(directory)
        output = directory / "note.md"
        with tempfile.TemporaryDirectory() as other_name:
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "render",
                    "--toolchain",
                    str(toolchain_path),
                    "--leg",
                    str(meson_path),
                    "--leg",
                    str(cmake_path),
                    "--output",
                    str(output),
                ],
                cwd=other_name,
                text=True,
                capture_output=True,
            )
        if result.returncode != 0:
            require_exit(result.stderr.strip() or result.stdout.strip())
        text = output.read_text(encoding="utf-8")
        meson_line = recorder.meson_summary(meson_counts)
        ctest_line = recorder.ctest_summary(ctest_counts)
        require(meson_line in text, "note omitted the parsed meson summary")
        require(ctest_line in text, "note omitted the parsed ctest summary")
        require("|Ok|10|" in text and "|Total|13|" in text, "meson table does not match the log")
        require("|Passed|8|" in text and "|Total|8|" in text, "ctest table does not match the log")
        require("Fail 1" not in text and "Ok 1," not in text, "note kept an earlier meson summary")
        require("1 tests failed" not in text, "note kept an earlier ctest summary")
        require("-Dwith_rpc=true" in text and "|with_fast_nlist|disabled|" in text, "meson options were dropped")
        require("|RGPOT_WITH_RPC|ON|" in text, "cmake cache value was dropped")
        require("hidden-meson.log" not in text and "hidden-ctest.log" not in text, "note leaked a log path")
        require("/usr/bin" not in text and str(directory) not in text, "note leaked a filesystem path")
        require("These counts are environment-specific." in text, "counts are not labeled environment-specific")
        require("92" not in text and "94" not in text, "note cites an unsubstantiated aggregate")
        require("\r" not in text and text.endswith("\n"), "note is not LF text")

        mismatched = json.loads(meson_path.read_text(encoding="utf-8"))
        mismatched["setup"] = list(mismatched["setup"])
        mismatched["setup"][mismatched["setup"].index("-Dwith_rpc=true")] = "-Dwith_rpc=false"
        bad_leg = directory / "bad-leg.json"
        write(bad_leg, json.dumps(mismatched) + "\n")
        rejected = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "render",
                "--toolchain",
                str(toolchain_path),
                "--leg",
                str(bad_leg),
                "--output",
                str(directory / "bad.md"),
            ],
            cwd=directory,
            text=True,
            capture_output=True,
        )
        rejected_text = (rejected.stdout + rejected.stderr).strip()
        require(
            rejected.returncode != 0 and "with_rpc" in rejected_text,
            f"mismatched option was not rejected: {rejected_text}",
        )

        bad_toolchain = toolchain()
        bad_toolchain["cxx"] = "/usr/bin/g++"
        bad_toolchain_path = directory / "bad-toolchain.json"
        write(bad_toolchain_path, json.dumps(bad_toolchain) + "\n")
        leaked = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "render",
                "--toolchain",
                str(bad_toolchain_path),
                "--leg",
                str(meson_path),
                "--output",
                str(directory / "leaked.md"),
            ],
            cwd=directory,
            text=True,
            capture_output=True,
        )
        leaked_text = (leaked.stdout + leaked.stderr).strip()
        require(
            leaked.returncode != 0 and "absolute path" in leaked_text,
            f"absolute toolchain path was not rejected: {leaked_text}",
        )


if __name__ == "__main__":
    main()

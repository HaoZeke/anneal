#!/usr/bin/env python3
"""Write the crossover verification note from Meson and CTest logs.

The counts in the note are parsed from the logs. Edit the legs and re-run
this program to change them. The note describes one toolchain.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

MESON_KEYS = (
    "Ok",
    "Expected Fail",
    "Fail",
    "Unexpected Pass",
    "Skipped",
    "Ignored",
    "Timeout",
)
RESOLVED_FEATURES = (
    "with_fortran_pots",
    "with_fast_nlist",
    "with_mpi",
    "with_ipi",
)
TOOLCHAIN_FIELDS = (
    ("cc", "C compiler"),
    ("cxx", "C++ compiler"),
    ("gfortran", "gfortran"),
    ("rustc", "rustc"),
    ("ase", "ASE"),
    ("pytest", "pytest"),
    ("mpi", "MPI"),
    ("meson", "Meson"),
    ("cmake", "CMake"),
    ("ninja", "Ninja"),
)
_TRUE = {"true", "yes", "on", "1"}
_FALSE = {"false", "no", "off", "0"}


def fail(message: str) -> None:
    raise SystemExit(message)


def require_recordable(label: str, value: str) -> str:
    if not isinstance(value, str) or value == "" or value != value.strip():
        fail(f"{label} is empty or padded")
    if "\n" in value or "\r" in value:
        fail(f"{label} contains a line break")
    if value.startswith("/") or value.startswith("\\"):
        fail(f"{label} contains an absolute path")
    if ".." in value.split("/"):
        fail(f"{label} contains a parent path")
    return value


def norm_option(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip()
    lowered = text.lower()
    if lowered in _TRUE:
        return "true"
    if lowered in _FALSE:
        return "false"
    return text


def display_option(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return require_recordable("configured value", str(value).strip())


def d_options(argv: list[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for arg in argv:
        if not arg.startswith("-D"):
            continue
        body = arg[2:]
        if "=" not in body:
            fail(f"option has no value: {arg}")
        name, value = body.split("=", 1)
        if name == "" or value == "":
            fail(f"option is incomplete: {arg}")
        found[name] = value
    return found


def format_command(argv: list[str]) -> str:
    if len(argv) < 2:
        fail("command is too short")
    for arg in argv:
        require_recordable("command argument", arg)
        if any(character.isspace() for character in arg):
            fail("command arguments must not contain whitespace")
    lines = [f"{argv[0]} {argv[1]} \\"]
    for arg in argv[2:-1]:
        lines.append(f"  {arg} \\")
    lines.append(f"  {argv[-1]}")
    return "\n".join(lines)


_SUMMARY_LINE = re.compile(
    r"^(Ok|Expected Fail|Fail|Unexpected Pass|Skipped|Ignored|Timeout):\s+(\d+)$"
)


def parse_meson_summary(text: str) -> dict[str, int]:
    """Parse the last Meson summary.

    Meson always prints Ok and Fail. It prints the other rows only when
    their count is not zero. Missing rows are zero.
    """
    blocks: list[dict[str, int]] = []
    current: dict[str, int] = {}

    def close() -> None:
        nonlocal current
        if "Ok" in current and "Fail" in current:
            blocks.append(current)
        current = {}

    for line in text.splitlines():
        match = _SUMMARY_LINE.fullmatch(line.strip())
        if match is None:
            close()
            continue
        key = match.group(1)
        if key in current:
            close()
        current[key] = int(match.group(2))
    close()
    if not blocks:
        fail("Meson log has no summary")
    last = blocks[-1]
    return {key: last.get(key, 0) for key in MESON_KEYS}


def parse_ctest_summary(text: str) -> dict[str, int]:
    found: list[dict[str, int]] = []
    pattern = re.compile(r"^(\d+)% tests passed, (\d+) tests failed out of (\d+)$")
    for line in text.splitlines():
        match = pattern.fullmatch(line.strip())
        if match is None:
            continue
        percent, failed, total = (int(group) for group in match.groups())
        if total < 1 or failed > total:
            fail("CTest summary counts are inconsistent")
        passed = total - failed
        ratio = 100.0 * passed / total
        if int(round(ratio)) != percent and int(ratio) != percent:
            fail("CTest summary percent does not match the counts")
        found.append({"passed": passed, "failed": failed, "total": total})
    if not found:
        fail("CTest log has no summary")
    return found[-1]


def meson_summary(counts: dict[str, int]) -> str:
    total = sum(counts[key] for key in MESON_KEYS)
    parts = ", ".join(f"{key} {counts[key]}" for key in MESON_KEYS)
    return f"{parts} ({total} total)"


def ctest_summary(counts: dict[str, int]) -> str:
    return f"{counts['passed']} passed, {counts['failed']} failed ({counts['total']} total)"


def load_toolchain(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        fail("toolchain must be an object")
    expected = {key for key, _label in TOOLCHAIN_FIELDS}
    if set(payload) != expected:
        fail("toolchain fields do not match the required set")
    return {key: require_recordable(key, payload[key]) for key, _label in TOOLCHAIN_FIELDS}


def meson_configured(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        fail("Meson buildoptions must be a list")
    options: dict[str, object] = {}
    for item in payload:
        if not isinstance(item, dict) or "name" not in item or "value" not in item:
            fail("Meson buildoptions entry is missing name or value")
        options[str(item["name"])] = item["value"]
    return options


def cmake_configured(path: Path) -> dict[str, str]:
    options: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(("//", "#")) or "=" not in line or ":" not in line:
            continue
        left, _, value = line.partition("=")
        name, _, _kind = left.partition(":")
        if name:
            options[name] = value
    return options


def check_options(argv: list[str], configured: dict[str, object]) -> None:
    for name, value in d_options(argv).items():
        if name not in configured:
            fail(f"configured build has no option {name}")
        if norm_option(value) != norm_option(configured[name]):
            fail(
                f"option {name} is {configured[name]!s} in the build and {value} on the command"
            )


def load_commands(leg: dict[str, object]) -> tuple[list[str], list[str], list[str]]:
    setup = leg["setup"]
    compile_cmd = leg["compile"]
    test_cmd = leg["test"]
    commands = (setup, compile_cmd, test_cmd)
    for cmd in commands:
        if not isinstance(cmd, list) or not cmd or not all(isinstance(arg, str) for arg in cmd):
            fail("leg commands must be string lists")
    return setup, compile_cmd, test_cmd


def feature_state(text: str, name: str) -> str:
    if text not in {"auto", "enabled", "disabled"}:
        fail(f"{name} value {text} is not a feature state")
    return text


def fast_from_setup(text: str) -> str:
    enabled = "Message: fast pair list: enabled" in text
    disabled = "Message: fast pair list: disabled" in text
    if enabled == disabled:
        fail("setup log does not resolve with_fast_nlist")
    return "enabled" if enabled else "disabled"


_MPI_FOUND = re.compile(r"Dependency mpi for host machine found:\s+(YES|NO)\b")


def mpi_from_setup(text: str) -> str:
    """Resolve with_mpi from the setup log.

    Meson prints ``Dependency mpi for host machine found: YES`` or ``NO``.
    The ``found:`` prefix is not enough: a ``NO`` line must stay disabled.
    ``mpic++ found: YES`` is a separate probe and does not override ``NO``.
    """
    found_yes = False
    found_no = False
    for line in text.splitlines():
        match = _MPI_FOUND.search(line)
        if match is None:
            continue
        if match.group(1) == "YES":
            found_yes = True
        else:
            found_no = True
    skipped = "Dependency mpi for host machine skipped: feature with_mpi disabled" in text
    mpicxx = "mpic++ found: YES" in text
    enabled = found_yes or (mpicxx and not found_no)
    disabled = found_no or skipped
    if enabled == disabled:
        fail("setup log does not resolve with_mpi")
    return "enabled" if enabled else "disabled"


def fortran_from_setup(text: str) -> str:
    found = False
    for line in text.splitlines():
        if line.startswith("Fortran compiler for the host machine:"):
            found = "not found" not in line.lower()
    return "enabled" if found else "disabled"


def ipi_from_pkgconfig(text: str) -> str:
    """Resolve with_ipi from the pkg-config file the build wrote.

    The Cflags line carries ``-DRGPOT_HAS_IPI=TRUE`` only when the build
    enabled the socket server. The option value alone is not a record of
    that choice.
    """
    cflags = [line for line in text.splitlines() if line.startswith("Cflags:")]
    if len(cflags) != 1:
        fail("pkg-config record has no single Cflags line")
    tokens = cflags[0].split()
    return "enabled" if "-DRGPOT_HAS_IPI=TRUE" in tokens else "disabled"


def resolved_features(
    setup_text: str,
    configured: dict[str, object],
    system: str,
    pkgconfig_text: str,
) -> list[tuple[str, str, str]]:
    ipi = ipi_from_pkgconfig(pkgconfig_text)
    if system == "windows" and ipi == "enabled":
        fail("with_ipi is enabled in the pkg-config record on windows")
    detected = {
        "with_fast_nlist": fast_from_setup(setup_text),
        "with_mpi": mpi_from_setup(setup_text),
        "with_fortran_pots": fortran_from_setup(setup_text),
        "with_ipi": ipi,
    }
    rows: list[tuple[str, str, str]] = []
    for name in RESOLVED_FEATURES:
        state = feature_state(display_option(configured[name]), name)
        resolved = detected[name]
        if state in {"enabled", "disabled"} and state != resolved:
            fail(f"{name} is {state} in the build and {resolved} in the setup log")
        rows.append((name, state, resolved))
    return rows


def host_system(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    system = payload.get("host", {}).get("system") if isinstance(payload, dict) else None
    if system not in {"linux", "darwin", "windows", "cygwin"}:
        fail("meson machines host system is missing")
    return str(system)


def load_meson_leg(leg: dict[str, object], root: Path) -> dict[str, object]:
    build = require_recordable("build", leg["build"])
    setup, compile_cmd, test_cmd = load_commands(leg)
    if setup[:2] != ["meson", "setup"] or len(setup) < 3 or setup[2] != build:
        fail("meson setup must be: meson setup <build> [options]")
    if compile_cmd != ["meson", "compile", "-C", build]:
        fail("meson compile command does not match the build directory")
    if test_cmd != ["meson", "test", "-C", build, "--print-errorlogs"]:
        fail("meson test command does not match the build directory")
    configured = meson_configured(root / str(leg["configured"]))
    check_options(setup, configured)
    for name in RESOLVED_FEATURES:
        if name not in configured:
            fail(f"Meson buildoptions omit {name}")
    setup_text = (root / str(leg["setup_log"])).read_text(encoding="utf-8")
    system = host_system(root / str(leg["machines"]))
    pkgconfig_text = (root / str(leg["pkgconfig"])).read_text(encoding="utf-8")
    counts = parse_meson_summary((root / str(leg["log"])).read_text(encoding="utf-8"))
    return {
        "runner": "meson",
        "title": require_recordable("title", leg["title"]),
        "commands": [setup, compile_cmd, test_cmd],
        "resolved": resolved_features(setup_text, configured, system, pkgconfig_text),
        "counts": counts,
        "summary": meson_summary(counts),
    }


def cmake_detection(text: str) -> list[tuple[str, str]]:
    if "Performing Test RGPOT_MPI_COMPILES - Success" in text:
        mpi = "ON"
    elif "Performing Test RGPOT_MPI_COMPILES - Failed" in text:
        mpi = "OFF"
    elif "Found MPI:" not in text:
        mpi = "OFF"
    else:
        fail("cmake setup log does not resolve RGPOT_HAVE_MPI")
    fortran = "OFF"
    marker = "The Fortran compiler identification is "
    for line in text.splitlines():
        if marker in line:
            fortran = require_recordable("fortran identification", line.split(marker, 1)[1].strip())
            break
    return [("Fortran", fortran), ("RGPOT_HAVE_MPI", mpi)]


def load_cmake_leg(leg: dict[str, object], root: Path) -> dict[str, object]:
    build = require_recordable("build", leg["build"])
    setup, compile_cmd, test_cmd = load_commands(leg)
    if setup[:5] != ["cmake", "-S", ".", "-B", build]:
        fail("cmake setup must be: cmake -S . -B <build> [options]")
    parallel = compile_cmd[4] if len(compile_cmd) == 5 else ""
    if compile_cmd[:4] != ["cmake", "--build", build, "--parallel"] or not parallel.isdigit():
        fail("cmake build command must be: cmake --build <build> --parallel <jobs>")
    if test_cmd != ["ctest", "--test-dir", build, "--output-on-failure"]:
        fail("ctest command does not match the build directory")
    configured = cmake_configured(root / str(leg["configured"]))
    check_options(setup, configured)
    setup_text = (root / str(leg["setup_log"])).read_text(encoding="utf-8")
    counts = parse_ctest_summary((root / str(leg["log"])).read_text(encoding="utf-8"))
    cache_rows = [(name, require_recordable(name, configured[name])) for name in d_options(setup)]
    return {
        "runner": "cmake",
        "title": require_recordable("title", leg["title"]),
        "commands": [setup, compile_cmd, test_cmd],
        "cache": cache_rows,
        "detected": cmake_detection(setup_text),
        "counts": counts,
        "summary": ctest_summary(counts),
    }


def load_leg(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        fail("leg must be an object")
    common = {"title", "runner", "build", "setup", "compile", "test", "log", "configured"}
    runner = payload.get("runner")
    if runner == "meson":
        expected = common | {"setup_log", "machines", "pkgconfig"}
    elif runner == "cmake":
        expected = common | {"setup_log"}
    else:
        fail("leg runner must be meson or cmake")
    if set(payload) != expected:
        fail("leg fields do not match the required set")
    root = path.parent
    if runner == "meson":
        return load_meson_leg(payload, root)
    if runner == "cmake":
        return load_cmake_leg(payload, root)
    fail("leg runner must be meson or cmake")
    return {}


def count_table(rows: list[tuple[str, str]]) -> list[str]:
    lines = ["|Result|Count|", "|---|---:|"]
    lines.extend(f"|{name}|{value}|" for name, value in rows)
    return lines


def render_verification(toolchain: dict[str, str], legs: list[dict[str, object]]) -> str:
    if not legs:
        fail("at least one leg is required")
    lines = [
        "# OpenMP crossover v3e test verification",
        "",
        "These counts are environment-specific.",
        "`bench/record_omp_crossover_v3_verification.py` parsed them from the",
        "Meson and CTest logs of the legs below. Another compiler, gfortran,",
        "rustc, ASE, pytest, or MPI install can register a different set. The",
        "configure arguments are the ones that produced each log, and the",
        "resolved values are the ones the build recorded.",
        "",
        "## Toolchain",
        "",
        "|Item|Value|",
        "|---|---|",
    ]
    for key, label in TOOLCHAIN_FIELDS:
        lines.append(f"|{label}|{toolchain[key]}|")
    lines += ["", "## Counts", "", "|Leg|Summary|", "|---|---|"]
    for leg in legs:
        lines.append(f"|{leg['title']}|{leg['summary']}|")
    for leg in legs:
        lines += ["", f"## {leg['title']}", "", "Configure, build, and test:", "", "```text"]
        rendered = [format_command(cmd) for cmd in leg["commands"]]
        lines.append("\n\n".join(rendered))
        lines += ["```"]
        if leg["runner"] == "meson":
            lines += [
                "",
                "Resolved features other than with_ipi are parsed from the",
                "setup log. with_ipi is parsed from the pkg-config Cflags",
                "the build wrote. Configured is the option value.",
                "",
                "|Option|Configured|Resolved|",
                "|---|---|---|",
            ]
            for name, configured_text, resolved in leg["resolved"]:
                lines.append(f"|{name}|{configured_text}|{resolved}|")
            counts = leg["counts"]
            total = sum(counts[key] for key in MESON_KEYS)
            rows = [(key, str(counts[key])) for key in MESON_KEYS]
            rows.append(("Total", str(total)))
        else:
            lines += [
                "",
                "CMake runs find_package(MPI) even when the configure",
                "arguments do not name MPI. RGPOT_HAVE_MPI is ON only when",
                "the compile check succeeds. Cache values below were checked",
                "against the configure arguments. Fortran and RGPOT_HAVE_MPI",
                "are parsed from the CMake setup log. with_fast_nlist is a",
                "Meson option.",
                "",
                "|Cache variable|Value|",
                "|---|---|",
            ]
            for name, value in leg["cache"]:
                lines.append(f"|{name}|{value}|")
            lines += ["", "|Detection|Value|", "|---|---|"]
            for name, value in leg["detected"]:
                lines.append(f"|{name}|{value}|")
            counts = leg["counts"]
            rows = [
                ("Passed", str(counts["passed"])),
                ("Failed", str(counts["failed"])),
                ("Total", str(counts["total"])),
            ]
        lines += ["", *count_table(rows)]
    lines += [
        "",
        "The crossover checks are `test_omp_crossover_v3_provenance.py`,",
        "`test_omp_crossover_v3_provenance_cases.py`,",
        "`test_omp_crossover_v3_report.py`, `test_omp_crossover_v3_analysis.py`,",
        "and `test_omp_crossover_v3_verification.py`. A bare provenance check",
        "from the repository root is:",
        "",
        "```text",
        "python3 bench/test_omp_crossover_v3_provenance.py",
        "```",
        "",
        "These checks are entries in `CppCore/tests/registry.txt`, which the",
        "Meson and CMake test runs consume.",
        "",
    ]
    return "\n".join(lines)


def first_line(text: str, label: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return require_recordable(label, stripped)
    fail(f"{label} probe produced no output")
    return ""


def probe_present(command: list[str], label: str) -> str:
    if shutil.which(command[0]) is None:
        return "no"
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        return "no"
    return "yes, " + first_line(result.stdout or result.stderr, label)


def probe_toolchain() -> dict[str, str]:
    def version(command: list[str], label: str) -> str:
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode != 0:
            fail(f"{label} probe failed")
        return first_line(result.stdout or result.stderr, label)

    python = sys.executable
    return {
        "cc": version(["gcc", "--version"], "cc"),
        "cxx": version(["g++", "--version"], "cxx"),
        "gfortran": probe_present(["gfortran", "--version"], "gfortran"),
        "rustc": version(["rustc", "--version"], "rustc"),
        "ase": probe_present([python, "-c", "import ase; print(ase.__version__)"], "ase"),
        "pytest": probe_present(
            [python, "-c", "import pytest; print(pytest.__version__)"],
            "pytest",
        ),
        "mpi": probe_mpi(),
        "meson": version(["meson", "--version"], "meson"),
        "cmake": version(["cmake", "--version"], "cmake"),
        "ninja": version(["ninja", "--version"], "ninja"),
    }


def probe_mpi() -> str:
    if shutil.which("mpirun") is None:
        return "no"
    result = subprocess.run(["mpirun", "--version"], text=True, capture_output=True)
    if result.returncode != 0:
        return "no"
    text = result.stdout or result.stderr
    for line in text.splitlines():
        if "Open MPI" not in line:
            continue
        version = line.split("Open MPI", 1)[1].strip(" ()")
        return require_recordable("mpi", f"yes, Open MPI {version}")
    fail("MPI probe did not name Open MPI")
    return ""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    probe = subparsers.add_parser("probe")
    probe.add_argument("--output", type=Path, required=True)
    render = subparsers.add_parser("render")
    render.add_argument("--toolchain", type=Path, required=True)
    render.add_argument("--leg", type=Path, action="append", required=True)
    render.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "probe":
        payload = probe_toolchain()
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
        return
    toolchain = load_toolchain(args.toolchain)
    legs = [load_leg(path) for path in args.leg]
    args.output.write_text(render_verification(toolchain, legs), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()

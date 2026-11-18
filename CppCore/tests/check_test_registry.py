#!/usr/bin/env python3
"""Compare the tests a configured build registered with registry.txt.

registry.txt lists every test meson and CMake register, the features that
enable it and its options. Each build system runs this script as the
test_list_parity test and passes the features it was configured with:

  check_test_registry.py meson BUILD_DIR FEATURE=0|1...
  check_test_registry.py cmake BUILD_DIR --ctest CTEST FEATURE=0|1...
  check_test_registry.py lint
  check_test_registry.py selftest

It fails when a registered test has no entry, when an entry whose gate
holds was not registered, when a test was registered although its gate
does not hold, when CMake registered a meson-only entry, and when a
test's timeout, suite, working directory, environment, serial flag or
should_fail flag differs from its entry. Meson's introspection does not
report should_fail, so only CMake's is compared. lint checks the
registry alone.

--list FILE (or RGPOT_TEST_LIST) writes the features that gate the entries
both build systems share, then the shared entries the build registered, so
a meson and a CMake build can be compared: where the features agree, the
tests must too.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = Path(__file__).resolve().with_name("registry.txt")
DEFAULT_TIMEOUT = 30
# Positional arguments each kind takes; py takes the script and its arguments.
ARITY = {"unit": 2, "fortran": 2, "nlist": 2, "rerun": 1, "custom": 0}
SOURCE_DIRS = {
    "unit": "CppCore/tests",
    "fortran": "CppCore/rgpot/fortran",
    "nlist": "CppCore/examples",
}
PROBE = "probe"
WORD = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.+-]*")


@dataclass
class Entry:
    name: str
    line: int
    gate: list[list[str]]
    kind: str
    args: list[str] = field(default_factory=list)
    timeout: int = DEFAULT_TIMEOUT
    suite: str | None = None
    workdir: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    serial: bool = False
    should_fail: bool = False
    reason: str | None = None

    def gate_text(self) -> str:
        return " ".join(",".join(clause) for clause in self.gate)

    def holds(self, features: dict[str, bool], probe: bool) -> bool:
        """probe stands for an optional program: True asks whether the test
        may be registered, False whether it must be."""
        return all(
            any(probe if atom == PROBE else features[atom] for atom in clause)
            for clause in self.gate
        )


@dataclass
class Test:
    name: str
    timeout: float | None
    suites: set[str]
    workdir: str | None
    env: dict[str, str]
    serial: bool
    should_fail: bool | None


def parse_options(entry: Entry, words: list[str], where: str) -> list[str]:
    errors = []
    options = False
    for word in words:
        key, eq, value = word.partition("=")
        if not eq and word not in ("serial", "should_fail"):
            if options:
                errors.append(f"{where}: argument {word!r} after an option")
            entry.args.append(word)
            continue
        options = True
        if word in ("serial", "should_fail"):
            setattr(entry, word, True)
        elif key == "timeout" and value.isdigit() and int(value) > 0:
            entry.timeout = int(value)
        elif key == "suite" and WORD.fullmatch(value):
            entry.suite = value
        elif key == "workdir" and value:
            entry.workdir = value
        elif key == "env" and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=[^=]*", value):
            var, _, setting = value.partition("=")
            entry.env[var] = setting
        elif word == "fp_contract=off" and entry.kind == "unit":
            pass
        else:
            errors.append(f"{where}: bad option {word!r}")
    return errors


def parse(path: Path) -> tuple[dict[str, Entry], list[str]]:
    entries: dict[str, Entry] = {}
    reasons: dict[str, str] = {}
    errors: list[str] = []
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        where = f"{path.name}:{number}"
        # CMake reads the file as a list of lines, comments included.
        if ";" in line or line.count("[") != line.count("]"):
            errors.append(f"{where}: ';' or an unmatched bracket splits CMake's list of lines")
            continue
        if not line or line.startswith("#"):
            continue
        if '"' in line:
            errors.append(f"{where}: '\"' quotes words when CMake splits an entry")
            continue
        if line.startswith("@"):
            key, _, text = line[1:].partition(" ")
            if not WORD.fullmatch(key) or not text.strip():
                errors.append(f"{where}: expected @reason text")
            elif key in reasons:
                errors.append(f"{where}: @{key} defined twice")
            reasons[key] = text.strip()
            continue
        fields = [part.strip() for part in line.split("|")]
        if len(fields) != 4:
            errors.append(f"{where}: expected name | gate | kind | reason")
            continue
        name, gate, spec, reason = fields
        words = spec.split()
        if not name or name in entries:
            errors.append(f"{where}: missing or repeated name {name!r}")
            continue
        clauses = [clause.split(",") for clause in gate.split()]
        if not clauses or not all(WORD.fullmatch(a) for c in clauses for a in c):
            errors.append(f"{where}: bad gate {gate!r}")
            continue
        if not words or words[0] not in (*ARITY, "py"):
            errors.append(f"{where}: unknown kind in {spec!r}")
            continue
        entry = Entry(name=name, line=number, gate=clauses, kind=words[0])
        errors += parse_options(entry, words[1:], where)
        want = ARITY.get(entry.kind)
        if (want is None and not entry.args) or (want is not None and len(entry.args) != want):
            errors.append(f"{where}: wrong number of arguments for {entry.kind}")
        elif entry.kind in SOURCE_DIRS:
            source = ROOT / SOURCE_DIRS[entry.kind] / entry.args[1]
            if not source.is_file():
                errors.append(f"{where}: no {source.relative_to(ROOT)}")
        elif entry.kind == "py" and not (ROOT / entry.args[0]).is_file():
            errors.append(f"{where}: no {entry.args[0]}")
        if entry.kind == "unit" and entry.workdir is None:
            entry.workdir = "."
        if entry.workdir not in (None, ".") and not (ROOT / entry.workdir).is_dir():
            errors.append(f"{where}: no directory {entry.workdir}")
        if reason:
            if not reason.startswith("@"):
                errors.append(f"{where}: a reason is @name")
            entry.reason = reason[1:]
        entries[name] = entry
    for entry in entries.values():
        where = f"{path.name}:{entry.line}"
        if entry.reason is not None and entry.reason not in reasons:
            errors.append(f"{where}: @{entry.reason} is not defined")
        if entry.kind == "rerun":
            target = entries.get(entry.args[0])
            if target is None or target.kind != "unit":
                errors.append(f"{where}: rerun names no unit entry")
    unused = set(reasons) - {e.reason for e in entries.values()}
    errors += [f"{path.name}: @{key} is not used" for key in sorted(unused)]
    return entries, errors


def meson_tests(build: Path) -> list[Test]:
    info = build / "meson-info"
    project = json.loads((info / "intro-projectinfo.json").read_text())
    name = project["descriptive_name"]
    tests = []
    for test in json.loads((info / "intro-tests.json").read_text()):
        suites = [s for s in test["suite"] if s == name or s.startswith(name + ":")]
        if not suites:
            continue
        tests.append(
            Test(
                name=test["name"],
                timeout=test["timeout"],
                suites={s.partition(":")[2] for s in suites if ":" in s},
                workdir=test["workdir"],
                env=test["env"],
                serial=not test["is_parallel"],
                should_fail=None,
            )
        )
    return tests


def inside(path: str | Path, directory: Path) -> bool:
    return Path(os.path.realpath(path)).is_relative_to(os.path.realpath(directory))


def cmake_tests(build: Path, ctest: str) -> list[Test]:
    shown = subprocess.run(
        [ctest, "--show-only=json-v1"],
        cwd=build,
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(shown.stdout)
    graph = data["backtraceGraph"]
    tests = []
    for test in data["tests"]:
        # Tests a fetched dependency registers are defined under the build tree.
        defined_in = graph["files"][graph["nodes"][test["backtrace"]]["file"]]
        if not inside(defined_in, ROOT) or inside(defined_in, build):
            continue
        props = {p["name"]: p["value"] for p in test.get("properties", [])}
        env = dict(item.partition("=")[::2] for item in props.get("ENVIRONMENT", []))
        tests.append(
            Test(
                name=test["name"],
                timeout=props.get("TIMEOUT"),
                suites=set(props.get("LABELS", [])),
                workdir=props.get("WORKING_DIRECTORY"),
                env=env,
                serial=bool(props.get("RUN_SERIAL", False)),
                should_fail=bool(props.get("WILL_FAIL", False)),
            )
        )
    return tests


def compare_options(entry: Entry, test: Test, build: Path) -> list[str]:
    problems = []
    timeout_scale = int(test.env.get("RGPOT_TEST_TIMEOUT_SCALE", "1"))
    expected_timeout = entry.timeout * timeout_scale
    if test.timeout is None or float(test.timeout) != expected_timeout:
        problems.append(
            f"timeout {test.timeout}, the registry says {entry.timeout}"
            f" times sanitizer scale {timeout_scale}"
        )
    suites = {entry.suite} if entry.suite else set()
    if test.suites != suites:
        problems.append(f"suite {sorted(test.suites)}, the registry says {sorted(suites)}")
    if entry.workdir is None:
        if test.workdir is not None and not inside(test.workdir, build):
            problems.append(f"runs in {test.workdir}, not in a build directory")
    else:
        want = ROOT / entry.workdir
        if test.workdir is None or os.path.realpath(test.workdir) != os.path.realpath(want):
            problems.append(f"runs in {test.workdir}, the registry says {want}")
    for var, setting in entry.env.items():
        if test.env.get(var) != setting:
            problems.append(f"{var}={test.env.get(var)!r}, the registry says {setting!r}")
    if test.serial != entry.serial:
        problems.append(f"serial is {test.serial}, the registry says {entry.serial}")
    if test.should_fail is not None and test.should_fail != entry.should_fail:
        problems.append(
            f"should_fail is {test.should_fail}, the registry says {entry.should_fail}"
        )
    return [f"{entry.name}: {problem}" for problem in problems]


def compare(
    entries: dict[str, Entry],
    tests: list[Test],
    features: dict[str, bool],
    system: str,
    build: Path,
) -> list[str]:
    used = {a for e in entries.values() for c in e.gate for a in c} - {PROBE}
    absent = sorted(used - set(features))
    if absent:
        return [f"{system} did not pass the features {', '.join(absent)}"]
    problems = []
    registered: dict[str, Test] = {}
    for test in tests:
        if test.name in registered:
            problems.append(f"{test.name}: registered twice")
        registered[test.name] = test
    for name, test in sorted(registered.items()):
        entry = entries.get(name)
        if entry is None:
            problems.append(f"{name}: registered, but registry.txt has no entry")
            continue
        if system == "cmake" and entry.reason is not None:
            problems.append(f"{name}: CMake registered a meson-only entry (@{entry.reason})")
        elif not entry.holds(features, probe=True):
            problems.append(f"{name}: registered, but its gate '{entry.gate_text()}' fails")
        problems += compare_options(entry, test, build)
    for name, entry in sorted(entries.items()):
        if name in registered or (system == "cmake" and entry.reason is not None):
            continue
        if entry.holds(features, probe=False):
            problems.append(
                f"{name}: gate '{entry.gate_text()}' holds, but {system} did not register it"
            )
    return problems


def parse_features(words: list[str]) -> tuple[dict[str, bool], list[str]]:
    features, errors = {}, []
    for word in words:
        name, eq, value = word.partition("=")
        if not eq or value not in ("0", "1"):
            errors.append(f"expected FEATURE=0|1, got {word!r}")
        features[name] = value == "1"
    return features, errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("system", choices=("meson", "cmake", "lint", "selftest"))
    parser.add_argument("rest", nargs="*", metavar="FEATURE=0|1")
    parser.add_argument("--ctest", default="ctest")
    parser.add_argument("--list", type=Path, default=os.environ.get("RGPOT_TEST_LIST"))
    return parser


def parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_intermixed_args(argv)
    if args.system in ("lint", "selftest"):
        if args.rest:
            parser.error(f"{args.system} takes no positional arguments: {' '.join(args.rest)}")
        args.build = None
        args.features = []
    else:
        if not args.rest:
            parser.error(f"{args.system} needs the build directory")
        args.build = Path(args.rest[0])
        args.features = args.rest[1:]
    del args.rest
    return args


def selftest() -> int:
    import contextlib
    import io
    import stat
    import tempfile
    import unittest

    class CliAndParitySelfTest(unittest.TestCase):
        def test_cmake_accepts_features_after_ctest_flag(self) -> None:
            args = parse_cli(
                [
                    "cmake",
                    "/tmp/build",
                    "--ctest",
                    "/usr/bin/ctest",
                    "tests=1",
                    "core=0",
                    "python=1",
                ]
            )
            self.assertEqual(args.system, "cmake")
            self.assertEqual(args.build, Path("/tmp/build"))
            self.assertEqual(args.ctest, "/usr/bin/ctest")
            self.assertEqual(args.features, ["tests=1", "core=0", "python=1"])

        def test_intermixed_options_anywhere_in_argv(self) -> None:
            args = parse_cli(
                [
                    "meson",
                    "/tmp/meson-build",
                    "tests=1",
                    "--list",
                    "/tmp/out.txt",
                    "core=1",
                ]
            )
            self.assertEqual(args.system, "meson")
            self.assertEqual(args.build, Path("/tmp/meson-build"))
            self.assertEqual(args.list, Path("/tmp/out.txt"))
            self.assertEqual(args.features, ["tests=1", "core=1"])

            before = parse_cli(
                [
                    "--ctest",
                    "custom-ctest",
                    "cmake",
                    "/tmp/cmake-build",
                    "tests=0",
                ]
            )
            self.assertEqual(before.ctest, "custom-ctest")
            self.assertEqual(before.build, Path("/tmp/cmake-build"))
            self.assertEqual(before.features, ["tests=0"])

        def test_missing_build_or_extra_lint_args_fail(self) -> None:
            for bad in (
                ["cmake"],
                ["cmake", "--ctest", "/usr/bin/ctest"],
                ["meson"],
                ["lint", "tests=1"],
                ["selftest", "extra"],
            ):
                with self.subTest(bad=bad):
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as ctx:
                            parse_cli(bad)
                    self.assertEqual(ctx.exception.code, 2)

        def test_cmake_main_with_intermixed_ctest_and_features(self) -> None:
            entries, problems = parse(REGISTRY)
            self.assertEqual(problems, [])
            all_atoms = sorted(
                {a for e in entries.values() for c in e.gate for a in c} - {PROBE}
            )
            with tempfile.TemporaryDirectory() as tmp:
                build = Path(tmp) / "build"
                build.mkdir()
                fake_ctest = Path(tmp) / "fake_ctest.py"
                payload = {
                    "backtraceGraph": {
                        "files": [str(ROOT / "CppCore" / "tests" / "tests.cmake")],
                        "nodes": [{"file": 0}],
                    },
                    "tests": [],
                }
                fake_ctest.write_text(
                    f"#!{sys.executable}\n"
                    "import json, sys\n"
                    "assert sys.argv[1:] == ['--show-only=json-v1']\n"
                    f"print(json.dumps({payload!r}))\n"
                )
                fake_ctest.chmod(fake_ctest.stat().st_mode | stat.S_IXUSR)
                out_list = Path(tmp) / "cmake-list.txt"
                feature_args = [f"{atom}=0" for atom in all_atoms]
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = main(
                        [
                            "cmake",
                            str(build),
                            "--ctest",
                            str(fake_ctest),
                            *feature_args[:3],
                            "--list",
                            str(out_list),
                            *feature_args[3:],
                        ]
                    )
                self.assertEqual(rc, 0)
                self.assertTrue(out_list.is_file())
                self.assertIn("feature tests=0\n", out_list.read_text())

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CliAndParitySelfTest)
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_cli(argv)
    if args.system == "selftest":
        return selftest()

    entries, problems = parse(REGISTRY)
    features, errors = parse_features(args.features)
    problems += errors
    tests: list[Test] = []
    if args.system != "lint" and not problems:
        if args.system == "meson":
            tests = meson_tests(args.build)
        else:
            tests = cmake_tests(args.build, args.ctest)
        problems += compare(entries, tests, features, args.system, args.build)
        if args.list:
            shared = {e.name for e in entries.values() if e.reason is None}
            gating = {a for n in shared for c in entries[n].gate for a in c} - {PROBE}
            lines = [f"feature {a}={int(features[a])}" for a in sorted(gating & set(features))]
            lines += [f"test {name}" for name in sorted(t.name for t in tests if t.name in shared)]
            args.list.write_text("".join(f"{line}\n" for line in lines))
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print(
            f"{len(problems)} problem(s) against {REGISTRY.relative_to(ROOT)}. A new"
            " test needs an entry there; a test both build systems can run is"
            " registered in both.",
            file=sys.stderr,
        )
        return 1
    if args.system == "lint":
        print(f"{len(entries)} entries")
    else:
        print(f"{args.system}: {len(tests)} tests registered, all match {len(entries)} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""OpenMP stays inside CppCore/rgpot/Slices.hpp and Slices.cc.

The helpers there cut work into slices and run every slice on whatever team
the runtime starts. A region written elsewhere that reads its own thread
number indexes buffers by threads that may never run, and loses their work
when OMP_THREAD_LIMIT, a nested region or dynamic teams start a short team.
This check fails when a library source spells OpenMP outside the helpers.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOTS = ("CppCore/rgpot", "include", "bind", "engines", "python")
ALLOWED = ("CppCore/rgpot/Slices.hpp", "CppCore/rgpot/Slices.cc")
C_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".cu",
    ".cuh",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".inc",
    ".ipp",
    ".tpp",
}
F_SUFFIXES = {".f", ".f90", ".f95", ".f03", ".f08"}

C_RULES = (
    (re.compile(r"#\s*pragma\s+omp\b"), "an OpenMP pragma"),
    (re.compile(r"\b_Pragma\s*\(\s*\"\s*omp\b"), "an OpenMP _Pragma"),
    (re.compile(r"\b__pragma\s*\(\s*omp\b"), "an OpenMP __pragma"),
    (re.compile(r"\b(?:omp|kmp)_\w+\s*\("), "an OpenMP runtime call"),
    (re.compile(r"#\s*include\s*[<\"]omp\.h[>\"]"), "omp.h"),
    (re.compile(r"\b_OPENMP\b"), "the _OPENMP macro"),
)
F_RULES = (
    (re.compile(r"^\s*!\$omp\b", re.IGNORECASE), "an OpenMP directive"),
    (re.compile(r"\buse\s+omp_lib\b", re.IGNORECASE), "omp_lib"),
    (re.compile(r"\bomp_\w+\s*\(", re.IGNORECASE), "an OpenMP runtime call"),
)


def strip_c_comments(text: str) -> str:
    """Blank out comments, keeping newlines so line numbers still match."""
    out = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            end = text.find("\n", i)
            i = n if end < 0 else end
        elif c == "/" and nxt == "*":
            end = text.find("*/", i + 2)
            stop = n if end < 0 else end + 2
            out.append("\n" * text.count("\n", i, stop))
            i = stop
        elif c == "R" and nxt == '"' and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")):
            open_paren = text.find("(", i + 2)
            if open_paren < 0:
                out.append(c)
                i += 1
                continue
            close = ")" + text[i + 2 : open_paren] + '"'
            end = text.find(close, open_paren + 1)
            stop = n if end < 0 else end + len(close)
            out.append(text[i:stop])
            i = stop
        elif c == '"' or (c == "'" and not (i > 0 and text[i - 1].isalnum())):
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            stop = min(j + 1, n)
            out.append(text[i:stop])
            i = stop
        else:
            out.append(c)
            i += 1
    return "".join(out)


def c_findings(text: str) -> list[tuple[int, str]]:
    found = []
    for number, line in enumerate(strip_c_comments(text).splitlines(), start=1):
        for pattern, what in C_RULES:
            if pattern.search(line):
                found.append((number, what))
    return found


def fortran_findings(text: str) -> list[tuple[int, str]]:
    found = []
    for number, line in enumerate(text.splitlines(), start=1):
        directive = F_RULES[0][0].search(line)
        if directive:
            found.append((number, F_RULES[0][1]))
            continue
        code = line.split("!", 1)[0]
        for pattern, what in F_RULES[1:]:
            if pattern.search(code):
                found.append((number, what))
    return found


def self_check() -> None:
    caught = {
        "#pragma omp parallel": "an OpenMP pragma",
        "  #  pragma   omp for schedule(static)": "an OpenMP pragma",
        '_Pragma("omp parallel")': "an OpenMP _Pragma",
        "__pragma(omp parallel)": "an OpenMP __pragma",
        "const int t = omp_get_thread_num();": "an OpenMP runtime call",
        "n = omp_get_max_threads ();": "an OpenMP runtime call",
        "#include <omp.h>": "omp.h",
        "#ifdef _OPENMP": "the _OPENMP macro",
    }
    for line, what in caught.items():
        hits = [w for _, w in c_findings(line + "\n")]
        if what not in hits:
            raise AssertionError(f"rule '{what}' misses: {line}")
    quiet = (
        "// omp_get_thread_num() would read the team\n",
        "/* #pragma omp parallel\n   _OPENMP */ int x;\n",
        'const char *v = "OMP_NUM_THREADS";\n',
        'const char *url = "http://x//y"; int omp = 1;\n',
        "const long n = 1'000; // omp_get_thread_num()\n",
    )
    for text in quiet:
        if c_findings(text):
            raise AssertionError(f"rule fires on prose: {text!r}")
    if [w for _, w in fortran_findings("!$omp parallel do\n")] != ["an OpenMP directive"]:
        raise AssertionError("Fortran directive rule misses !$omp")
    if fortran_findings("! omp_get_thread_num() in a comment\n"):
        raise AssertionError("Fortran rule fires on a comment")


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    self_check()
    for name in ALLOWED:
        if not (root / name).is_file():
            print(f"{name} is missing; update ALLOWED with the helper's new home", file=sys.stderr)
            return 1
    allowed = {root / name for name in ALLOWED}
    failures = []
    scanned = 0
    for top in ROOTS:
        base = root / top
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path in allowed:
                continue
            suffix = path.suffix.lower()
            if suffix in C_SUFFIXES:
                finder = c_findings
            elif suffix in F_SUFFIXES:
                finder = fortran_findings
            else:
                continue
            scanned += 1
            text = path.read_text(encoding="utf-8", errors="replace")
            folded = text.lower()
            if "omp" not in folded and "kmp_" not in folded:
                continue
            for number, what in finder(text):
                failures.append(f"{path.relative_to(root).as_posix()}:{number}: {what}")
    if failures:
        print("OpenMP outside CppCore/rgpot/Slices.hpp and Slices.cc:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        print(
            "Open regions with rgpot::forEachSlice, forEachSlicePhase or onFullTeam,"
            " and index buffers by the slice they pass.",
            file=sys.stderr,
        )
        return 1
    print(f"{scanned} sources keep OpenMP inside the Slices helpers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

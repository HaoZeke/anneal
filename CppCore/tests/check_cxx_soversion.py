#!/usr/bin/env python3
"""Fail when the C++ layout stamp moves without a librgpot soversion bump.

kCxxSoversion in rgpot/abi/CxxLayout.hpp is the ELF soversion of librgpot.
Meson and CMake read that constant. kCxxSoversionTable records
header_stamp().hash for each platform tag, without RGPOT_HAS_CACHE.
The ledger records the soversion and a fingerprint of the stamp's inputs.
The fingerprint is platform-independent. Shipped ledger lines are
immutable: a layout change appends a higher soversion.

A live stamp is checked only when its tag is in the table and features
is 0. An unlisted platform prints the stamp and passes. Cache builds
(features != 0) are not compared to the no-cache table.

The C ABI stamp (RGPOT_ABI_*) is not this number. It did not change, and
the C API is exported from librgpot, so it has no soversion of its own.
librgpot_mpi stays at soversion 3 (RGPOT_MPI_ABI_VERSION 1).

Usage:
  check_cxx_soversion.py [soversion-from-meson] [path-to-librgpot|-]
                         [print_cxx_layout_stamp]
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAYOUT = ROOT / "CppCore" / "rgpot" / "abi" / "CxxLayout.hpp"
LEDGER = ROOT / "CppCore" / "rgpot" / "abi" / "cxx_soversion_ledger.txt"
MESON = ROOT / "CppCore" / "meson.build"
CMAKE = ROOT / "CMakeLists.txt"

# POD types whose sizeof or offsetof the stamp folds in. Method bodies of
# the potential classes are not included: a comment there is not a layout
# change. Virtual counts are the constants in CxxLayout.hpp.
STRUCTS = (
    ("CppCore/rgpot/pot_caps.hpp", "PotCaps"),
    ("CppCore/rgpot/ForceStructs.hpp", "ForceInput"),
    ("CppCore/rgpot/ForceStructs.hpp", "ForceOut"),
    ("CppCore/rgpot/ForceStructs.hpp", "ForceBatch"),
    ("CppCore/rgpot/CalculatorGroup.hpp", "CalculatorGroup"),
    ("CppCore/rgpot/LennardJones/LJPot.hpp", "LJConfig"),
    ("CppCore/rgpot/LennardJones/LJClusterPot.hpp", "LJClusterConfig"),
    ("CppCore/rgpot/Morse/MorsePot.hpp", "MorseConfig"),
    ("CppCore/rgpot/ZBL/ZBLPot.hpp", "ZBLConfig"),
    ("CppCore/rgpot/SuttonChen/SuttonChenPot.hpp", "SuttonChenConfig"),
    ("CppCore/rgpot/D2Pot/D2Pot.hpp", "D2Config"),
)

# Classes whose sizeof the stamp folds. Only instance data members are
# hashed: a method body or a static constexpr does not change the layout.
# Neighbours and CachedPairs are members of LJPot and MorsePot, so a field
# added there moves those sizes too.
CLASSES = (
    ("CppCore/rgpot/LennardJones/LJPot.hpp", "LJPot"),
    ("CppCore/rgpot/LennardJones/LJClusterPot.hpp", "LJClusterPot"),
    ("CppCore/rgpot/Morse/MorsePot.hpp", "MorsePot"),
    ("CppCore/rgpot/ZBL/ZBLPot.hpp", "ZBLPot"),
    ("CppCore/rgpot/SuttonChen/SuttonChenPot.hpp", "SuttonChenPot"),
    ("CppCore/rgpot/SumPot/SumPot.hpp", "SumPot"),
    ("CppCore/rgpot/D2Pot/D2Pot.hpp", "D2Pot"),
    ("CppCore/rgpot/nlist/Neighbours.hpp", "Neighbours"),
    ("CppCore/rgpot/simd/from_cache.hpp", "CachedPairs"),
)


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def brace_body(text: str, open_at: int) -> str:
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_at : i + 1]
    fail("unbalanced brace")
    return ""


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_compute_hash(text: str) -> str:
    stripped = strip_comments(text)
    match = re.search(r"constexpr std::uint64_t compute_hash\(\)", stripped)
    if not match:
        fail("detail::compute_hash() not found in CxxLayout.hpp")
    brace = stripped.find("{", match.end())
    return normalize(stripped[match.start() : brace] + brace_body(stripped, brace))


def extract_constants(text: str) -> str:
    stripped = strip_comments(text)
    names = (
        "kCxxLayoutRevision",
        "kPotentialBaseVirtuals",
        "kPotentialBaseCacheVirtuals",
        "kPotentialVirtuals",
    )
    lines = []
    for name in names:
        match = re.search(rf"{name}\s*=\s*\d+", stripped)
        if not match:
            fail(f"{name} not found in CxxLayout.hpp")
        lines.append(match.group(0))
    return "\n".join(lines)


def extract_struct(path: Path, name: str) -> str:
    stripped = strip_comments(path.read_text())
    match = re.search(rf"\b(?:struct|class)\s+{name}\b", stripped)
    if match:
        brace = stripped.find("{", match.end())
        return normalize(brace_body(stripped, brace))
    # typedef struct { ... } Name;
    match = re.search(rf"\}}\s*{name}\s*;", stripped)
    if not match:
        fail(f"{name} not found in {path}")
    close = match.start()
    depth = 0
    for i in range(close, -1, -1):
        if stripped[i] == "}":
            depth += 1
        elif stripped[i] == "{":
            depth -= 1
            if depth == 0:
                return normalize(stripped[i : match.end()])
    fail(f"{name} brace not found in {path}")
    return ""


def _function_body(buf: str) -> bool:
    text = buf.rstrip()
    text = re.sub(
        r"(?:\bconst\b|\boverride\b|\bfinal\b|\bnoexcept\b(?:\([^)]*\))?|"
        r"&&|&|\bvolatile\b|\s)+$",
        "",
        text,
    )
    return text.endswith(")")


def _data_member(stmt: str) -> bool:
    if not stmt or stmt.startswith("static ") or stmt.startswith("using "):
        return False
    if stmt.startswith("typedef ") or stmt.startswith("friend "):
        return False
    if "(" in stmt:
        return False
    return True


def extract_data_members(path: Path, name: str) -> str:
    """Instance data members of a class, with function bodies removed."""
    stripped = strip_comments(path.read_text())
    match = re.search(rf"\b(?:struct|class)\s+{name}\b", stripped)
    if not match:
        fail(f"{name} not found in {path}")
    brace = stripped.find("{", match.end())
    body = brace_body(stripped, brace)
    members: list[str] = []
    buf: list[str] = []
    depth = 0
    i = 1
    while i < len(body) - 1:
        char = body[i]
        if char == "{":
            if depth == 0 and _function_body("".join(buf)):
                skip = 1
                i += 1
                while i < len(body) - 1 and skip:
                    if body[i] == "{":
                        skip += 1
                    elif body[i] == "}":
                        skip -= 1
                    i += 1
                buf = []
                continue
            depth += 1
            buf.append(char)
        elif char == "}":
            depth -= 1
            buf.append(char)
        elif char == ";" and depth == 0:
            stmt = normalize("".join(buf))
            buf = []
            if _data_member(stmt):
                members.append(stmt)
        elif char == ":" and depth == 0 and re.fullmatch(
            r"\s*(public|private|protected)\s*", "".join(buf)
        ):
            buf = []
        else:
            buf.append(char)
        i += 1
    if not members:
        fail(f"{name} in {path} has no data members for the layout fingerprint")
    return "\n".join(members)


def fingerprint(layout_text: str) -> str:
    parts = [
        extract_compute_hash(layout_text),
        extract_constants(layout_text),
    ]
    for rel, name in STRUCTS:
        parts.append(extract_struct(ROOT / rel, name))
    for rel, name in CLASSES:
        parts.append(extract_data_members(ROOT / rel, name))
    digest = hashlib.sha256("\n".join(parts).encode()).hexdigest()
    return digest


def parse_soversion(text: str) -> int:
    if re.search(r"\bkCxxSoversionHash\b", text):
        fail("kCxxSoversionHash is removed; hashes belong in kCxxSoversionTable")
    found = re.findall(r"\bkCxxSoversion\s*=\s*(\d+)", text)
    if len(found) != 1:
        fail("kCxxSoversion must be assigned exactly once")
    return int(found[0])


def parse_table(text: str) -> list[tuple[str, int]]:
    rows = re.findall(
        r'\{\s*"([^"]+)"\s*,\s*(0x[0-9a-fA-F]+)ull\s*\}',
        text,
    )
    seen = set()
    parsed = []
    for tag, hashed in rows:
        if tag in seen:
            fail(f"kCxxSoversionTable repeats {tag}")
        seen.add(tag)
        parsed.append((tag, int(hashed, 16)))
    listed = re.findall(
        r'#define RGPOT_CXX_PLATFORM_LISTED_TAG "([^"]+)"',
        text,
    )
    if set(listed) != {tag for tag, _ in parsed}:
        fail(
            "kCxxSoversionTable tags and RGPOT_CXX_PLATFORM_LISTED_TAG guards "
            f"differ: table={sorted(tag for tag, _ in parsed)} listed={sorted(set(listed))}"
        )
    if len(listed) != len(set(listed)):
        fail("RGPOT_CXX_PLATFORM_LISTED_TAG repeats a tag")
    return parsed


def parse_ledger(text: str) -> list[tuple[int, str]]:
    rows = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            fail(f"ledger line must be 'soversion fingerprint': {raw}")
        so, digest = parts
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            fail(f"ledger fingerprint must be 64 hex digits: {raw}")
        rows.append((int(so), digest))
    if not rows:
        fail("cxx_soversion_ledger.txt has no entries")
    return rows


def git_show(rev: str, path: str) -> str | None:
    probe = subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-e", f"{rev}:{path}"],
        capture_output=True,
    )
    if probe.returncode != 0:
        return None
    show = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{rev}:{path}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return show.stdout


def base_rev() -> str | None:
    candidates = []
    import os

    if os.environ.get("GITHUB_BASE_REF"):
        candidates.append("origin/" + os.environ["GITHUB_BASE_REF"])
    candidates.extend(["origin/main", "main"])
    for rev in candidates:
        if (
            subprocess.run(
                ["git", "-C", str(ROOT), "rev-parse", "--verify", "--quiet", rev],
                capture_output=True,
            ).returncode
            == 0
        ):
            return rev
    if os.environ.get("CI") == "true":
        subprocess.run(
            ["git", "-C", str(ROOT), "fetch", "--depth=1", "origin", "main"],
            capture_output=True,
        )
        if (
            subprocess.run(
                ["git", "-C", str(ROOT), "rev-parse", "--verify", "--quiet", "origin/main"],
                capture_output=True,
            ).returncode
            == 0
        ):
            return "origin/main"
    return None


def check_against_base(rows: list[tuple[int, str]], layout_text: str) -> None:
    rev = base_rev()
    if rev is None:
        print("soversion ledger: no main ref, skipped the history check")
        return
    old = git_show(rev, "CppCore/rgpot/abi/cxx_soversion_ledger.txt")
    if old is None:
        print(f"soversion ledger: {rev} has no ledger yet")
        return
    old_rows = parse_ledger(old)
    if len(rows) < len(old_rows):
        fail("cxx_soversion_ledger.txt dropped a shipped soversion line")
    for index, previous in enumerate(old_rows):
        if rows[index] != previous:
            fail(
                "layout stamp changed under soversion "
                f"{previous[0]} ({rev}). Append a new soversion line and "
                "bump kCxxSoversion. Do not edit a shipped line."
            )
    if len(rows) == len(old_rows):
        old_layout = git_show(rev, "CppCore/rgpot/abi/CxxLayout.hpp")
        if old_layout is None:
            return
        if extract_compute_hash(old_layout) != extract_compute_hash(layout_text):
            fail(
                "detail::compute_hash() changed and kCxxSoversion did not. "
                "Bump kCxxSoversion and append a ledger line."
            )


def check_soname(lib: Path, soversion: int) -> None:
    if not lib.is_file():
        fail(f"librgpot not found: {lib}")
    # macOS meson passes -Ddefault_library=static, so the artifact is
    # librgpot.a and the runner has no readelf. A static archive has no
    # SONAME. Windows import libs are the same kind of file.
    if lib.suffix in {".a", ".lib"}:
        print(f"soversion: {lib.name} is a static archive, skipped SONAME check")
        return
    if lib.suffix == ".dll":
        print(f"soversion: {lib.name} is a Windows DLL, skipped ELF SONAME check")
        return
    if lib.suffix == ".dylib":
        check_install_name(lib, soversion)
        return
    try:
        readelf = subprocess.run(
            ["readelf", "-d", str(lib)],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print(f"soversion: readelf is not on PATH, skipped SONAME check for {lib.name}")
        return
    if readelf.returncode != 0:
        print(f"soversion: {lib} is not an ELF object, skipped SONAME check")
        return
    match = re.search(r"\(SONAME\)\s+Library soname: \[([^\]]+)\]", readelf.stdout)
    if not match:
        fail(f"no SONAME in {lib}")
    want = f"librgpot.so.{soversion}"
    if match.group(1) != want:
        fail(f"SONAME is {match.group(1)}, expected {want}")
    print(f"SONAME {match.group(1)}")


def check_install_name(lib: Path, soversion: int) -> None:
    try:
        otool = subprocess.run(
            ["otool", "-D", str(lib)],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print(f"soversion: otool is not on PATH, skipped install-name check for {lib.name}")
        return
    if otool.returncode != 0:
        fail(f"otool -D failed for {lib}: {otool.stderr.strip()}")
    names = [
        line.strip()
        for line in otool.stdout.splitlines()
        if line.strip() and not line.strip().endswith(":")
    ]
    if not names:
        fail(f"no install name in {lib}")
    want = f"librgpot.{soversion}.dylib"
    got = Path(names[0]).name
    if got != want:
        fail(f"install name is {names[0]}, expected {want}")
    print(f"install name {want}")


def check_build_files(soversion: int) -> None:
    meson = MESON.read_text()
    if "soversion: rgpot_cxx_soversion" not in meson:
        fail("CppCore/meson.build must set librgpot soversion from rgpot_cxx_soversion")
    if meson.count("soversion: '3'") != 1:
        fail("librgpot_mpi is the only library that keeps soversion 3")
    if "kCxxSoversion = " not in meson:
        fail("CppCore/meson.build must parse kCxxSoversion")
    cmake = CMAKE.read_text()
    if "SOVERSION ${RGPOT_CXX_SOVERSION}" not in cmake:
        fail("CMakeLists.txt must set the rgpot SOVERSION from RGPOT_CXX_SOVERSION")
    if "kCxxSoversion = ([0-9]+)" not in cmake:
        fail("CMakeLists.txt must parse kCxxSoversion")
    if not re.search(r"rgpot_mpi PROPERTIES[\s\S]{0,200}SOVERSION 3", cmake):
        fail("librgpot_mpi must keep SOVERSION 3")
    for rel in (
        "scripts/rgpot_repair_wheel.sh",
        "scripts/rgpot_sync_soname_copies.sh",
        "pyproject.toml",
    ):
        text = (ROOT / rel).read_text()
        if "librgpot.so.3" in text:
            fail(f"{rel} still hardcodes librgpot.so.3")
    print(f"soversion {soversion}: meson, cmake, and wheel scripts agree")


STAMP_RE = re.compile(
    r"RGPOT_CXX_LAYOUT_STAMP tag=(\S+) hash=([0-9a-fA-F]+)"
    r" hash_dec=(\d+) features=(\d+)"
)


def check_stamp(binary: Path, table: list[tuple[str, int]]) -> None:
    if not binary.is_file():
        fail(f"layout stamp printer not found: {binary}")
    run = subprocess.run(
        [str(binary)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = (run.stdout or "") + (run.stderr or "")
    print(output.strip())
    if run.returncode != 0:
        fail(f"layout stamp printer exited {run.returncode}")
    match = STAMP_RE.search(output)
    if not match:
        fail("layout stamp printer did not print RGPOT_CXX_LAYOUT_STAMP")
    tag, hashed, decimal, features = match.groups()
    if int(hashed, 16) != int(decimal):
        fail(f"stamp hash {hashed} does not match hash_dec {decimal}")
    if int(features) != 0:
        print(f"cxx layout stamp {tag} features={features}; cache build, hash not enforced")
        return
    known = {row_tag: row_hash for row_tag, row_hash in table}
    if tag not in known:
        print(
            f"cxx layout stamp {tag} hash {hashed} is not in "
            "kCxxSoversionTable; not enforced"
        )
        return
    if known[tag] != int(hashed, 16):
        fail(
            f"layout hash for {tag} is {hashed}, table has {known[tag]:016x}. "
            "Bump kCxxSoversion and record the new hash."
        )
    print(f"cxx layout stamp {tag} hash {hashed} matches the table")


def main() -> None:
    layout_text = LAYOUT.read_text()
    soversion = parse_soversion(layout_text)
    table = parse_table(layout_text)
    if len(sys.argv) > 1 and int(sys.argv[1]) != soversion:
        fail(
            f"meson parsed soversion {sys.argv[1]} but kCxxSoversion is {soversion}"
        )
    rows = parse_ledger(LEDGER.read_text())
    for index in range(1, len(rows)):
        if rows[index][0] <= rows[index - 1][0]:
            fail("ledger soversions must increase")
    current = rows[-1]
    live = fingerprint(layout_text)
    if current[0] != soversion:
        fail(
            f"ledger soversion {current[0]} does not match kCxxSoversion {soversion}"
        )
    if current[1] != live:
        fail(
            "C++ layout stamp inputs changed and the soversion ledger does not "
            f"record them.\nLive fingerprint {live}\n"
            "Bump kCxxSoversion, replace kCxxSoversionTable from the CI "
            "stamp lines, and append a ledger line. Do not edit a shipped line."
        )
    check_against_base(rows, layout_text)
    check_build_files(soversion)
    if len(sys.argv) > 2 and sys.argv[2] != "-":
        check_soname(Path(sys.argv[2]), soversion)
    if len(sys.argv) > 3:
        check_stamp(Path(sys.argv[3]), table)
    tags = ", ".join(tag for tag, _ in table) or "none yet"
    print(f"cxx soversion {soversion} platforms [{tags}]")


if __name__ == "__main__":
    main()

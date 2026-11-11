#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Line, region, and branch coverage for the touched production files.

Nightly is date-pinned and passes -Z coverage-options=branch.
Rust 1.88 reports lines and regions. No file in scope is excluded.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

NIGHTLY = "nightly-2026-11-10"
STABLE = "1.88.0"
FILES = (
    "src/methods/lattice_search.rs",
    "examples/lj_requench_xyz.rs",
)
FIXTURE = "tests/fixtures/lj104_gm.xyz"


def run(cmd: list[str], env: dict[str, str] | None = None, cwd: Path | None = None) -> str:
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env, cwd=cwd)
    if proc.returncode != 0:
        raise SystemExit((proc.stderr or proc.stdout or "command failed")[-2000:])
    return proc.stdout


def tool(toolchain: str, name: str) -> str:
    sysroot = run(["rustup", "run", toolchain, "rustc", "--print", "sysroot"]).strip()
    path = Path(sysroot) / "lib" / "rustlib" / "x86_64-unknown-linux-gnu" / "bin" / name
    if not path.is_file():
        raise SystemExit(f"missing {name} for {toolchain}")
    return str(path)


def bindir(toolchain: str) -> Path:
    cargo = run(["rustup", "which", "--toolchain", toolchain, "cargo"]).strip()
    return Path(cargo).parent


def version(toolchain: str, prog: str) -> str:
    return run([str(bindir(toolchain) / prog), "-V"]).strip()


def sanitize(text: str) -> str:
    return re.sub(r"(^|\s)(/[^\s]+)", r"\1PATH", text)


def pct(hit: str, total: str) -> float:
    if total == "0":
        return 100.0
    return 100.0 * int(hit) / int(total)


def parse_report(text: str) -> dict[str, dict[str, str]]:
    found: dict[str, dict[str, str]] = {}
    pat = re.compile(
        r"(\d+)\s+(\d+)\s+[\d.]+%\s+"
        r"(\d+)\s+(\d+)\s+[\d.]+%\s+"
        r"(\d+)\s+(\d+)\s+([\d.]+)%\s+"
        r"(\d+)\s+(\d+)\s+([\d.]+%|-)"
    )
    for line in text.splitlines():
        for name in FILES:
            if Path(name).name not in line:
                continue
            match = pat.search(line)
            if not match:
                continue
            lines_total = int(match.group(5))
            lines_missed = int(match.group(6))
            branches_total = int(match.group(8))
            branches_missed = int(match.group(9))
            found[name] = {
                "lines": f"{lines_total - lines_missed}/{lines_total}",
                "line_pct": match.group(7),
                "branches": f"{branches_total - branches_missed}/{branches_total}",
                "branch_pct": match.group(10),
            }
    return found


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    out_dir = root / "docs" / "prereg" / "lj104-sota"
    lines = ["Copyright (c) anneal developers", ""]
    for toolchain in (NIGHTLY, STABLE):
        lines.append(f"Toolchain {toolchain}.")
        lines.append(version(toolchain, "rustc") + ".")
        lines.append(version(toolchain, "cargo") + ".")
    # Branch coverage on the pinned nightly, then lines and regions on 1.88.
    reports = []
    for toolchain, flags, label in (
        (NIGHTLY, "-C instrument-coverage -Z coverage-options=branch", "branch"),
        (STABLE, "-C instrument-coverage", "lines-regions"),
    ):
        prof_dir = root.parent / "tmp" / f"cov-{label}"
        prof_dir.mkdir(parents=True, exist_ok=True)
        for old in prof_dir.glob("*.profraw"):
            old.unlink()
        env = os.environ.copy()
        env["RUSTFLAGS"] = flags
        env["LLVM_PROFILE_FILE"] = str(prof_dir / "prof-%p-%m.profraw")
        target = root.parent / "tmp" / f"cov-target-{label}"
        env["CARGO_TARGET_DIR"] = str(target)
        env["PATH"] = str(bindir(toolchain)) + ":" + env.get("PATH", "")
        cargo = str(bindir(toolchain) / "cargo")
        run([cargo, "test", "--offline", "--lib", "--", "lattice_search"], env=env, cwd=root)
        run([cargo, "test", "--offline", "--example", "lj_requench_xyz"], env=env, cwd=root)
        fixture = root / FIXTURE
        run(
            [cargo, "run", "--offline", "--example", "lj_requench_xyz", "--", str(fixture)],
            env=env,
            cwd=root,
        )
        profdata = prof_dir / "coverage.profdata"
        run([tool(toolchain, "llvm-profdata"), "merge", "-sparse", "-o", str(profdata), *map(str, prof_dir.glob("*.profraw"))])
        # Objects: the lib test binary and the example.
        objects = [
            p
            for p in (target / "debug" / "deps").glob("anneal_core-*")
            if p.suffix == "" and p.is_file()
        ]
        objects.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        example = target / "debug" / "examples" / "lj_requench_xyz"
        cmd = [
            tool(toolchain, "llvm-cov"),
            "report",
            "--instr-profile",
            str(profdata),
            "--show-region-summary",
        ]
        if label == "branch":
            cmd.append("--show-branch-summary")
        if objects:
            cmd.append(f"--object={objects[0]}")
        if example.is_file():
            cmd.append(f"--object={example}")
        cmd.append("--sources")
        cmd.extend(str(root / name) for name in FILES)
        report = run(cmd)
        reports.append((label, toolchain, report))
        parsed = parse_report(report)
        lines.append(f"Mode {label}.")
        for name in FILES:
            row = parsed.get(name)
            if row is None:
                raise SystemExit(f"{label} report missed {name}")
            lines.append(f"File {Path(name).name}.")
            lines.append(f"Lines {row['lines']} at {row['line_pct']} percent.")
            lines.append(f"Branches {row['branches']} at {row['branch_pct']}.")
    branch_rows = parse_report(reports[0][2])
    line_rows = parse_report(reports[1][2])
    for name in FILES:
        b = branch_rows[name]["lines"].split("/")
        s = line_rows[name]["lines"].split("/")
        gap = abs(pct(*b) - pct(*s))
        if gap > 0.5:
            lines.append(
                f"File {Path(name).name} line rates differ by {gap:.2f} points."
            )
            lines.append("The nightly binary and the 1.88 binary instrument different builds.")
    lines.append("")
    text = "\n".join(lines)
    # Drop absolute paths before the note is committed.
    text = sanitize(text)
    (out_dir / "COVERAGE.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    sys.exit(main() or 0)

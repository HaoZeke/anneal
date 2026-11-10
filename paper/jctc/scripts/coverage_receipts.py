#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Write per-file coverage receipts from two llvm-cov JSON exports.

The line export is produced by rustc 1.88.0.
The branch export is produced by a date-pinned nightly with
-Zcoverage-options=branch. This script does not write a toolchain file.
Branch totals are omitted rather than reported as 0/0 when the export
has no branch counters.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def version(toolchain: str, tool: str) -> str:
    return subprocess.check_output(
        ["rustup", "run", toolchain, tool, "-V"], text=True
    ).strip()


def load(path: Path) -> dict[str, dict[str, tuple[int, int]]]:
    data = json.loads(path.read_text())
    files: dict[str, dict[str, tuple[int, int]]] = {}
    for entry in data.get("data", []):
        for item in entry.get("files", []):
            name = item.get("filename", "")
            totals = item.get("summary", {}).get("totals", {})
            files[name] = {
                key: (
                    int(totals.get(key, {}).get("covered", 0)),
                    int(totals.get(key, {}).get("count", 0)),
                )
                for key in ("lines", "regions", "branches")
            }
    return files


def percent(covered: int, count: int) -> str:
    if count == 0:
        return "none"
    return f"{100.0 * covered / count:.2f}"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 4:
        raise SystemExit(
            "usage: coverage_receipts.py LINES_JSON BRANCH_JSON OUTDIR FILELIST"
        )
    lines = load(Path(args[0]))
    branches = load(Path(args[1]))
    out = Path(args[2])
    out.mkdir(parents=True, exist_ok=True)
    wanted = [line.strip() for line in Path(args[3]).read_text().splitlines() if line.strip()]
    pinned = "1.88.0"
    nightly = "nightly-2026-11-01"
    header = [
        f"rustc_188 {version(pinned, 'rustc')}",
        f"cargo_188 {version(pinned, 'cargo')}",
        f"rustc_nightly {version(nightly, 'rustc')}",
        f"cargo_nightly {version(nightly, 'cargo')}",
        "branch_flag -Zcoverage-options=branch",
        "branch_toolchain nightly-2026-11-01",
    ]
    for name in wanted:
        line = lines.get(name)
        branch = branches.get(name)
        text = list(header)
        text.append(f"file {name}")
        if line is None:
            text.append("lines_188 absent")
            text.append("regions_188 absent")
        else:
            covered, count = line["lines"]
            text.append(f"lines_188 {covered}/{count} {percent(covered, count)}")
            covered, count = line["regions"]
            text.append(f"regions_188 {covered}/{count} {percent(covered, count)}")
        if branch is None:
            text.append("lines_nightly absent")
            text.append("regions_nightly absent")
            text.append("branches absent")
        else:
            covered, count = branch["lines"]
            text.append(f"lines_nightly {covered}/{count} {percent(covered, count)}")
            covered, count = branch["regions"]
            text.append(f"regions_nightly {covered}/{count} {percent(covered, count)}")
            covered, count = branch["branches"]
            if count == 0:
                text.append("branches unavailable")
            else:
                text.append(f"branches {covered}/{count} {percent(covered, count)}")
        if line and branch and line["lines"][1] and branch["lines"][1]:
            left = 100.0 * line["lines"][0] / line["lines"][1]
            right = 100.0 * branch["lines"][0] / branch["lines"][1]
            gap = abs(left - right)
            text.append(f"line_gap {gap:.2f}")
            if gap > 0.5:
                text.append("line_gap_reason the nightly and 1.88 runs instrument different counters")
        safe = name.replace("/", "__")
        (out / f"{safe}.txt").write_text("\n".join(text) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

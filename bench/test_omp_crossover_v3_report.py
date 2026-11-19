#!/usr/bin/env python3
"""Require the committed v3 report to match a fresh render byte for byte."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    harness = repo / args.harness
    source_json = repo / args.json
    source_report = repo / args.report
    expected = source_report.read_bytes()
    with tempfile.TemporaryDirectory() as directory:
        rendered = Path(directory) / "report.md"
        subprocess.run(
            [
                sys.executable,
                str(harness),
                "render",
                "--input-json",
                str(source_json),
                "--output-report",
                str(rendered),
            ],
            cwd=repo,
            check=True,
        )
        actual = rendered.read_bytes()
    if actual != expected:
        raise SystemExit("committed report differs from a fresh JSON render")


if __name__ == "__main__":
    main()

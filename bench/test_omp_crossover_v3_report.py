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
    parser.add_argument("--analysis-report", type=Path)
    parser.add_argument("--clarified-report", type=Path)
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
    if args.analysis_report is not None:
        analysis_report = repo / args.analysis_report
        expected_analysis = analysis_report.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            rendered = Path(directory) / "analysis-report.md"
            subprocess.run(
                [
                    sys.executable,
                    str(harness),
                    "analyze",
                    "--input-json",
                    str(source_json),
                    "--output-report",
                    str(rendered),
                ],
                cwd=repo,
                check=True,
            )
            actual_analysis = rendered.read_bytes()
        if actual_analysis != expected_analysis:
            raise SystemExit("corrected report differs from a fresh analysis")
    if args.clarified_report is not None:
        clarified_report = repo / args.clarified_report
        expected_clarified = clarified_report.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            rendered = Path(directory) / "clarified-report.md"
            subprocess.run(
                [
                    sys.executable,
                    str(harness),
                    "analyze-v3e",
                    "--input-json",
                    str(source_json),
                    "--output-report",
                    str(rendered),
                ],
                cwd=repo,
                check=True,
            )
            actual_clarified = rendered.read_bytes()
        if actual_clarified != expected_clarified:
            raise SystemExit("clarified report differs from a fresh analysis")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Require each recorded arm tree to belong to its recorded commit."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

REPORT_NAME = "rgpot_omp_crossover_v3_report.md"


def arms_in_report(text: str) -> set[str]:
    lines = text.splitlines()
    start = next((index for index, line in enumerate(lines) if line.startswith("|Arm|")), None)
    if start is None:
        raise SystemExit("report arm table is missing")
    names: set[str] = set()
    for row in lines[start + 1 :]:
        if row.startswith("|---"):
            continue
        if not row.startswith("|"):
            break
        name = row.split("|")[1].strip()
        if name:
            names.add(name)
    if not names:
        raise SystemExit("report arm table is empty")
    return names


def resolved_commit_tree(repo: Path, arm: str, sha: str) -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "--verify", f"{sha}^{{commit}}"],
        cwd=repo,
        text=True,
        capture_output=True,
    )
    if commit.returncode != 0:
        raise SystemExit(f"{arm}: commit does not resolve: {sha}")
    return subprocess.run(
        ["git", "rev-parse", f"{sha}^{{tree}}"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def tree_file(repo: Path, tree: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{tree}:{path}"],
        cwd=repo,
        text=True,
        capture_output=True,
    )
    return result.stdout if result.returncode == 0 else ""


def tree_threshold(repo: Path, arm: str, tree: str) -> int:
    options = tree_file(repo, tree, "meson_options.txt")
    match = re.search(
        r"option\(\s*'team_min_atoms'.*?value:\s*([0-9]+)",
        options,
        flags=re.DOTALL,
    )
    if match is None:
        slices = tree_file(repo, tree, "CppCore/rgpot/Slices.hpp")
        match = re.search(r"kTeamMinAtoms\s*=\s*([0-9]+)", slices)
    if match is None:
        raise SystemExit(f"{arm}: tree {tree} has no kTeamMinAtoms value")
    return int(match.group(1))


def check_provenance(repo: Path, payload: dict, report_text: str) -> None:
    record = payload["metadata"]["arms"]
    record_names = [arm["name"] for arm in record]
    report_names = arms_in_report(report_text)
    if len(record_names) != len(report_names) or set(record_names) != report_names:
        missing = sorted(report_names.difference(record_names))
        extra = sorted(set(record_names).difference(report_names))
        raise SystemExit(f"arm set mismatch: {', '.join(missing + extra)}")
    for arm in record:
        name = arm["name"]
        recorded = arm.get("tree")
        actual = resolved_commit_tree(repo, name, arm["sha"])
        if recorded != actual:
            raise SystemExit(f"{name}: recorded tree {recorded} does not match commit tree {actual}")
        actual_threshold = tree_threshold(repo, name, recorded)
        if arm["threshold"] != actual_threshold:
            raise SystemExit(
                f"{name}: recorded threshold {arm['threshold']} "
                f"does not match tree threshold {actual_threshold}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--json-relative", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    json_path = args.json
    if args.json_relative is not None:
        json_path = json_path / args.json_relative
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    report_path = args.report or json_path.with_name(REPORT_NAME)
    if not report_path.is_absolute():
        report_path = args.repo / report_path
    if not report_path.is_file():
        raise SystemExit(f"report is missing: {report_path.name}")
    check_provenance(args.repo, payload, report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

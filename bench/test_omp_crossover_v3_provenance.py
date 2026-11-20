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


def has_git(repo: Path) -> bool:
    if not (repo / ".git").exists():
        return False
    probe = subprocess.run(
        ["git", "rev-parse", "--git-dir"],
        cwd=repo,
        text=True,
        capture_output=True,
    )
    return probe.returncode == 0


def recorded_commit_trees(record: list[dict]) -> dict[str, str]:
    out: dict[str, str] = {}
    for arm in record:
        sha = arm.get("commit_sha")
        commit_tree = arm.get("commit_tree")
        if sha and commit_tree and sha not in out:
            out[sha] = commit_tree
    return out


def recorded_tree_thresholds(record: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for arm in record:
        tree = arm.get("commit_tree") or arm.get("tree")
        threshold = arm.get("tree_threshold")
        if tree and threshold is not None and tree not in out:
            out[tree] = int(threshold)
    return out


def resolved_commit_tree(
    repo: Path, arm: str, sha: str, fallback_trees: dict[str, str]
) -> str:
    if has_git(repo):
        commit = subprocess.run(
            ["git", "rev-parse", "--verify", f"{sha}^{{commit}}"],
            cwd=repo,
            text=True,
            capture_output=True,
        )
        if commit.returncode == 0:
            return subprocess.run(
                ["git", "rev-parse", f"{sha}^{{tree}}"],
                cwd=repo,
                check=True,
                text=True,
                capture_output=True,
            ).stdout.strip()
        raise SystemExit(f"{arm}: commit does not resolve: {sha}")
    actual = fallback_trees.get(sha)
    if not actual:
        raise SystemExit(f"{arm}: commit does not resolve: {sha}")
    return actual


def tree_file(repo: Path, tree: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{tree}:{path}"],
        cwd=repo,
        text=True,
        capture_output=True,
    )
    return result.stdout if result.returncode == 0 else ""


def tree_threshold(
    repo: Path, arm: str, tree: str, fallback_thresholds: dict[str, int]
) -> int:
    if has_git(repo):
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
    actual = fallback_thresholds.get(tree)
    if actual is None:
        raise SystemExit(f"{arm}: tree {tree} has no kTeamMinAtoms value")
    return actual


def check_provenance(repo: Path, payload: dict, report_text: str) -> None:
    record = payload["metadata"]["arms"]
    record_names = [arm["name"] for arm in record]
    report_names = arms_in_report(report_text)
    if len(record_names) != len(report_names) or set(record_names) != report_names:
        missing = sorted(report_names.difference(record_names))
        extra = sorted(set(record_names).difference(report_names))
        missing_text = ", ".join(missing) if missing else "(none)"
        extra_text = ", ".join(extra) if extra else "(none)"
        raise SystemExit(f"arm set mismatch: missing {missing_text}; extra {extra_text}")
    fallback_trees = recorded_commit_trees(record)
    fallback_thresholds = recorded_tree_thresholds(record)
    for arm in record:
        name = arm["name"] if isinstance(arm.get("name"), str) and arm.get("name") else "<unnamed>"
        if "sha" not in arm:
            raise SystemExit(f"{name}: arm record is missing required key sha")
        recorded = arm.get("tree")
        actual = resolved_commit_tree(repo, name, arm["sha"], fallback_trees)
        if recorded != actual:
            raise SystemExit(f"{name}: recorded tree {recorded} does not match commit tree {actual}")
        actual_threshold = tree_threshold(repo, name, recorded, fallback_thresholds)
        if arm["threshold"] != actual_threshold:
            raise SystemExit(
                f"{name}: recorded threshold {arm['threshold']} "
                f"does not match tree threshold {actual_threshold}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("bench/results/rgpot_omp_crossover_v3_results.json"),
    )
    parser.add_argument("--json-relative", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    json_path = args.json if args.json.is_absolute() else repo / args.json
    if args.json_relative is not None:
        json_path = json_path / args.json_relative
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if args.report is None:
        report_path = json_path.with_name(REPORT_NAME)
    elif args.report.is_absolute():
        report_path = args.report
    else:
        report_path = repo / args.report
    if not report_path.is_file():
        raise SystemExit(f"report is missing: {report_path.name}")
    check_provenance(repo, payload, report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

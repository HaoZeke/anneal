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


def resolved_commit_tree(repo: Path, arm: str, sha: str, fallback_tree: str) -> str:
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
    return fallback_tree


def tree_file(repo: Path, tree: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{tree}:{path}"],
        cwd=repo,
        text=True,
        capture_output=True,
    )
    return result.stdout if result.returncode == 0 else ""


def tree_threshold(repo: Path, arm: str, tree: str, fallback_threshold: int) -> int:
    if has_git(repo):
        present = subprocess.run(
            ["git", "cat-file", "-e", f"{tree}^{{tree}}"],
            cwd=repo,
            text=True,
            capture_output=True,
        )
        if present.returncode == 0:
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
    return fallback_threshold


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
    for arm in record:
        name = arm["name"] if isinstance(arm.get("name"), str) and arm.get("name") else "<unnamed>"
        if "sha" not in arm:
            raise SystemExit(f"{name}: arm record is missing required key sha")
        if "commit_sha" not in arm:
            raise SystemExit(f"{name}: arm record is missing required key commit_sha")
        if arm["sha"] != arm["commit_sha"]:
            raise SystemExit(
                f"{name}: recorded sha {arm['sha']} does not match commit_sha {arm['commit_sha']}"
            )
        if "commit_tree" not in arm:
            raise SystemExit(f"{name}: arm record is missing required key commit_tree")
        if "tree" not in arm:
            raise SystemExit(f"{name}: arm record is missing required key tree")
        if arm["tree"] != arm["commit_tree"]:
            raise SystemExit(
                f"{name}: recorded tree {arm['tree']} does not match commit_tree {arm['commit_tree']}"
            )
        if "tree_threshold" not in arm:
            raise SystemExit(f"{name}: arm record is missing required key tree_threshold")
        if "threshold" not in arm:
            raise SystemExit(f"{name}: arm record is missing required key threshold")
        if arm["threshold"] != arm["tree_threshold"]:
            raise SystemExit(
                f"{name}: recorded threshold {arm['threshold']} "
                f"does not match tree_threshold {arm['tree_threshold']}"
            )
        recorded = arm.get("tree")
        actual = resolved_commit_tree(repo, name, arm["sha"], arm["commit_tree"])
        if recorded != actual:
            raise SystemExit(f"{name}: recorded tree {recorded} does not match commit tree {actual}")
        actual_threshold = tree_threshold(repo, name, recorded, int(arm["tree_threshold"]))
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

#!/usr/bin/env python3
"""Require each recorded arm tree to belong to its recorded commit."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

REPORT_NAME = "rgpot_omp_crossover_v3_report.md"
PINS_PATH = Path(__file__).with_name("omp_crossover_v3_pins.json")


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

def is_shallow_git(repo: Path) -> bool:
    if not has_git(repo):
        return False
    probe = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"],
        cwd=repo,
        text=True,
        capture_output=True,
    )
    return probe.returncode == 0 and probe.stdout.strip() == "true"


def has_known_recorded_commit(repo: Path, recorded_trees: dict[str, str]) -> bool:
    for candidate in recorded_trees:
        probe = subprocess.run(
            ["git", "cat-file", "-e", f"{candidate}^{{commit}}"],
            cwd=repo,
            text=True,
            capture_output=True,
        )
        if probe.returncode == 0:
            return True
    return False


def require_non_empty_string(arm: str, field: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise SystemExit(f"{arm}: arm record has invalid {field}: {value!r}")
    return value


def require_int_not_bool(arm: str, field: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise SystemExit(f"{arm}: arm record has invalid {field}: {value!r}")
    return value


def recorded_commit_trees(record: list[dict]) -> dict[str, str]:
    out: dict[str, str] = {}
    for arm in record:
        sha = arm.get("commit_sha")
        commit_tree = arm.get("commit_tree")
        if isinstance(sha, str) and sha and isinstance(commit_tree, str) and commit_tree and sha not in out:
            out[sha] = commit_tree
    return out


def resolved_commit_tree(
    repo: Path,
    arm: str,
    sha: str,
    recorded_trees: dict[str, str],
    pins: dict[str, dict[str, str]],
) -> tuple[str, bool]:
    resolved: str | None = None
    if has_git(repo):
        commit = subprocess.run(
            ["git", "rev-parse", "--verify", f"{sha}^{{commit}}"],
            cwd=repo,
            text=True,
            capture_output=True,
        )
        if commit.returncode == 0:
            resolved = subprocess.run(
                ["git", "rev-parse", f"{sha}^{{tree}}"],
                cwd=repo,
                check=True,
                text=True,
                capture_output=True,
            ).stdout.strip()
        elif has_known_recorded_commit(repo, recorded_trees) and not is_shallow_git(repo):
            raise SystemExit(f"{arm}: commit does not resolve: {sha}")
    arm_pins = pins.get(arm)
    pinned: str | None = None
    if isinstance(arm_pins, dict):
        raw_pin = arm_pins.get(sha)
        if isinstance(raw_pin, str) and raw_pin:
            pinned = raw_pin
    # A resolved object still has to be the commit named by the pin. A full
    # history may disagree with a forged pin value; a shallow one may not.
    if resolved is not None:
        if isinstance(arm_pins, dict) and pinned is None:
            raise SystemExit(f"{arm}: commit is not pinned: {sha}")
        if pinned != resolved:
            raise SystemExit(
                f"{arm}: pinned tree {pinned} does not match commit tree {resolved}"
            )
        return resolved, True
    if not isinstance(arm_pins, dict):
        raise SystemExit(f"{arm}: pinned manifest has no arm entry")
    if pinned is None:
        raise SystemExit(f"{arm}: commit does not resolve and has no pin: {sha}")
    return pinned, False


def tree_file(repo: Path, tree: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{tree}:{path}"],
        cwd=repo,
        text=True,
        capture_output=True,
    )
    return result.stdout if result.returncode == 0 else ""


def tree_threshold(
    repo: Path, arm: str, tree: str, pinned_threshold: int, commit_present: bool
) -> int:
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
            parsed = int(match.group(1))
            if parsed != pinned_threshold:
                raise SystemExit(
                    f"{arm}: pinned kTeamMinAtoms {pinned_threshold} "
                    f"does not match tree threshold {parsed}"
                )
            return parsed
    if commit_present:
        raise SystemExit(f"{arm}: resolved commit tree is unavailable: {tree}")
    return pinned_threshold


def check_provenance(
    repo: Path,
    payload: dict,
    report_text: str,
    pins: dict[str, dict[str, str]],
    thresholds: dict[str, int],
) -> None:
    record = payload["metadata"]["arms"]
    record_names = [arm["name"] for arm in record]
    report_names = arms_in_report(report_text)
    if len(record_names) != len(report_names) or set(record_names) != report_names:
        missing = sorted(report_names.difference(record_names))
        extra = sorted(set(record_names).difference(report_names))
        missing_text = ", ".join(missing) if missing else "(none)"
        extra_text = ", ".join(extra) if extra else "(none)"
        raise SystemExit(f"arm set mismatch: missing {missing_text}; extra {extra_text}")
    recorded_trees = recorded_commit_trees(record)
    for arm in record:
        name = arm["name"] if isinstance(arm.get("name"), str) and arm.get("name") else "<unnamed>"
        if "sha" not in arm:
            raise SystemExit(f"{name}: arm record is missing required key sha")
        if "commit_sha" not in arm:
            raise SystemExit(f"{name}: arm record is missing required key commit_sha")
        sha = require_non_empty_string(name, "sha", arm["sha"])
        commit_sha = require_non_empty_string(name, "commit_sha", arm["commit_sha"])
        if sha != commit_sha:
            raise SystemExit(
                f"{name}: recorded sha {arm['sha']} does not match commit_sha {arm['commit_sha']}"
            )
        if "commit_tree" not in arm:
            raise SystemExit(f"{name}: arm record is missing required key commit_tree")
        if "tree" not in arm:
            raise SystemExit(f"{name}: arm record is missing required key tree")
        tree = require_non_empty_string(name, "tree", arm["tree"])
        commit_tree = require_non_empty_string(name, "commit_tree", arm["commit_tree"])
        if tree != commit_tree:
            raise SystemExit(
                f"{name}: recorded tree {tree} does not match commit_tree {commit_tree}"
            )
        if "tree_threshold" not in arm:
            raise SystemExit(f"{name}: arm record is missing required key tree_threshold")
        if "threshold" not in arm:
            raise SystemExit(f"{name}: arm record is missing required key threshold")
        require_int_not_bool(name, "tree_threshold", arm["tree_threshold"])
        threshold = require_int_not_bool(name, "threshold", arm["threshold"])
        if arm["threshold"] != arm["tree_threshold"]:
            raise SystemExit(
                f"{name}: recorded threshold {arm['threshold']} "
                f"does not match tree_threshold {arm['tree_threshold']}"
            )
        recorded = tree
        actual, commit_present = resolved_commit_tree(repo, name, sha, recorded_trees, pins)
        if recorded != actual:
            raise SystemExit(f"{name}: recorded tree {recorded} does not match commit tree {actual}")
        if name not in thresholds:
            raise SystemExit(f"{name}: pinned manifest has no kTeamMinAtoms")
        actual_threshold = tree_threshold(
            repo, name, recorded, thresholds[name], commit_present
        )
        if threshold != actual_threshold:
            raise SystemExit(
                f"{name}: recorded threshold {threshold} "
                f"does not match tree threshold {actual_threshold}"
            )


def parse_pins(raw: object) -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    if not isinstance(raw, dict):
        raise SystemExit("pinned manifest must be a JSON object")
    threshold_raw = raw.get("kTeamMinAtoms")
    if not isinstance(threshold_raw, dict):
        raise SystemExit("pinned manifest is missing kTeamMinAtoms")
    thresholds: dict[str, int] = {}
    for arm, value in threshold_raw.items():
        if not isinstance(arm, str) or isinstance(value, bool) or not isinstance(value, int):
            raise SystemExit(f"pinned manifest has invalid kTeamMinAtoms for {arm!r}")
        thresholds[arm] = value
    commits: dict[str, dict[str, str]] = {}
    for arm, entry in raw.items():
        if arm == "kTeamMinAtoms":
            continue
        if not isinstance(arm, str) or not isinstance(entry, dict):
            raise SystemExit("pinned manifest must map arms to objects")
        arm_commits: dict[str, str] = {}
        for sha, tree in entry.items():
            if not isinstance(sha, str) or not sha or not isinstance(tree, str) or not tree:
                raise SystemExit(f"{arm}: pinned manifest has an invalid commit entry")
            arm_commits[sha] = tree
        commits[arm] = arm_commits
    return commits, thresholds


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
    parser.add_argument("--pins", type=Path, default=PINS_PATH)
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
    pins_path = args.pins if args.pins.is_absolute() else repo / args.pins
    if not pins_path.is_file():
        raise SystemExit(f"pinned manifest is missing: {pins_path}")
    commits, thresholds = parse_pins(json.loads(pins_path.read_text(encoding="utf-8")))
    check_provenance(
        repo, payload, report_path.read_text(encoding="utf-8"), commits, thresholds
    )


if __name__ == "__main__":
    main()

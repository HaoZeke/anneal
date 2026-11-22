#!/usr/bin/env python3
"""Require each recorded arm tree to belong to its recorded commit."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPORT_NAME = "rgpot_omp_crossover_v3_report.md"
PINS_PATH = Path(__file__).with_name("omp_crossover_v3_pins.json")
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
HISTORY_ABSENT_REPORTED = False


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
        if not name:
            raise SystemExit("report arm table has an empty arm name")
        if any(char.isspace() for char in name):
            raise SystemExit(f"report arm name has whitespace: {name!r}")
        if name in names:
            raise SystemExit(f"report arm table has duplicate arm: {name}")
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

def has_known_commit(repo: Path, sha: str) -> bool:
    probe = subprocess.run(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
        cwd=repo,
        text=True,
        capture_output=True,
    )
    return probe.returncode == 0


def load_canonical_commit_shas(record: list[dict]) -> set[str]:
    out: set[str] = set()
    for arm in record:
        sha = arm.get("commit_sha")
        if isinstance(sha, str) and HEX40_RE.fullmatch(sha):
            out.add(sha)
    return out


def repo_has_history(repo: Path, canonical_commits: set[str]) -> bool:
    if not has_git(repo):
        return False
    for sha in canonical_commits:
        if has_known_commit(repo, sha):
            return True
    return False


def note_history_absent_once() -> None:
    global HISTORY_ABSENT_REPORTED
    if HISTORY_ABSENT_REPORTED:
        return
    HISTORY_ABSENT_REPORTED = True
    print(
        "provenance: repository history is absent; using pinned manifest fallback",
        file=sys.stderr,
    )


def require_hex40(arm: str, field: str, value: object) -> str:
    if not isinstance(value, str) or not HEX40_RE.fullmatch(value):
        raise SystemExit(f"{arm}: arm record has invalid {field}: {value!r}")
    return value


def require_arm_name(value: object, source: str) -> str:
    if not isinstance(value, str) or not value:
        raise SystemExit(f"{source}: arm name is invalid: {value!r}")
    if any(char.isspace() for char in value):
        raise SystemExit(f"{source}: arm name has whitespace: {value!r}")
    return value


def require_pin_hex40(arm: str, role: str, value: object) -> str:
    if not isinstance(value, str) or any(char.isspace() for char in value) or not HEX40_RE.fullmatch(value):
        raise SystemExit(f"{arm}: pinned manifest has invalid {role}: {value!r}")
    return value


def require_positive_int(name: str, value: object, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SystemExit(f"{source} has invalid {name}: {value!r}")
    return value


def require_configured_team_min_atoms(payload: dict) -> int:
    identity = payload.get("identity")
    if not isinstance(identity, dict):
        raise SystemExit("identity block is missing")
    return require_positive_int(
        "configured_team_min_atoms",
        identity.get("configured_team_min_atoms"),
        "identity",
    )


def require_record_arms(payload: dict) -> list[dict]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise SystemExit("metadata block is missing")
    record = metadata.get("arms")
    if not isinstance(record, list) or not record:
        raise SystemExit("metadata arms are missing or empty")
    return record


def require_unique_record_arm_names(record: list[dict]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for arm in record:
        if not isinstance(arm, dict):
            raise SystemExit("metadata arm entry is invalid")
        name = require_arm_name(arm.get("name"), "metadata arm")
        if name in seen:
            raise SystemExit(f"metadata arms contain duplicate arm: {name}")
        seen.add(name)
        names.append(name)
    return names


def require_threshold_presence(name: str, thresholds: dict[str, int]) -> int:
    if name not in thresholds:
        raise SystemExit(f"{name}: pinned manifest has no kTeamMinAtoms")
    return thresholds[name]


def require_pin_entry(
    arm: str, sha: str, pins: dict[str, dict[str, str]], commit_present: bool
) -> str:
    arm_pins = pins.get(arm)
    if not isinstance(arm_pins, dict):
        raise SystemExit(f"{arm}: pinned manifest has no arm entry")
    if sha not in arm_pins:
        if commit_present:
            raise SystemExit(f"{arm}: commit is not pinned: {sha}")
        raise SystemExit(f"{arm}: commit does not resolve and has no pin: {sha}")
    return arm_pins[sha]


def resolve_commit_tree(
    repo: Path,
    arm: str,
    sha: str,
    pins: dict[str, dict[str, str]],
    canonical_commits: set[str],
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
        elif repo_has_history(repo, canonical_commits):
            raise SystemExit(f"{arm}: commit does not resolve: {sha}")
        else:
            note_history_absent_once()
    else:
        note_history_absent_once()
    pinned = require_pin_entry(arm, sha, pins, resolved is not None)
    if resolved is not None and pinned != resolved:
        raise SystemExit(f"{arm}: pinned tree {pinned} does not match commit tree {resolved}")
    return (resolved if resolved is not None else pinned), resolved is not None


def require_int_not_bool(arm: str, field: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise SystemExit(f"{arm}: arm record has invalid {field}: {value!r}")
    return value


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
    configured_team_min_atoms = require_configured_team_min_atoms(payload)
    record = require_record_arms(payload)
    record_names = require_unique_record_arm_names(record)
    report_names = arms_in_report(report_text)
    if len(record_names) != len(report_names) or set(record_names) != report_names:
        missing = sorted(report_names.difference(record_names))
        extra = sorted(set(record_names).difference(report_names))
        missing_text = ", ".join(missing) if missing else "(none)"
        extra_text = ", ".join(extra) if extra else "(none)"
        raise SystemExit(f"arm set mismatch: missing {missing_text}; extra {extra_text}")
    canonical_commits = load_canonical_commit_shas(record)
    for arm in record:
        name = arm["name"]
        if "sha" not in arm:
            raise SystemExit(f"{name}: arm record is missing required key sha")
        if "commit_sha" not in arm:
            raise SystemExit(f"{name}: arm record is missing required key commit_sha")
        sha = require_hex40(name, "sha", arm["sha"])
        commit_sha = require_hex40(name, "commit_sha", arm["commit_sha"])
        if sha != commit_sha:
            raise SystemExit(
                f"{name}: recorded sha {arm['sha']} does not match commit_sha {arm['commit_sha']}"
            )
        if "commit_tree" not in arm:
            raise SystemExit(f"{name}: arm record is missing required key commit_tree")
        if "tree" not in arm:
            raise SystemExit(f"{name}: arm record is missing required key tree")
        tree = require_hex40(name, "tree", arm["tree"])
        commit_tree = require_hex40(name, "commit_tree", arm["commit_tree"])
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
        if threshold < 1:
            raise SystemExit(f"{name}: arm record has invalid threshold: {threshold!r}")
        if arm["threshold"] != arm["tree_threshold"]:
            raise SystemExit(
                f"{name}: recorded threshold {arm['threshold']} "
                f"does not match tree_threshold {arm['tree_threshold']}"
            )
        if threshold != configured_team_min_atoms and name == "team-1":
            raise SystemExit(
                f"{name}: recorded threshold {threshold} "
                f"does not match identity configured_team_min_atoms {configured_team_min_atoms}"
            )
        recorded = tree
        actual, commit_present = resolve_commit_tree(repo, name, sha, pins, canonical_commits)
        if recorded != actual:
            raise SystemExit(f"{name}: recorded tree {recorded} does not match commit tree {actual}")
        pinned_threshold = require_threshold_presence(name, thresholds)
        actual_threshold = tree_threshold(
            repo, name, recorded, pinned_threshold, commit_present
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
        arm_name = require_arm_name(arm, "pinned manifest kTeamMinAtoms")
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise SystemExit(f"pinned manifest has invalid kTeamMinAtoms for {arm!r}")
        thresholds[arm_name] = value
    commits: dict[str, dict[str, str]] = {}
    for arm, entry in raw.items():
        if arm == "kTeamMinAtoms":
            continue
        arm_name = require_arm_name(arm, "pinned manifest")
        if not isinstance(entry, dict):
            raise SystemExit("pinned manifest must map arms to objects")
        arm_commits: dict[str, str] = {}
        for sha, tree in entry.items():
            pin_sha = require_pin_hex40(arm_name, "commit entry sha", sha)
            pin_tree = require_pin_hex40(arm_name, "commit entry tree", tree)
            arm_commits[pin_sha] = pin_tree
        commits[arm_name] = arm_commits
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

#!/usr/bin/env python3
"""The provenance checker must reject a record that names the wrong tree."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "bench" / "test_omp_crossover_v3_provenance.py"
SOURCE_JSON = ROOT / "bench" / "results" / "rgpot_omp_crossover_v3_results.json"
SOURCE_REPORT = ROOT / "bench" / "results" / "rgpot_omp_crossover_v3_report.md"
REPORT_NAME = "rgpot_omp_crossover_v3_report.md"


def commit_tree(sha: str) -> str:
    return subprocess.run(
        ["git", "rev-parse", f"{sha}^{{tree}}"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def object_tree(revision: str) -> str:
    return subprocess.run(
        ["git", "rev-parse", revision],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def run_checker(payload: dict) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as directory:
        record = Path(directory) / SOURCE_JSON.name
        report = Path(directory) / REPORT_NAME
        record.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        report.write_bytes(SOURCE_REPORT.read_bytes())
        return subprocess.run(
            [sys.executable, str(CHECKER), "--repo", str(ROOT), "--json", str(record)],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )


def run_bare_checker() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


def require_tree_rejection(result: subprocess.CompletedProcess[str], arm: str, recorded: str, actual: str) -> None:
    text = result.stdout + result.stderr
    if result.returncode == 0 or arm not in text or recorded not in text or actual not in text:
        raise SystemExit(f"{arm} was not rejected with recorded tree {recorded} and commit tree {actual}")


def require_exact_rejection(result: subprocess.CompletedProcess[str], expected: str) -> None:
    text = (result.stdout + result.stderr).strip()
    if result.returncode == 0 or text != expected:
        raise SystemExit(f"expected exact rejection {expected!r}, got {result.returncode} {text!r}")


def require_missing_arm(result: subprocess.CompletedProcess[str], arm: str) -> None:
    require_exact_rejection(result, f"arm set mismatch: missing {arm}; extra (none)")


def require_named_rejection(
    result: subprocess.CompletedProcess[str], missing: str, extra: str
) -> None:
    require_exact_rejection(result, f"arm set mismatch: missing {missing}; extra {extra}")


def require_missing_sha(result: subprocess.CompletedProcess[str], arm: str) -> None:
    require_exact_rejection(result, f"{arm}: arm record is missing required key sha")


def require_commit_rejection(
    result: subprocess.CompletedProcess[str], arm: str, sha: str
) -> None:
    text = result.stdout + result.stderr
    if result.returncode == 0 or arm not in text or sha not in text:
        raise SystemExit(f"unresolvable commit for {arm} was not rejected: {sha}")


def require_threshold_rejection(
    result: subprocess.CompletedProcess[str], arm: str, recorded: int, actual: int
) -> None:
    text = result.stdout + result.stderr
    if (
        result.returncode == 0
        or arm not in text
        or str(recorded) not in text
        or str(actual) not in text
    ):
        raise SystemExit(
            f"{arm} was not rejected with recorded threshold {recorded} and tree threshold {actual}"
        )


def main() -> None:
    bare = run_bare_checker()
    if bare.returncode != 0:
        raise SystemExit(f"bare provenance check failed: {bare.stderr.strip()}")

    original = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))
    accepted = run_checker(json.loads(json.dumps(original)))
    if accepted.returncode != 0:
        raise SystemExit(accepted.stderr.strip())

    arms = {arm["name"]: arm for arm in original["metadata"]["arms"]}
    team = arms["team-1"]
    stock = arms["stock"]
    team_1024 = arms["team-1024"]
    team_tree = commit_tree(team["sha"])
    stock_tree = commit_tree(stock["sha"])
    team_1024_tree = commit_tree(team_1024["sha"])

    swapped_tree = json.loads(json.dumps(original))
    swapped_team = next(arm for arm in swapped_tree["metadata"]["arms"] if arm["name"] == "team-1")
    swapped_team["tree"] = team_1024_tree
    require_tree_rejection(run_checker(swapped_tree), "team-1", team_1024_tree, team_tree)

    stock_as_team = json.loads(json.dumps(original))
    changed_team = next(arm for arm in stock_as_team["metadata"]["arms"] if arm["name"] == "team-1")
    changed_team["tree"] = stock_tree
    require_tree_rejection(run_checker(stock_as_team), "team-1", stock_tree, team_tree)

    head_as_team = json.loads(json.dumps(original))
    changed_team = next(arm for arm in head_as_team["metadata"]["arms"] if arm["name"] == "team-1")
    head_tree = object_tree("HEAD^{tree}")
    changed_team["tree"] = head_tree
    require_tree_rejection(run_checker(head_as_team), "team-1", head_tree, team_tree)

    subdirectory_as_team = json.loads(json.dumps(original))
    changed_team = next(
        arm for arm in subdirectory_as_team["metadata"]["arms"] if arm["name"] == "team-1"
    )
    subdirectory_tree = object_tree(f"{team['sha']}:CppCore")
    changed_team["tree"] = subdirectory_tree
    require_tree_rejection(
        run_checker(subdirectory_as_team), "team-1", subdirectory_tree, team_tree
    )

    swapped_commit = json.loads(json.dumps(original))
    swapped_team = next(arm for arm in swapped_commit["metadata"]["arms"] if arm["name"] == "team-1")
    swapped_team["sha"] = stock["sha"]
    require_tree_rejection(run_checker(swapped_commit), "team-1", team_tree, stock_tree)

    omitted_sha = json.loads(json.dumps(original))
    changed_team = next(arm for arm in omitted_sha["metadata"]["arms"] if arm["name"] == "team-1")
    del changed_team["sha"]
    require_missing_sha(run_checker(omitted_sha), "team-1")

    unresolved_commit = json.loads(json.dumps(original))
    changed_team = next(
        arm for arm in unresolved_commit["metadata"]["arms"] if arm["name"] == "team-1"
    )
    missing_sha = "0" * 40
    changed_team["sha"] = missing_sha
    require_commit_rejection(run_checker(unresolved_commit), "team-1", missing_sha)

    wrong_threshold = json.loads(json.dumps(original))
    changed_team = next(
        arm for arm in wrong_threshold["metadata"]["arms"] if arm["name"] == "team-1"
    )
    changed_team["threshold"] = 1024
    require_threshold_rejection(run_checker(wrong_threshold), "team-1", 1024, 1)

    missing_arm = json.loads(json.dumps(original))
    missing_arm["metadata"]["arms"] = [
        arm for arm in missing_arm["metadata"]["arms"] if arm["name"] != "team-1"
    ]
    require_missing_arm(run_checker(missing_arm), "team-1")

    renamed_arm = json.loads(json.dumps(original))
    changed_team = next(
        arm for arm in renamed_arm["metadata"]["arms"] if arm["name"] == "team-1"
    )
    changed_team["name"] = "renamed-team"
    require_named_rejection(run_checker(renamed_arm), "team-1", "renamed-team")


if __name__ == "__main__":
    main()

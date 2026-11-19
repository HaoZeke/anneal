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


def require_tree_rejection(result: subprocess.CompletedProcess[str], arm: str, recorded: str, actual: str) -> None:
    text = result.stdout + result.stderr
    if result.returncode == 0 or arm not in text or recorded not in text or actual not in text:
        raise SystemExit(f"{arm} was not rejected with recorded tree {recorded} and commit tree {actual}")


def require_missing_arm(result: subprocess.CompletedProcess[str], arm: str) -> None:
    text = result.stdout + result.stderr
    if result.returncode == 0 or arm not in text:
        raise SystemExit(f"missing arm {arm} was not rejected")


def main() -> None:
    original = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))
    accepted = run_checker(json.loads(json.dumps(original)))
    if accepted.returncode != 0:
        raise SystemExit(accepted.stderr.strip())

    arms = {arm["name"]: arm for arm in original["metadata"]["arms"]}
    team = arms["team-1"]
    stock = arms["stock"]
    team_tree = commit_tree(team["sha"])
    stock_tree = commit_tree(stock["sha"])

    swapped_tree = json.loads(json.dumps(original))
    swapped_team = next(arm for arm in swapped_tree["metadata"]["arms"] if arm["name"] == "team-1")
    swapped_team["tree"] = stock_tree
    require_tree_rejection(run_checker(swapped_tree), "team-1", stock_tree, team_tree)

    swapped_commit = json.loads(json.dumps(original))
    swapped_team = next(arm for arm in swapped_commit["metadata"]["arms"] if arm["name"] == "team-1")
    swapped_team["sha"] = stock["sha"]
    require_tree_rejection(run_checker(swapped_commit), "team-1", team_tree, stock_tree)

    missing_arm = json.loads(json.dumps(original))
    missing_arm["metadata"]["arms"] = [
        arm for arm in missing_arm["metadata"]["arms"] if arm["name"] != "team-1"
    ]
    require_missing_arm(run_checker(missing_arm), "team-1")


if __name__ == "__main__":
    main()

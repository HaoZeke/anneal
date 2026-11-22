#!/usr/bin/env python3
"""The provenance checker must reject a record that names the wrong tree."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "bench" / "test_omp_crossover_v3_provenance.py"
SOURCE_JSON = ROOT / "bench" / "results" / "rgpot_omp_crossover_v3_results.json"
SOURCE_REPORT = ROOT / "bench" / "results" / "rgpot_omp_crossover_v3_report.md"
SOURCE_PINS = ROOT / "bench" / "omp_crossover_v3_pins.json"
REPORT_NAME = "rgpot_omp_crossover_v3_report.md"


def has_git() -> bool:
    if not (ROOT / ".git").exists():
        return False
    return (
        subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        ).returncode
        == 0
    )


def repo_has_git(repo: Path) -> bool:
    if not (repo / ".git").exists():
        return False
    return (
        subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=repo,
            text=True,
            capture_output=True,
        ).returncode
        == 0
    )


def repo_is_shallow(repo: Path) -> bool:
    if not repo_has_git(repo):
        return False
    probe = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"],
        cwd=repo,
        text=True,
        capture_output=True,
    )
    return probe.returncode == 0 and probe.stdout.strip() == "true"


def repo_enforces_commit_resolution(repo: Path, original: dict[str, Any]) -> bool:
    if not repo_has_git(repo) or repo_is_shallow(repo):
        return False
    present: list[bool] = []
    for arm in original["metadata"]["arms"]:
        sha = arm.get("commit_sha")
        if not isinstance(sha, str) or not sha:
            continue
        probe = subprocess.run(
            ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
            cwd=repo,
            text=True,
            capture_output=True,
        )
        present.append(probe.returncode == 0)
    return bool(present) and all(present)


def commit_tree(arm: dict) -> str:
    if has_git():
        resolved = subprocess.run(
            ["git", "rev-parse", "--verify", f"{arm['sha']}^{{tree}}"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        if resolved.returncode == 0:
            return resolved.stdout.strip()
    return str(arm.get("commit_tree") or arm["tree"])


def object_tree(revision: str, fallback_seed: str) -> str:
    if has_git():
        resolved = subprocess.run(
            ["git", "rev-parse", "--verify", revision],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        if resolved.returncode == 0:
            return resolved.stdout.strip()
    return hashlib.sha1(fallback_seed.encode("utf-8")).hexdigest()


def run_checker(
    payload: dict,
    repo: Path = ROOT,
    pins: dict[str, Any] | None = None,
    report_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as directory:
        record = Path(directory) / SOURCE_JSON.name
        report = Path(directory) / REPORT_NAME
        record.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        if report_text is None:
            report.write_bytes(SOURCE_REPORT.read_bytes())
        else:
            report.write_text(report_text, encoding="utf-8")
        command = [
            sys.executable,
            str(CHECKER),
            "--repo",
            str(repo),
            "--json",
            str(record),
        ]
        if pins is not None:
            pins_path = Path(directory) / SOURCE_PINS.name
            pins_path.write_text(json.dumps(pins) + "\n", encoding="utf-8")
            command.extend(["--pins", str(pins_path)])
        return subprocess.run(
            command,
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


def run_checker_from_other_directory() -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as directory:
        return subprocess.run(
            [sys.executable, str(CHECKER), "--repo", str(ROOT)],
            cwd=directory,
            text=True,
            capture_output=True,
        )


def require_tree_rejection(result: subprocess.CompletedProcess[str], arm: str, recorded: str, actual: str) -> None:
    text = result.stdout + result.stderr
    if result.returncode == 0 or arm not in text or recorded not in text or actual not in text:
        raise SystemExit(f"{arm} was not rejected with recorded tree {recorded} and commit tree {actual}")


def require_exact_rejection(result: subprocess.CompletedProcess[str], expected: str) -> None:
    merged = (result.stdout + result.stderr).strip()
    lines = [
        line
        for line in merged.splitlines()
        if not line.startswith("provenance: repository history is absent;")
    ]
    text = "\n".join(lines).strip()
    if result.returncode == 0 or text != expected:
        raise SystemExit(f"expected exact rejection {expected!r}, got {result.returncode} {merged!r}")


def require_missing_arm(result: subprocess.CompletedProcess[str], arm: str) -> None:
    require_exact_rejection(result, f"arm set mismatch: missing {arm}; extra (none)")


def require_named_rejection(
    result: subprocess.CompletedProcess[str], missing: str, extra: str
) -> None:
    require_exact_rejection(result, f"arm set mismatch: missing {missing}; extra {extra}")


def require_acceptance(result: subprocess.CompletedProcess[str], context: str) -> None:
    if result.returncode != 0:
        raise SystemExit(f"{context} failed: {(result.stdout + result.stderr).strip()}")


def require_missing_sha(result: subprocess.CompletedProcess[str], arm: str) -> None:
    require_exact_rejection(result, f"{arm}: arm record is missing required key sha")


def require_invalid_field(
    result: subprocess.CompletedProcess[str], arm: str, field: str, value: object
) -> None:
    require_exact_rejection(result, f"{arm}: arm record has invalid {field}: {value!r}")


def require_contains_rejection(result: subprocess.CompletedProcess[str], text: str) -> None:
    merged = result.stdout + result.stderr
    if result.returncode == 0 or text not in merged:
        raise SystemExit(f"expected rejection containing {text!r}, got {result.returncode} {merged!r}")


def require_commit_rejection(
    result: subprocess.CompletedProcess[str], arm: str, sha: str
) -> None:
    text = result.stdout + result.stderr
    if result.returncode == 0 or arm not in text or sha not in text:
        raise SystemExit(f"unresolvable commit for {arm} was not rejected: {sha}")


def require_sha_mismatch(
    result: subprocess.CompletedProcess[str], arm: str, recorded: str, committed: str
) -> None:
    text = result.stdout + result.stderr
    if result.returncode == 0 or arm not in text or recorded not in text or committed not in text:
        raise SystemExit(
            f"{arm} was not rejected with recorded sha {recorded} and commit_sha {committed}"
        )


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


def check_cases(original: dict, repo: Path) -> None:
    accepted = run_checker(json.loads(json.dumps(original)), repo=repo)
    require_acceptance(accepted, "original provenance record")

    arms = {arm["name"]: arm for arm in original["metadata"]["arms"]}
    team = arms["team-1"]
    stock = arms["stock"]
    team_1024 = arms["team-1024"]
    team_tree = commit_tree(team)
    stock_tree = commit_tree(stock)
    team_1024_tree = commit_tree(team_1024)

    swapped_tree = json.loads(json.dumps(original))
    swapped_team = next(arm for arm in swapped_tree["metadata"]["arms"] if arm["name"] == "team-1")
    swapped_team["tree"] = team_1024_tree
    require_tree_rejection(
        run_checker(swapped_tree, repo=repo), "team-1", team_1024_tree, team_tree
    )

    stock_as_team = json.loads(json.dumps(original))
    changed_team = next(arm for arm in stock_as_team["metadata"]["arms"] if arm["name"] == "team-1")
    changed_team["tree"] = stock_tree
    require_tree_rejection(
        run_checker(stock_as_team, repo=repo), "team-1", stock_tree, team_tree
    )

    head_as_team = json.loads(json.dumps(original))
    changed_team = next(arm for arm in head_as_team["metadata"]["arms"] if arm["name"] == "team-1")
    head_tree = object_tree("HEAD^{tree}", "head-tree")
    changed_team["tree"] = head_tree
    require_tree_rejection(
        run_checker(head_as_team, repo=repo), "team-1", head_tree, team_tree
    )

    subdirectory_as_team = json.loads(json.dumps(original))
    changed_team = next(
        arm for arm in subdirectory_as_team["metadata"]["arms"] if arm["name"] == "team-1"
    )
    subdirectory_tree = object_tree(f"{team['sha']}:CppCore", f"{team['sha']}:CppCore")
    changed_team["tree"] = subdirectory_tree
    require_tree_rejection(
        run_checker(subdirectory_as_team, repo=repo),
        "team-1",
        subdirectory_tree,
        team_tree,
    )

    swapped_commit = json.loads(json.dumps(original))
    swapped_team = next(arm for arm in swapped_commit["metadata"]["arms"] if arm["name"] == "team-1")
    swapped_team["sha"] = stock["sha"]
    require_sha_mismatch(
        run_checker(swapped_commit, repo=repo),
        "team-1",
        stock["sha"],
        team["commit_sha"],
    )

    coordinated_swap = json.loads(json.dumps(original))
    swapped_team = next(arm for arm in coordinated_swap["metadata"]["arms"] if arm["name"] == "team-1")
    swapped_team["sha"] = stock["sha"]
    swapped_team["tree"] = stock["tree"]
    swapped_team["threshold"] = stock["threshold"]
    require_sha_mismatch(
        run_checker(coordinated_swap, repo=repo),
        "team-1",
        stock["sha"],
        team["commit_sha"],
    )

    omitted_sha = json.loads(json.dumps(original))
    changed_team = next(arm for arm in omitted_sha["metadata"]["arms"] if arm["name"] == "team-1")
    del changed_team["sha"]
    require_missing_sha(run_checker(omitted_sha, repo=repo), "team-1")

    unresolved_commit = json.loads(json.dumps(original))
    changed_team = next(
        arm for arm in unresolved_commit["metadata"]["arms"] if arm["name"] == "team-1"
    )
    missing_sha = "0" * 40
    changed_team["sha"] = missing_sha
    changed_team["commit_sha"] = missing_sha
    require_commit_rejection(run_checker(unresolved_commit, repo=repo), "team-1", missing_sha)

    empty_sha = json.loads(json.dumps(original))
    changed_team = next(arm for arm in empty_sha["metadata"]["arms"] if arm["name"] == "team-1")
    changed_team["sha"] = ""
    changed_team["commit_sha"] = ""
    require_invalid_field(run_checker(empty_sha, repo=repo), "team-1", "sha", "")

    null_sha = json.loads(json.dumps(original))
    changed_team = next(arm for arm in null_sha["metadata"]["arms"] if arm["name"] == "team-1")
    changed_team["sha"] = None
    changed_team["commit_sha"] = None
    require_invalid_field(run_checker(null_sha, repo=repo), "team-1", "sha", None)

    empty_tree = json.loads(json.dumps(original))
    changed_team = next(arm for arm in empty_tree["metadata"]["arms"] if arm["name"] == "team-1")
    changed_team["tree"] = ""
    changed_team["commit_tree"] = ""
    require_invalid_field(run_checker(empty_tree, repo=repo), "team-1", "tree", "")

    null_tree = json.loads(json.dumps(original))
    changed_team = next(arm for arm in null_tree["metadata"]["arms"] if arm["name"] == "team-1")
    changed_team["tree"] = None
    changed_team["commit_tree"] = None
    require_invalid_field(run_checker(null_tree, repo=repo), "team-1", "tree", None)

    wrong_type_tree = json.loads(json.dumps(original))
    changed_team = next(
        arm for arm in wrong_type_tree["metadata"]["arms"] if arm["name"] == "team-1"
    )
    changed_team["tree"] = 123
    changed_team["commit_tree"] = 123
    require_invalid_field(run_checker(wrong_type_tree, repo=repo), "team-1", "tree", 123)

    bool_threshold = json.loads(json.dumps(original))
    changed_team = next(
        arm for arm in bool_threshold["metadata"]["arms"] if arm["name"] == "team-1"
    )
    changed_team["threshold"] = True
    changed_team["tree_threshold"] = True
    require_invalid_field(
        run_checker(bool_threshold, repo=repo), "team-1", "tree_threshold", True
    )

    wrong_threshold = json.loads(json.dumps(original))
    changed_team = next(
        arm for arm in wrong_threshold["metadata"]["arms"] if arm["name"] == "team-1"
    )
    changed_team["threshold"] = 1024
    require_threshold_rejection(
        run_checker(wrong_threshold, repo=repo), "team-1", 1024, 1
    )

    missing_arm = json.loads(json.dumps(original))
    missing_arm["metadata"]["arms"] = [
        arm for arm in missing_arm["metadata"]["arms"] if arm["name"] != "team-1"
    ]
    require_missing_arm(run_checker(missing_arm, repo=repo), "team-1")

    renamed_arm = json.loads(json.dumps(original))
    changed_team = next(
        arm for arm in renamed_arm["metadata"]["arms"] if arm["name"] == "team-1"
    )
    changed_team["name"] = "renamed-team"
    require_named_rejection(run_checker(renamed_arm, repo=repo), "team-1", "renamed-team")

    duplicate_arm = json.loads(json.dumps(original))
    duplicate_arm["metadata"]["arms"].append(json.loads(json.dumps(duplicate_arm["metadata"]["arms"][0])))
    require_exact_rejection(
        run_checker(duplicate_arm, repo=repo),
        "metadata arms contain duplicate arm: stock",
    )

    whitespace_name = json.loads(json.dumps(original))
    changed_team = next(arm for arm in whitespace_name["metadata"]["arms"] if arm["name"] == "team-1")
    changed_team["name"] = "team 1"
    require_exact_rejection(
        run_checker(whitespace_name, repo=repo),
        "metadata arm: arm name has whitespace: 'team 1'",
    )

    empty_name = json.loads(json.dumps(original))
    changed_team = next(arm for arm in empty_name["metadata"]["arms"] if arm["name"] == "team-1")
    changed_team["name"] = ""
    require_exact_rejection(
        run_checker(empty_name, repo=repo),
        "metadata arm: arm name is invalid: ''",
    )

    bad_hex_sha = json.loads(json.dumps(original))
    changed_team = next(arm for arm in bad_hex_sha["metadata"]["arms"] if arm["name"] == "team-1")
    changed_team["sha"] = "0" * 39 + "g"
    changed_team["commit_sha"] = "0" * 39 + "g"
    require_invalid_field(
        run_checker(bad_hex_sha, repo=repo), "team-1", "sha", "0" * 39 + "g"
    )

    bad_hex_tree = json.loads(json.dumps(original))
    changed_team = next(arm for arm in bad_hex_tree["metadata"]["arms"] if arm["name"] == "team-1")
    changed_team["tree"] = "A" * 40
    changed_team["commit_tree"] = "A" * 40
    require_invalid_field(run_checker(bad_hex_tree, repo=repo), "team-1", "tree", "A" * 40)

    bad_identity = json.loads(json.dumps(original))
    bad_identity["identity"]["configured_team_min_atoms"] = 0
    require_exact_rejection(
        run_checker(bad_identity, repo=repo),
        "identity has invalid configured_team_min_atoms: 0",
    )

    identity_mismatch = json.loads(json.dumps(original))
    identity_mismatch["identity"]["configured_team_min_atoms"] = 2
    require_exact_rejection(
        run_checker(identity_mismatch, repo=repo),
        "team-1: recorded threshold 1 does not match identity configured_team_min_atoms 2",
    )

    report_lines = SOURCE_REPORT.read_text(encoding="utf-8").splitlines()
    table_start = next(i for i, line in enumerate(report_lines) if line.startswith("|Arm|"))
    table_end = table_start + 2
    while table_end < len(report_lines) and report_lines[table_end].startswith("|"):
        table_end += 1
    duplicate_report_lines = report_lines[:table_end] + ["|team-1|duplicate|"] + report_lines[table_end:]
    duplicate_report = "\n".join(duplicate_report_lines) + "\n"
    require_exact_rejection(
        run_checker(original, repo=repo, report_text=duplicate_report),
        "report arm table has duplicate arm: team-1",
    )

    empty_report_lines = report_lines[:table_end] + ["||blank|"] + report_lines[table_end:]
    empty_report = "\n".join(empty_report_lines) + "\n"
    require_exact_rejection(
        run_checker(original, repo=repo, report_text=empty_report),
        "report arm table has an empty arm name",
    )


def check_absent_commit_forgeries(original: dict[str, Any], repo: Path) -> None:
    arms = {arm["name"]: arm for arm in original["metadata"]["arms"]}
    team = arms["team-1"]
    forged_tree = "1" * 40

    coordinated_tree_forgery = json.loads(json.dumps(original))
    changed_team = next(
        arm
        for arm in coordinated_tree_forgery["metadata"]["arms"]
        if arm["name"] == "team-1"
    )
    changed_team["tree"] = forged_tree
    changed_team["commit_tree"] = forged_tree
    require_tree_rejection(
        run_checker(coordinated_tree_forgery, repo=repo),
        "team-1",
        forged_tree,
        team["tree"],
    )

    coordinated_commit_forgery = json.loads(json.dumps(original))
    changed_team = next(
        arm
        for arm in coordinated_commit_forgery["metadata"]["arms"]
        if arm["name"] == "team-1"
    )
    forged_sha = "0" * 40
    changed_team["sha"] = forged_sha
    changed_team["commit_sha"] = forged_sha
    changed_team["tree"] = forged_tree
    changed_team["commit_tree"] = forged_tree
    require_exact_rejection(
        run_checker(coordinated_commit_forgery, repo=repo),
        f"team-1: commit does not resolve and has no pin: {forged_sha}",
    )

    missing_arm_pin = json.loads(SOURCE_PINS.read_text(encoding="utf-8"))
    del missing_arm_pin["team-1"]
    require_exact_rejection(
        run_checker(original, repo=repo, pins=missing_arm_pin),
        "team-1: pinned manifest has no arm entry",
    )

    missing_commit_pin = json.loads(SOURCE_PINS.read_text(encoding="utf-8"))
    del missing_commit_pin["team-1"][team["sha"]]
    missing_pin_result = run_checker(original, repo=repo, pins=missing_commit_pin)
    if commit_resolves(repo, team["sha"]):
        require_exact_rejection(
            missing_pin_result,
            f"team-1: commit is not pinned: {team['sha']}",
        )
    else:
        require_exact_rejection(
            missing_pin_result,
            f"team-1: commit does not resolve and has no pin: {team['sha']}",
        )

    mismatched_pin = json.loads(SOURCE_PINS.read_text(encoding="utf-8"))
    mismatched_pin["team-1"][team["sha"]] = forged_tree
    require_tree_rejection(
        run_checker(original, repo=repo, pins=mismatched_pin),
        "team-1",
        team["tree"],
        forged_tree,
    )

    whitespace_arm_pin = json.loads(SOURCE_PINS.read_text(encoding="utf-8"))
    whitespace_arm_pin["team 1"] = whitespace_arm_pin["team-1"]
    del whitespace_arm_pin["team-1"]
    require_exact_rejection(
        run_checker(original, repo=repo, pins=whitespace_arm_pin),
        "pinned manifest: arm name has whitespace: 'team 1'",
    )

    bad_thresholds = json.loads(SOURCE_PINS.read_text(encoding="utf-8"))
    bad_thresholds["kTeamMinAtoms"]["team-1"] = 0
    require_exact_rejection(
        run_checker(original, repo=repo, pins=bad_thresholds),
        "pinned manifest has invalid kTeamMinAtoms for 'team-1'",
    )


def main() -> None:
    bare = run_bare_checker()
    if bare.returncode != 0:
        raise SystemExit(f"bare provenance check failed: {bare.stderr.strip()}")

    away = run_checker_from_other_directory()
    if away.returncode != 0:
        raise SystemExit(
            "provenance check with --repo from another directory failed: "
            + (away.stderr or away.stdout).strip()
        )

    original = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))
    check_cases(original, ROOT)
    forged_pins = json.loads(SOURCE_PINS.read_text(encoding="utf-8"))
    for arm_name, arm_pins in forged_pins.items():
        if arm_name == "kTeamMinAtoms":
            continue
        for sha in arm_pins:
            arm_pins[sha] = "f" * 40
    del forged_pins["team-1"]
    forged = run_checker(original, repo=ROOT, pins=forged_pins)
    stock = next(arm for arm in original["metadata"]["arms"] if arm["name"] == "stock")
    require_tree_rejection(forged, "stock", stock["tree"], "f" * 40)
    with tempfile.TemporaryDirectory() as partial_git_dir:
        partial_repo = Path(partial_git_dir)
        subprocess.run(
            ["git", "init", "-q"],
            cwd=partial_repo,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=test",
                "-c",
                "user.email=test@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-q",
                "--allow-empty",
                "-m",
                "initial",
            ],
            cwd=partial_repo,
            check=True,
            capture_output=True,
            text=True,
        )
        check_cases(original, partial_repo)
        check_absent_commit_forgeries(original, partial_repo)
    with tempfile.TemporaryDirectory() as shallow_root:
        shallow_repo, head_sha, head_tree = make_shallow_repo(Path(shallow_root))
        check_substituted_head(original, shallow_repo, head_sha, head_tree)
        with tempfile.TemporaryDirectory() as no_git_repo:
            no_git_path = Path(no_git_repo)
            no_git_ok = run_checker(original, repo=no_git_path)
            require_acceptance(no_git_ok, "original provenance record without git history")
            if "repository history is absent" not in (no_git_ok.stdout + no_git_ok.stderr):
                raise SystemExit("history-absent fallback was not reported explicitly")
            check_cases(original, no_git_path)
            check_absent_commit_forgeries(original, no_git_path)
            check_substituted_head(original, no_git_path, head_sha, head_tree)


def commit_resolves(repo: Path, sha: str) -> bool:
    if not repo_has_git(repo):
        return False
    probe = subprocess.run(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
        cwd=repo,
        check=False,
        text=True,
        capture_output=True,
    )
    return probe.returncode == 0


def substituted_team(
    original: dict[str, Any], sha: str, tree: str, threshold: int
) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(json.dumps(original))
    changed = next(arm for arm in payload["metadata"]["arms"] if arm["name"] == "team-1")
    changed["sha"] = sha
    changed["commit_sha"] = sha
    changed["tree"] = tree
    changed["commit_tree"] = tree
    changed["threshold"] = threshold
    changed["tree_threshold"] = threshold
    return payload


def inflated_team_threshold(original: dict[str, Any], threshold: int) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(json.dumps(original))
    changed = next(arm for arm in payload["metadata"]["arms"] if arm["name"] == "team-1")
    changed["threshold"] = threshold
    changed["tree_threshold"] = threshold
    return payload


def check_substituted_head(
    original: dict[str, Any], repo: Path, head_sha: str, head_tree: str
) -> None:
    configured = original["identity"]["configured_team_min_atoms"]
    swapped = run_checker(
        substituted_team(original, head_sha, head_tree, configured), repo=repo
    )
    merged = swapped.stdout + swapped.stderr
    if "stock: commit does not resolve" in merged:
        require_contains_rejection(swapped, "commit does not resolve")
    elif commit_resolves(repo, head_sha):
        require_exact_rejection(swapped, f"team-1: commit is not pinned: {head_sha}")
    else:
        require_exact_rejection(
            swapped, f"team-1: commit does not resolve and has no pin: {head_sha}"
        )
    require_threshold_rejection(
        run_checker(inflated_team_threshold(original, 999999), repo=repo),
        "team-1",
        999999,
        1,
    )


def make_shallow_repo(root: Path) -> tuple[Path, str, str]:
    seed = root / "seed"
    shallow = root / "shallow"
    seed.mkdir()
    header = seed / "CppCore" / "rgpot"
    header.mkdir(parents=True)
    (header / "Slices.hpp").write_text(
        "inline constexpr int kTeamMinAtoms = 8192;\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "init", "-q"],
        cwd=seed,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "add", "--", "CppCore/rgpot/Slices.hpp"],
        cwd=seed,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "-m",
            "threshold",
        ],
        cwd=seed,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "clone", "--depth", "1", "--no-local", seed.as_uri(), str(shallow)],
        check=True,
        capture_output=True,
        text=True,
    )
    if not repo_is_shallow(shallow):
        raise SystemExit("shallow provenance fixture is not shallow")
    head_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=shallow,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    head_tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=shallow,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return shallow, head_sha, head_tree


if __name__ == "__main__":
    main()

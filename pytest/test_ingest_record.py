# Copyright (c) anneal developers
"""A synthetic record is installed only when its harness commit is held.

The fixture token lives in the temporary record and must not be written
into the paper.
"""
from __future__ import annotations

import importlib.util
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "FIXTURE-NEVER-PRINT"
HELD = "61fa67cf380e46876b569da45e7cc091647a775b"
BRANCH = "record/fixture-never-print"
ROW = "bench/fixture-never-print/rows.csv"


def load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ingest():
    return load("ingest_record_test", "scripts/ingest_record.py")


def commit_record(remote: Path, revision: str) -> str:
    subprocess.run(["git", "init", "-b", BRANCH], cwd=remote, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.com"], cwd=remote, check=True)
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=remote, check=True)
    path = remote / ROW
    path.parent.mkdir(parents=True)
    path.write_text(
        "system,host,code_revision\n"
        f"fixture,{TOKEN},{revision}\n"
    )
    subprocess.run(["git", "add", ROW], cwd=remote, check=True)
    subprocess.run(["git", "commit", "-m", "fixture record"], cwd=remote, check=True, capture_output=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=remote, check=True, capture_output=True, text=True,
    ).stdout.strip()


def catalog(module):
    return {
        BRANCH: module.Record(key="fixture-never-print", paths=(ROW,), reason="rerun pending", refresh=lambda root: None),
    }


def git_repo(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)


def test_a_held_fixture_lands_and_is_not_printed(tmp_path: Path):
    module = ingest()
    remote = tmp_path / "remote"
    remote.mkdir()
    sha = commit_record(remote, HELD)
    root = tmp_path / "tree"
    root.mkdir()
    git_repo(root)
    module.ingest(
        BRANCH, sha, remote=str(remote), root=root, git_dir=root,
        catalog=catalog(module), reachable_repo=ROOT,
    )
    landed = (root / ROW).read_text()
    assert TOKEN in landed
    assert HELD in landed
    paper = ROOT / "paper" / "jctc"
    leaked = [
        path.relative_to(ROOT).as_posix()
        for path in paper.rglob("*")
        if path.is_file() and TOKEN in path.read_text(errors="ignore")
    ]
    assert leaked == []
    assert TOKEN not in (ROOT / "scripts" / "ingest_record.py").read_text()
    assert BRANCH not in module.production_catalog()


def test_an_unreachable_harness_writes_nothing(tmp_path: Path):
    module = ingest()
    remote = tmp_path / "remote"
    remote.mkdir()
    sha = commit_record(remote, "b" * 40)
    root = tmp_path / "tree"
    root.mkdir()
    git_repo(root)
    with pytest.raises(SystemExit, match="not reachable"):
        module.ingest(
            BRANCH, sha, remote=str(remote), root=root, git_dir=root,
            catalog=catalog(module), reachable_repo=ROOT,
        )
    assert not (root / ROW).exists()


def test_a_sha_that_is_not_the_branch_tip_is_refused(tmp_path: Path):
    module = ingest()
    remote = tmp_path / "remote"
    remote.mkdir()
    commit_record(remote, HELD)
    root = tmp_path / "tree"
    root.mkdir()
    git_repo(root)
    with pytest.raises(SystemExit, match="not"):
        module.ingest(
            BRANCH, "a" * 40, remote=str(remote), root=root, git_dir=root,
            catalog=catalog(module), reachable_repo=ROOT,
        )
    assert not (root / ROW).exists()


def test_empty_marks_take_a_table_and_clear_it():
    ase = load("ase_panel_splice", "scripts/ase_panel/summarize.py")
    text = "before\n% ase-panel table: start\n% ase-panel table: end\nafter\n"
    block = "\\begin{table}\n\\label{tab:ase}\n\\end{table}"
    filled = ase.splice_at(text, ase.TEX_MARKS, block)
    assert "\\label{tab:ase}" in filled
    assert filled.startswith("before\n")
    assert filled.endswith("after\n")
    cleared = ase.splice_at(filled, ase.TEX_MARKS, "")
    assert "\\label{tab:ase}" not in cleared
    assert ase.TEX_MARKS[0] in cleared and ase.TEX_MARKS[1] in cleared
    assert cleared.startswith("before\n") and cleared.endswith("after\n")


def test_withheld_block_is_not_written():
    module = ingest()
    assert module.withheld_block([
        ("tab:ase", "rerun pending"),
        ("tab:sc", "harness not reachable"),
    ]) == ""
    assert module.withheld_block([]) == ""


def test_the_si_does_not_narrate_unquoted_comparisons():
    module = ingest()
    ase = load("ase_panel_empty", "scripts/ase_panel/summarize.py")
    items = module.withheld_items()
    assert [name for name, _reason in items] == [
        "tab:ase", "Morse against GMIN", "tab:sc",
    ]
    assert {reason for _name, reason in items} <= {"harness not reachable", "rerun pending"}
    text = (ROOT / "paper" / "jctc" / "si.tex").read_text()
    assert "Withheld results" not in text
    assert "rerun pending" not in text
    assert "\\label{tab:gmin-tuned}" in text
    assert module.WITHHELD_MARKS[0] not in text
    assert module.withheld_block() == ""
    assert ase.held_prose() == ""
    assert ase.tex_table() == ""


def test_a_fetch_into_the_same_repo_does_not_hold_a_side_commit(tmp_path: Path):
    module = ingest()
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=repo, check=True)
    (repo / "README").write_text("anchor\n")
    subprocess.run(["git", "add", "README"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "anchor"], cwd=repo, check=True, capture_output=True)
    remote = tmp_path / "remote"
    remote.mkdir()
    subprocess.run(["git", "init", "-b", BRANCH], cwd=remote, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.com"], cwd=remote, check=True)
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=remote, check=True)
    (remote / "note").write_text("side\n")
    subprocess.run(["git", "add", "note"], cwd=remote, check=True)
    subprocess.run(["git", "commit", "-m", "side"], cwd=remote, check=True, capture_output=True)
    side = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=remote, check=True, capture_output=True, text=True,
    ).stdout.strip()
    row = remote / ROW
    row.parent.mkdir(parents=True)
    row.write_text(f"system,host,code_revision\nfixture,{TOKEN},{side}\n")
    subprocess.run(["git", "add", ROW], cwd=remote, check=True)
    subprocess.run(["git", "commit", "-m", "rows"], cwd=remote, check=True, capture_output=True)
    tip = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=remote, check=True, capture_output=True, text=True,
    ).stdout.strip()
    with pytest.raises(SystemExit, match="not reachable"):
        module.ingest(
            BRANCH, tip, remote=str(remote), root=repo, git_dir=repo,
            catalog=catalog(module), reachable_repo=repo,
        )
    assert not (repo / ROW).exists()
    present = subprocess.run(
        ["git", "cat-file", "-e", f"{side}^{{commit}}"], cwd=repo, capture_output=True,
    )
    assert present.returncode != 0


def test_named_remote_fetches_via_scratch_repo_with_default_reachable_repo(tmp_path: Path, monkeypatch):
    module = ingest()
    bare = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", str(bare)], check=True, capture_output=True)

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=repo, check=True)
    (repo / "README").write_text("anchor\n")
    subprocess.run(["git", "add", "README"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "anchor"], cwd=repo, check=True, capture_output=True)
    held = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip()
    subprocess.run(["git", "remote", "add", "origin", str(bare)], cwd=repo, check=True)

    producer = tmp_path / "producer"
    producer.mkdir()
    subprocess.run(["git", "init", "-b", BRANCH], cwd=producer, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.com"], cwd=producer, check=True)
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=producer, check=True)
    row = producer / ROW
    row.parent.mkdir(parents=True)
    row.write_text(f"system,host,code_revision\nfixture,{TOKEN},{held}\n")
    subprocess.run(["git", "add", ROW], cwd=producer, check=True)
    subprocess.run(["git", "commit", "-m", "rows"], cwd=producer, check=True, capture_output=True)
    tip = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=producer, check=True, capture_output=True, text=True,
    ).stdout.strip()
    subprocess.run(["git", "push", str(bare), f"HEAD:refs/heads/{BRANCH}"], cwd=producer, check=True, capture_output=True)

    refreshed: list[Path] = []
    custom_catalog = {
        BRANCH: module.Record(
            key="fixture-never-print",
            paths=(ROW,),
            reason="rerun pending",
            refresh=refreshed.append,
        ),
    }
    monkeypatch.setattr(module, "REPO", repo)
    module.ingest(BRANCH, tip, catalog=custom_catalog)
    assert (repo / ROW).read_text() == f"system,host,code_revision\nfixture,{TOKEN},{held}\n"
    assert refreshed == [repo]
    present = subprocess.run(
        ["git", "cat-file", "-e", f"{tip}^{{commit}}"], cwd=repo, capture_output=True,
    )
    assert present.returncode != 0


def test_pins_and_harness_fields_match_the_audit():
    module = ingest()
    audit = load("paper_audit_pins", "scripts/paper_audit.py")
    tables = load("h2h_pins", "paper/jctc/scripts/h2h_tables.py")
    assert module.EXTERNAL_REVISIONS == tuple(audit.EXTERNAL_REVISIONS)
    assert module.EXTERNAL_REVISIONS == tuple(tables.EXTERNAL_PINS)
    source = inspect.getsource(audit._row_shas)
    for field in module.HARNESS_FIELDS:
        assert field in source


def test_tab_ase_expected_failure_stays_until_the_table_is_present():
    source = (ROOT / "scripts" / "paper_audit.py").read_text()
    assert 'audit.expect(SI, "tab:ase expected failure"' in source
    assert "bool(panel.tex_table())" in source

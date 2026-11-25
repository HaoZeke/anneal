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


def test_the_si_does_not_narrate_unquoted_comparisons(monkeypatch):
    module = ingest()
    ase = load("ase_panel_empty", "scripts/ase_panel/summarize.py")
    items = module.withheld_items()
    assert [name for name, _reason in items] == ["tab:sc"]
    assert {reason for _name, reason in items} <= {"harness not reachable", "rerun pending"}
    text = (ROOT / "paper" / "jctc" / "si.tex").read_text()
    assert "Withheld results" not in text
    assert "rerun pending" not in text
    assert "\\label{tab:gmin-tuned}" in text
    assert module.WITHHELD_MARKS[0] not in text
    assert module.withheld_block() == ""
    assert ase.tex_table() and ase.tex_table() in text
    assert ase.held_prose() and ase.held_prose() in text
    monkeypatch.setattr(ase, "commit_held", lambda revision: False)
    assert ase.held_prose() == ""
    assert ase.tex_table() == ""


def test_a_withheld_ase_table_and_morse_panel_are_named_and_not_narrated(monkeypatch):
    module = ingest()
    ase = module._module("ase_panel_withheld", "scripts/ase_panel/summarize.py")
    panel = module._module("gmin_panel_withheld", "scripts/gmin_panel/run_panel.py")
    monkeypatch.setattr(ase, "commit_held", lambda revision: False)
    monkeypatch.setattr(panel, "_commit_held", lambda revision: not revision.startswith("9837358a"))
    monkeypatch.setattr(module, "_ITEMS", None)
    monkeypatch.setattr(module, "_module", lambda name, path: {
        "ase_panel_withheld": ase, "gmin_panel_withheld": panel,
    }.get(name) or load(name, path))
    items = module.withheld_items()
    assert [name for name, _reason in items][:2] == ["tab:ase", "Morse against GMIN"]
    assert {reason for _name, reason in items} <= {"harness not reachable", "rerun pending"}
    assert module.withheld_block(items) == ""
    monkeypatch.setattr(module, "_ITEMS", None)


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


TUNED = "bench/gmin-tuned/rows-shard2.csv"


@pytest.fixture
def side_commit(tmp_path: Path, monkeypatch) -> str:
    """A commit that is in the object store of this checkout and is not an ancestor of HEAD.

    It is written to a temporary object directory that git is told to read
    through, so the checkout itself gains nothing.
    """
    objects = tmp_path / "objects"
    objects.mkdir()
    main_objects = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-path", "objects"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    env = {
        **__import__("os").environ,
        "GIT_OBJECT_DIRECTORY": str(objects),
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": main_objects,
        "GIT_AUTHOR_NAME": "Fixture", "GIT_AUTHOR_EMAIL": "fixture@example.com",
        "GIT_COMMITTER_NAME": "Fixture", "GIT_COMMITTER_EMAIL": "fixture@example.com",
    }
    sha = subprocess.run(
        ["git", "commit-tree", "HEAD^{tree}", "-p", "HEAD", "-m", "unmerged side commit"],
        cwd=ROOT, check=True, capture_output=True, text=True, env=env,
    ).stdout.strip()
    monkeypatch.setenv("GIT_ALTERNATE_OBJECT_DIRECTORIES", str(objects))
    present = subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=ROOT, capture_output=True)
    assert present.returncode == 0
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", sha, "HEAD"], cwd=ROOT, capture_output=True)
    assert ancestor.returncode != 0
    return sha


def test_a_commit_that_is_present_but_not_an_ancestor_is_held_by_no_reader(side_commit: str):
    shared = load("revision_reach_test", "scripts/revision_reach.py")
    audit = load("paper_audit_reach", "scripts/paper_audit.py")
    ase = load("ase_panel_reach", "scripts/ase_panel/summarize.py")
    tables = load("h2h_tables_reach", "paper/jctc/scripts/h2h_tables.py")
    panel = load("gmin_panel_reach", "scripts/gmin_panel/run_panel.py")
    module = ingest()
    readers = {
        "revision_reach": lambda rev: shared.commit_held(rev, ROOT),
        "paper_audit.commit_held": audit.commit_held,
        "paper_audit.harness_reachable": audit.harness_reachable,
        "ase summarize": ase.commit_held,
        "h2h_tables": tables.commit_held,
        "gmin run_panel": panel._commit_held,
        "ingest_record": lambda rev: module.harness_reachable(rev, ROOT),
    }
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    held = {name: read(head) for name, read in readers.items()}
    unheld = {name: read(side_commit) for name, read in readers.items()}
    assert held == dict.fromkeys(readers, True)
    assert unheld == dict.fromkeys(readers, False)


def test_tab_ase_is_held_only_for_an_ancestor_by_the_table_and_the_audit(side_commit: str, monkeypatch):
    ase = load("ase_panel_held", "scripts/ase_panel/summarize.py")
    module = ingest()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()

    def rows_at(revision: str):
        row = {"code_revision": revision, "hit": "1", "stop": "wall"}
        return {arm: [dict(row)] for arm in ("recommended", *ase.ASE_ARMS)}

    monkeypatch.setattr(ase, "load", lambda system: rows_at(side_commit))
    assert ase.quoted_campaigns() == {}
    assert ase.held_prose() == ""
    monkeypatch.setattr(ase, "load", lambda system: rows_at(head))
    assert set(ase.quoted_campaigns()) == set(ase.SYSTEMS)
    assert module.harness_reachable(side_commit, ROOT) is False
    assert module.harness_reachable(head, ROOT) is True


def test_a_record_naming_a_present_but_unmerged_commit_is_refused(side_commit: str):
    module = ingest()
    blob = f"system,harness_revision,code_revision\nfixture,{HELD},{side_commit}\n".encode()
    with pytest.raises(SystemExit, match=f"harness {side_commit} is not reachable"):
        module.require_reachable({ROW: blob}, ROOT)


def record_blob(header: str, *rows: str) -> bytes:
    return (header + "\n" + "\n".join(rows) + "\n").encode()


HEADER = "system,harness_revision,code_revision"


@pytest.mark.parametrize(
    ("row", "field", "why"),
    [
        (f"fixture,{HELD},", "code_revision", "blank"),
        (f"fixture,{HELD},unknown", "code_revision", "unknown"),
        (f"fixture,{HELD},Unknown", "code_revision", "unknown"),
        (f"fixture,{HELD},latest", "code_revision", "not a commit"),
        (f"fixture,{HELD},not-a-commit", "code_revision", "not a commit"),
        (f"fixture,{HELD},1234", "code_revision", "not a commit"),
        (f"fixture,{HELD},{HELD.upper()}", "code_revision", "not a commit"),
        (f"fixture,{HELD},{HELD} zzz", "code_revision", "not a commit"),
        (f"fixture,{HELD},{HELD};unknown", "code_revision", "unknown"),
        (f"fixture,,{HELD}", "harness_revision", "blank"),
        (f"fixture,unknown,{HELD}", "harness_revision", "unknown"),
        (f"fixture,zzzzzzz,{HELD}", "harness_revision", "not a commit"),
        (f"fixture,{HELD}", "code_revision", "blank"),
    ],
)
def test_a_malformed_revision_field_is_refused_beside_a_valid_commit(row: str, field: str, why: str):
    module = ingest()
    with pytest.raises(SystemExit, match=f"row 1 {field} is {why}"):
        module.require_reachable({ROW: record_blob(HEADER, row)}, ROOT)


def test_a_malformed_revision_field_in_a_later_row_is_refused():
    module = ingest()
    good = f"fixture,{HELD},{HELD}"
    bad = f"fixture,{HELD},unknown"
    with pytest.raises(SystemExit, match="row 3 code_revision is unknown"):
        module.require_reachable({ROW: record_blob(HEADER, good, good, bad, good)}, ROOT)


def test_a_malformed_field_is_refused_in_each_revision_column_the_audit_reads():
    module = ingest()
    for field in module.REVISION_FIELDS:
        header = f"system,harness_revision,{field}" if field != "harness_revision" else "system,harness_revision,head"
        column = field if field != "harness_revision" else "head"
        for value, why in (("", "blank"), ("unknown", "unknown"), ("g1", "not a commit")):
            with pytest.raises(SystemExit, match=f"{column} is {why}"):
                module.require_reachable({ROW: record_blob(header, f"fixture,{HELD},{value}")}, ROOT)


def test_a_record_with_every_revision_field_well_formed_and_reachable_is_accepted():
    module = ingest()
    pin = module.EXTERNAL_REVISIONS[0]
    external = "https://example.org/team/code.git@" + "a" * 40
    rows = (
        f"fixture,{HELD},{HELD}",
        f"fixture,{HELD};{pin},{pin}",
        f"fixture,{HELD} {external},{external}",
    )
    module.require_reachable({ROW: record_blob(HEADER, *rows)}, ROOT)
    for blob in (record_blob("system,host", "fixture,here"), record_blob(HEADER)):
        with pytest.raises(SystemExit, match="no harness commit"):
            module.require_reachable({ROW: blob}, ROOT)


def tuned_rows():
    import csv
    return list(csv.DictReader((ROOT / TUNED).read_text().splitlines()))


def test_a_committed_tuned_record_is_accepted_as_the_audit_accepts_it():
    module = ingest()
    audit = load("paper_audit_tuned", "scripts/paper_audit.py")
    rows = tuned_rows()
    audited = [row for row in rows if row["system"] in audit.TUNED_SYSTEMS and row["code"] in audit.TUNED_CODES]
    controls = [row for row in rows if row not in audited]
    assert audited and controls
    assert all(not audit.harness_reachable(row["harness_commit"]) for row in audited), \
        "the fixture names a harness that is not an ancestor, as a tuned side commit does"
    module.require_reachable({TUNED: (ROOT / TUNED).read_bytes()}, ROOT)
    assert all(module.tuned_verified(row, ROOT) for row in audited)
    assert all(audit.tuned_gmin_harness_verified(row) for row in audited)
    assert not any(audit.tuned_gmin_harness_verified(row) for row in controls)
    old = module.harness_reachable
    assert not any(old(row["harness_commit"], ROOT) for row in rows)


def test_a_control_row_the_audit_does_not_read_needs_a_verified_row_with_its_harness_and_data(tmp_path: Path):
    module = ingest()
    lines = (ROOT / TUNED).read_text().splitlines()
    header = lines[0]
    only_controls = [lines[0], *[line for line in lines[1:] if line.startswith("lj98aa,")]]
    assert len(only_controls) > 1
    with pytest.raises(SystemExit, match="verified tuned-GMIN side commit"):
        module.require_reachable({TUNED: ("\n".join(only_controls) + "\n").encode()}, ROOT)
    audited = [line for line in lines[1:] if line.startswith(("lj98,", "lj104,"))]
    module.require_reachable({TUNED: ("\n".join([header, *audited, only_controls[1]]) + "\n").encode()}, ROOT)
    other = only_controls[1]
    fields = header.split(",")
    columns = other.split(",")
    columns[fields.index("data_blob")] = "0" * 40
    with pytest.raises(SystemExit, match="verified tuned-GMIN side commit"):
        module.require_reachable({TUNED: ("\n".join([header, *audited, ",".join(columns)]) + "\n").encode()}, ROOT)


def test_a_tuned_record_is_refused_when_the_audit_would_refuse_a_row():
    module = ingest()
    audit = load("paper_audit_tuned_bad", "scripts/paper_audit.py")
    text = (ROOT / TUNED).read_text()
    rows = tuned_rows()
    row = next(row for row in rows if row["system"] in audit.TUNED_SYSTEMS and row["code"] in audit.TUNED_CODES)
    for field, value in (("data_blob", "0" * 40), ("harness_commit", "b" * 40)):
        broken = text.replace(row[field], value, 1)
        assert broken != text
        changed = {**row, field: value}
        assert audit.tuned_gmin_harness_verified(changed) is False
        with pytest.raises(SystemExit, match="verified tuned-GMIN side commit|not reachable"):
            module.require_reachable({TUNED: broken.encode()}, ROOT)


def test_a_tuned_record_is_judged_against_the_repository_it_lands_in(tmp_path: Path):
    module = ingest()
    other = tmp_path / "other"
    other.mkdir()
    git_repo(other)
    with pytest.raises(SystemExit, match="not reachable"):
        module.require_reachable({TUNED: (ROOT / TUNED).read_bytes()}, other)


def two_file_record(remote: Path, first: str, second: str) -> str:
    subprocess.run(["git", "init", "-b", BRANCH], cwd=remote, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.com"], cwd=remote, check=True)
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=remote, check=True)
    for name, body in ((ROW, first), (ROW.replace("rows", "more"), second)):
        path = remote / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
        subprocess.run(["git", "add", name], cwd=remote, check=True)
    subprocess.run(["git", "commit", "-m", "two rows"], cwd=remote, check=True, capture_output=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=remote, check=True, capture_output=True, text=True,
    ).stdout.strip()


MORE = ROW.replace("rows", "more")


def two_file_catalog(module, refresh):
    return {BRANCH: module.Record(key="fixture-never-print", paths=(ROW, MORE), reason="rerun pending", refresh=refresh)}


def test_a_failed_regeneration_leaves_every_original_row_file_intact(tmp_path: Path):
    module = ingest()
    remote = tmp_path / "remote"
    remote.mkdir()
    body = f"system,host,code_revision\nfixture,{TOKEN},{HELD}\n"
    sha = two_file_record(remote, body, body)
    root = tmp_path / "tree"
    root.mkdir()
    git_repo(root)
    (root / ROW).parent.mkdir(parents=True)
    (root / ROW).write_text("old rows\n")
    (root / ROW).chmod(0o640)

    def failing(_root):
        assert (root / ROW).read_text() == body
        raise RuntimeError("the generator failed")

    with pytest.raises(RuntimeError, match="generator failed"):
        module.ingest(
            BRANCH, sha, remote=str(remote), root=root, git_dir=root,
            catalog=two_file_catalog(module, failing), reachable_repo=ROOT,
        )
    assert (root / ROW).read_text() == "old rows\n"
    assert (root / ROW).stat().st_mode & 0o777 == 0o640
    assert not (root / MORE).exists()
    assert sorted(path.name for path in (root / ROW).parent.iterdir()) == ["rows.csv"]


def test_a_failure_while_moving_a_row_file_into_place_leaves_the_originals(tmp_path: Path, monkeypatch):
    module = ingest()
    remote = tmp_path / "remote"
    remote.mkdir()
    body = f"system,host,code_revision\nfixture,{TOKEN},{HELD}\n"
    sha = two_file_record(remote, body, body)
    root = tmp_path / "tree"
    root.mkdir()
    git_repo(root)
    (root / ROW).parent.mkdir(parents=True)
    (root / ROW).write_text("old rows\n")
    (root / MORE).write_text("old more\n")
    real = __import__("os").replace
    calls = []

    def second_move_fails(source, target):
        calls.append(target)
        if len(calls) == 2:
            raise OSError("disk full")
        return real(source, target)

    monkeypatch.setattr(module.os, "replace", second_move_fails)
    refreshed = []
    with pytest.raises(OSError, match="disk full"):
        module.ingest(
            BRANCH, sha, remote=str(remote), root=root, git_dir=root,
            catalog=two_file_catalog(module, refreshed.append), reachable_repo=ROOT,
        )
    assert refreshed == []
    assert (root / ROW).read_text() == "old rows\n"
    assert (root / MORE).read_text() == "old more\n"
    assert sorted(path.name for path in (root / ROW).parent.iterdir()) == ["more.csv", "rows.csv"]


def test_a_write_that_fails_keeps_the_old_file_and_leaves_no_temporary_file(tmp_path: Path, monkeypatch):
    atomic = load("atomic_write_test", "scripts/atomic_write.py")
    target = tmp_path / "table.csv"
    target.write_text("old\n")
    atomic.write_atomic(target, "new\n")
    assert target.read_text() == "new\n"
    assert [path.name for path in tmp_path.iterdir()] == ["table.csv"]

    def refuse(*_args):
        raise OSError("refused")

    monkeypatch.setattr(atomic.os, "replace", refuse)
    with pytest.raises(OSError, match="refused"):
        atomic.write_atomic(target, "newer\n")
    assert target.read_text() == "new\n"
    assert [path.name for path in tmp_path.iterdir()] == ["table.csv"]
    monkeypatch.undo()
    monkeypatch.setattr(atomic.os, "fsync", refuse)
    with pytest.raises(OSError, match="refused"):
        atomic.write_atomic(target, "newest\n")
    assert target.read_text() == "new\n"
    assert [path.name for path in tmp_path.iterdir()] == ["table.csv"]


def test_the_generators_write_their_tables_atomically(tmp_path: Path, monkeypatch):
    panel = load("gmin_panel_atomic", "scripts/gmin_panel/run_panel.py")
    table = tmp_path / "summary.csv"
    table.write_text("old\n")
    document = tmp_path / "doc.md"
    document.write_text("a\n<!-- s -->\nold\n<!-- e -->\nz\n")
    monkeypatch.setattr(panel.os, "replace", lambda *_a: (_ for _ in ()).throw(OSError("refused")))
    with pytest.raises(OSError, match="refused"):
        panel.write_csv(table, ["a"], [{"a": "1"}])
    with pytest.raises(OSError, match="refused"):
        panel.splice(document, ("<!-- s -->", "<!-- e -->"), "new")
    assert table.read_text() == "old\n"
    assert document.read_text() == "a\n<!-- s -->\nold\n<!-- e -->\nz\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["doc.md", "summary.csv"]
    monkeypatch.undo()
    panel.write_csv(table, ["a"], [{"a": "1"}])
    assert table.read_text() == "a\n1\n"

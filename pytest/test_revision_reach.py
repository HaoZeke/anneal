# Copyright (c) anneal developers
"""One definition of a held revision, shared by the audit, the ingest and the generators.

The table generators decide whether a result may be quoted, the paper audit
decides whether it was quoted correctly and the ingest decides whether a record
may land. They answer through the same functions, so one row has one verdict.
A revision a row cites is a full SHA, never an abbreviation.
"""
from __future__ import annotations

import csv
import importlib.util
import inspect
import io
import re
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TUNED = "bench/gmin-tuned/rows-shard2.csv"
HELD = "61fa67cf380e46876b569da45e7cc091647a775b"
URL = "https://example.org/team/code.git@"

if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import revision_reach  # noqa: E402


def load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def paths():
    return {
        "revision_reach": revision_reach.tuned_gmin_harness_verified,
        "paper_audit": load("paper_audit_shared", "scripts/paper_audit.py").tuned_gmin_harness_verified,
        "h2h_tables": load("h2h_tables_shared", "paper/jctc/scripts/h2h_tables.py").tuned_gmin_harness_verified,
        "ingest_record": lambda row: load("ingest_record_shared", "scripts/ingest_record.py").tuned_verified(row, ROOT),
    }


def tuned_rows() -> list[dict[str, str]]:
    return list(csv.DictReader((ROOT / TUNED).read_text().splitlines()))


def audited_row() -> dict[str, str]:
    return next(row for row in tuned_rows() if row["system"] in revision_reach.TUNED_SYSTEMS and row["code"] == "anneal")


def test_the_gate_the_refresh_and_the_ingest_reach_one_tuned_verifier(monkeypatch, paths):
    for answer in (True, False):
        monkeypatch.setattr(revision_reach, "tuned_gmin_harness_verified", lambda row, repo=ROOT, held=None: answer)
        verdicts = {name: bool(check(audited_row())) for name, check in paths.items() if name != "revision_reach"}
        assert verdicts == {"paper_audit": answer, "h2h_tables": answer, "ingest_record": answer}


def variants() -> list[tuple[str, dict[str, str], bool]]:
    row = audited_row()
    code, harness = row["code_revision"], row["harness_commit"]
    return [
        ("a committed row", row, True),
        ("a code revision as a URL at a full SHA", {**row, "code_revision": URL + code}, True),
        ("a harness as a URL at a full SHA", {**row, "harness_commit": URL + harness}, True),
        ("an abbreviated code revision", {**row, "code_revision": code[:12]}, False),
        ("an abbreviated harness commit", {**row, "harness_commit": harness[:12]}, False),
        ("a harness named HEAD", {**row, "harness_commit": "HEAD"}, False),
        ("an upper-case harness commit", {**row, "harness_commit": harness.upper()}, False),
        ("a harness git could read as an option", {**row, "harness_commit": "-" + harness[1:]}, False),
        ("two harness commits", {**row, "harness_commit": f"{harness} {harness}"}, False),
        ("an unknown code revision", {**row, "code_revision": "unknown"}, False),
        ("a blank data blob", {**row, "data_blob": ""}, False),
    ]


@pytest.mark.parametrize("label, row, verdict", variants(), ids=[v[0] for v in variants()])
def test_every_path_gives_one_row_the_same_verdict(paths, label, row, verdict):
    assert {name: bool(check(row)) for name, check in paths.items()} == dict.fromkeys(paths, verdict), label


def test_every_committed_tuned_row_gets_one_verdict_from_every_path(paths):
    rows = tuned_rows()
    assert rows
    for row in rows:
        assert len({bool(check(row)) for check in paths.values()}) == 1, row


def test_a_url_form_revision_was_a_row_on_which_the_table_generator_disagreed_with_the_audit(paths):
    row = {**audited_row(), "code_revision": URL + audited_row()["code_revision"]}
    assert paths["paper_audit"](row) and paths["h2h_tables"](row) and paths["ingest_record"](row)


def test_no_second_copy_of_the_verifier_remains():
    audit = load("paper_audit_no_copy", "scripts/paper_audit.py")
    tables = load("h2h_tables_no_copy", "paper/jctc/scripts/h2h_tables.py")
    for function in (audit.tuned_gmin_harness_verified, tables.tuned_gmin_harness_verified):
        assert "revision_reach.tuned_gmin_harness_verified(" in inspect.getsource(function)
        assert "git" not in inspect.getsource(function).replace("tuned_gmin", "")
    for relative in ("scripts/ingest_record.py", "paper/jctc/scripts/h2h_tables.py"):
        source = (ROOT / relative).read_text()
        for command in ("merge-base", "--batch-check", "is-ancestor"):
            assert command not in source, (relative, command)


def test_the_pins_and_forms_are_defined_once():
    audit = load("paper_audit_pins_once", "scripts/paper_audit.py")
    tables = load("h2h_tables_pins_once", "paper/jctc/scripts/h2h_tables.py")
    ingest = load("ingest_record_pins_once", "scripts/ingest_record.py")
    assert audit.EXTERNAL_REVISIONS is revision_reach.EXTERNAL_REVISIONS
    assert ingest.EXTERNAL_REVISIONS == tables.EXTERNAL_PINS == tuple(revision_reach.EXTERNAL_REVISIONS)
    assert audit.external_form is ingest.external_form is revision_reach.external_form
    assert audit.external_pin is ingest.external_pin is revision_reach.external_pin


@pytest.mark.parametrize(
    "value, full",
    [
        (HELD, True),
        (HELD[:39], False),
        (HELD[:7], False),
        (HELD + "0", False),
        (HELD.upper(), False),
        (" " + HELD, False),
        ("", False),
        (None, False),
        ("unknown", False),
        ("HEAD", False),
    ],
)
def test_a_full_sha_is_forty_lowercase_hex_characters_and_nothing_else(value, full):
    assert revision_reach.is_full_sha(value) is full


def test_an_abbreviated_sha_is_not_a_held_revision_though_git_resolves_it():
    assert revision_reach.commit_held(HELD[:12], ROOT)
    assert not revision_reach.revision_reachable(HELD[:12], ROOT)
    assert revision_reach.revision_reachable(HELD, ROOT)
    pin = next(iter(revision_reach.EXTERNAL_REVISIONS))
    assert revision_reach.revision_reachable(pin, ROOT)
    assert not revision_reach.revision_reachable(pin[:12], ROOT)
    assert revision_reach.external_pin(pin[:12], abbreviated=True)
    assert not revision_reach.external_pin(pin[:6], abbreviated=True)


def record_blob(header: str, *rows: str) -> bytes:
    return (header + "\n" + "\n".join(rows) + "\n").encode()


def test_a_record_citing_an_abbreviated_sha_is_refused_in_every_revision_field():
    ingest = load("ingest_record_abbreviated", "scripts/ingest_record.py")
    for field in ingest.REVISION_FIELDS:
        header = f"system,harness_revision,{field}" if field != "harness_revision" else "system,harness_revision,head"
        column = field if field != "harness_revision" else "head"
        for short in (HELD[:7], HELD[:12], HELD[:39]):
            with pytest.raises(SystemExit, match=f"{column} is abbreviated SHA"):
                ingest.require_reachable({"bench/x/rows.csv": record_blob(header, f"fixture,{HELD},{short}")}, ROOT)
            with pytest.raises(SystemExit, match="abbreviated SHA"):
                ingest.require_reachable({"bench/x/rows.csv": record_blob(header, f"fixture,{HELD},{HELD} {short}")}, ROOT)


def test_the_record_sha_is_checked_by_the_shared_validator():
    ingest = load("ingest_record_name", "scripts/ingest_record.py")
    source = (ROOT / "scripts/ingest_record.py").read_text()
    assert "revision_reach.is_full_sha(sha)" in source
    assert "{7,40}" not in source and not hasattr(ingest, "HEX")


def test_the_committed_record_rows_cite_full_shas_only():
    ingest = load("ingest_record_committed", "scripts/ingest_record.py")
    seen = 0
    for name, record in ingest.production_catalog().items():
        for relative in record.paths:
            path = ROOT / relative
            if not path.exists():
                continue
            rows = list(csv.DictReader(io.StringIO(path.read_text())))
            for number, row in enumerate(rows, start=2):
                for field in ingest.REVISION_FIELDS:
                    if field in row:
                        seen += 1
                        assert ingest.malformed_revision(row[field]) is None, (name, relative, number, field)
    assert seen > 1000


def copy_ase_tree(tmp_path: Path, system: str, shorten: bool) -> Path:
    source = ROOT / "bench" / f"ase-{system}" / "rows.csv"
    rows = list(csv.DictReader(source.open(newline="")))
    target = tmp_path / "bench" / f"ase-{system}"
    target.mkdir(parents=True)
    with (target / "rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "code_revision": row["code_revision"][:12] if shorten else row["code_revision"]})
    return tmp_path


def test_the_ase_refresh_refuses_an_abbreviated_revision_and_reads_the_full_one(tmp_path, monkeypatch):
    ase = load("ase_panel_summary_shared", "scripts/ase_panel/summarize.py")
    monkeypatch.setattr(ase, "ROOT", copy_ase_tree(tmp_path / "full", "lj38", shorten=False))
    assert ase.load("lj38")
    monkeypatch.setattr(ase, "ROOT", copy_ase_tree(tmp_path / "short", "lj38", shorten=True))
    with pytest.raises(SystemExit, match="invalid revision"):
        ase.load("lj38")


def test_the_nonlj_refresh_refuses_an_abbreviated_revision(tmp_path, monkeypatch):
    nonlj = load("ase_nonlj_summary_shared", "scripts/ase_nonlj/summarize.py")
    source = ROOT / "bench" / "ase-nonlj" / "rerun-20261119" / "rows.csv"
    rows = list(csv.DictReader(source.open(newline="")))
    for label, shorten in (("short", True), ("full", False)):
        target = tmp_path / label / "bench" / "ase-nonlj" / "rerun-20261119"
        target.mkdir(parents=True)
        with (target / "rows.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, "code_revision": row["code_revision"][:12] if shorten else row["code_revision"]})
    monkeypatch.setattr(nonlj, "ROOT", tmp_path / "full")
    assert nonlj.load_rows()
    monkeypatch.setattr(nonlj, "ROOT", tmp_path / "short")
    with pytest.raises(SystemExit, match="invalid revision"):
        nonlj.load_rows()


def test_a_runner_takes_its_revision_from_the_shared_validator():
    for relative in ("scripts/ase_panel/run_one.py", "scripts/ase_panel/summarize.py", "scripts/ase_nonlj/summarize.py"):
        source = (ROOT / relative).read_text()
        assert "revision_reach.is_full_sha(" in source, relative


def test_one_validator_serves_the_python_readers_of_a_revision():
    """A pattern for forty hex digits appears outside the shared module only where the text read is not a row."""
    allowed = {
        "scripts/revision_reach.py",
        "scripts/paper_audit.py",
        "paper/jctc/scripts/paper_constants.py",
    }
    found = set()
    for base in ("scripts", "paper"):
        for path in (ROOT / base).rglob("*.py"):
            if re.search(r"\[0-9a-f\]\{(?:7|8),?40\}|\[0-9a-f\]\{40\}", path.read_text()):
                found.add(path.relative_to(ROOT).as_posix())
    assert found <= allowed, sorted(found - allowed)

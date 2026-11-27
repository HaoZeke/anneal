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
import os
import re
import shutil
import subprocess
import sys
import types
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


@pytest.mark.parametrize(("script", "subdir"), (
    ("summarize.py", ""),
    ("summarize_rerun.py", "rerun-20261119"),
))
def test_the_nonlj_refresh_refuses_an_abbreviated_revision(tmp_path, monkeypatch, script, subdir):
    nonlj = load(f"ase_nonlj_summary_shared_{script[:-3]}", f"scripts/ase_nonlj/{script}")
    source = ROOT / "bench" / "ase-nonlj" / subdir / "rows.csv"
    rows = list(csv.DictReader(source.open(newline="")))
    for label, shorten in (("short", True), ("full", False)):
        target = tmp_path / label / "bench" / "ase-nonlj" / subdir
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
    for relative in ("scripts/ase_panel/run_one.py", "scripts/ase_panel/summarize.py", "scripts/ase_nonlj/summarize.py",
                     "scripts/ase_nonlj/summarize_rerun.py"):
        source = (ROOT / relative).read_text()
        assert "revision_reach.is_full_sha(" in source, relative


SHA_CHECKS = (
    re.compile(r"\{40\}|\{\d*,40\}|\{40,\d*\}"),
    re.compile(r"len\([^)]*\)\s*(?:==|!=|<=|>=|<|>)\s*40\b"),
    re.compile(r"0123456789abcdef|hexdigits|a-fA-F|\[a-f0-9\]|ABCDEF"),
)
MACHINE_CHECKS = (
    re.compile(
        r"/etc/machine-id|product_uuid|IOPlatformUUID|\bioreg\b|machdep\.cpu|/proc/cpuinfo"
        r"|ANNEAL_MACHINE_ID|ANNEAL_CPU_MODEL|MACHINE_ID_SALT"
    ),
)
DIGEST_CHECK = 'if len(first) == 64 and all(c in "0123456789abcdef" for c in first):'
DIGEST_REASON = "a 64-character sha256 digest of a data file, not a commit id"
# (file, exact line content) -> why the line is not a commit id or host identity check.
EXEMPT = {
    ("paper/jctc/scripts/fig_landfold_maps.py", DIGEST_CHECK): DIGEST_REASON,
    ("paper/jctc/scripts/lj75_gmin_table.py", DIGEST_CHECK): DIGEST_REASON,
    ("paper/jctc/scripts/results_jcc.py", DIGEST_CHECK): DIGEST_REASON,
    ("paper/jctc/scripts/lj_csm_sizes_table.py", DIGEST_CHECK): DIGEST_REASON,
    ("paper/jctc/scripts/system_visuals.py", DIGEST_CHECK): DIGEST_REASON,
    (
        "scripts/h2h_gmin_board.sh",
        "flags=$(awk -F: '/^flags/ {gsub(/^[ \\t]+/, \"\", $2); print $2; exit}' /proc/cpuinfo)",
    ): "the CPU feature flags of a log file, not a machine identity",
    (
        "scripts/h2h_receipt.sh",
        "echo \"mhz=$(awk -F': ' '/cpu MHz/{print $2; exit}' /proc/cpuinfo)\"",
    ): "the clock speed of a receipt, not a machine identity",
    (
        "scripts/h2o16_v9_one_bench.sh",
        "awk '/^cpu MHz/ {print}' /proc/cpuinfo > \"$dest/${when}_mhz.raw\"",
    ): "raw clock speeds of a run, not a machine identity",
    (
        "scripts/run_cuh2_fcc_rec.sh",
        "awk -F: '/^[Cc]pu MHz/ { gsub(/^[ \\t]+/, \"\", $2); print $2 }' /proc/cpuinfo",
    ): "clock speeds of a run, not a machine identity",
    (
        "paper/jctc/scripts/water_h2h.py",
        'for line in Path("/proc/cpuinfo").read_text().splitlines():',
    ): "a free-text description for a log header (family and model number too), not a row column",
}
HOMES = {"scripts/revision_reach.py": SHA_CHECKS, "scripts/machine_provenance.py": MACHINE_CHECKS}


def guarded_lines():
    for base in ("scripts", "bench", "paper/jctc/scripts"):
        for path in sorted((ROOT / base).rglob("*")):
            relative = path.relative_to(ROOT).as_posix()
            if path.suffix not in {".py", ".sh"} or "__pycache__" in path.parts or not path.is_file():
                continue
            for number, line in enumerate(path.read_text().splitlines(), start=1):
                yield relative, number, line.strip()


def test_no_local_commit_id_check_or_machine_identity_read_appears_outside_its_home_module():
    """A commit id is judged by revision_reach and a machine by machine_provenance, whatever the file.

    There is no exemption for a file. A line that is neither is exempt by its exact content and a reason.
    """
    found = []
    used = set()
    for relative, number, line in guarded_lines():
        for pattern in (*SHA_CHECKS, *MACHINE_CHECKS):
            if pattern.search(line) and not any(
                pattern in checks and relative == home for home, checks in HOMES.items()
            ):
                if (relative, line) in EXEMPT:
                    used.add((relative, line))
                else:
                    found.append(f"{relative}:{number}: {line}")
                break
    assert not found, "validate commit ids with revision_reach and read the machine through machine_provenance:\n" + "\n".join(found)
    assert used == set(EXEMPT), sorted(set(EXEMPT) - used)
    assert all(reason.strip() and "\n" not in reason for reason in EXEMPT.values())


def test_the_guard_flags_the_checks_it_exists_for():
    flagged = (
        'ok = re.fullmatch(r"[0-9a-f]{40}", rgpot)',
        "ok = re.fullmatch(r'[0-9a-f]{7,40}', value)",
        "if len(sha) != 40:",
        "if [[ ! $REV =~ ^[0-9a-f]{40}$ ]]; then",
        'return len(name) == 40 and not name.strip("0123456789abcdef")',
        'text = Path("/etc/machine-id").read_text()',
        'os.environ.get("ANNEAL_MACHINE_ID")',
    )
    for line in flagged:
        assert any(pattern.search(line) for pattern in (*SHA_CHECKS, *MACHINE_CHECKS)), line
    for line in ("if len(group) != 12:", 'digest = hashlib.sha256(data).hexdigest()', "r'[0-9a-f]{64}'"):
        assert not any(pattern.search(line) for pattern in (*SHA_CHECKS, *MACHINE_CHECKS)), line


def test_the_shell_check_of_a_full_sha_answers_through_the_shared_validator():
    import subprocess

    def run(*values: str) -> int:
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "revision_reach.py"), "full-sha", *values],
            capture_output=True, check=False,
        ).returncode

    assert run(HELD) == 0
    assert run(HELD, HELD) == 0
    assert run(HELD[:12]) == 1
    assert run(HELD, HELD.upper()) == 1
    assert run("") == 1
    assert run() == 2


def test_the_commit_id_pattern_is_the_one_place_that_names_a_prefix_length():
    assert re.fullmatch(revision_reach.commit_id_pattern(), HELD)
    assert not re.fullmatch(revision_reach.commit_id_pattern(), HELD[:39])
    assert re.fullmatch(revision_reach.commit_id_pattern(7), HELD[:7])
    assert not re.fullmatch(revision_reach.commit_id_pattern(8), HELD[:7])


def git_text(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def test_the_listed_tuned_rewrite_is_what_accepts_the_committed_rows(monkeypatch):
    head = git_text("rev-parse", "HEAD:scripts/gmin_panel/tuned.py")
    harness = audited_row()["harness_commit"]
    old = git_text("rev-parse", f"{harness}:scripts/gmin_panel/tuned.py")
    recorded_old, heads = revision_reach.TUNED_SOURCE_REWRITES[harness]
    assert old != head
    assert recorded_old == old
    assert head in heads and len(heads) == 2
    legacy = next(blob for blob in heads if blob != head)
    assert revision_reach.tuned_source_accepted(harness, old, head)
    assert revision_reach.tuned_source_accepted(harness, old, legacy)
    assert not revision_reach.tuned_source_accepted(harness, old, "0" * 40)
    assert not revision_reach.tuned_source_accepted(harness, "0" * 40, head)
    assert not revision_reach.tuned_source_accepted(HELD, old, head)
    assert not revision_reach.tuned_source_accepted(HELD, old, legacy)
    assert revision_reach.tuned_gmin_harness_verified(audited_row())
    monkeypatch.setattr(revision_reach, "TUNED_SOURCE_REWRITES", {})
    assert not revision_reach.tuned_gmin_harness_verified(audited_row())
    monkeypatch.setattr(revision_reach, "TUNED_SOURCE_REWRITES", {harness: (old, frozenset({"0" * 40}))})
    assert not revision_reach.tuned_gmin_harness_verified(audited_row())
    monkeypatch.setattr(revision_reach, "TUNED_SOURCE_REWRITES", {HELD: (old, frozenset({head, legacy}))})
    assert not revision_reach.tuned_gmin_harness_verified(audited_row())


def test_a_committed_tuned_row_verifies_at_head(paths):
    rows = [row for row in tuned_rows() if row["system"] in revision_reach.TUNED_SYSTEMS]
    assert rows
    for row in rows:
        assert {name: bool(check(row)) for name, check in paths.items()} == dict.fromkeys(paths, True)


def side_commit(repo: Path, parent: str, files: dict[str, str]) -> str:
    """A commit on ``parent`` in ``repo`` with ``files`` replaced, made without a ref or a checkout."""
    env = {
        "GIT_INDEX_FILE": str(repo / "side.index"),
        "GIT_AUTHOR_NAME": "fixture", "GIT_AUTHOR_EMAIL": "fixture@example.org",
        "GIT_COMMITTER_NAME": "fixture", "GIT_COMMITTER_EMAIL": "fixture@example.org",
    }

    def git(*args: str, stdin: str | None = None) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args], input=stdin, capture_output=True, text=True, check=True,
            env={**os.environ, **env},
        ).stdout.strip()

    git("read-tree", parent)
    for path, text in files.items():
        blob = git("hash-object", "-w", "--stdin", stdin=text)
        git("update-index", "--add", "--cacheinfo", f"100644,{blob},{path}")
    return git("commit-tree", git("write-tree"), "-p", parent, "-m", "side commit")


@pytest.fixture()
def shared_clone(tmp_path: Path) -> Path:
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(ROOT), str(clone)], check=True)
    subprocess.run(["git", "-C", str(clone), "checkout", "--quiet", "--detach", git_text("rev-parse", "HEAD")], check=True)
    return clone


def test_a_side_commit_that_keeps_the_tuned_source_but_changes_what_it_imports_is_refused(shared_clone):
    row = audited_row()
    harness = row["harness_commit"]
    panel = "scripts/gmin_panel/run_panel.py"
    recorded = git_text("show", f"{harness}:{panel}")
    assert "\nLJ_STEP = 0.40\n" in recorded
    stepped = recorded.replace("\nLJ_STEP = 0.40\n", "\nLJ_STEP = 999.0\n")
    head_tuned = git_text("show", "HEAD:scripts/gmin_panel/tuned.py") + "\n"
    assert revision_reach.tuned_gmin_harness_verified(row, shared_clone)
    for label, files in (
        ("the recorded tuned.py", {panel: stepped}),
        ("the tuned.py of HEAD", {panel: stepped, "scripts/gmin_panel/tuned.py": head_tuned}),
        ("the recorded tree under another commit", {}),
    ):
        side = side_commit(shared_clone, harness, files)
        assert side != harness
        assert not revision_reach.tuned_gmin_harness_verified({**row, "harness_commit": side}, shared_clone), label


def test_a_forged_row_naming_an_ancestor_or_url_harness_still_needs_the_recorded_data_and_tuned_source(paths):
    row = audited_row()
    head = git_text("rev-parse", "HEAD")
    assert revision_reach.commit_held(head, ROOT) and revision_reach.commit_held(HELD, ROOT)
    assert {name: bool(check({**row, "harness_commit": head})) for name, check in paths.items()} == dict.fromkeys(paths, True)
    for label, forged in (
        ("HEAD with a bogus data blob", {**row, "harness_commit": head, "data_blob": "0" * 40}),
        ("an ancestor without the tuned harness", {**row, "harness_commit": HELD}),
        ("an ancestor without the tuned harness and a bogus data blob", {**row, "harness_commit": HELD, "data_blob": "0" * 40}),
        ("a URL at HEAD with a bogus data blob", {**row, "harness_commit": URL + head, "data_blob": "0" * 40}),
        ("a URL at a commit this repository lacks", {**row, "harness_commit": URL + "a" * 40}),
    ):
        assert {name: bool(check(forged)) for name, check in paths.items()} == dict.fromkeys(paths, False), label


@pytest.mark.skipif(sys.version_info < (3, 12), reason="the recorded tuned.py needs Python 3.12 to run")
def test_the_rewritten_tuned_source_builds_the_same_commands(monkeypatch):
    harness = audited_row()["harness_commit"]
    real = ROOT / "scripts/gmin_panel/tuned.py"
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    monkeypatch.syspath_prepend(str(ROOT / "scripts/gmin_panel"))

    def run_source(name: str, source: str):
        module = types.ModuleType(name)
        module.__file__ = str(real)
        monkeypatch.setitem(sys.modules, name, module)
        exec(compile(source, str(real), "exec"), module.__dict__)
        return module

    recorded = run_source("tuned_recorded", git_text("show", f"{harness}:scripts/gmin_panel/tuned.py"))
    current = run_source("tuned_current", real.read_text())
    for name, kind, n, reference, rho, binary in (
        ("lj98", "lj", 98, -543.665361, 0.0, "h2h_gmin_proof"),
        ("lj104", "lj", 104, -582.086642, 0.0, "h2h_gmin_proof"),
        ("m80r6", "morse", 80, -378.333471, 6.0, "h2h_morse"),
    ):
        commands = [
            module.gmin_command(module.Cell(name, kind, n, reference, rho, 4_000_000, binary), 2000, 4_000_000, "DATA")
            for module in (recorded, current)
        ]
        assert commands[0] == commands[1] and '"$BIN/GMIN"' in commands[0], name
        runs = [
            module.anneal_command(module.Cell(name, kind, n, reference, rho, 4_000_000, binary), 2000, 4_000_000)
            for module in (recorded, current)
        ]
        assert runs[0] == runs[1] and f'"$BIN/{binary}"' in runs[0], name
        inputs = [module.data_shell_file(name, 2000) for module in (recorded, current)]
        assert inputs[0] == inputs[1] and f"inputs/{name}.data" in inputs[0], name

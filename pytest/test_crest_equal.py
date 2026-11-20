# Copyright (c) anneal developers
"""The committed rows predate the 4000-call cap.

They charge CREST the calls the driver served, under a ceiling of 50000
calls, and the 4000 in their budget column is a label. The tests of the
later protocol serve the CREST arm only ``BUDGET`` calls and never charge a
row less than CREST's own energy+grad total. The committed rows must pair
the arms seed for seed and agree with the sentence the README quotes.
"""

import csv
import importlib.util
import os
import shutil
import stat
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("crest_equal_panel", ROOT / "scripts" / "crest_equal" / "run_panel.py")
panel = importlib.util.module_from_spec(SPEC)
sys.modules["crest_equal_panel"] = panel
SPEC.loader.exec_module(panel)

PANEL = ROOT / "bench" / "crest-equal"
AUDIT_SPEC = importlib.util.spec_from_file_location("paper_audit_crest", ROOT / "scripts" / "paper_audit.py")
audit_mod = importlib.util.module_from_spec(AUDIT_SPEC)
AUDIT_SPEC.loader.exec_module(audit_mod)
CONSTANTS_SPEC = importlib.util.spec_from_file_location(
    "paper_constants_crest",
    ROOT / "paper" / "jctc" / "scripts" / "paper_constants.py",
)
constants_mod = importlib.util.module_from_spec(CONSTANTS_SPEC)
CONSTANTS_SPEC.loader.exec_module(constants_mod)
RGPOT_REVISION = constants_mod.values()["RgpotRevision"]


def committed(name: str) -> list[dict[str, str]]:
    with (PANEL / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_crest_log_total_is_read_from_its_own_line(tmp_path):
    log = tmp_path / "crest.out"
    log.write_text(" * Total number of energy+grad calls: 151074\n * wall-time: 0 h\n")
    assert panel.crest_log_calls(log) == 151074


def test_missing_or_empty_total_in_crest_log_is_refused(tmp_path):
    log = tmp_path / "crest.out"
    log.write_text("no total here\n")
    with pytest.raises(SystemExit):
        panel.crest_log_calls(log)
    with pytest.raises(SystemExit):
        panel.crest_log_calls(tmp_path / "missing.out")


def test_two_totals_in_one_log_are_refused(tmp_path):
    log = tmp_path / "crest.out"
    log.write_text(" * Total number of energy+grad calls: 5\n * Total number of energy+grad calls: 6\n")
    with pytest.raises(SystemExit):
        panel.crest_log_calls(log)


def test_a_row_is_charged_the_larger_of_the_served_and_the_logged_count():
    assert panel.charged_calls(3055, 3055) == 3055
    assert panel.charged_calls(3055, 14210) == 14210
    assert panel.charged_calls(4000, 3900) == 4000
    with pytest.raises(SystemExit):
        panel.charged_calls(3055, None)


def fake_tools(tmp_path: Path, log_total: int | None, served: int):
    driver = tmp_path / "driver"
    driver.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import sys, time
            from pathlib import Path
            command = sys.argv[1]
            if command == "serve":
                socket, rundir, molecules, budget, target = sys.argv[2:7]
                run = Path(rundir)
                run.mkdir(parents=True, exist_ok=True)
                Path(run / "budget-arg").write_text(budget)
                (run / "ready").write_text("ready")
                while not Path(socket + ".stopped").exists():
                    time.sleep(0.01)
                (run / "driver.txt").write_text(
                    "calls: {served}\\ninside_s: 1.5\\nbest_eV: -829.0\\nstop: none\\n"
                )
            elif command == "stop":
                Path(sys.argv[2] + ".stopped").write_text("x")
            """
        )
    )
    crest_line = (
        f'echo " * Total number of energy+grad calls: {log_total}"'
        if log_total is not None
        else 'echo "aborted before summary"'
    )
    crest = tmp_path / "crest"
    crest.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/sh
            {crest_line}
            """
        )
    )
    for tool in (driver, crest):
        tool.chmod(tool.stat().st_mode | stat.S_IEXEC)
    return driver, crest


@pytest.mark.skipif(shutil.which("taskset") is None, reason="needs taskset")
@pytest.mark.parametrize("log_total, served, charged", [(3100, 3050, 3100), (900, 3050, 3050)])
def test_the_crest_arm_is_served_the_ledger_and_billed_what_the_log_says(tmp_path, log_total, served, charged):
    driver, crest = fake_tools(tmp_path, log_total, served)
    start = tmp_path / "start.xyz"
    start.write_text("18\n\n" + "O 0 0 0\n" * 18)
    out = tmp_path / "out"
    result = panel.run_crest(crest, Path("/bin/true"), driver, start, out, "0", "crest")
    assert (out / "driver" / "budget-arg").read_text() == str(panel.BUDGET)
    assert result["driver_calls"] == served
    assert result["crest_log_calls"] == log_total
    assert result["charged"] == charged


@pytest.mark.skipif(shutil.which("taskset") is None, reason="needs taskset")
def test_run_crest_fails_loudly_when_crest_prints_no_total(tmp_path):
    driver, crest = fake_tools(tmp_path, None, 3050)
    start = tmp_path / "start.xyz"
    start.write_text("18\n\n" + "O 0 0 0\n" * 18)
    with pytest.raises(SystemExit):
        panel.run_crest(crest, Path("/bin/true"), driver, start, tmp_path / "out", "0", "crest")


def test_the_log_is_kept_gzipped_with_its_digest(tmp_path):
    import gzip

    source = tmp_path / "crest.out"
    source.write_text(" * Total number of energy+grad calls: 7\n")
    digest = panel.keep_log(source, tmp_path / "logs", "crest", 2000, "723b7c23" + "0" * 32)
    kept = tmp_path / "logs" / "crest-2000-723b7c23.out.gz"
    assert gzip.open(kept, "rt").read() == source.read_text()
    assert digest == panel.sha256(source)


def test_the_arms_are_paired_seed_for_seed_in_the_committed_rows():
    rows = committed("rows.csv")
    for revision in {row["code_revision"] for row in rows}:
        mine = [row for row in rows if row["code_revision"] == revision]
        by_arm = {arm: sorted(int(row["seed"]) for row in mine if row["arm"] == arm) for arm in panel.ARMS}
        assert all(seeds == list(panel.SEEDS) for seeds in by_arm.values()), by_arm


def test_no_committed_row_is_charged_above_the_ledger():
    assert max(int(row["charged"]) for row in committed("rows.csv")) <= panel.BUDGET


def test_the_readme_omits_the_sign_affected_hit_claim():
    sentence = panel.claim_sentence(committed("summary.csv"))
    readme = (PANEL / "README.md").read_text()
    assert "median 916.5 calls" in sentence
    assert sentence not in readme
    assert "hits 1 of 12" not in readme
    assert "hits 4 of 12" not in readme


def test_the_summary_is_what_the_rows_give():
    pytest.importorskip("scipy.stats")
    assert panel.summarise(committed("rows.csv")) == committed("summary.csv")


def test_a_row_charged_above_the_ledger_is_reported():
    audit = audit_mod.Audit()
    rows = audit.rows("bench/crest-equal/rows.csv")
    next(row for row in rows if row["arm"] == "crest")["charged"] = str(panel.BUDGET + 1)
    audit_mod.crest_equal_bench(audit)
    labels = {row[2] for row in audit.drifts()}
    assert "no row is charged above the budget" in labels


def test_an_unpaired_seed_is_reported():
    audit = audit_mod.Audit()
    rows = audit.rows("bench/crest-equal/rows.csv")
    next(row for row in rows if row["arm"] == "crest")["seed"] = "9999"
    audit_mod.crest_equal_bench(audit)
    assert any(row[2].endswith("crest seeds") for row in audit.drifts())


def test_a_drifted_sentence_is_reported():
    audit = audit_mod.Audit()
    path = audit_mod.CREST_EQUAL_README
    audit.text_cache[path] = audit.text(path).replace("2510 to 3531 calls", "2510 to 3530 calls")
    audit_mod.crest_equal_bench(audit)
    assert [row[2] for row in audit.drifts()] == ["prose"]

    injected = audit_mod.Audit()
    injected.text_cache[path] = injected.text(path) + "\n" + panel.claim_sentence(committed("summary.csv")) + "\n"
    audit_mod.crest_equal_bench(injected)
    assert [row[2] for row in injected.drifts()] == ["sign-affected hit-rate claim is absent from crest-equal README"]


def _sample_count_rows():
    return [
        {"arm": "anneal", "seed": "2000", "driver_calls": "3021", "crest_log_calls": "", "charged": "3021"},
        {"arm": "crest", "seed": "2000", "driver_calls": "3050", "crest_log_calls": "3100", "charged": "3100"},
        {"arm": "crest-wall", "seed": "2000", "driver_calls": "950", "crest_log_calls": "900", "charged": "950"},
    ]


def test_call_counts_rows_match_the_larger_of_driver_and_crest_log():
    audit = audit_mod.Audit()
    audit.read["bench/crest-equal/call-counts.csv"] = _sample_count_rows()
    audit_mod.crest_equal_bench(audit)
    assert not audit.drifts()


def test_a_call_counts_row_undercharged_below_crest_log_is_reported():
    audit = audit_mod.Audit()
    counts = _sample_count_rows()
    counts[1]["charged"] = "3050"
    audit.read["bench/crest-equal/call-counts.csv"] = counts
    audit_mod.crest_equal_bench(audit)
    assert [(row[1], row[2]) for row in audit.drifts()] == [
        ("bench/crest-equal/call-counts.csv", "row 2 crest seed 2000 charged")
    ]


def test_a_crest_call_counts_row_without_log_total_is_reported():
    audit = audit_mod.Audit()
    counts = _sample_count_rows()
    counts[1]["crest_log_calls"] = ""
    audit.read["bench/crest-equal/call-counts.csv"] = counts
    audit_mod.crest_equal_bench(audit)
    assert [(row[1], row[2]) for row in audit.drifts()] == [
        ("bench/crest-equal/call-counts.csv", "row 2 crest seed 2000 charged")
    ]


def _sample_record_pair():
    row = {
        "system": "h2o6",
        "arm": "crest",
        "seed": 2000,
        "budget": 4000,
        "hit": "false",
        "charged": 3100,
        "polished_Eh": "-30.49",
        "gap_Eh": "1.0e-03",
        "engine_eV": "-829.0",
        "wall_s": "10.0",
        "pot_s": "2.0",
        "outside_s": "8.0",
        "crest_len_ps": "0.90",
        "crest_exit": "0",
        "surrogate_calls": "0",
        "crest_flags": panel.CREST_FLAGS,
        "code_revision": "0" * 40,
        "host": "bench",
        "date": "2026-11-18",
    }
    count = {
        **row,
        "driver_calls": 3050,
        "driver_stop": "none",
        "crest_log_calls": 3100,
        "crest_out_sha256": "a" * 64,
    }
    return count, row


def test_record_row_writes_rows_before_call_counts(tmp_path, monkeypatch):
    counts_path = tmp_path / "call-counts.csv"
    rows_path = tmp_path / "rows.csv"
    count, row = _sample_record_pair()
    seen = []
    real_replace = os.replace

    def spy_replace(src, dst):
        real_replace(src, dst)
        seen.append((Path(dst).name, rows_path.is_file(), counts_path.is_file()))

    monkeypatch.setattr(panel.os, "replace", spy_replace)
    panel.record_row(counts_path, rows_path, count, row)
    assert seen == [
        ("rows.csv", True, False),
        ("call-counts.csv", True, True),
    ]
    assert len(panel.read_csv(counts_path)) == 1
    assert len(panel.read_csv(rows_path)) == 1


def test_record_row_never_writes_call_counts_when_rows_write_fails(tmp_path, monkeypatch):
    counts_path = tmp_path / "call-counts.csv"
    rows_path = tmp_path / "rows.csv"
    count, row = _sample_record_pair()
    real_replace = os.replace
    replace_calls = 0
    attempted = []

    def failing_replace(src, dst):
        nonlocal replace_calls
        replace_calls += 1
        attempted.append(Path(dst))
        if replace_calls == 1:
            raise OSError("disk full")
        real_replace(src, dst)

    monkeypatch.setattr(panel.os, "replace", failing_replace)
    with pytest.raises(OSError, match="disk full"):
        panel.record_row(counts_path, rows_path, count, row)
    assert attempted == [rows_path]
    assert not counts_path.exists()
    assert not rows_path.exists()
    assert list(tmp_path.iterdir()) == []


def test_record_row_retry_appends_without_replacing_the_first_row(tmp_path, monkeypatch):
    counts_path = tmp_path / "call-counts.csv"
    rows_path = tmp_path / "rows.csv"
    count, row = _sample_record_pair()
    real_replace = os.replace
    replace_calls = 0

    def failing_second_replace(src, dst):
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("disk full")
        real_replace(src, dst)

    monkeypatch.setattr(panel.os, "replace", failing_second_replace)
    with pytest.raises(OSError, match="disk full"):
        panel.record_row(counts_path, rows_path, count, row)
    assert not counts_path.exists()
    assert len(panel.read_csv(rows_path)) == 1
    assert panel.completed_runs(rows_path, counts_path) == set()
    assert [p.name for p in tmp_path.iterdir()] == ["rows.csv"]

    monkeypatch.setattr(panel.os, "replace", real_replace)
    panel.record_row(counts_path, rows_path, count, row)
    assert len(panel.read_csv(rows_path)) == 2
    assert len(panel.read_csv(counts_path)) == 1
    assert panel.completed_runs(rows_path, counts_path) == {(2000, "crest", "0" * 40)}


def test_record_row_second_write_appends_every_ledger(tmp_path):
    counts_path = tmp_path / "call-counts.csv"
    rows_path = tmp_path / "rows.csv"
    sidecar_path = tmp_path / "rgpot-revisions.csv"
    count, row = _sample_record_pair()
    sidecar = {
        "system": row["system"],
        "arm": row["arm"],
        "seed": row["seed"],
        "code_revision": row["code_revision"],
        "rgpot_revision": "1" * 40,
        "host": row["host"],
    }
    panel.record_row(counts_path, rows_path, count, row, sidecar_path, sidecar)
    before = {path: path.read_bytes() for path in (sidecar_path, rows_path, counts_path)}
    panel.record_row(counts_path, rows_path, count, row, sidecar_path, sidecar)
    for path, original in before.items():
        assert path.read_bytes().startswith(original)
        assert len(path.read_bytes()) > len(original)


def test_record_row_validates_both_headers_before_writing(tmp_path):
    counts_path = tmp_path / "call-counts.csv"
    rows_path = tmp_path / "rows.csv"
    rows_path.write_text("bad,header\n")
    count, row = _sample_record_pair()
    with pytest.raises(SystemExit):
        panel.record_row(counts_path, rows_path, count, row)
    assert not counts_path.exists()
    assert rows_path.read_text() == "bad,header\n"


def test_append_rows_is_atomic_on_write_failure(tmp_path, monkeypatch):
    rows_path = tmp_path / "rows.csv"
    _, row = _sample_record_pair()
    panel.append_rows(rows_path, panel.ROW_FIELDS, [row])
    before = rows_path.read_bytes()

    def failing_replace(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr(panel.os, "replace", failing_replace)
    with pytest.raises(OSError, match="replace failed"):
        panel.append_rows(rows_path, panel.ROW_FIELDS, [{**row, "seed": 2001}])
    assert rows_path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["rows.csv"]


def test_rgpot_revision_is_resolved_from_sidecar_or_refused(tmp_path, monkeypatch):
    monkeypatch.delenv("RGPOT_REVISION", raising=False)
    monkeypatch.delenv("RGPOT_ROOT", raising=False)
    engine = tmp_path / "libxtb_engine.so"
    engine.write_text("so\n")
    with pytest.raises(SystemExit, match="missing rgpot revision"):
        panel.rgpot_revision_of(engine)
    (tmp_path / "rgpot.sha").write_text(f"{RGPOT_REVISION}\n")
    assert panel.rgpot_revision_of(engine) == RGPOT_REVISION
    assert panel.rgpot_revision_of(engine, "ab" * 20) == "ab" * 20


def test_record_row_writes_rgpot_sidecar_before_rows(tmp_path, monkeypatch):
    counts_path = tmp_path / "call-counts.csv"
    rows_path = tmp_path / "rows.csv"
    sidecar_path = tmp_path / "rgpot-revisions.csv"
    writes = []
    real_replace = os.replace

    def spy_replace(src, dst):
        real_replace(src, dst)
        writes.append(Path(dst).name)

    monkeypatch.setattr(panel.os, "replace", spy_replace)
    count, row = _sample_record_pair()
    sidecar_entry = {
        "system": row["system"],
        "arm": row["arm"],
        "seed": row["seed"],
        "code_revision": row["code_revision"],
        "rgpot_revision": "1" * 40,
        "host": row["host"],
    }
    panel.record_row(
        counts_path,
        rows_path,
        count,
        row,
        sidecar_path=sidecar_path,
        sidecar_row=sidecar_entry,
    )
    assert writes == ["rgpot-revisions.csv", "rows.csv", "call-counts.csv"]
    assert "rgpot_revision" not in rows_path.read_text().splitlines()[0]
    assert panel.read_csv(sidecar_path) == [
        {k: str(v) for k, v in sidecar_entry.items()}
    ]


def test_crest_equal_rgpot_sidecar_drift_is_reported():
    audit = audit_mod.Audit()
    sidecar = audit_mod.CREST_EQUAL_RGPOT_SIDECAR
    audit.text_cache[sidecar] = audit.text(sidecar).replace(",bench\n", ",badrev,bench\n", 1)
    audit_mod.crest_equal_bench(audit)
    assert any(row[1] == sidecar for row in audit.drifts())

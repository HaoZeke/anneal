# Copyright (c) anneal developers
"""The (H2O)6 CREST panel charges CREST what it spent and caps it at the ledger.

The panel compares anneal and CREST on calls, so the CREST arm must be served
only ``BUDGET`` calls, and a row must never be charged less than CREST's own
energy+grad total. The committed rows must pair the arms seed for seed and
agree with the sentence the README quotes.
"""

import csv
import importlib.util
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


def committed(name: str) -> list[dict[str, str]]:
    with (PANEL / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_crest_log_total_is_read_from_its_own_line(tmp_path):
    log = tmp_path / "crest.out"
    log.write_text(" * Total number of energy+grad calls: 151074\n * wall-time: 0 h\n")
    assert panel.crest_log_calls(log) == 151074
    log.write_text("no total here\n")
    assert panel.crest_log_calls(log) is None
    assert panel.crest_log_calls(tmp_path / "missing.out") is None


def test_two_totals_in_one_log_are_refused(tmp_path):
    log = tmp_path / "crest.out"
    log.write_text(" * Total number of energy+grad calls: 5\n * Total number of energy+grad calls: 6\n")
    with pytest.raises(SystemExit):
        panel.crest_log_calls(log)


def test_a_row_is_charged_the_larger_of_the_served_and_the_logged_count():
    assert panel.charged_calls(3055, None) == 3055
    assert panel.charged_calls(3055, 3055) == 3055
    assert panel.charged_calls(3055, 14210) == 14210
    assert panel.charged_calls(4000, 3900) == 4000


def fake_tools(tmp_path: Path, log_total: int, served: int):
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
    crest = tmp_path / "crest"
    crest.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/sh
            echo " * Total number of energy+grad calls: {log_total}"
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


def test_the_readme_quotes_the_sentence_the_summary_gives():
    sentence = panel.claim_sentence(committed("summary.csv"))
    assert sentence in (PANEL / "README.md").read_text()
    assert "median 916.5 calls" in sentence


def test_the_summary_is_what_the_rows_give():
    assert panel.summarise(committed("rows.csv")) == committed("summary.csv")


def test_a_row_charged_above_the_ledger_is_reported():
    audit = audit_mod.Audit()
    rows = audit.rows("bench/crest-equal/rows.csv")
    next(row for row in rows if row["arm"] == "crest")["charged"] = str(panel.BUDGET + 1)
    audit_mod.crest_equal(audit)
    labels = {row[2] for row in audit.drifts()}
    assert "no row is charged above the budget" in labels


def test_an_unpaired_seed_is_reported():
    audit = audit_mod.Audit()
    rows = audit.rows("bench/crest-equal/rows.csv")
    next(row for row in rows if row["arm"] == "crest")["seed"] = "9999"
    audit_mod.crest_equal(audit)
    assert any(row[2].endswith("crest seeds") for row in audit.drifts())


def test_a_drifted_sentence_is_reported():
    audit = audit_mod.Audit()
    path = audit_mod.CREST_EQUAL_README
    audit.text_cache[path] = audit.text(path).replace("median 916.5 calls", "median 916 calls")
    audit_mod.crest_equal(audit)
    assert [row[2] for row in audit.drifts()] == ["prose"]

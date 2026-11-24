# Copyright (c) anneal developers
"""The same-session gate summaries: an arm short of seeds loses the gate,
and the committed gates read as their README states. CREST's rescore
against another reference."""

import argparse
import csv
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FLOOR_GATE_ROWS = ROOT / "bench" / "covalent-h2h" / "floor-gate" / "gfn2.csv"
PANEL_ROWS = ROOT / "bench" / "covalent-h2h" / "floor-gate" / "panel.csv"


def load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


floor_gate = load("gfn2_floor_gate", "scripts/covalent_h2h/gfn2_floor_gate.py")
arm_gate = load("arm_gate", "scripts/paper_0120/arm_gate.py")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def verdict(capsys) -> str:
    return capsys.readouterr().out.strip().splitlines()[-1]


def test_a_floor_gate_arm_short_of_a_seed_loses(tmp_path, capsys):
    rows = [r for r in read(FLOOR_GATE_ROWS) if r["system"] == "c20"]
    rows = [r for r in rows if not (r["arm"] == "floor" and r["seed"] == "2003")]
    floor_gate.summary(argparse.Namespace(rows=write(tmp_path / "rows.csv", rows), arms=["base", "floor"]))
    assert verdict(capsys) == "floor loses: c20 seeds"


def test_voided_si13_rows_do_not_enter_the_floor_gate(tmp_path, capsys):
    rows = read(FLOOR_GATE_ROWS)
    rows = [
        r for r in rows
        if not (r["system"] == "c20" and r["arm"] == "floor" and r["seed"] == "2003")
    ]
    floor_gate.summary(argparse.Namespace(rows=write(tmp_path / "rows.csv", rows), arms=["base", "floor"]))
    assert verdict(capsys) == "floor loses: c20 seeds"


def test_a_floor_gate_run_without_calls_is_summarised(tmp_path, capsys):
    pytest.importorskip("scipy.stats")
    rows = [r for r in read(FLOOR_GATE_ROWS) if r["system"] == "c20"]
    for row in rows:
        row["charged"] = "0"
    floor_gate.summary(argparse.Namespace(rows=write(tmp_path / "rows.csv", rows), arms=["base", "floor"]))
    out = capsys.readouterr().out
    assert "SCC failures 0/0 (no calls)" in out
    assert out.strip().splitlines()[-1] == "floor loses: nowhere"


def test_the_committed_floor_gate_loses_nowhere(capsys):
    pytest.importorskip("scipy.stats")
    floor_gate.summary(argparse.Namespace(rows=FLOOR_GATE_ROWS, arms=["base", "floor"]))
    out = capsys.readouterr().out
    assert "si13" not in out
    assert "c20 base: hits 0/12" in out
    assert "c20 floor: hits 0/12" in out
    assert out.strip().splitlines()[-1] == "floor loses: nowhere"


def test_a_panel_system_one_arm_left_out_or_short_of_a_seed_loses(tmp_path, capsys):
    rows = [r for r in read(PANEL_ROWS) if r["system"] in ("lj38", "lj55")]
    rows = [r for r in rows if not (r["arm"] == "floor" and r["system"] == "lj38")]
    rows = [r for r in rows if not (r["arm"] == "develop" and r["system"] == "lj55" and r["seed"] == "2005")]
    arm_gate.summary(argparse.Namespace(rows=write(tmp_path / "panel.csv", rows), arms=["develop", "floor"]))
    assert verdict(capsys) == "floor loses: lj38 h2h_gmin_proof seeds, lj55 h2h_gmin_proof seeds"


def test_floor_gate_references_are_cited_and_compared_by_tolerance():
    crest = load("gfn2_crest_ref", "scripts/covalent_h2h/gfn2_crest.py")
    references = {"sample": -12.5}
    sources = {"sample": "synthetic fixture"}
    floor_gate.validate_reference_sources(references, sources)
    assert crest.energy_hits(references["sample"], -12.5, 0.0)
    assert crest.energy_hits(-12.5, references["sample"], 0.0)


def test_a_floor_gate_reference_without_a_source_is_refused():
    with pytest.raises(ValueError, match="sample has no reference source"):
        floor_gate.validate_reference_sources({"sample": -12.5}, {"sample": ""})


@pytest.mark.parametrize(
    ("references", "sources"),
    [
        ({"sample": -12.5}, {}),
        ({}, {"sample": "synthetic fixture"}),
    ],
)
def test_floor_gate_reference_source_key_mismatches_are_refused(references, sources):
    with pytest.raises(ValueError, match="reference and source systems differ"):
        floor_gate.validate_reference_sources(references, sources)


def synthetic_rescore_rows():
    return [
        {
            "system": "sample",
            "code": "anneal-general",
            "seed": "1",
            "budget": "100",
            "charged": "10",
            "best": "-10.0",
        },
        {
            "system": "sample",
            "code": "anneal-general",
            "seed": "2",
            "budget": "100",
            "charged": "0",
            "best": "inf",
        },
        {
            "system": "sample",
            "code": "crest",
            "seed": "1",
            "budget": "20",
            "charged": "20",
            "best": "-9.0",
        },
        {
            "system": "sample",
            "code": "crest",
            "seed": "2",
            "budget": "30",
            "charged": "5",
            "best": "-8.0",
        },
        {
            "system": "sample",
            "code": "crest",
            "seed": "3",
            "budget": "0",
            "charged": "0",
            "best": "",
        },
    ]


def test_rescore_labels_stop_modes_and_no_results(tmp_path, monkeypatch):
    pytest.importorskip("scipy.stats")
    crest = load("gfn2_crest", "scripts/covalent_h2h/gfn2_crest.py")
    monkeypatch.setitem(crest.REFERENCE, "sample", -10.0)
    source = write(tmp_path / "rows.csv", synthetic_rescore_rows())
    out = tmp_path / "rescore.csv"
    crest.rescore([source], out, "sample", -9.5, "0" * 40, "")
    by_code = {row["code"]: row for row in read(out)}

    anneal = by_code["anneal-general"]
    assert (anneal["runs"], anneal["hits"]) == ("2", "1")
    assert anneal["no_result"] == "1"
    assert anneal["no_result_seeds"] == "2"
    assert "1 runs stopped at the -10.00000000 stop target" in anneal["stop_note"]
    assert anneal["median_charged_all"] == "5.0"
    assert anneal["median_charged_ran"] == "10.0"

    other = by_code["crest"]
    assert (other["runs"], other["hits"]) == ("3", "0")
    assert other["no_result"] == "1"
    assert other["no_result_seeds"] == "3"
    assert "1 runs ran to their own end" in other["stop_note"]
    assert "1 runs stopped short of their recorded budget" in other["stop_note"]
    assert other["median_budget_all"] == "20.0"
    assert other["median_budget_ran"] == "25.0"
    with pytest.raises(SystemExit, match="no CREST rows"):
        crest.rescore([source], out, "other", -1.0, "0" * 40, "")


def test_rescore_counts_zero_charge_finite_energy_only_as_no_result(
    tmp_path, monkeypatch
):
    crest = load("gfn2_crest_exclusive", "scripts/covalent_h2h/gfn2_crest.py")
    monkeypatch.setitem(crest.REFERENCE, "sample", -9.5)
    rows = [
        {
            "system": "sample",
            "code": "anneal-general",
            "seed": "1",
            "budget": "100",
            "charged": "0",
            "best": "-9.5",
        },
        {
            "system": "sample",
            "code": "anneal-general",
            "seed": "2",
            "budget": "100",
            "charged": "50",
            "best": "-9.6",
        },
        {
            "system": "sample",
            "code": "anneal-general",
            "seed": "3",
            "budget": "80",
            "charged": "80",
            "best": "-1.0",
        },
        {
            "system": "sample",
            "code": "crest",
            "seed": "1",
            "budget": "10",
            "charged": "10",
            "best": "-1.0",
        },
        {
            "system": "sample",
            "code": "crest",
            "seed": "4",
            "budget": "40",
            "charged": "0",
            "best": "-100.0",
        },
    ]
    source = write(tmp_path / "rows.csv", rows)
    out = tmp_path / "rescore.csv"
    crest.rescore([source], out, "sample", -9.5, "0" * 40, "")
    by_code = {row["code"]: row for row in read(out)}

    anneal = by_code["anneal-general"]
    assert (anneal["runs"], anneal["hits"], anneal["no_result"]) == ("3", "1", "1")
    assert anneal["no_result_seeds"] == "1"
    assert "1 produced no result (seeds 1)" in anneal["stop_note"]
    assert "stopped at the -9.50000000 stop target (seeds 2" in anneal["stop_note"]
    assert "1 runs ran to their own end (seeds 3" in anneal["stop_note"]

    other = by_code["crest"]
    assert (other["runs"], other["hits"], other["no_result"]) == ("2", "0", "1")
    assert other["no_result_seeds"] == "4"
    assert "1 produced no result (seeds 4)" in other["stop_note"]
    assert "1 runs ran to their own end (seeds 1" in other["stop_note"]
    assert "stop target" not in other["stop_note"]

    assert crest.energy_hits(-9.5, -9.5, crest.HIT_TOL)
    assert crest.energy_hits(-100.0, -9.5, crest.HIT_TOL)
    assert [crest._rescore_class(row, -9.5) for row in rows] == [
        "no_result",
        "hit",
        "miss",
        "miss",
        "no_result",
    ]
    for code, written in by_code.items():
        labels = [
            crest._rescore_class(row, -9.5)
            for row in rows
            if row["code"] == code
        ]
        hits = int(written["hits"])
        missing = int(written["no_result"])
        rest = int(written["runs"]) - hits - missing
        assert labels.count("hit") == hits
        assert labels.count("no_result") == missing
        assert labels.count("miss") == rest
        assert hits + missing + rest == len(labels)


def test_arm_calls_without_a_stop_classifies_only_row_endings():
    crest = load("gfn2_crest_no_stop", "scripts/covalent_h2h/gfn2_crest.py")
    rows = synthetic_rescore_rows()[2:4]

    calls = crest.arm_calls(rows, None)

    assert "1 runs ran to their own end" in calls["stop_note"]
    assert "1 runs stopped short of their recorded budget" in calls["stop_note"]
    assert "stop target" not in calls["stop_note"]


def test_arm_calls_with_all_no_results_has_empty_ran_medians():
    crest = load("gfn2_crest_no_results", "scripts/covalent_h2h/gfn2_crest.py")
    rows = [synthetic_rescore_rows()[1], synthetic_rescore_rows()[4]]

    calls = crest.arm_calls(rows, None)

    assert calls["no_result"] == "2"
    assert calls["median_budget_ran"] == ""
    assert calls["median_charged_ran"] == ""
    assert "2 produced no result" in calls["stop_note"]


def test_rescore_filters_revision_and_false_hits(tmp_path, monkeypatch):
    crest = load("gfn2_crest_filters", "scripts/covalent_h2h/gfn2_crest.py")
    monkeypatch.setitem(crest.REFERENCE, "sample", -10.0)
    rows = synthetic_rescore_rows()
    for row in rows:
        row["code_revision"] = "keep"
    stale = dict(rows[0], seed="3", code_revision="stale", best="-20.0")
    source = write(tmp_path / "rows.csv", [*rows, stale])
    false_hits = write(
        tmp_path / "false-hits.csv",
        [{
            "system": "sample",
            "code": "anneal-general",
            "code_revision": "keep",
            "seed": "1",
            "recorded_eV": "-10.0",
            "xtb_eV": "-8.0",
            "evidence": "fixture",
        }],
    )
    out = tmp_path / "rescore.csv"

    crest.rescore(
        [source],
        out,
        "sample",
        -9.5,
        "0" * 40,
        "",
        anneal_revision="keep",
        false_hits=false_hits,
    )

    by_code = {row["code"]: row for row in read(out)}
    assert (by_code["anneal-general"]["runs"], by_code["anneal-general"]["hits"]) == ("2", "0")
    assert by_code["anneal-general"]["seeds"] == "1 2"
    assert by_code["crest"]["runs"] == "3"


def test_rescore_preserves_legacy_and_appends_v2_rows(tmp_path, monkeypatch):
    pytest.importorskip("scipy.stats")
    crest = load("gfn2_crest_legacy", "scripts/covalent_h2h/gfn2_crest.py")
    monkeypatch.setitem(crest.REFERENCE, "sample", -10.0)
    source = write(tmp_path / "rows.csv", synthetic_rescore_rows())
    committed = ROOT / "bench" / "covalent-h2h" / "gfn2" / "rescore.csv"
    legacy = tmp_path / "rescore.csv"
    legacy.write_bytes(committed.read_bytes())
    before = legacy.read_bytes()

    crest.rescore([source], legacy, "sample", -9.5, "0" * 40, "")
    crest.rescore([source], legacy, "sample", -9.5, "1" * 40, "")

    assert legacy.read_bytes() == before
    future = tmp_path / "rescore-v2.csv"
    written = read(future)
    assert len(written) == 4
    assert list(written[0]) == crest.RESCORE_FIELDS
    assert written[0]["machine_id"]
    assert written[0]["cpu_model"]
    assert {row["code_revision"] for row in written} == {"0" * 40, "1" * 40}


def test_gfn2_crest_second_write_appends_without_replacing(tmp_path):
    crest = load("gfn2_crest_append", "scripts/covalent_h2h/gfn2_crest.py")
    path = tmp_path / "rows.csv"
    fields = ["system", "seed"]
    crest.append_rows(path, fields, [{"system": "si13", "seed": "2000"}])
    first = path.read_bytes()
    crest.append_rows(path, fields, [{"system": "si13", "seed": "2000"}])
    assert path.read_bytes().startswith(first)
    assert read(path) == [
        {"system": "si13", "seed": "2000"},
        {"system": "si13", "seed": "2000"},
    ]


def test_the_committed_panel_is_the_same_on_every_seed(capsys):
    pytest.importorskip("scipy.stats")
    arm_gate.summary(argparse.Namespace(rows=PANEL_ROWS, arms=["develop", "floor"]))
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-1] == "floor loses: nowhere"
    assert len(lines) == 13
    assert all(line.endswith("same on 12/12 seeds") for line in lines[:-1])

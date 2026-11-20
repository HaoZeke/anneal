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
    rows = [r for r in read(FLOOR_GATE_ROWS) if r["system"] == "si13"]
    rows = [r for r in rows if not (r["arm"] == "floor" and r["seed"] == "2003")]
    floor_gate.summary(argparse.Namespace(rows=write(tmp_path / "rows.csv", rows), arms=["base", "floor"]))
    assert verdict(capsys) == "floor loses: si13 seeds"


def test_a_system_only_the_base_arm_ran_loses_the_floor_gate(tmp_path, capsys):
    rows = [r for r in read(FLOOR_GATE_ROWS) if not (r["arm"] == "floor" and r["system"] == "c20")]
    rows = [r for r in rows if r["system"] == "c20" or r["arm"] == "base"]
    floor_gate.summary(argparse.Namespace(rows=write(tmp_path / "rows.csv", rows), arms=["base", "floor"]))
    assert verdict(capsys) == "floor loses: si13 seeds, c20 seeds"


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
    assert "si13 base: hits 12/12, median calls to hit 1896.5, SCC failures 1646/27222 (6.05%)" in out
    assert "si13 floor: hits 12/12, median calls to hit 2206.5, SCC failures 878/30392 (2.89%)" in out
    assert out.strip().splitlines()[-1] == "floor loses: nowhere"


def test_a_panel_system_one_arm_left_out_or_short_of_a_seed_loses(tmp_path, capsys):
    rows = [r for r in read(PANEL_ROWS) if r["system"] in ("lj38", "lj55")]
    rows = [r for r in rows if not (r["arm"] == "floor" and r["system"] == "lj38")]
    rows = [r for r in rows if not (r["arm"] == "develop" and r["system"] == "lj55" and r["seed"] == "2005")]
    arm_gate.summary(argparse.Namespace(rows=write(tmp_path / "panel.csv", rows), arms=["develop", "floor"]))
    assert verdict(capsys) == "floor loses: lj38 h2h_gmin_proof seeds, lj55 h2h_gmin_proof seeds"


def test_the_floor_gate_reference_matches_minima():
    pytest.importorskip("scipy.stats")
    crest = load("gfn2_crest_ref", "scripts/covalent_h2h/gfn2_crest.py")
    minima = ROOT / "bench" / "covalent-h2h" / "gfn2" / "minima.csv"
    rows = read(minima)
    expect = {}
    for row in rows:
        assert row["source"].strip()
        expect[row["system"]] = float(row["energy_eV"])
    assert floor_gate.REFERENCE == expect
    assert set(expect) == {"si10", "c10", "c20", "si13", "si20"}
    campaign = {"si10": -460.68373809, "c10": -571.32273783, "c20": -1148.85383633}
    cited = {row["system"]: row["source"] for row in rows}
    for system, energy in campaign.items():
        assert expect[system] == energy == crest.REFERENCE[system]
        assert "scripts/covalent_h2h/gfn2_crest.py" in cited[system]
    assert expect["si13"] == -624.70446339
    assert expect["si20"] == -922.98227162
    assert expect["si13"] != crest.REFERENCE["si13"]
    assert "minima/si13-ring.xyz" in cited["si13"]
    assert "minima/si20-ring.xyz" in cited["si20"]


def test_a_reference_without_a_source_is_refused(tmp_path):
    path = tmp_path / "minima.csv"
    path.write_text("system,energy_eV,source\nc20,-1148.85383633,\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="no reference source"):
        floor_gate.read_minima(path)


def test_si13_summary_hits_are_the_stop_target_score(tmp_path):
    pytest.importorskip("scipy.stats")
    crest = load("gfn2_crest_summary", "scripts/covalent_h2h/gfn2_crest.py")
    gfn2 = ROOT / "bench" / "covalent-h2h" / "gfn2"
    out = tmp_path / "summaries.csv"
    crest.summarise([gfn2 / "rows.csv"], out, "0" * 40, "bench", systems=["si13"])
    fresh = {row["code"]: row for row in read(out)}
    stop = f"{crest.REFERENCE['si13']:.8f}"
    assert fresh["anneal-general"]["stop_target_eV"] == stop
    assert fresh["crest"]["stop_target_eV"] == stop
    assert (fresh["anneal-general"]["runs"], fresh["anneal-general"]["hits"]) == ("6", "6")
    assert (fresh["crest"]["runs"], fresh["crest"]["hits"]) == ("6", "4")
    assert fresh["crest"]["fisher_p"] == "0.454545"
    assert fresh["crest"]["mannwhitney_p"] == "0.00499812"
    committed = [row for row in read(gfn2 / "summaries.csv") if row["system"] == "si13"]
    assert {(row["code"], row["runs"], row["hits"], row["stop_target_eV"]) for row in committed} == {
        ("anneal-general", "6", "6", stop),
        ("crest", "6", "4", stop),
    }


def test_crest_rescore_reproduces_the_committed_row(tmp_path):
    pytest.importorskip("scipy.stats")
    crest = load("gfn2_crest", "scripts/covalent_h2h/gfn2_crest.py")
    gfn2 = ROOT / "bench" / "covalent-h2h" / "gfn2"
    history = [gfn2 / "seeds.csv", gfn2 / "rows.csv"]
    out = tmp_path / "rescore.csv"
    crest.rescore(history, out, "si13", -624.70446339, "0" * 40, "bench")
    crest.rescore(history, out, "si13", crest.REFERENCE["si13"], "0" * 40, "bench")
    ring, reference = read(out)
    keys = ("system", "code", "reference_eV", "runs", "hits", "seeds")
    assert {k: ring[k] for k in keys} == {k: read(gfn2 / "rescore.csv")[0][k] for k in keys}
    assert (reference["runs"], reference["hits"]) == ("6", "4")
    with pytest.raises(SystemExit, match="no CREST rows"):
        crest.rescore(history, out, "si20", -922.98227162, "0" * 40, "bench")


def test_the_committed_panel_is_the_same_on_every_seed(capsys):
    pytest.importorskip("scipy.stats")
    arm_gate.summary(argparse.Namespace(rows=PANEL_ROWS, arms=["develop", "floor"]))
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-1] == "floor loses: nowhere"
    assert len(lines) == 13
    assert all(line.endswith("same on 12/12 seeds") for line in lines[:-1])

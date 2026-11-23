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


def test_gfn2_final_geometry_is_kept_beside_the_row(tmp_path, monkeypatch):
    pytest.importorskip("scipy.stats")
    crest = load("gfn2_geometry", "scripts/covalent_h2h/gfn2_crest.py")
    source = tmp_path / "run" / "best.xyz"
    source.parent.mkdir()
    source.write_text("2\nfinal\nSi 0 0 0\nSi 2.25 0 0\n")
    monkeypatch.setattr(
        crest, "standalone_rescore", lambda _xtb, _geometry: (True, -12.5)
    )
    rows = tmp_path / "results" / "rows-v2.csv"
    row = {"system": "si20", "code": "anneal-general", "seed": 2000}
    fields = crest.record_final_geometry(
        rows, row, source, tmp_path / "xtb", "1" * 40
    )
    geometry = rows.parent / fields["geometry"]
    assert geometry.read_bytes() == source.read_bytes()
    assert fields["standalone_scc_converged"] == "true"
    assert fields["standalone_xtb_eV"] == "-12.50000000"
    assert fields["min_distance_A"] == "2.25000000"


def test_unconverged_standalone_final_cannot_count_as_a_hit():
    crest = load("gfn2_hit", "scripts/covalent_h2h/gfn2_crest.py")
    row = {
        "budget": 100_000,
        "hit": "true",
        "evals_to_hit": 745,
        "standalone_scc_converged": "false",
    }

    crest.enforce_converged_hit(row)

    assert row["hit"] == "false"
    assert row["evals_to_hit"] == 100_000


def test_gfn2_v2_rows_use_the_append_schema():
    crest = load("gfn2_schema", "scripts/covalent_h2h/gfn2_crest.py")
    path = ROOT / "bench" / "covalent-h2h" / "gfn2" / "rows-v2.csv"

    with path.open(newline="") as handle:
        header = next(csv.reader(handle))

    assert header == crest.ROW_FIELDS + crest.PROVENANCE_FIELDS


def test_the_committed_panel_is_the_same_on_every_seed(capsys):
    pytest.importorskip("scipy.stats")
    arm_gate.summary(argparse.Namespace(rows=PANEL_ROWS, arms=["develop", "floor"]))
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-1] == "floor loses: nowhere"
    assert len(lines) == 13
    assert all(line.endswith("same on 12/12 seeds") for line in lines[:-1])

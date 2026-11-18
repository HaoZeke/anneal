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
    floor_gate.summary(argparse.Namespace(rows=[write(tmp_path / "rows.csv", rows)], arms=["base", "floor"]))
    assert verdict(capsys) == "floor loses: si13 seeds"


def test_a_system_only_the_base_arm_ran_loses_the_floor_gate(tmp_path, capsys):
    rows = [r for r in read(FLOOR_GATE_ROWS) if not (r["arm"] == "floor" and r["system"] == "c20")]
    rows = [r for r in rows if r["system"] == "c20" or r["arm"] == "base"]
    floor_gate.summary(argparse.Namespace(rows=[write(tmp_path / "rows.csv", rows)], arms=["base", "floor"]))
    assert verdict(capsys) == "floor loses: si13 seeds, c20 seeds"


def test_a_floor_gate_run_without_calls_is_summarised(tmp_path, capsys):
    pytest.importorskip("scipy.stats")
    rows = [r for r in read(FLOOR_GATE_ROWS) if r["system"] == "c20"]
    for row in rows:
        row["charged"] = "0"
    floor_gate.summary(argparse.Namespace(rows=[write(tmp_path / "rows.csv", rows)], arms=["base", "floor"]))
    out = capsys.readouterr().out
    assert "SCC failures 0/0 (no calls)" in out
    assert out.strip().splitlines()[-1] == "floor loses: nowhere"


def test_a_floor_gate_run_takes_the_cores_from_the_first_core(tmp_path, monkeypatch):
    cores = []

    def run_one(binary, system, seed, budget, core, where):
        cores.append(core)
        return None

    monkeypatch.setattr(floor_gate, "run_one", run_one)
    floor_gate.run(argparse.Namespace(
        arm=["grad=/bin/true@" + "0" * 40], systems=["si13"], seed0=2000, n=4,
        budget=1, cores=2, first_core=3, out=tmp_path, rows=tmp_path / "rows.csv"))
    assert len(cores) == 4
    assert set(cores) <= {3, 4}


def test_a_floor_gate_run_refuses_a_rows_file_with_another_header(tmp_path, monkeypatch):
    rows = tmp_path / "rows.csv"
    rows.write_text(",".join(["arm", *floor_gate.EXAMPLE_FIELDS[:-1], "code_revision", "host"]) + "\n")
    before = rows.read_bytes()
    monkeypatch.setattr(floor_gate, "run_one", lambda *args: pytest.fail("ran a job"))
    monkeypatch.setattr("sys.argv", [
        "gfn2_floor_gate.py", "run", "--arm", "scc=/bin/true@" + "0" * 40, "--n", "1",
        "--cores", "1", "--out", str(tmp_path), "--rows", str(rows)])
    with pytest.raises(SystemExit):
        floor_gate.main()
    assert rows.read_bytes() == before


def test_a_floor_gate_row_of_another_width_is_not_appended(tmp_path, monkeypatch, capsys):
    rows = tmp_path / "rows.csv"
    row = next(r for r in read(FLOOR_GATE_ROWS) if r["arm"] == "grad")
    old = [row[f] for f in floor_gate.EXAMPLE_FIELDS[:-1]]
    monkeypatch.setattr(floor_gate, "run_one", lambda *args: old)
    floor_gate.run(argparse.Namespace(
        arm=["grad=/bin/true@" + "0" * 40], systems=["si13"], seed0=2000, n=1,
        budget=1, cores=1, first_core=0, out=tmp_path, rows=rows))
    assert not rows.exists()
    assert "16 fields, not 17" in capsys.readouterr().out


def test_a_floor_gate_reads_scc_failed_calls_as_the_per_call_rate(tmp_path, capsys):
    pytest.importorskip("scipy.stats")
    rows = [r for r in read(FLOOR_GATE_ROWS) if r["arm"] in ("base", "floor")]
    for row in rows:
        row["scc_rejected"] = row["scc_failed_calls"] = str(int(row["scc_rejected"]) // 2)
    floor_gate.summary(argparse.Namespace(rows=[write(tmp_path / "rows.csv", rows)], arms=["base", "floor"]))
    out = capsys.readouterr().out
    assert "si13 base: hits 12/12, median calls to hit 1896.5, SCC-failed calls 823/27222 (3.02%), 823 xtb evaluations" in out
    assert "si13 floor: hits 12/12, median calls to hit 2206.5, SCC-failed calls 439/30392 (1.44%), 439 xtb evaluations" in out
    assert out.strip().splitlines()[-1] == "floor loses: nowhere"


def test_a_floor_gate_across_rows_files_counts_failures_per_arm(tmp_path, capsys):
    pytest.importorskip("scipy.stats")
    rows = [r for r in read(FLOOR_GATE_ROWS) if r["arm"] == "grad"]
    halved = [str(int(r["scc_rejected"]) // 2) for r in rows]
    per_call = [dict(r, arm="scc", scc_rejected=h, scc_failed_calls=h) for r, h in zip(rows, halved)]
    floor_gate.summary(argparse.Namespace(
        rows=[FLOOR_GATE_ROWS, write(tmp_path / "rows.csv", per_call)], arms=["grad", "scc"]))
    out = capsys.readouterr().out
    assert "si13 grad: hits 12/12, median calls to hit 1972.5, SCC failures 1424/28585 (4.98%)" in out
    assert "si13 scc: hits 12/12, median calls to hit 1972.5, SCC-failed calls 712/28585 (2.49%), 712 xtb evaluations" in out


def test_the_committed_floor_gate_loses_nowhere(capsys):
    pytest.importorskip("scipy.stats")
    floor_gate.summary(argparse.Namespace(rows=[FLOOR_GATE_ROWS], arms=["base", "floor"]))
    out = capsys.readouterr().out
    assert "si13 base: hits 12/12, median calls to hit 1896.5, SCC failures 1646/27222 (6.05%)" in out
    assert "si13 floor: hits 12/12, median calls to hit 2206.5, SCC failures 878/30392 (2.89%)" in out
    assert out.strip().splitlines()[-1] == "floor loses: nowhere"


def test_the_rerun_arms_agree_and_repeat_the_committed_c20_rows():
    fields = ("hit", "evals_to_hit", "charged", "best", "scc_rejected", "geometry_rejected")
    seen = {}
    for row in read(FLOOR_GATE_ROWS):
        seen.setdefault((row["system"], row["seed"]), {})[row["arm"]] = tuple(row[f] for f in fields)
    assert len(seen) == 24
    for (system, _), arms in seen.items():
        assert arms["floor-rerun"] == arms["series"] == arms["grad"]
        if system == "c20":
            assert arms["grad"] == arms["floor"]
        else:
            assert arms["grad"][1] != arms["floor"][1]


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


def test_the_committed_panel_is_the_same_on_every_seed(capsys):
    pytest.importorskip("scipy.stats")
    arm_gate.summary(argparse.Namespace(rows=PANEL_ROWS, arms=["develop", "floor"]))
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-1] == "floor loses: nowhere"
    assert len(lines) == 13
    assert all(line.endswith("same on 12/12 seeds") for line in lines[:-1])

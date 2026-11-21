# Copyright (c) anneal developers
"""A benchmark row names its host by label, never by the machine's own hostname."""

import os
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1] / "scripts" / "openconf_h2h"
sys.path.insert(0, str(HARNESS))

import bench_host  # noqa: E402


def uname_as(monkeypatch, nodename):
    real = os.uname()
    monkeypatch.setattr(
        bench_host.os,
        "uname",
        lambda: os.uname_result((real.sysname, nodename, real.release, real.version, real.machine)),
    )


def test_a_set_label_is_used_as_given(monkeypatch):
    uname_as(monkeypatch, "workstation-17")
    monkeypatch.setenv("BENCH_HOST", "cloud-vm")
    assert bench_host.host_label() == "cloud-vm"


def test_an_unlabelled_machine_is_not_named(monkeypatch):
    monkeypatch.delenv("BENCH_HOST", raising=False)
    uname_as(monkeypatch, "workstation-17")
    assert bench_host.host_label() == "unlabelled"
    monkeypatch.setenv("BENCH_HOST", "  ")
    assert bench_host.host_label() == "unlabelled"


def test_the_bench_namespace_keeps_its_name(monkeypatch):
    monkeypatch.delenv("BENCH_HOST", raising=False)
    uname_as(monkeypatch, "bench")
    assert bench_host.host_label() == "bench"


def test_no_harness_script_reads_the_machine_name():
    for path in sorted(HARNESS.glob("*.py")):
        if path.name == "bench_host.py":
            continue
        text = path.read_text()
        assert "nodename" not in text and "gethostname" not in text, path.name
        assert '"hostname"' not in text, path.name


def test_a_job_without_a_host_reads_as_unlabelled():
    assert bench_host.run_host({"host": "cloud-vm"}) == "cloud-vm"
    assert bench_host.run_host({}) == "unlabelled"
    assert bench_host.run_host({"host": None}) == "unlabelled"
    assert bench_host.run_host({"host": " "}) == "unlabelled"
    assert bench_host.run_host({"hostname": "workstation-17"}) == "unlabelled"


def test_scoring_a_job_written_before_the_host_label_does_not_raise(tmp_path, monkeypatch):
    import csv
    import json

    import numpy as np
    import pytest

    pytest.importorskip("rdkit")
    pytest.importorskip("scipy")
    import h2h_common as hc
    import paired_score
    from h2h_score import Molecule

    monkeypatch.setattr(hc, "molecules", lambda: {"fccf": {"name": "fccf", "smiles": "FCCF"}})
    paired_score._MOLS.clear()
    m = Molecule({"smiles": "FCCF"})
    hc.embed(m.mol, 3)
    x, e, valid = m.polish(np.asarray(m.mol.GetConformer().GetPositions()))
    assert valid
    job_dir = tmp_path / "runs" / "fccf" / "1" / "anneal"
    job_dir.mkdir(parents=True)
    old = {
        "molecule": "fccf", "arm": "anneal", "seed": 1, "energy_calls": 10, "grad_calls": 10,
        "wall_s": 1.0, "pot_s": 0.5, "outside_s": 0.5, "hostname": "workstation-17",
    }
    (job_dir / "job.json").write_text(json.dumps(old))
    np.savez(job_dir / "polished.npz", x=x[None], e=np.array([e]), ok=np.array([True]))
    rows = tmp_path / "rows.csv"
    paired_score.score_runs(tmp_path / "runs", rows, 1)
    out = list(csv.DictReader(rows.open()))
    assert [r["host"] for r in out] == ["unlabelled"]
    assert "workstation-17" not in rows.read_text()

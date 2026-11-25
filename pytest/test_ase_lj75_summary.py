# Copyright (c) anneal developers
"""The LJ75 summary selects one campaign row per seed."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ase_lj75" / "summarize.py"
SPEC = importlib.util.spec_from_file_location("ase_lj75_summary", SCRIPT)
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def test_cpu_cap_row_replaces_older_row_for_seed(monkeypatch, tmp_path):
    with SUMMARY.ROWS.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = tuple(rows[0])
    replacement = dict(next(
        row for row in rows
        if row["arm"] == "ase_mh" and row["seed"] == "2000"
    ))
    replacement["cap_s"] = "43200.000000"
    replacement["stop"] = "cap reached (12 h CPU)"
    replacement["best"] = "-397.0"
    rows.append(replacement)

    path = tmp_path / "rows.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(SUMMARY, "ROWS", path)

    grouped = SUMMARY.load()
    selected = next(
        row for row in grouped["ase_mh"] if row["seed"] == "2000"
    )
    assert selected["cap_s"] == "43200.000000"
    assert selected["best"] == "-397.0"

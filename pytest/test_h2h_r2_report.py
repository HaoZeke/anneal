# Copyright (c) anneal developers
"""The TIP4P ABBA harness writes its records once and reports mismatches."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench" / "tip4p-h2h"


def load_harness():
    spec = importlib.util.spec_from_file_location("run_h2h_r2", BENCH / "run_h2h_r2.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def report_args(harness, rows: Path, out: Path):
    return harness.build_parser().parse_args(
        [
            "report",
            "--rows",
            str(rows),
            "--reference-rows",
            str(BENCH / "rows-h2h.csv"),
            "--check-out",
            str(out / "check.md"),
            "--wall-out",
            str(out / "wall.csv"),
        ]
    )


def test_report_regenerates_the_committed_summaries(tmp_path):
    harness = load_harness()
    args = report_args(harness, BENCH / "rows-h2h-r2.csv", tmp_path)
    assert args.handler(args) == 0
    assert (tmp_path / "wall.csv").read_text() == (BENCH / "wall-r2.csv").read_text()
    assert (tmp_path / "check.md").read_text() == (BENCH / "check-r2.md").read_text()


def test_report_marks_a_size_without_a_reference_unscored(tmp_path):
    harness = load_harness()
    args = report_args(harness, BENCH / "rows-h2h-r2.csv", tmp_path)
    assert args.handler(args) == 0
    with (tmp_path / "wall.csv").open() as handle:
        rows = {int(row["molecules"]): row for row in csv.DictReader(handle)}
    assert rows[25]["anneal_hits_n"] == "unscored"
    assert rows[25]["gmin_hits_n"] == "unscored"
    assert rows[21]["anneal_hits_n"] == "0/12"
    assert rows[21]["gmin_hits_n"] == "0/12"


def test_report_is_idempotent_and_refuses_to_rewrite(tmp_path):
    harness = load_harness()
    args = report_args(harness, BENCH / "rows-h2h-r2.csv", tmp_path)
    assert args.handler(args) == 0
    assert args.handler(args) == 0
    (tmp_path / "wall.csv").write_text("changed\n")
    with pytest.raises(SystemExit, match="refusing to rewrite"):
        args.handler(args)
    assert (tmp_path / "wall.csv").read_text() == "changed\n"


def test_report_exits_nonzero_when_a_row_departs_from_the_reference(tmp_path):
    harness = load_harness()
    with (BENCH / "rows-h2h-r2.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["calls"] = str(int(rows[0]["calls"]) + 1)
    tampered = tmp_path / "rows.csv"
    with tampered.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    args = report_args(harness, tampered, tmp_path)
    assert args.handler(args) == 1
    assert "- 8,2000,anneal,hit:1!=1,calls:15600!=15599" in (tmp_path / "check.md").read_text()


def test_run_refuses_existing_records(tmp_path):
    harness = load_harness()
    kept = tmp_path / "rows.csv"
    kept.write_text("kept\n")
    with pytest.raises(SystemExit, match="refusing to overwrite existing records"):
        harness.refuse_existing(kept, tmp_path / "sessions.jsonl")
    assert kept.read_text() == "kept\n"
    harness.refuse_existing(tmp_path / "absent.csv")

# Copyright (c) anneal developers
"""Consolidated G2 budget rows match the single-seed cells."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_g2_budget_rows as rows  # noqa: E402


def test_strip_drops_the_named_field_and_keeps_the_other_bytes():
    original = (
        '{"arm": "ase_mh", "machine_id_sha256": "abc", '
        '"model_name": "Intel(R) Xeon(R) Processor", "stop": "hit"}'
    )
    replaced = (
        '{"arm": "ase_mh", "host": "host-1", '
        '"model_name": "Intel(R) Xeon(R) Processor", "stop": "hit"}'
    )
    stripped = rows.strip_field(original, "machine_id_sha256")
    assert stripped == rows.strip_field(replaced, "host")
    assert "machine_id_sha256" not in stripped
    assert json.loads(stripped)["stop"] == "hit"
    assert json.loads(stripped)["model_name"] == "Intel(R) Xeon(R) Processor"


def test_absent_rows_are_an_error(tmp_path):
    assert rows.check(tmp_path) == ["no consolidated rows"]


def test_consolidated_rows_match_the_manifest():
    root = rows.resolve_root()
    errors = rows.check(root)
    assert errors == [], "\n".join(errors)

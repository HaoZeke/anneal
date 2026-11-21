# Copyright (c) anneal developers
"""Regression tests for OCCat campaign result parsing."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "occat_h2h" / "run_campaign.py"
SPEC = importlib.util.spec_from_file_location("occat_h2h_run_campaign", RUNNER)
CAMPAIGN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAMPAIGN)


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (
            "pt38,anneal,2000,400000,true,123,123,2.5,-199.4,1.25,1.25,0.01,0.01\n",
            "1.25",
        ),
        (
            "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,"
            "pot_s,outside_s,pot_per_eval_s,outside_per_eval_s,status\n"
            "pt38,anneal,2000,400000,true,123,123,2.5,-199.4,"
            "1.5,1.0,0.01,0.01,ok\n",
            "1.5",
        ),
        (
            "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,"
            "pot_s,outside_s,pot_per_eval_s,outside_per_eval_s,status,search_error\n"
            "pt38,anneal,2000,400000,false,400000,123,2.5,inf,"
            "1.75,0.75,0.01,0.01,failed,no finite minimum\n",
            "1.75",
        ),
    ],
    ids=("old-13-column", "status-14-column", "extended-15-column"),
)
def test_parse_metal_rows_by_header_or_legacy_schema(output: str, expected: str):
    row = CAMPAIGN.parse_anneal_row(output, "pt38", "pt38", "gupta")
    assert row["pot_s"] == expected
